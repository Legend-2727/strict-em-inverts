"""
KV Cache Steering for VLMs.

This module implements contrastive KV cache steering:
1.  **Extraction** - compute contrastive steering vectors from positive/negative
    prompt pairs by extracting and diffing their KV caches.
2.  **Application** - apply steering vectors to a pre-filled KV cache.
3.  **Generation** - generate text with steering applied transparently.

All functions operate on ``transformers.DynamicCache`` objects and are
model-agnostic (tested with Qwen2.5-VL).
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import DynamicCache, AutoProcessor
from PIL import Image

logger = logging.getLogger(__name__)


# =============================================================================
# VLM INPUT PROCESSING
# =============================================================================

def prepare_vlm_inputs(
    processor: AutoProcessor,
    messages: List[Dict],
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """
    Prepare inputs for Qwen2.5-VL from message format.

    Args:
        processor: Qwen2.5-VL processor.
        messages: List of message dicts in Qwen format.
        device: Target device.

    Returns:
        Dictionary of model inputs (input_ids, attention_mask, pixel_values, …).
    """
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Manually extract images (qwen_vl_utils is optional)
    image_inputs: list = []
    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    img_data = item.get("image")
                    if img_data is not None and isinstance(img_data, Image.Image):
                        image_inputs.append(img_data)
    video_inputs = None
    if not image_inputs:
        image_inputs = None

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return inputs.to(device)


# =============================================================================
# KV CACHE EXTRACTION
# =============================================================================

def get_kv_cache_from_inputs(
    model,
    inputs: Dict[str, torch.Tensor],
) -> DynamicCache:
    """
    Run a forward pass and return the KV cache.

    Args:
        model: Qwen2.5-VL model.
        inputs: Prepared inputs dictionary (from :func:`prepare_vlm_inputs`).

    Returns:
        ``DynamicCache`` containing keys and values for all layers.
    """
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
    return outputs.past_key_values


def extract_kv(
    kv_cache: DynamicCache,
    location: int = -1,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """
    Extract a single token's keys and values from all layers.

    Args:
        kv_cache: ``DynamicCache`` from a forward pass.
        location: Token index to extract (``-1`` for last token).

    Returns:
        ``(keys_dict, values_dict)`` mapping ``layer_idx → [num_heads, head_dim]``.
    """
    keys_last: Dict[int, torch.Tensor] = {}
    values_last: Dict[int, torch.Tensor] = {}

    if hasattr(kv_cache, "layers"):
        num_layers = len(kv_cache.layers)
        for layer_idx in range(num_layers):
            k = kv_cache.layers[layer_idx].keys
            v = kv_cache.layers[layer_idx].values
            keys_last[layer_idx] = k[:, :, location, :].squeeze(0)
            values_last[layer_idx] = v[:, :, location, :].squeeze(0)
    elif hasattr(kv_cache, "key_cache"):
        num_layers = len(kv_cache.key_cache)
        for layer_idx in range(num_layers):
            k = kv_cache.key_cache[layer_idx]
            v = kv_cache.value_cache[layer_idx]
            keys_last[layer_idx] = k[:, :, location, :].squeeze(0)
            values_last[layer_idx] = v[:, :, location, :].squeeze(0)
    else:
        raise ValueError("Unknown cache format")

    return keys_last, values_last


# =============================================================================
# CONTRASTIVE STEERING VECTOR EXTRACTION
# =============================================================================

def extract_steering_kv_from_contrastive_pairs(
    model,
    processor: AutoProcessor,
    contrastive_pairs: List[Dict],
    device: str = "cuda",
    aggregation_method: str = "mean",
    location: int = -1,
) -> Dict[str, Dict[int, torch.Tensor]]:
    """
    Extract contrastive KV steering vectors from contrastive pairs.

    For each layer computes::

        key_steering   = aggregate(keys_positive)   - aggregate(keys_negative)
        value_steering = aggregate(values_positive)  - aggregate(values_negative)

    Args:
        model: VLM model.
        processor: VLM processor.
        contrastive_pairs: List of dicts with ``'positive'`` and ``'negative'`` message lists.
        device: Target device.
        aggregation_method: ``"mean"`` supported.
        location: Token index to extract (``-1`` → last token).

    Returns:
        ``{"keys": {layer_idx: tensor}, "values": {layer_idx: tensor}}``.
    """
    logger.info(
        f"Extracting steering vectors from {len(contrastive_pairs)} contrastive pairs...")

    keys_pos_all: Dict[int, list] = defaultdict(list)
    values_pos_all: Dict[int, list] = defaultdict(list)
    keys_neg_all: Dict[int, list] = defaultdict(list)
    values_neg_all: Dict[int, list] = defaultdict(list)

    for pair_idx, pair in enumerate(tqdm(contrastive_pairs, desc="Extracting KV caches")):
        positive_messages = pair["positive"]
        negative_messages = pair["negative"]

        try:
            pos_inputs = prepare_vlm_inputs(
                processor, positive_messages, device)
            pos_cache = get_kv_cache_from_inputs(model, pos_inputs)
            pos_keys, pos_values = extract_kv(pos_cache, location=location)
            for layer_idx in pos_keys:
                keys_pos_all[layer_idx].append(pos_keys[layer_idx].cpu())
                values_pos_all[layer_idx].append(pos_values[layer_idx].cpu())
            del pos_cache, pos_inputs

            neg_inputs = prepare_vlm_inputs(
                processor, negative_messages, device)
            neg_cache = get_kv_cache_from_inputs(model, neg_inputs)
            neg_keys, neg_values = extract_kv(neg_cache, location=location)
            for layer_idx in neg_keys:
                keys_neg_all[layer_idx].append(neg_keys[layer_idx].cpu())
                values_neg_all[layer_idx].append(neg_values[layer_idx].cpu())
            del neg_cache, neg_inputs
            torch.cuda.empty_cache()

        except Exception as e:
            logger.warning(f"Error processing pair {pair_idx}: {e}")
            continue

    if not keys_pos_all:
        raise ValueError("No steering vectors could be extracted!")

    num_layers = len(keys_pos_all)
    key_steering: Dict[int, torch.Tensor] = {}
    value_steering: Dict[int, torch.Tensor] = {}

    for layer_idx in range(num_layers):
        k_pos = torch.stack(keys_pos_all[layer_idx], dim=0)
        k_neg = torch.stack(keys_neg_all[layer_idx], dim=0)
        v_pos = torch.stack(values_pos_all[layer_idx], dim=0)
        v_neg = torch.stack(values_neg_all[layer_idx], dim=0)

        if aggregation_method == "mean":
            k_pos_agg = k_pos.mean(dim=0)
            k_neg_agg = k_neg.mean(dim=0)
            v_pos_agg = v_pos.mean(dim=0)
            v_neg_agg = v_neg.mean(dim=0)
        else:
            raise ValueError(
                f"Unsupported aggregation method: {aggregation_method}")

        key_steering[layer_idx] = k_pos_agg - k_neg_agg
        value_steering[layer_idx] = v_pos_agg - v_neg_agg

    logger.info(f"Computed steering vectors for {num_layers} layers")
    key_norms = [key_steering[i].norm().item() for i in range(num_layers)]
    value_norms = [value_steering[i].norm().item() for i in range(num_layers)]
    logger.info(f"  Key vector mean norm: {sum(key_norms)/len(key_norms):.4f}")
    logger.info(
        f"  Value vector mean norm: {sum(value_norms)/len(value_norms):.4f}")

    return {"keys": key_steering, "values": value_steering}


# =============================================================================
# KV CACHE STEERING APPLICATION
# =============================================================================

def steer_kv_cache(
    kv_cache: DynamicCache,
    key_steer_vectors: Dict[int, torch.Tensor],
    value_steer_vectors: Dict[int, torch.Tensor],
    key_scale: float = 0.0,
    value_scale: float = 1.0,
    steer_layers: Optional[List[int]] = None,
    application_token_idx: int = -1,
) -> DynamicCache:
    """
    Apply steering vectors to a KV cache **in-place**.

    Args:
        kv_cache: The KV cache to steer.
        key_steer_vectors: ``{layer_idx: tensor}`` for keys.
        value_steer_vectors: ``{layer_idx: tensor}`` for values.
        key_scale: Scaling factor for key steering.
        value_scale: Scaling factor for value steering.
        steer_layers: Which layers to steer (``None`` → all).
        application_token_idx: Token position to apply steering to (``-1`` → last).

    Returns:
        The modified ``DynamicCache``.
    """
    if hasattr(kv_cache, "layers"):
        num_layers = len(kv_cache.layers)
        use_layers_attr = True
    elif hasattr(kv_cache, "key_cache"):
        num_layers = len(kv_cache.key_cache)
        use_layers_attr = False
    else:
        raise ValueError("Unknown cache format")

    layers_to_steer = steer_layers if steer_layers is not None else list(
        range(num_layers))

    for layer_idx in layers_to_steer:
        if layer_idx >= num_layers or layer_idx not in key_steer_vectors:
            continue

        if use_layers_attr:
            k_cache = kv_cache.layers[layer_idx].keys
            v_cache = kv_cache.layers[layer_idx].values
        else:
            k_cache = kv_cache.key_cache[layer_idx]
            v_cache = kv_cache.value_cache[layer_idx]

        k_steer = key_steer_vectors[layer_idx].to(
            device=k_cache.device, dtype=k_cache.dtype)
        v_steer = value_steer_vectors[layer_idx].to(
            device=v_cache.device, dtype=v_cache.dtype)

        if use_layers_attr:
            kv_cache.layers[layer_idx].keys[:, :,
                                            application_token_idx, :] += key_scale * k_steer
            kv_cache.layers[layer_idx].values[:, :,
                                              application_token_idx, :] += value_scale * v_steer
        else:
            kv_cache.key_cache[layer_idx][:, :,
                                          application_token_idx, :] += key_scale * k_steer
            kv_cache.value_cache[layer_idx][:, :,
                                            application_token_idx, :] += value_scale * v_steer

    return kv_cache


# =============================================================================
# PREFILL
# =============================================================================

def precompute_kv_cache_vlm(
    model,
    inputs: Dict[str, torch.Tensor],
) -> DynamicCache:
    """
    Pre-fill KV cache for all but the last token (VLM inputs).

    Computes the cache for the context tokens so that steering can be applied
    to the last cached position before generation continues.

    Args:
        model: VLM model.
        inputs: Full input dict (input_ids, attention_mask, pixel_values, …).

    Returns:
        ``DynamicCache`` with prefill values.
    """
    past_key_values = DynamicCache()

    sequence_keys = ["input_ids", "attention_mask",
                     "token_type_ids", "position_ids"]
    visual_keys = ["pixel_values", "image_grid_thw",
                   "video_grid_thw", "pixel_values_videos"]

    cache_input: dict = {}
    for k, v in inputs.items():
        if k in sequence_keys:
            cache_input[k] = v[:, :-1]
        elif k in visual_keys:
            cache_input[k] = v

    with torch.no_grad():
        model(**cache_input, past_key_values=past_key_values, use_cache=True)

    return past_key_values


# =============================================================================
# STEERED GENERATION
# =============================================================================

def generate_with_kv_steering(
    model,
    processor: AutoProcessor,
    messages: List[Dict],
    key_steer_vectors: Optional[Dict[int, torch.Tensor]] = None,
    value_steer_vectors: Optional[Dict[int, torch.Tensor]] = None,
    key_scale: float = 0.0,
    value_scale: float = 1.0,
    steer_layers: Optional[List[int]] = None,
    max_new_tokens: int = 64,
    device: str = "cuda",
) -> str:
    """
    Generate text with optional KV cache steering applied.

    When steering vectors are provided the pipeline is:

    1. **Prefill** - compute KV cache for all but last token.
    2. **Steer** - add scaled steering vectors to the last cached position.
    3. **Generate** - continue generation from the steered cache.

    Without steering vectors, standard ``model.generate`` is used.

    Args:
        model: VLM model.
        processor: Model processor.
        messages: Messages in Qwen format.
        key_steer_vectors: Optional key steering vectors.
        value_steer_vectors: Optional value steering vectors.
        key_scale: Scale for key steering.
        value_scale: Scale for value steering.
        steer_layers: Which layers to steer (``None`` → all).
        max_new_tokens: Maximum tokens to generate.
        device: Target device.

    Returns:
        Generated text response.
    """
    inputs = prepare_vlm_inputs(processor, messages, device)

    apply_steering = key_steer_vectors is not None and value_steer_vectors is not None

    generation_kwargs = {
        k: v
        for k, v in inputs.items()
        if k not in ["input_ids", "attention_mask", "position_ids"]
    }

    with torch.no_grad():
        if apply_steering:
            past_key_values = precompute_kv_cache_vlm(model, inputs)
            past_key_values = steer_kv_cache(
                past_key_values,
                key_steer_vectors,
                value_steer_vectors,
                key_scale=key_scale,
                value_scale=value_scale,
                steer_layers=steer_layers,
                application_token_idx=-1,
            )
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                position_ids=inputs.get("position_ids"),
                past_key_values=past_key_values,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                **generation_kwargs,
            )
        else:
            generated_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens)

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return output_text[0].lstrip()
