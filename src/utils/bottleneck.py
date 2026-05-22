"""Evidence bottleneck helpers for image-token suppression during answer loss."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch


_IMAGE_TOKEN_CANDIDATES = [
    "<|image_pad|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|image_start|>",
    "<|image_end|>",
    "<|image|>",
    "<image>",
]


def resolve_image_token_ids(tokenizer) -> List[int]:
    """Best-effort discovery of image-related token IDs for Qwen-VL tokenizers."""
    ids: set[int] = set()
    unk_id = getattr(tokenizer, "unk_token_id", None)

    vocab = {}
    if hasattr(tokenizer, "get_vocab"):
        try:
            vocab = tokenizer.get_vocab() or {}
        except Exception:
            vocab = {}

    for token in _IMAGE_TOKEN_CANDIDATES:
        token_id = None
        if token in vocab:
            token_id = vocab[token]
        else:
            try:
                token_id = tokenizer.convert_tokens_to_ids(token)
            except Exception:
                token_id = None

        if token_id is None:
            continue
        if unk_id is not None and token_id == unk_id and token not in vocab:
            continue
        if int(token_id) >= 0:
            ids.add(int(token_id))

    for token in getattr(tokenizer, "all_special_tokens", []) or []:
        token_text = str(token)
        lowered = token_text.lower()
        if "image" not in lowered and "vision" not in lowered:
            continue
        try:
            token_id = tokenizer.convert_tokens_to_ids(token_text)
        except Exception:
            continue
        if token_id is None:
            continue
        if unk_id is not None and token_id == unk_id and token_text not in vocab:
            continue
        if int(token_id) >= 0:
            ids.add(int(token_id))

    return sorted(ids)


def build_output_label_masks(
    *,
    input_ids: torch.Tensor,
    tokenizer,
    target_text: str,
    evidence_text: Optional[str] = None,
) -> Dict[str, Any]:
    """Create full/evidence/answer label masks for teacher-forced training."""
    if input_ids.ndim != 2:
        raise ValueError("input_ids must be rank-2 [batch, seq]")

    output_token_count = len(tokenizer.encode(target_text, add_special_tokens=False)) + 2
    seq_len = int(input_ids.shape[1])
    start_idx = max(0, seq_len - output_token_count)

    labels_full = torch.full_like(input_ids, -100)
    labels_full[:, start_idx:] = input_ids[:, start_idx:]

    evidence_token_count = 0
    if evidence_text:
        evidence_token_count = max(1, len(tokenizer.encode(evidence_text, add_special_tokens=False)))

    evidence_end = min(seq_len, start_idx + evidence_token_count)

    labels_evidence = torch.full_like(input_ids, -100)
    labels_answer = torch.full_like(input_ids, -100)

    if evidence_token_count > 0 and evidence_end > start_idx:
        labels_evidence[:, start_idx:evidence_end] = input_ids[:, start_idx:evidence_end]
    if evidence_end < seq_len:
        labels_answer[:, evidence_end:] = input_ids[:, evidence_end:]

    evidence_label_tokens = int((labels_evidence != -100).sum().item())
    answer_label_tokens = int((labels_answer != -100).sum().item())
    full_label_tokens = int((labels_full != -100).sum().item())

    return {
        "labels_full": labels_full,
        "labels_evidence": labels_evidence,
        "labels_answer": labels_answer,
        "output_token_count": int(output_token_count),
        "evidence_token_count": int(evidence_token_count),
        "output_start_idx": int(start_idx),
        "evidence_end_idx": int(evidence_end),
        "full_label_tokens": full_label_tokens,
        "evidence_label_tokens": evidence_label_tokens,
        "answer_label_tokens": answer_label_tokens,
    }


def forward_with_image_token_zeroing(
    *,
    model,
    inputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    image_token_ids: List[int],
    **forward_kwargs,
):
    """Run forward pass while zeroing image token embeddings via a temporary hook."""
    if not image_token_ids:
        return model(**inputs, labels=labels, **forward_kwargs)

    input_ids = inputs.get("input_ids")
    if input_ids is None:
        return model(**inputs, labels=labels, **forward_kwargs)

    image_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for token_id in image_token_ids:
        image_mask |= input_ids.eq(int(token_id))

    if not bool(image_mask.any().item()):
        return model(**inputs, labels=labels, **forward_kwargs)

    embedding_layer = model.get_input_embeddings()

    def _hook(_module, _inputs, outputs):
        if not torch.is_tensor(outputs):
            return outputs
        masked = outputs.clone()
        masked[image_mask] = 0.0
        return masked

    handle = embedding_layer.register_forward_hook(_hook)
    try:
        return model(**inputs, labels=labels, **forward_kwargs)
    finally:
        handle.remove()
