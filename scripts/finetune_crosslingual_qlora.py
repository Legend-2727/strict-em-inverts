#!/usr/bin/env python3
"""
Cross-Lingual Attention-Aligned QLoRA (CAAL-QLoRA) Fine-Tuning
==============================================================

**Novel contribution**: Standard VQA fine-tuning optimises only the
language-modelling loss, which does not explicitly address the
cross-lingual representation gap identified by our VAR analysis.  This
script adds a **Cross-Lingual Attention Alignment (CAAL) loss** that
penalises divergence between the hidden-state representations produced
for paired English/Hindi inputs at the same query position, targeting
the deep transformer layers (19-27) where VAR analysis found the largest
cross-lingual shift norms (0.7–3.7 vs <0.2 in early layers).

By aligning the decoder-layer outputs that are produced by the
attention + FFN pipeline, we implicitly align the attention patterns
that generate them — without ever materialising the expensive
[heads, seq, seq] attention matrices.

The combined objective is:

.. math::

    \\mathcal{L} = \\mathcal{L}_{\\text{LM}} + \\lambda \\cdot \\mathcal{L}_{\\text{CAAL}}

where :math:`\\mathcal{L}_{\\text{CAAL}}` is the mean cosine distance of
hidden-state representations at the last token between paired EN/HI
samples, averaged over targeted layers.

**Key design choices**:

1. QLoRA (NF4 base + LoRA adapters) keeps VRAM ≤ 20 GiB on A100 40 GB.
2. LoRA targets attention projection layers (`q_proj`, `v_proj`) in
   the language model only — the vision encoder is frozen.
3. CAAL loss hooks into **decoder layer outputs** (not attention
   matrices), so it works with flash attention and gradient
   checkpointing.  No ``output_attentions=True`` is needed.
4. CAAL piggybacks on the LM forward passes (2 passes total, not 4),
   halving training memory vs a naïve approach.
5. Train/eval split: Since MMCricBench has no training split, we hold
   out 20% of paired samples for validation.
6. Gradient checkpointing reduces peak memory during backprop.

Usage (A100 40 GB)::

    python scripts/finetune_crosslingual_qlora.py \\
        --model-size 7b --epochs 3 --lr 2e-4 \\
        --caal-lambda 0.1 --caal-layers 19 20 21 22 23 24 25 26 27 \\
        --output-dir results/finetune_caal

Usage (A100 80 GB — faster, larger batch)::

    python scripts/finetune_crosslingual_qlora.py \\
        --model-size 7b --epochs 5 --lr 1e-4 --batch-size 2 \\
        --caal-lambda 0.1 --caal-layers 19 20 21 22 23 24 25 26 27 \\
        --output-dir results/finetune_caal
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Ensure the repo root is importable when running from Colab/scripts/
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data import load_mmcricbench, language_from_id
from src.data.prompts import NEGATIVE_PROMPT_TEMPLATE
from src.evaluation import (
    create_paired_dataset,
    evaluate_paired_model,
    prepare_fixed_paired_eval_split,
    exclude_pairs,
    load_fixed_eval_split,
)
from src.utils.checkpoints import (
    adapters_match,
    collect_adapter_sha256,
    materialize_eval_aliases,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGE_TOKEN_ID = 151655  # Qwen2.5-VL image placeholder token
MODEL_NAMES = {
    "3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "7b": "Qwen/Qwen2.5-VL-7B-Instruct",
}
PROMPT_TEMPLATE = NEGATIVE_PROMPT_TEMPLATE  # "Answer precisely in 1-2 words …"


class CAALDataset(Dataset):
    """Wraps paired EN/HI samples for the CAAL fine-tuning loop."""

    def __init__(self, pairs: List[Dict]):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def build_chat_input(processor, sample, question: str, device: str):
    """Build model inputs from a sample (image + question → direct answer)."""
    prompt = PROMPT_TEMPLATE.format(question=question)
    content = []
    for img in sample["images"]:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[img for img in sample["images"]],
        return_tensors="pt",
        padding=True,
    )
    return {k: v.to(device) for k, v in inputs.items()}


def build_training_input(processor, sample, question: str, answer: str,
                         device: str):
    """Build inputs WITH the answer appended for teacher-forced LM loss."""
    prompt = PROMPT_TEMPLATE.format(question=question)
    content = []
    for img in sample["images"]:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": prompt})
    messages = [
        {"role": "user", "content": content},
        {"role": "assistant", "content": str(answer)},
    ]
    text = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=False)
    inputs = processor(
        text=[text],
        images=[img for img in sample["images"]],
        return_tensors="pt",
        padding=True,
    )
    return {k: v.to(device) for k, v in inputs.items()}


# ═══════════════════════════════════════════════════════════════════════════
# CAAL LOSS — Cross-Lingual Attention Alignment (via hidden states)
# ═══════════════════════════════════════════════════════════════════════════

def _find_decoder_layers(model):
    """Find the ModuleList of decoder layers in a (possibly PEFT-wrapped) model."""
    for _name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) > 20:
            if hasattr(mod[0], "self_attn"):
                return mod
    raise ValueError("Cannot locate decoder layers in model")


class CAALHookManager:
    """Captures hidden-state representations at the last token position
    for specified decoder layers.

    Hooks into **decoder layer outputs** (not attention), so no
    ``output_attentions=True`` is needed.  This makes training fully
    compatible with flash attention and gradient checkpointing, and
    uses negligible extra memory (~0.1 MB for 9 layers).

    By aligning the decoder-layer output representations, we implicitly
    align the attention routing that produces them.
    """

    def __init__(self, model, target_layers: List[int]):
        self.model = model
        self.target_layers = sorted(target_layers)
        self.hooks: list = []
        self.storage: Dict[int, torch.Tensor] = {}
        self._decoder_layers = _find_decoder_layers(model)

    def _hook_fn(self, layer_idx: int, module, args, output):
        """Capture hidden state at the last token.  Shape: [hidden_dim]."""
        # Decoder-layer output is (hidden_states, ...) with hidden_states
        # shape [batch, seq, hidden_dim].  Take the last token.
        hidden = output[0] if isinstance(output, tuple) else output
        self.storage[layer_idx] = hidden[0, -1, :]  # [hidden_dim]

    def attach(self):
        for idx in self.target_layers:
            hook = self._decoder_layers[idx].register_forward_hook(
                partial(self._hook_fn, idx)
            )
            self.hooks.append(hook)
        return self

    def detach(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def clear(self):
        self.storage.clear()


def compute_caal_loss(
    en_reprs: Dict[int, torch.Tensor],
    hi_reprs: Dict[int, torch.Tensor],
) -> torch.Tensor:
    """Cosine-distance alignment loss between EN and HI hidden states.

    Returns mean(1 - cos_sim) across target layers.  EN representations
    are expected to be **detached** (the model learns to move HI towards
    EN, not the reverse).
    """
    losses = []
    for layer_idx in en_reprs:
        if layer_idx not in hi_reprs:
            continue
        en = en_reprs[layer_idx]  # [hidden_dim], detached
        hi = hi_reprs[layer_idx]  # [hidden_dim], has grad
        cos_sim = F.cosine_similarity(en.unsqueeze(0), hi.unsqueeze(0))
        losses.append(1.0 - cos_sim.squeeze())

    if not losses:
        return torch.tensor(0.0, requires_grad=True)
    return torch.stack(losses).mean()


def compute_xlsd_loss(
    en_logits: torch.Tensor,
    hi_logits: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """KL-divergence between EN (teacher, detached) and HI (student) logits.

    Temperature scaling softens distributions, transferring inter-class
    relationships ('dark knowledge').  The T^2 factor keeps gradient
    magnitudes comparable across temperature values.
    """
    en_soft = F.softmax(en_logits / temperature, dim=-1)
    hi_log_soft = F.log_softmax(hi_logits / temperature, dim=-1)
    kl = F.kl_div(hi_log_soft, en_soft, reduction='batchmean')
    return kl * (temperature ** 2)


def _log_eval_summary(results: Dict, label: str) -> None:
    strict_metrics = results["strict_metrics"]
    normalized_metrics = results["normalized_metrics"]
    logger.info(
        "%s strict EM: EN %.1f%%  HI %.1f%%  Gap %.1fpp  Overall %.1f%%",
        label,
        strict_metrics["english_accuracy"],
        strict_metrics["hindi_accuracy"],
        strict_metrics["gap_pp"],
        strict_metrics["overall_accuracy"],
    )
    logger.info(
        "%s normalized EM: EN %.1f%%  HI %.1f%%  Gap %.1fpp  Overall %.1f%%",
        label,
        normalized_metrics["english_accuracy"],
        normalized_metrics["hindi_accuracy"],
        normalized_metrics["gap_pp"],
        normalized_metrics["overall_accuracy"],
    )


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model,
    processor,
    optimizer,
    scheduler,
    train_pairs: List[Dict],
    caal_hooks: Optional[CAALHookManager],
    caal_lambda: float,
    device: str,
    epoch: int,
    grad_accum_steps: int = 4,
    xlsd_alpha: float = 0.0,
    xlsd_temperature: float = 2.0,
) -> Dict:
    """Train for one epoch with LM + optional CAAL + optional XLSD loss.

    Both alignment losses piggyback on the LM forward passes:

    1. Forward EN with labels → LM loss + hidden states (detached) + logits (detached)
    2. Forward HI with labels → LM loss + hidden states (with grad) + logits (with grad)
    3. CAAL loss = cosine distance of hidden-state representations
    4. XLSD loss = KL divergence of answer-position logit distributions
    5. Backward on combined loss
    """
    model.train()
    total_lm_loss = 0.0
    total_caal_loss = 0.0
    total_xlsd_loss = 0.0
    total_steps = 0
    optimizer.zero_grad()

    indices = list(range(len(train_pairs)))
    np.random.shuffle(indices)

    pbar = tqdm(indices, desc=f"Epoch {epoch}")
    for step_idx, pair_idx in enumerate(pbar):
        pair = train_pairs[pair_idx]

        def _forward_with_labels(sample, return_answer_logits=False):
            """Build inputs, create labels, and run forward pass."""
            answer = str(sample["answer"])
            question = str(sample["question"])
            inputs = build_training_input(
                processor, sample, question, answer, device
            )
            input_ids = inputs["input_ids"]
            # Labels = input_ids with non-answer tokens masked to -100.
            # Do NOT shift — HuggingFace internally shifts labels in the
            # loss computation (predict token[i+1] from position[i]).
            labels = torch.full_like(input_ids, -100)
            answer_ids = processor.tokenizer.encode(
                answer, add_special_tokens=False
            )
            n_answer = len(answer_ids) + 2  # +2 for safety margin
            if input_ids.shape[1] > n_answer:
                labels[:, -n_answer:] = input_ids[:, -n_answer:]
            else:
                labels = input_ids.clone()
            outputs = model(**inputs, labels=labels)
            lm_loss = outputs.loss
            answer_logits = None
            if return_answer_logits and n_answer > 0:
                # Extract only answer-position logits; clone() ensures
                # the full [seq_len, vocab_size] tensor can be freed
                answer_logits = outputs.logits[0, -n_answer:, :].clone()
            return lm_loss, answer_logits

        # ---- EN forward: LM loss + capture hidden states (detached) ----
        if caal_hooks is not None:
            caal_hooks.clear()

        use_xlsd = xlsd_alpha > 0
        en_lm_loss, en_answer_logits = _forward_with_labels(
            pair["en_sample"], return_answer_logits=use_xlsd
        )
        if en_answer_logits is not None:
            en_answer_logits = en_answer_logits.detach()

        en_reprs = None
        if caal_hooks is not None and caal_lambda > 0:
            en_reprs = {k: v.detach() for k, v in caal_hooks.storage.items()}
            caal_hooks.clear()

        # ---- HI forward: LM loss + capture hidden states (with grad) ----
        hi_lm_loss, hi_answer_logits = _forward_with_labels(
            pair["hi_sample"], return_answer_logits=use_xlsd
        )

        lm_loss = (en_lm_loss + hi_lm_loss) / 2.0

        # ---- CAAL Loss (representation alignment) ----
        caal_loss_val = torch.tensor(0.0, device=device)
        if en_reprs is not None and caal_hooks is not None:
            hi_reprs = caal_hooks.storage  # keep grad
            caal_loss_val = compute_caal_loss(en_reprs, hi_reprs)

        # ---- XLSD Loss (cross-lingual self-distillation) ----
        xlsd_loss_val = torch.tensor(0.0, device=device)
        if en_answer_logits is not None and hi_answer_logits is not None:
            # Match lengths (safety for rare tokenization edge cases)
            min_len = min(en_answer_logits.shape[0], hi_answer_logits.shape[0])
            xlsd_loss_val = compute_xlsd_loss(
                en_answer_logits[-min_len:],
                hi_answer_logits[-min_len:],
                xlsd_temperature,
            )

        # ---- Combined loss ----
        # In model-parallel mode, auxiliary losses may be on different devices.
        target_device = lm_loss.device
        if caal_loss_val.device != target_device:
            caal_loss_val = caal_loss_val.to(target_device)
        if xlsd_loss_val.device != target_device:
            xlsd_loss_val = xlsd_loss_val.to(target_device)

        loss = (lm_loss + caal_lambda * caal_loss_val + xlsd_alpha * xlsd_loss_val) / grad_accum_steps
        loss.backward()

        if (step_idx + 1) % grad_accum_steps == 0 or (step_idx + 1) == len(indices):
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_lm_loss += lm_loss.item()
        total_caal_loss += caal_loss_val.item()
        total_xlsd_loss += xlsd_loss_val.item()
        total_steps += 1

        postfix = {
            "lm": f"{lm_loss.item():.3f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}",
        }
        if caal_lambda > 0:
            postfix["caal"] = f"{caal_loss_val.item():.4f}"
        if xlsd_alpha > 0:
            postfix["xlsd"] = f"{xlsd_loss_val.item():.4f}"
        pbar.set_postfix(postfix)

        # Free memory
        del en_lm_loss, hi_lm_loss, lm_loss, caal_loss_val, xlsd_loss_val, loss
        if (step_idx + 1) % 50 == 0:
            torch.cuda.empty_cache()

    return {
        "avg_lm_loss": total_lm_loss / max(total_steps, 1),
        "avg_caal_loss": total_caal_loss / max(total_steps, 1),
        "avg_xlsd_loss": total_xlsd_loss / max(total_steps, 1),
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="CAAL-QLoRA: Cross-Lingual Attention-Aligned Fine-Tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument("--model-size", choices=["3b", "7b"], default="7b")
    parser.add_argument("--model-parallel", action="store_true",
                        help="Shard model across all visible GPUs to reduce per-GPU memory")
    parser.add_argument("--max-memory-per-gpu-gb", type=int, default=None,
                        help="Optional per-GPU memory cap (GiB) when --model-parallel is enabled")
    parser.add_argument("--lora-r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32,
                        help="LoRA alpha scaling")
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-targets", nargs="+",
                        default=["q_proj", "v_proj"],
                        help="LoRA target modules in attention")

    # CAAL
    parser.add_argument("--caal-lambda", type=float, default=0.1,
                        help="Weight for cross-lingual attention alignment loss")
    parser.add_argument("--caal-layers", nargs="+", type=int,
                        default=list(range(19, 28)),
                        help="Layers for CAAL loss (default: 19-27)")

    # XLSD (Cross-Lingual Self-Distillation)
    parser.add_argument("--xlsd-alpha", type=float, default=0.0,
                        help="Weight for XLSD KL-divergence loss (0=disabled)")
    parser.add_argument("--xlsd-temperature", type=float, default=2.0,
                        help="Temperature for XLSD softmax scaling")

    # DG-LoRA (Diagnostic-Guided LoRA)
    parser.add_argument("--layers-to-transform", type=int, nargs="+",
                        default=None,
                        help="Apply LoRA only to these layers (e.g. 19 20 ... 27)")

    # Training
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Effective batch size (grad accum adjusts)")
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-pixels", type=int, default=2_000_000,
                        help="Max pixels for image processing")
    parser.add_argument("--max-new-tokens", type=int, default=32,
                        help="Max tokens to generate (answers are 1-2 words)")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="Fraction of pairs for validation")
    parser.add_argument("--eval-pairs", type=int, default=50,
                        help="Pairs for mid-training eval (0=all, full eval at end)")

    # I/O
    parser.add_argument("--output-dir", type=str,
                        default="results/finetune_caal")
    parser.add_argument("--seed", type=int, default=42)

    # Ablation
    parser.add_argument("--no-caal", action="store_true",
                        help="Ablation: standard QLoRA without CAAL loss")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only evaluate (requires saved adapter)")
    parser.add_argument("--adapter-path", type=str, default=None,
                        help="Path to saved LoRA adapter for eval-only")
    parser.add_argument("--full-eval", action="store_true",
                        help="Evaluate on ALL pairs (not just val split)")
    parser.add_argument("--fixed_eval_split", type=str, default=None,
                        help="JSON file containing fixed paired eval IDs")
    parser.add_argument("--create_fixed_eval_split", action="store_true",
                        help="Create and save the fixed paired eval split before use")
    parser.add_argument("--fixed_eval_size", type=int, default=200,
                        help="Number of EN/HI pairs in the fixed eval split")
    parser.add_argument("--eval_checkpoint_tag", choices=["base", "best", "final"],
                        default="final",
                        help="Artifact tag for eval-only runs")
    parser.add_argument("--save_predictions_dir", type=str, default=None,
                        help="Directory for machine-readable evaluation artifacts")
    parser.add_argument("--save_judge_ready", action="store_true",
                        help="Also save judge-ready JSON derived from the same predictions")
    eval_group = parser.add_mutually_exclusive_group()
    eval_group.add_argument("--eval-best-only", dest="eval_best_only", action="store_true",
                            help="Evaluate only the paper-facing best adapter after training")
    eval_group.add_argument("--eval-both", dest="eval_best_only", action="store_false",
                            help="Also evaluate the final adapter when it differs from best")
    parser.set_defaults(eval_best_only=True)

    args = parser.parse_args()

    # ---- Setup ----
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"caal_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Save config
    config = vars(args)
    config["timestamp"] = timestamp
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    predictions_dir = args.save_predictions_dir or os.path.join(output_dir, "evaluations")

    # ---- Load model with QLoRA ----
    model_name = MODEL_NAMES[args.model_size]
    logger.info(f"Loading model: {model_name}")

    from transformers import (
        Qwen2_5_VLForConditionalGeneration,
        AutoProcessor,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    # Processor
    processor = AutoProcessor.from_pretrained(
        model_name,
        max_pixels=args.max_pixels,
    )

    # NF4 base
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    device_map = "auto"
    max_memory = None
    if args.model_parallel and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        device_map = "balanced"
        if args.max_memory_per_gpu_gb is not None:
            max_memory = {
                gpu_idx: f"{args.max_memory_per_gpu_gb}GiB"
                for gpu_idx in range(torch.cuda.device_count())
            }
        logger.info(
            "Model parallel enabled across %d GPUs (device_map=%s)",
            torch.cuda.device_count(),
            device_map,
        )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        max_memory=max_memory,
        attn_implementation="sdpa",  # CAAL uses hidden states, not attention matrices
    )

    # Patch vision encoder to SDPA (same fix as in wrapper)
    if hasattr(model, "visual"):
        if hasattr(model.visual, "config"):
            model.visual.config._attn_implementation = "sdpa"
        for block in model.visual.blocks:
            if hasattr(block, "attn"):
                block.attn._attn_implementation = "sdpa"
        logger.info("Vision encoder → SDPA attention")

    # Prepare for k-bit training (handles gradient checkpointing etc.)
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # LoRA config
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_targets,
        layers_to_transform=args.layers_to_transform,
        bias="none",
        task_type="CAUSAL_LM",
    )

    if args.eval_only:
        if args.eval_checkpoint_tag == "base":
            logger.info("Eval-only base checkpoint: using raw base model without adapters")
        elif args.adapter_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.adapter_path)
            logger.info(f"Loaded adapter from {args.adapter_path}")
        else:
            raise ValueError(
                "--eval-only with checkpoint_tag best/final requires --adapter-path"
            )
    else:
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Log memory ----
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        logger.info(f"GPU memory after model load: {alloc:.1f} GiB")

    # ---- Load data ----
    dataset = load_mmcricbench()

    if args.eval_only and args.fixed_eval_split and not args.create_fixed_eval_split and not args.full_eval:
        split_payload = load_fixed_eval_split(args.fixed_eval_split)
        requested_pair_ids = [str(pair_id) for pair_id in split_payload["pair_ids"]]
        all_pairs = create_paired_dataset(dataset["test_single"], pair_ids=requested_pair_ids)
        fixed_eval_pairs = list(all_pairs)
        fixed_eval_split_path = args.fixed_eval_split
        logger.info(
            "Loaded %d fixed-split pairs directly for eval-only",
            len(fixed_eval_pairs),
        )
    else:
        all_pairs = create_paired_dataset(dataset["test_single"])
        logger.info(f"Total EN-HI pairs: {len(all_pairs)}")

        fixed_eval_pairs, fixed_eval_split_path = prepare_fixed_paired_eval_split(
            all_pairs=all_pairs,
            split_path=args.fixed_eval_split,
            create_split=args.create_fixed_eval_split,
            split_name="mmcricbench_test_single_paired",
            eval_size=args.fixed_eval_size,
            seed=args.seed,
        )

    # Train/val split (by pair, so EN-HI pairs stay together)
    if fixed_eval_pairs is not None:
        val_pairs = list(fixed_eval_pairs)
        train_pairs = exclude_pairs(all_pairs, fixed_eval_pairs)
        logger.info("Using fixed eval split: %s (%d pairs)", fixed_eval_split_path, len(val_pairs))
    else:
        n_val = max(1, int(len(all_pairs) * args.val_ratio))
        np.random.shuffle(all_pairs)
        val_pairs = all_pairs[:n_val]
        train_pairs = all_pairs[n_val:]
    logger.info(f"Train pairs: {len(train_pairs)}, Val pairs: {len(val_pairs)}")
    if not train_pairs and not args.eval_only:
        raise ValueError("No training pairs remain after selecting the eval split")

    # ---- Eval-only mode ----
    if args.eval_only:
        eval_data = all_pairs if args.full_eval else val_pairs
        logger.info(f"=== EVAL ONLY ({len(eval_data)} pairs) ===")
        results = evaluate_paired_model(
            model=model,
            processor=processor,
            pairs=eval_data,
            device=device,
            build_chat_input_fn=build_chat_input,
            max_new_tokens=args.max_new_tokens,
            split_name="fixed_eval" if fixed_eval_split_path else ("full_eval" if args.full_eval else "val"),
            checkpoint_tag=args.eval_checkpoint_tag,
            save_predictions_dir=predictions_dir,
            split_file=fixed_eval_split_path,
            generation_config={"max_new_tokens": args.max_new_tokens, "do_sample": False},
            save_judge_ready=args.save_judge_ready,
        )
        _log_eval_summary(results, f"[{args.eval_checkpoint_tag}]")
        return

    # ---- Setup CAAL hooks ----
    caal_hooks = None
    if not args.no_caal and args.caal_lambda > 0:
        caal_hooks = CAALHookManager(model, args.caal_layers)
        caal_hooks.attach()
        logger.info(f"CAAL hooks attached at layers {args.caal_layers} "
                    f"(hidden-state alignment, no output_attentions needed)")

    if args.xlsd_alpha > 0:
        logger.info(f"XLSD enabled: alpha={args.xlsd_alpha}, "
                    f"temperature={args.xlsd_temperature}")

    if args.layers_to_transform:
        logger.info(f"DG-LoRA: adapting only layers {args.layers_to_transform}")

    # ---- Optimizer + scheduler ----
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    total_steps = len(train_pairs) * args.epochs // args.grad_accum_steps
    warmup_steps = int(total_steps * args.warmup_ratio)

    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    # ---- Pre-training eval ----
    eval_n = len(val_pairs) if fixed_eval_split_path else (
        min(args.eval_pairs, len(val_pairs)) if args.eval_pairs > 0 else len(val_pairs)
    )
    eval_subset = val_pairs[:eval_n]
    logger.info(f"=== Pre-training evaluation ({eval_n} pairs) ===")
    pre_results = evaluate_paired_model(
        model=model,
        processor=processor,
        pairs=eval_subset,
        device=device,
        build_chat_input_fn=build_chat_input,
        max_new_tokens=args.max_new_tokens,
        split_name="fixed_eval" if fixed_eval_split_path else "val_subset",
        checkpoint_tag="base",
        save_predictions_dir=predictions_dir,
        split_file=fixed_eval_split_path,
        generation_config={"max_new_tokens": args.max_new_tokens, "do_sample": False},
        save_judge_ready=args.save_judge_ready,
    )
    _log_eval_summary(pre_results, "[base]")

    # ---- Training loop ----
    best_overall = -1.0
    best_epoch = -1
    best_results = None
    history = []

    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"EPOCH {epoch}/{args.epochs}")
        logger.info(f"{'='*60}")

        train_metrics = train_one_epoch(
            model, processor, optimizer, scheduler,
            train_pairs, caal_hooks, args.caal_lambda,
            device, epoch, args.grad_accum_steps,
            xlsd_alpha=args.xlsd_alpha,
            xlsd_temperature=args.xlsd_temperature,
        )
        logger.info(f"  LM loss: {train_metrics['avg_lm_loss']:.4f}  "
                    f"CAAL loss: {train_metrics['avg_caal_loss']:.4f}  "
                    f"XLSD loss: {train_metrics['avg_xlsd_loss']:.4f}")

        # Eval on subset (hooks are lightweight, no need to detach)
        eval_results = evaluate_paired_model(
            model=model,
            processor=processor,
            pairs=eval_subset,
            device=device,
            build_chat_input_fn=build_chat_input,
            max_new_tokens=args.max_new_tokens,
            split_name="fixed_eval" if fixed_eval_split_path else "val_subset",
            checkpoint_tag="best",
            save_predictions_dir=predictions_dir,
            split_file=fixed_eval_split_path,
            generation_config={"max_new_tokens": args.max_new_tokens, "do_sample": False},
            save_judge_ready=args.save_judge_ready,
        )
        _log_eval_summary(eval_results, f"[epoch {epoch}]")

        epoch_record = {
            "epoch": epoch,
            **train_metrics,
            "strict_metrics": eval_results["strict_metrics"],
            "normalized_metrics": eval_results["normalized_metrics"],
            "counts": eval_results["counts"],
        }
        history.append(epoch_record)

        # Save best adapter (highest normalized overall accuracy)
        if eval_results["normalized_metrics"]["overall_accuracy"] > best_overall:
            best_overall = eval_results["normalized_metrics"]["overall_accuracy"]
            best_epoch = epoch
            best_results = eval_results
            adapter_path = os.path.join(output_dir, "best_adapter")
            model.save_pretrained(adapter_path)
            processor.save_pretrained(adapter_path)
            logger.info(
                "  \u2713 New best! Normalized overall=%.1f%% Gap=%.1fpp (saved)",
                best_overall,
                eval_results["normalized_metrics"]["gap_pp"],
            )

    # ---- Cleanup ----
    if caal_hooks:
        caal_hooks.detach()

    # Save final adapter
    final_path = os.path.join(output_dir, "final_adapter")
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)

    best_adapter_path = os.path.join(output_dir, "best_adapter")
    best_adapter_exists = os.path.exists(os.path.join(best_adapter_path, "adapter_model.safetensors"))
    adapter_sha256 = collect_adapter_sha256(
        best_adapter_dir=best_adapter_path,
        final_adapter_dir=final_path,
    )

    posttrain_best_results = None
    posttrain_final_results = None
    posttrain_final_eq_best_results = None

    posttrain_split_name = "fixed_eval" if fixed_eval_split_path else "val_full"
    posttrain_pairs = val_pairs
    if best_adapter_exists and adapters_match(adapter_sha256):
        logger.info("[dedupe] best == final, single eval.")
        if not args.eval_best_only:
            logger.warning(
                "--eval-both requested but best/final adapters are SHA-identical; "
                "running one deduped eval"
            )
        posttrain_final_eq_best_results = evaluate_paired_model(
            model=model,
            processor=processor,
            pairs=posttrain_pairs,
            device=device,
            build_chat_input_fn=build_chat_input,
            max_new_tokens=args.max_new_tokens,
            split_name=posttrain_split_name,
            checkpoint_tag="final_eq_best",
            save_predictions_dir=predictions_dir,
            split_file=fixed_eval_split_path,
            generation_config={"max_new_tokens": args.max_new_tokens, "do_sample": False},
            save_judge_ready=args.save_judge_ready,
        )
        materialize_eval_aliases(posttrain_final_eq_best_results, "best", "final")
        _log_eval_summary(posttrain_final_eq_best_results, "[final_eq_best]")
        posttrain_best_results = posttrain_final_eq_best_results
        posttrain_final_results = posttrain_final_eq_best_results
    else:
        run_final_eval = (not best_adapter_exists) or (not args.eval_best_only)
        if run_final_eval:
            posttrain_final_results = evaluate_paired_model(
                model=model,
                processor=processor,
                pairs=posttrain_pairs,
                device=device,
                build_chat_input_fn=build_chat_input,
                max_new_tokens=args.max_new_tokens,
                split_name=posttrain_split_name,
                checkpoint_tag="final",
                save_predictions_dir=predictions_dir,
                split_file=fixed_eval_split_path,
                generation_config={"max_new_tokens": args.max_new_tokens, "do_sample": False},
                save_judge_ready=args.save_judge_ready,
            )
            _log_eval_summary(posttrain_final_results, "[final]")

        if best_adapter_exists:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            from transformers import (
                Qwen2_5_VLForConditionalGeneration,
                AutoProcessor,
                BitsAndBytesConfig,
            )
            from peft import PeftModel

            best_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                max_memory=max_memory,
                attn_implementation="sdpa",
            )
            if hasattr(best_model, "visual"):
                if hasattr(best_model.visual, "config"):
                    best_model.visual.config._attn_implementation = "sdpa"
                for block in best_model.visual.blocks:
                    if hasattr(block, "attn"):
                        block.attn._attn_implementation = "sdpa"
            best_model = PeftModel.from_pretrained(best_model, best_adapter_path)
            posttrain_best_results = evaluate_paired_model(
                model=best_model,
                processor=processor,
                pairs=posttrain_pairs,
                device=device,
                build_chat_input_fn=build_chat_input,
                max_new_tokens=args.max_new_tokens,
                split_name=posttrain_split_name,
                checkpoint_tag="best",
                save_predictions_dir=predictions_dir,
                split_file=fixed_eval_split_path,
                generation_config={"max_new_tokens": args.max_new_tokens, "do_sample": False},
                save_judge_ready=args.save_judge_ready,
            )
            _log_eval_summary(posttrain_best_results, "[best]")

    if posttrain_final_eq_best_results is not None:
        posttrain_eval_mode = "deduped_final_eq_best"
    elif posttrain_best_results is not None and posttrain_final_results is not None:
        posttrain_eval_mode = "both"
    elif posttrain_best_results is not None:
        posttrain_eval_mode = "best_only"
    else:
        posttrain_eval_mode = "final_only"

    # Save history
    summary = {
        "config": config,
        "fixed_eval_split": fixed_eval_split_path,
        "pre_training": {k: v for k, v in pre_results.items() if k != "predictions"},
        "history": history,
        "best_epoch": best_epoch,
        "best_overall_accuracy": best_overall,
        "adapter_sha256": adapter_sha256,
        "post_training_eval_mode": posttrain_eval_mode,
    }
    if posttrain_best_results is not None:
        summary["best_eval"] = {
            k: v for k, v in posttrain_best_results.items() if k != "predictions"
        }
    elif best_results is not None:
        summary["best_eval"] = {k: v for k, v in best_results.items() if k != "predictions"}
    if posttrain_final_results is not None:
        summary["final_eval"] = {
            k: v for k, v in posttrain_final_results.items() if k != "predictions"
        }
    if posttrain_final_eq_best_results is not None:
        summary["final_eq_best_eval"] = {
            k: v for k, v in posttrain_final_eq_best_results.items() if k != "predictions"
        }
    with open(os.path.join(output_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING COMPLETE")
    logger.info(f"  Best epoch: {best_epoch}, Best overall: {best_overall:.1f}%")
    logger.info(
        "  Base normalized gap: %.1fpp",
        pre_results["normalized_metrics"]["gap_pp"],
    )
    logger.info(f"  Saved to: {output_dir}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
