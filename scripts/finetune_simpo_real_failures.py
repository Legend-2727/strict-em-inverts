#!/usr/bin/env python3
"""Day-12: SimPO trainer for (chosen, rejected) preference pairs built from
REAL base-model failures (data/processed/dpo_pairs_real_failures/{train,val}.jsonl).

This is a minimal trainer that imports the loss/log-prob helpers from
`finetune_xpeg2_simpo.py` and applies them to plain-text chosen/rejected
completions (no <evidence> formatting, no Vertex verdicts). The intent is
to show that training Qwen2.5-VL-7B QLoRA on REAL hallucination pairs
lifts non-Latin EM beyond the 'fixed test-time budget' shown in §5.

Usage:
  /home/ubuntu/xLingual/.venv/bin/python scripts/finetune_simpo_real_failures.py \
      --train-pairs data/processed/dpo_pairs_real_failures/train.jsonl \
      --val-pairs data/processed/dpo_pairs_real_failures/val.jsonl \
      --output-dir results/dpo_real_failures \
      --model-size 7b --quantization nf4 --attn-impl sdpa \
      --epochs 1 --beta 2.0 --gamma 0.5 --lr 5e-5 --lora-r 16
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

# Reuse the well-tested helpers from the X-PEG² trainer
from finetune_xpeg2_simpo import (  # noqa: E402
    teacher_forced_logprob, simpo_loss, _read_jsonl,
)


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-pairs", required=True, type=Path)
    p.add_argument("--val-pairs", type=Path, default=None)
    p.add_argument("--output-dir", required=True, type=Path)

    p.add_argument("--model-size", choices=["3b", "7b"], default="7b")
    p.add_argument("--quantization", choices=["nf4", "8bit", "none"], default="nf4")
    p.add_argument("--attn-impl",
                   choices=["sdpa", "flash_attention_2", "eager"], default="sdpa")
    p.add_argument("--max-pixels", type=int, default=1_500_000)

    # LoRA
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-target-modules", nargs="+",
                   default=["q_proj", "k_proj", "v_proj", "o_proj"])

    # Optim
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)

    # SimPO
    p.add_argument("--beta", type=float, default=2.0,
                   help="SimPO inverse-temperature scaling.")
    p.add_argument("--gamma", type=float, default=0.5,
                   help="SimPO target margin.")
    p.add_argument("--anchor-lambda", type=float, default=0.1,
                   help="Weight on -NLL(chosen) term to anchor chosen log-prob.")

    # Eval / logging
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--eval-every-steps", type=int, default=100)
    p.add_argument("--save-every-steps", type=int, default=200)
    p.add_argument("--resume", action="store_true",
                   help="If set, resume from the latest adapter_step{N} checkpoint in --output-dir "
                        "(restores optimizer, RNG, epoch, and in-epoch position).")

    return p


def load_model_and_processor(args, resume_adapter_dir: Optional[Path] = None):
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen2_5_VLForConditionalGeneration,
    )
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

    repo = ("Qwen/Qwen2.5-VL-3B-Instruct" if args.model_size == "3b"
            else "Qwen/Qwen2.5-VL-7B-Instruct")
    print(f"[model] loading {repo}", flush=True)
    quant_cfg = None
    if args.quantization == "nf4":
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif args.quantization == "8bit":
        quant_cfg = BitsAndBytesConfig(load_in_8bit=True)

    processor = AutoProcessor.from_pretrained(repo)
    if hasattr(processor, "tokenizer") and processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    kwargs = {"torch_dtype": torch.bfloat16, "attn_implementation": args.attn_impl}
    if quant_cfg is not None:
        kwargs["quantization_config"] = quant_cfg
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(repo, **kwargs)
    if quant_cfg is None:
        model.to("cuda")

    if quant_cfg is not None:
        model = prepare_model_for_kbit_training(model)

    if resume_adapter_dir is not None:
        print(f"[model] resuming PEFT adapter from {resume_adapter_dir}", flush=True)
        model = PeftModel.from_pretrained(model, str(resume_adapter_dir), is_trainable=True)
    else:
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=args.lora_target_modules,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.train()
    return model, processor


def absolute_image_path(rel_or_abs: str) -> str:
    p = Path(rel_or_abs)
    if p.is_absolute() or p.exists():
        return str(p)
    return str(REPO_ROOT / rel_or_abs)


def step_pair(model, processor, pair, args, device="cuda"):
    image_path = absolute_image_path(pair["image_path"])
    chosen_lp, chosen_n = teacher_forced_logprob(
        model, processor, image_path, pair["prompt_text"], pair["chosen"],
        device=device, max_pixels=args.max_pixels)
    rejected_lp, rejected_n = teacher_forced_logprob(
        model, processor, image_path, pair["prompt_text"], pair["rejected"],
        device=device, max_pixels=args.max_pixels)
    loss = simpo_loss(
        chosen_lp, chosen_n, rejected_lp, rejected_n,
        beta=args.beta, gamma=args.gamma)
    chosen_avg = chosen_lp / max(chosen_n, 1)
    rejected_avg = rejected_lp / max(rejected_n, 1)
    margin = (chosen_avg - rejected_avg).detach()
    if args.anchor_lambda > 0:
        # Add anchor: positive -log p(chosen) keeps chosen log-prob from drifting
        # downward when SimPO incentivizes only the GAP.
        anchor = -chosen_avg
        loss = loss + args.anchor_lambda * anchor
    return loss, margin.item(), chosen_avg.item(), rejected_avg.item()


@torch.no_grad()
def evaluate(model, processor, pairs, args, device="cuda", max_eval=None):
    model.eval()
    margins = []
    chosen_lps = []
    rejected_lps = []
    em_chosen_higher = 0
    n = 0
    for pair in pairs[:max_eval] if max_eval else pairs:
        try:
            image_path = absolute_image_path(pair["image_path"])
            c_lp, c_n = teacher_forced_logprob(
                model, processor, image_path, pair["prompt_text"],
                pair["chosen"], device=device, max_pixels=args.max_pixels)
            r_lp, r_n = teacher_forced_logprob(
                model, processor, image_path, pair["prompt_text"],
                pair["rejected"], device=device, max_pixels=args.max_pixels)
            c_avg = c_lp.item() / max(c_n, 1)
            r_avg = r_lp.item() / max(r_n, 1)
            margins.append(c_avg - r_avg)
            chosen_lps.append(c_avg)
            rejected_lps.append(r_avg)
            if c_avg > r_avg:
                em_chosen_higher += 1
            n += 1
        except Exception as e:
            print(f"[eval-warn] {e}", flush=True)
            continue
    model.train()
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "mean_margin": sum(margins) / n,
        "win_rate_chosen": em_chosen_higher / n,
        "mean_chosen_lp": sum(chosen_lps) / n,
        "mean_rejected_lp": sum(rejected_lps) / n,
    }


def _epoch_order(n: int, seed: int, epoch: int) -> List[int]:
    """Deterministic per-epoch permutation. Reproducing this on resume is what
    lets us skip already-trained pairs in the current epoch without storing the
    full shuffled list."""
    order = list(range(n))
    rng = random.Random(seed * 1_000_003 + epoch)
    rng.shuffle(order)
    return order


def _save_checkpoint(model, optim, out_dir: Path, step: int, epoch: int,
                     in_epoch_idx: int, args) -> Path:
    """Save adapter + optimizer + RNG + position so training can resume here."""
    ck = out_dir / f"adapter_step{step}"
    model.save_pretrained(ck)
    state = {
        "step": step,
        "epoch": epoch,
        "in_epoch_idx": in_epoch_idx,
        "optimizer_state_dict": optim.state_dict(),
        "rng_python": random.getstate(),
        "rng_torch": torch.get_rng_state(),
        "rng_torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "args_seed": args.seed,
    }
    torch.save(state, ck / "trainer_state.pt")
    # Mirror to a stable "latest" pointer so resume code can find it without globbing
    (out_dir / "latest_checkpoint.txt").write_text(ck.name, encoding="utf-8")
    return ck


def _find_latest_checkpoint(out_dir: Path) -> Optional[Path]:
    pointer = out_dir / "latest_checkpoint.txt"
    if pointer.exists():
        name = pointer.read_text(encoding="utf-8").strip()
        ck = out_dir / name
        if (ck / "trainer_state.pt").exists():
            return ck
    # Fallback: scan adapter_step* and pick the highest step
    candidates = []
    for d in out_dir.glob("adapter_step*"):
        if (d / "trainer_state.pt").exists():
            try:
                n = int(d.name.replace("adapter_step", ""))
                candidates.append((n, d))
            except ValueError:
                continue
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def main():
    args = build_parser().parse_args()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_pairs = _read_jsonl(args.train_pairs)
    val_pairs = _read_jsonl(args.val_pairs) if args.val_pairs else []
    print(f"[data] train={len(train_pairs)}  val={len(val_pairs)}", flush=True)

    resume_ck: Optional[Path] = None
    resume_state: Optional[Dict[str, Any]] = None
    if args.resume:
        resume_ck = _find_latest_checkpoint(out_dir)
        if resume_ck is None:
            print(f"[resume] no checkpoint with trainer_state.pt found in {out_dir} — starting fresh", flush=True)
        else:
            resume_state = torch.load(resume_ck / "trainer_state.pt", map_location="cpu", weights_only=False)
            print(f"[resume] loading checkpoint {resume_ck} "
                  f"(step={resume_state['step']} epoch={resume_state['epoch']} idx={resume_state['in_epoch_idx']})", flush=True)

    model, processor = load_model_and_processor(args, resume_adapter_dir=resume_ck)

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay)

    start_step = 0
    start_epoch = 0
    start_in_epoch_idx = 0
    if resume_state is not None:
        optim.load_state_dict(resume_state["optimizer_state_dict"])
        random.setstate(resume_state["rng_python"])
        torch.set_rng_state(resume_state["rng_torch"])
        if torch.cuda.is_available() and resume_state.get("rng_torch_cuda") is not None:
            torch.cuda.set_rng_state_all(resume_state["rng_torch_cuda"])
        start_step = resume_state["step"]
        start_epoch = resume_state["epoch"]
        start_in_epoch_idx = resume_state["in_epoch_idx"]

    log_path = out_dir / "training_log.jsonl"
    log_fh = log_path.open("a" if resume_state is not None else "w", encoding="utf-8")

    def log(obj):
        obj = {**obj, "ts": datetime.now(timezone.utc).isoformat()}
        log_fh.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
        log_fh.flush()

    if resume_state is not None:
        log({"event": "resume", "step": start_step, "epoch": start_epoch,
              "in_epoch_idx": start_in_epoch_idx, "ckpt": str(resume_ck)})
    else:
        log({"event": "start", "args": vars(args), "n_train": len(train_pairs),
              "n_val": len(val_pairs)})

    # Initial eval (skipped on resume — we already did this on the original start)
    if val_pairs and resume_state is None:
        m = evaluate(model, processor, val_pairs, args, max_eval=min(50, len(val_pairs)))
        print(f"[eval init] n={m['n']}  mean_margin={m.get('mean_margin', 0):+.4f}  "
              f"win_rate={m.get('win_rate_chosen', 0):.4f}", flush=True)
        log({"event": "eval", "step": 0, **m})

    step = start_step
    t0 = time.time()
    try:
        for epoch in range(start_epoch, args.epochs):
            order = _epoch_order(len(train_pairs), args.seed, epoch)
            begin_idx = start_in_epoch_idx if epoch == start_epoch else 0
            for in_epoch_idx in range(begin_idx, len(order)):
                pair = train_pairs[order[in_epoch_idx]]
                optim.zero_grad()
                try:
                    loss, margin, c_avg, r_avg = step_pair(model, processor, pair, args)
                except Exception as e:
                    print(f"[step-warn] {e}", flush=True)
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optim.step()
                step += 1
                if step % args.log_every == 0:
                    el = time.time() - t0
                    print(f"[step {step}] epoch={epoch} idx={in_epoch_idx+1}/{len(order)}  "
                          f"loss={loss.item():.4f}  margin={margin:+.4f}  "
                          f"chosen_lp_avg={c_avg:+.4f}  rejected_lp_avg={r_avg:+.4f}  "
                          f"t={el:.0f}s", flush=True)
                    log({"event": "step", "step": step, "epoch": epoch,
                          "in_epoch_idx": in_epoch_idx + 1,
                          "loss": float(loss.item()), "margin": float(margin),
                          "chosen_lp_avg": c_avg, "rejected_lp_avg": r_avg})

                if val_pairs and step % args.eval_every_steps == 0:
                    m = evaluate(model, processor, val_pairs, args,
                                  max_eval=min(50, len(val_pairs)))
                    print(f"[eval step {step}] n={m['n']}  "
                          f"mean_margin={m['mean_margin']:+.4f}  "
                          f"win_rate={m['win_rate_chosen']:.4f}", flush=True)
                    log({"event": "eval", "step": step, **m})

                if step % args.save_every_steps == 0:
                    ck = _save_checkpoint(model, optim, out_dir, step, epoch,
                                          in_epoch_idx + 1, args)
                    print(f"[ckpt] {ck}", flush=True)
                    log({"event": "save", "step": step, "path": str(ck)})
            # reset for next epoch
            start_in_epoch_idx = 0
    except KeyboardInterrupt:
        print("[interrupt] saving emergency checkpoint", flush=True)
        ck = _save_checkpoint(model, optim, out_dir, step, epoch, in_epoch_idx + 1, args)
        log({"event": "interrupt_save", "step": step, "path": str(ck)})
        log_fh.close()
        raise

    # Final save + eval
    final_ck = out_dir / "adapter_final"
    model.save_pretrained(final_ck)
    print(f"[final ckpt] {final_ck}", flush=True)
    if val_pairs:
        m = evaluate(model, processor, val_pairs, args)
        print(f"[eval final] n={m['n']}  mean_margin={m['mean_margin']:+.4f}  "
              f"win_rate={m['win_rate_chosen']:.4f}", flush=True)
        log({"event": "eval_final", "step": step, **m})

    log({"event": "done", "step": step, "elapsed_s": time.time() - t0})
    log_fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
