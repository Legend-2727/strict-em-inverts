#!/usr/bin/env python3
"""Day-11: run the OCR-prepended-prompt intervention on InternVL2.5-8B
non-Latin EXTRACTIVE pool. Tests whether the 'fixed correct-answer budget'
pattern (5 interventions → ΔEM ≈ 0 on Qwen-VL-7B) also holds on InternVL.

For InternVL we cannot inject logit-bias / VCD / rep-eng directly because
the runner uses model.chat() which hides logits. We can do:
  - Plain prompt baseline (already covered by diagnose_mtvqa_internvl.py)
  - OCR-prepended prompt (this script)

Two-intervention test on InternVL is sufficient to claim
'pattern generalizes across families' if both deltas are within noise.

Usage:
  /home/ubuntu/xLingual/.venv/bin/python scripts/intervention_sweep_internvl.py \
      --output-dir results/internvl_sweep_ocrprepend \
      --languages ar ja kr ru th --task-type EXTRACTIVE --max-per-language 50 \
      --prompt-mode ocr_prepended
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from diagnose_mtvqa_per_language import (  # noqa: E402
    PROMPT_INSTRUCTION, classify, dominant_script, extract_answer,
    load_ocr, normalize, script_compatible, text_in_ocr, token_f1,
)
from diagnose_mtvqa_internvl import (  # noqa: E402
    classify_row, read_jsonl, reconstruct_image_path, summarize,
)
from evaluate_xfund_qa_zeroshot_vlm import InternVLRunner  # noqa: E402

VAL_PATH = REPO_ROOT / "data/processed/mtvqa/val.jsonl"


def filter_pool(val_rows, languages, task_type, max_per_language):
    pool = []
    for r in val_rows:
        if languages and r["language"] not in languages:
            continue
        gold = (r.get("answer") or "").strip()
        if task_type and task_type != "ALL":
            ocr_concat, ocr_spans = load_ocr(r["image_uid"])
            ocr_concat_norm = normalize(ocr_concat)
            ocr_spans_norm = [normalize(s) for s in ocr_spans]
            gold_in_ocr = text_in_ocr(gold, ocr_concat_norm, ocr_spans_norm)
            tt = "EXTRACTIVE" if gold_in_ocr else "ABSTRACTIVE"
            if tt != task_type:
                continue
        pool.append(r)
    if max_per_language:
        from collections import defaultdict
        by = defaultdict(list)
        for r in pool:
            by[r["language"]].append(r)
        pool = []
        for lang, rs in by.items():
            pool.extend(rs[:max_per_language])
    return pool


def build_instruction_plain() -> str:
    """Instruction-only text; runner appends 'Question: {question}'."""
    return PROMPT_INSTRUCTION


def build_instruction_ocr(ocr_text: str, max_chars: int) -> str:
    if max_chars and len(ocr_text) > max_chars:
        ocr_text = ocr_text[:max_chars] + " ..."
    return (
        "You are looking at an image. The text visible in the image is "
        "provided below as OCR output. Use both the image and this text to "
        "answer the question. Answer briefly in the same language as the "
        "question — output just the answer, no explanation.\n\n"
        f"OCR text:\n{ocr_text}"
    )


def run_condition(runner, rows, prompt_mode, ocr_prompt_max_chars,
                   max_new_tokens, tag, out_dir):
    out_jsonl = out_dir / f"per_pair_{tag}.jsonl"
    fh = out_jsonl.open("w", encoding="utf-8")
    t0 = time.time()
    out_rows = []
    em_count = 0
    halluc_count = 0
    for i, r in enumerate(rows, 1):
        image_path = reconstruct_image_path(r["image_uid"])
        ocr_concat, _ = load_ocr(r["image_uid"])
        if prompt_mode == "plain":
            prompt_text = build_instruction_plain()
        else:
            prompt_text = build_instruction_ocr(ocr_concat, ocr_prompt_max_chars)
        try:
            # InternVLRunner constructs: "<image>\n{prompt_text}\nQuestion: {question}"
            gen_text = runner.generate(image_path, r["question"], max_new_tokens,
                                          prompt_text)
        except Exception as e:
            print(f"[warn] gen failed row {i}: {e}", flush=True)
            continue
        cls = classify_row(r, gen_text)
        if cls["em"]:
            em_count += 1
        if cls["failure_mode"] == "HALLUCINATION":
            halluc_count += 1
        out_rows.append(cls)
        fh.write(json.dumps(cls, ensure_ascii=False) + "\n")
        fh.flush()
        if i % 20 == 0 or i == len(rows):
            el = time.time() - t0
            print(f"  [{tag}] {i}/{len(rows)}  EM={em_count} HAL={halluc_count}  "
                  f"t={el:.0f}s", flush=True)
    fh.close()
    return out_rows


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--val-path", default=str(VAL_PATH))
    p.add_argument("--model-id", default="OpenGVLab/InternVL2_5-8B")
    p.add_argument("--languages", nargs="+",
                   default=["ar", "ja", "kr", "ru", "th"])
    p.add_argument("--task-type", choices=["ALL", "EXTRACTIVE", "ABSTRACTIVE"],
                   default="EXTRACTIVE")
    p.add_argument("--max-per-language", type=int, default=50)
    p.add_argument("--prompt-mode", choices=["plain", "ocr_prepended"],
                   default="ocr_prepended")
    p.add_argument("--ocr-prompt-max-chars", type=int, default=1500)
    p.add_argument("--include-baseline", action="store_true",
                   help="Also run plain prompt baseline.")
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    return p


def main():
    args = build_parser().parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    val_rows = read_jsonl(Path(args.val_path))
    pool = filter_pool(val_rows, args.languages, args.task_type,
                         args.max_per_language)
    print(f"[data] eval pool: {len(pool)} rows × "
          f"{len(set(r['language'] for r in pool))} langs (task={args.task_type})",
          flush=True)

    print(f"[model] loading {args.model_id}", flush=True)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    runner = InternVLRunner(model_id=args.model_id, device="cuda",
                              dtype=dtype, trust_remote_code=True)

    summary: Dict[str, Any] = {"args": vars(args), "cells": []}

    if args.include_baseline:
        print("\n[run] plain prompt baseline", flush=True)
        rows_b = run_condition(runner, pool, "plain", 1500,
                                 args.max_new_tokens, "baseline", out_dir)
        summary["cells"].append({"tag": "baseline", "overall": summarize(rows_b)})

    tag = args.prompt_mode
    print(f"\n[run] {tag}", flush=True)
    rows_t = run_condition(runner, pool, args.prompt_mode,
                             args.ocr_prompt_max_chars, args.max_new_tokens,
                             tag, out_dir)
    summary["cells"].append({"tag": tag, "overall": summarize(rows_t)})

    summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print(f"{'tag':<24s}  {'n':>4s}  {'EM':>6s}  {'HAL':>6s}  {'WSPAN':>6s}  {'predOCR':>8s}")
    print("-" * 80)
    for c in summary["cells"]:
        ov = c["overall"]
        if ov.get("n", 0) == 0:
            continue
        # Reconstruct from failure_modes since summarize() returns counts dict
        n = ov["n"]
        fm = ov.get("failure_modes", {})
        hal = fm.get("HALLUCINATION", 0) / n
        wspan = fm.get("WRONG_SPAN_FROM_IMAGE", 0) / n
        print(f"{c['tag']:<24s}  {n:>4d}  "
              f"{ov['em_rate']:.4f}  {hal:.4f}  {wspan:.4f}  "
              f"{ov['pred_in_ocr_rate']:.4f}")
    print("=" * 80)

    if args.include_baseline and len(summary["cells"]) >= 2:
        b = summary["cells"][0]["overall"]
        v = summary["cells"][-1]["overall"]
        d_em = v["em_rate"] - b["em_rate"]
        print(f"\n[ΔEM]={d_em:+.4f}  (cf. Qwen-7B Δ ≈ 0 → universality test)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
