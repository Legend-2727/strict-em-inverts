#!/usr/bin/env python3
"""Day-10: MTVQA per-language diagnostic on InternVL2.5-8B (non-Qwen family).

Mirrors `diagnose_mtvqa_per_language.py` but uses the existing InternVLRunner
from `scripts/evaluate_xfund_qa_zeroshot_vlm.py`. Output format identical so
`compare_models_diagnostic.py` can compare across models.

Goal: test if the "fixed correct-answer budget" pattern (5 test-time
interventions → ΔEM ≈ 0) is universal across VLM families, or specific to
Qwen-VL.

Usage:
  /home/ubuntu/xLingual/.venv/bin/python scripts/diagnose_mtvqa_internvl.py \
      --output-dir results/mtvqa_perlang_internvl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from diagnose_mtvqa_per_language import (  # noqa: E402
    PROMPT_INSTRUCTION, build_prompt, classify, dominant_script,
    extract_answer, load_ocr, normalize, script_compatible,
    text_in_ocr, token_f1,
)
from evaluate_xfund_qa_zeroshot_vlm import InternVLRunner  # noqa: E402

VAL_PATH = REPO_ROOT / "data/processed/mtvqa/val.jsonl"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in path.open("r", encoding="utf-8") if l.strip()]


def reconstruct_image_path(image_uid: str) -> str:
    if image_uid.startswith("train__"):
        split = "train"
    elif image_uid.startswith("test__"):
        split = "test"
    else:
        split = "val"
    return str(REPO_ROOT / f"data/processed/mtvqa/images/{split}/{image_uid}.png")


def classify_row(row: Dict[str, Any], gen_text: str) -> Dict[str, Any]:
    pred = extract_answer(gen_text)
    gold = (row.get("answer") or "").strip()
    p_norm, g_norm = normalize(pred), normalize(gold)
    em = (p_norm == g_norm) and bool(g_norm)
    sub = bool(g_norm) and ((g_norm in p_norm) or (p_norm in g_norm and len(p_norm) >= 2))
    f1 = token_f1(pred, gold)
    gold_script = dominant_script(gold)
    pred_script = dominant_script(pred)
    ocr_concat, ocr_spans = load_ocr(row["image_uid"])
    ocr_concat_norm = normalize(ocr_concat)
    ocr_spans_norm = [normalize(s) for s in ocr_spans]
    gold_in_ocr = text_in_ocr(gold, ocr_concat_norm, ocr_spans_norm)
    pred_in_ocr = text_in_ocr(pred, ocr_concat_norm, ocr_spans_norm) if pred else False
    if em:
        task_type = "EXTRACTIVE" if gold_in_ocr else "ABSTRACTIVE"
        failure = "OK"
    else:
        task_type, failure = classify(pred, gold, pred_script, gold_script,
                                       gold_in_ocr, pred_in_ocr, f1)
    return {
        "sample_id": row.get("sample_id"),
        "language": row["language"],
        "image_uid": row["image_uid"],
        "question": row["question"],
        "gold": gold,
        "gen_text": gen_text,
        "pred": pred,
        "em": em,
        "substring": sub,
        "f1": f1,
        "gold_script": gold_script,
        "pred_script": pred_script,
        "script_compatible": script_compatible(pred_script, gold_script),
        "gold_in_ocr": gold_in_ocr,
        "pred_in_ocr": pred_in_ocr,
        "task_type": task_type,
        "failure_mode": failure,
    }


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"n": 0}
    n = len(rows)
    fm = Counter(r["failure_mode"] for r in rows)
    return {
        "n": n,
        "em_rate": sum(r["em"] for r in rows) / n,
        "substring_rate": sum(r["substring"] for r in rows) / n,
        "f1_avg": sum(r["f1"] for r in rows) / n,
        "gold_in_ocr_rate": sum(r["gold_in_ocr"] for r in rows) / n,
        "pred_in_ocr_rate": sum(r["pred_in_ocr"] for r in rows) / n,
        "script_compatible_rate": sum(r["script_compatible"] for r in rows) / n,
        "failure_modes": dict(fm),
    }


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--val-path", default=str(VAL_PATH))
    p.add_argument("--model-id", default="OpenGVLab/InternVL2_5-8B")
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    return p


def main():
    args = build_parser().parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    val_rows = read_jsonl(Path(args.val_path))
    if args.max_rows:
        val_rows = val_rows[:args.max_rows]
    print(f"[data] {len(val_rows)} MTVQA val rows", flush=True)
    print(f"[data] languages: {Counter(r['language'] for r in val_rows)}",
          flush=True)

    print(f"[model] loading {args.model_id}", flush=True)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    runner = InternVLRunner(model_id=args.model_id, device="cuda", dtype=dtype,
                              trust_remote_code=True)

    out_jsonl = out_dir / "per_pair_base.jsonl"
    fh = out_jsonl.open("w", encoding="utf-8")

    all_rows: List[Dict[str, Any]] = []
    t0 = time.time()
    for i, r in enumerate(val_rows, 1):
        image_path = reconstruct_image_path(r["image_uid"])
        # InternVLRunner.generate expects (image_path, question, max_new_tokens, prompt_text)
        prompt_text = PROMPT_INSTRUCTION
        try:
            gen_text = runner.generate(image_path, r["question"],
                                          args.max_new_tokens, prompt_text)
        except Exception as e:
            print(f"[warn] gen failed at row {i} ({r['image_uid']}): {e}",
                  flush=True)
            continue
        cls = classify_row(r, gen_text)
        all_rows.append(cls)
        fh.write(json.dumps(cls, ensure_ascii=False) + "\n")
        fh.flush()
        if i % 50 == 0 or i == len(val_rows):
            el = time.time() - t0
            em_so_far = sum(x["em"] for x in all_rows) / max(len(all_rows), 1)
            rate = len(all_rows) / max(el, 1)
            print(f"  {i}/{len(val_rows)}  EM={em_so_far:.4f}  "
                  f"rate={rate:.2f}/s  t={el:.0f}s",
                  flush=True)
    fh.close()

    # Build summary
    by_lang = defaultdict(list)
    for r in all_rows:
        by_lang[r["language"]].append(r)

    summary = {
        "per_language": {lang: summarize(rs) for lang, rs in sorted(by_lang.items())},
        "per_language_extractive": {
            lang: summarize([r for r in rs if r["task_type"] == "EXTRACTIVE"])
            for lang, rs in sorted(by_lang.items())
        },
        "per_language_abstractive": {
            lang: summarize([r for r in rs if r["task_type"] == "ABSTRACTIVE"])
            for lang, rs in sorted(by_lang.items())
        },
        "overall": summarize(all_rows),
        "args": vars(args),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "summary_per_language.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[done] {out_dir}", flush=True)
    print(f"\nOverall: EM={summary['overall']['em_rate']:.4f}  "
          f"n={summary['overall']['n']}", flush=True)
    print("\nPer-language EM:", flush=True)
    for lang, s in summary["per_language"].items():
        print(f"  {lang}: n={s['n']}  EM={s['em_rate']:.4f}  "
              f"pred_in_OCR={s['pred_in_ocr_rate']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
