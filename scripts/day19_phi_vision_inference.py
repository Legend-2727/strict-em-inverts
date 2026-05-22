#!/usr/bin/env python3
"""Day-19 / P0-B (variant): Phi-3.5-vision-instruct inference on 358-row MTVQA
held-out. Third VLM family (Microsoft Phi) for the architecture-diversity claim.

Output: results/day19_phi35vision_heldout/per_pair_base.jsonl  (same schema as
the InternVL/Qwen per_pair_base files so existing analysis scripts work).
"""
from __future__ import annotations
import argparse, json, sys, time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import Image

# Patch DynamicCache for compatibility with Phi-3.5-vision's older modeling code.
# (transformers 4.57 removed several legacy methods that Phi's custom modeling expects)
from transformers.cache_utils import DynamicCache as _DC
if not hasattr(_DC, "seen_tokens"):
    _DC.seen_tokens = property(lambda self: self.get_seq_length())
if not hasattr(_DC, "get_max_length"):
    _DC.get_max_length = lambda self: None
if not hasattr(_DC, "get_usable_length"):
    _DC.get_usable_length = lambda self, new_seq_length, layer_idx=0: self.get_seq_length(layer_idx)

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from diagnose_mtvqa_per_language import (
    PROMPT_INSTRUCTION, classify, dominant_script, extract_answer,
    load_ocr, normalize, script_compatible, text_in_ocr, token_f1,
)

VAL_PATH = REPO / "data/processed/mtvqa/val.jsonl"
EVAL_IMGS = REPO / "data/processed/dpo_pairs_real_failures/eval_image_uids.json"
IMG_DIR = REPO / "data/processed/mtvqa/images"


def read_jsonl(p):
    return [json.loads(l) for l in p.open() if l.strip()]


def img_path(image_uid):
    if image_uid.startswith("train__"): split = "train"
    elif image_uid.startswith("test__"): split = "test"
    else: split = "val"
    return IMG_DIR / split / f"{image_uid}.png"


def classify_row(row, gen_text):
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
        "language": row["language"], "image_uid": row["image_uid"],
        "question": row["question"], "gold": gold, "gen_text": gen_text,
        "pred": pred, "em": em, "substring": sub, "f1": f1,
        "gold_script": gold_script, "pred_script": pred_script,
        "script_compatible": script_compatible(pred_script, gold_script),
        "gold_in_ocr": gold_in_ocr, "pred_in_ocr": pred_in_ocr,
        "task_type": task_type, "failure_mode": failure,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="results/day19_phi35vision_heldout")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--held-out-only", action="store_true", default=True)
    args = ap.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_imgs = set(json.loads(EVAL_IMGS.read_text()))
    rows = read_jsonl(VAL_PATH)
    if args.held_out_only:
        rows = [r for r in rows if r["image_uid"] in eval_imgs]
    if args.limit:
        rows = rows[:args.limit]
    print(f"[data] {len(rows)} MTVQA val rows ({Counter(r['language'] for r in rows)})",
          flush=True)

    print("[model] loading microsoft/Phi-3.5-vision-instruct ...", flush=True)
    from transformers import AutoModelForCausalLM, AutoProcessor
    model_id = "microsoft/Phi-3.5-vision-instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=torch.bfloat16,
        _attn_implementation="eager",
        device_map="cuda",
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True,
                                                num_crops=4)
    print(f"[model] loaded. params={sum(p.numel() for p in model.parameters())/1e9:.1f}B",
          flush=True)

    out_p = out_dir / "per_pair_base.jsonl"
    fh = out_p.open("w", encoding="utf-8")
    all_rows = []
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        path = img_path(r["image_uid"])
        if not path.exists():
            print(f"[skip] image missing: {path}", flush=True)
            continue
        try:
            image = Image.open(path).convert("RGB")
            messages = [
                {"role": "user", "content": f"<|image_1|>\n{PROMPT_INSTRUCTION}\n\nQuestion: {r['question']}"},
            ]
            prompt = processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(prompt, [image], return_tensors="pt").to("cuda")
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=80,
                                       do_sample=False, temperature=0.0,
                                       eos_token_id=processor.tokenizer.eos_token_id)
            gen_ids = gen[:, inputs["input_ids"].shape[1]:]
            text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
        except Exception as e:
            print(f"[warn] row {i} ({r['image_uid']}): {type(e).__name__}: {str(e)[:120]}",
                  flush=True)
            continue
        cls = classify_row(r, text)
        all_rows.append(cls)
        fh.write(json.dumps(cls, ensure_ascii=False) + "\n"); fh.flush()
        if i % 25 == 0 or i == len(rows):
            el = time.time() - t0
            em_so_far = sum(x["em"] for x in all_rows) / max(len(all_rows),1)
            rate = len(all_rows)/max(el,1)
            print(f"  {i}/{len(rows)}  EM={em_so_far:.4f}  rate={rate:.2f}/s  t={el:.0f}s",
                  flush=True)
    fh.close()

    by_lang = defaultdict(list)
    for r in all_rows:
        by_lang[r["language"]].append(r)
    summary = {
        "per_language": {lang: {"n": len(rs),
                                "em_rate": sum(r["em"] for r in rs)/max(len(rs),1)}
                          for lang, rs in sorted(by_lang.items())},
        "overall_n": len(all_rows),
        "overall_em": sum(r["em"] for r in all_rows)/max(len(all_rows),1),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "summary_per_language.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n[done] {out_dir}")
    print(f"Overall: EM={summary['overall_em']:.4f}  n={summary['overall_n']}")
    print("Per-language:")
    for lang, s in summary["per_language"].items():
        print(f"  {lang}: n={s['n']}  EM={s['em_rate']:.4f}")


if __name__ == "__main__":
    main()
