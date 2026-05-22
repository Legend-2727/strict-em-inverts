#!/usr/bin/env python3
"""Day-18: XFUND semantic-EM judge for within-script gap analysis.

Reads existing XFUND base Qwen2.5-VL-7B predictions (full test set 10,333 rows;
default uses scout50 stratified 50/lang × 7 langs = 350 rows for cost). For
each (image, question, ground_truth, extracted_prediction) call Gemini Flash
adjudicator → CORRECT / PARTIAL / INCORRECT.

Critical question: does the within-CJK Japanese–Chinese gap (~20pp strict-EM
reported in current draft) survive semantic eval? Or is it also artifact?

Outputs: results/day18_xfund_semantic/{judgements.jsonl, summary.md}
"""
from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict

from google import genai
from google.genai import types as genai_types

REPO = Path(__file__).resolve().parent.parent
IMG_ROOT = REPO / "data/raw/xfund"
OUT_DIR = REPO / "results/day18_xfund_semantic"

JUDGE_PROMPT = """You are evaluating a Visual Question Answering (VQA) model's prediction against a gold-standard answer.

Given the document image, the question, the gold answer, and the model's prediction, decide whether the prediction is:
  - CORRECT: semantically equivalent to gold (synonyms, whitespace, diacritics, alternate phrasing, number-vs-numeral, valid translation, lossless reformat — same meaning).
  - PARTIAL: contains correct information but is incomplete or less specific than gold (e.g., gives surname without first name; gives one item from a list).
  - INCORRECT: states something different from gold, fabricates information not supported by the image, or is unrelated.

You MUST look at the image. If the gold answer itself is wrong and the model's prediction matches the image better, judge against the IMAGE evidence, not the gold.

Respond with ONLY a valid JSON object with two fields, both short:
  {"verdict": "CORRECT"|"PARTIAL"|"INCORRECT", "reason": "<one short sentence, under 30 words>"}

Question: {question}
Gold answer: {gold}
Model prediction: {prediction}
"""


def resolve_image(document_id: str, language: str) -> Path:
    """document_id like 'de_test_0001_de_val_1' → data/raw/xfund/de.val/de_val_1.jpg
    Also handles 'de_train_0001_de_train_0' → data/raw/xfund/de.train/de_train_0.jpg"""
    m = re.match(r"^[a-z]+_(?:test|train)_\d+_([a-z]+_(val|train)_\d+)$", document_id)
    if not m:
        return None
    inner_id = m.group(1)
    split = m.group(2)
    return IMG_ROOT / f"{language}.{split}" / f"{inner_id}.jpg"


def strip_to_json(text):
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{[^{}]*\}", s, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        vm = re.search(r'"verdict"\s*:\s*"(CORRECT|PARTIAL|INCORRECT)"', s, re.IGNORECASE)
        if vm:
            return {"verdict": vm.group(1).upper(), "reason": "[recovered]"}
        return None


def judge_one(client, model_name, img_path, question, gold, prediction,
              max_tokens=4096, max_retries=6):
    with img_path.open("rb") as f:
        image_bytes = f.read()
    ext = img_path.suffix.lower()
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
    prompt = JUDGE_PROMPT.replace("{question}", question or "") \
                         .replace("{gold}", gold or "") \
                         .replace("{prediction}", prediction or "")
    contents = [
        genai_types.Part.from_bytes(data=image_bytes, mime_type=mime),
        genai_types.Part.from_text(text=prompt),
    ]
    last_err = None
    for attempt in range(max_retries):
        try:
            t0 = time.time()
            resp = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=genai_types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=max_tokens,
                    response_mime_type="application/json",
                ),
            )
            elapsed = time.time() - t0
            raw = resp.text.strip() if resp.text else ""
            return {"raw": raw, "parsed": strip_to_json(raw),
                    "elapsed_s": round(elapsed, 2)}
        except Exception as e:
            last_err = e
            msg = str(e)
            if any(k in msg for k in ("429", "RESOURCE_EXHAUSTED",
                                       "503", "UNAVAILABLE", "DEADLINE")):
                time.sleep((2 ** attempt) + 0.5)
                continue
            raise
    raise last_err


def load_cache(cache_path: Path):
    if not cache_path.exists():
        return {}
    out = {}
    for line in cache_path.open():
        s = line.strip().lstrip("\x00")
        if not s:
            continue
        try:
            r = json.loads(s)
        except json.JSONDecodeError:
            continue
        out[r["sample_id"]] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions",
                    default="results/xfund_qa/venc_occlusion_s42_scout50/source_baseline/predictions_source_baseline.jsonl",
                    help="XFUND predictions jsonl (default: scout50 base, 350 rows)")
    ap.add_argument("--judge-model", default="gemini-2.5-flash")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = OUT_DIR / "judgements.jsonl"
    cache = load_cache(cache_path)
    print(f"[cache] {len(cache)} existing", flush=True)

    client = genai.Client(vertexai=True,
                          project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
                          location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"))

    rows = []
    for line in Path(args.predictions).open():
        s = line.strip().lstrip("\x00")
        if not s:
            continue
        try:
            r = json.loads(s)
        except json.JSONDecodeError:
            continue
        rows.append(r)
    if args.limit:
        rows = rows[: args.limit]
    print(f"[load] {len(rows)} rows from {args.predictions}", flush=True)

    queue = [r for r in rows
             if r["sample_id"] not in cache or cache[r["sample_id"]].get("verdict") == "?"]
    print(f"[queue] {len(queue)} pending ({args.workers} workers)", flush=True)

    lock = threading.Lock()
    fh = cache_path.open("a", encoding="utf-8")
    done = [0]
    t0 = time.time()
    errors = [0]

    def worker(item):
        try:
            img = resolve_image(item["document_id"], item["language"])
            if not img or not img.exists():
                rec = {"sample_id": item["sample_id"], "language": item["language"],
                       "document_id": item["document_id"], "verdict": "INCORRECT",
                       "reason": "image-missing", "raw": "",
                       "gold": item.get("ground_truth", ""),
                       "pred": item.get("extracted_prediction", ""),
                       "strict_em": item.get("strict_em", False),
                       "normalized_em": item.get("normalized_em", False),
                       "elapsed_s": 0.0, "judged_at": time.time()}
            else:
                j = judge_one(client, args.judge_model, img,
                              item.get("question", ""),
                              item.get("ground_truth", ""),
                              item.get("extracted_prediction", ""))
                p = j["parsed"] or {}
                rec = {
                    "sample_id": item["sample_id"], "language": item["language"],
                    "document_id": item["document_id"],
                    "question": item.get("question", ""),
                    "gold": item.get("ground_truth", ""),
                    "pred": item.get("extracted_prediction", ""),
                    "strict_em": item.get("strict_em", False),
                    "normalized_em": item.get("normalized_em", False),
                    "verdict": (p.get("verdict") or "?").upper(),
                    "reason": p.get("reason", "")[:300],
                    "raw": j["raw"][:500],
                    "elapsed_s": j["elapsed_s"],
                    "judged_at": time.time(),
                }
            with lock:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                cache[item["sample_id"]] = rec
                done[0] += 1
                if done[0] % 20 == 0 or done[0] == len(queue):
                    el = time.time() - t0
                    rate = done[0] / el if el > 0 else 0
                    eta = (len(queue) - done[0]) / rate if rate > 0 else 0
                    print(f"  {done[0]}/{len(queue)}  rate={rate:.2f}/s  ETA={eta/60:.1f}min",
                          flush=True)
        except Exception as e:
            with lock:
                done[0] += 1
                errors[0] += 1
            print(f"[error] {item.get('sample_id')}: {type(e).__name__}: {str(e)[:100]}",
                  flush=True)

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(worker, queue))
    finally:
        fh.close()

    cache = load_cache(cache_path)
    relevant = [cache[r["sample_id"]] for r in rows if r["sample_id"] in cache]
    by_v = Counter(r["verdict"] for r in relevant)
    n = sum(by_v.values())
    print()
    print(f"[summary] XFUND scout50 (n={n}, errors={errors[0]}):")
    print(f"  " + "  ".join(f"{k}={v} ({v/n*100:.1f}%)" for k, v in by_v.most_common()))


if __name__ == "__main__":
    main()
