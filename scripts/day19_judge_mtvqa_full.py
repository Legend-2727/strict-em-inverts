#!/usr/bin/env python3
"""Day-19 / P0-C: Gemini semantic-EM judge on the full MTVQA val (n=2203).

Reads base Qwen2.5-VL-7B predictions from
results/mtvqa_perlang_diagnostic/per_pair_base.jsonl, judges with
Gemini-2.5-Flash via Vertex AI, writes to
results/day19_mtvqa_full_semantic/judgements.jsonl.

Cache-resumable. 4 workers, retry on 429 with exponential backoff.
"""
from __future__ import annotations
import argparse, json, os, re, threading, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from google import genai
from google.genai import types as genai_types

REPO = Path(__file__).resolve().parent.parent
IMG_DIR = REPO / "data/processed/mtvqa/images"
OUT_DIR = REPO / "results/day19_mtvqa_full_semantic"

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


def reconstruct_image_path(image_uid: str) -> Path:
    if image_uid.startswith("train__"):
        split = "train"
    elif image_uid.startswith("test__"):
        split = "test"
    else:
        split = "val"
    return IMG_DIR / split / f"{image_uid}.png"


def strip_to_json(text):
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    try: return json.loads(s)
    except: pass
    m = re.search(r"\{[^{}]*\}", s, re.DOTALL)
    if m:
        try: return json.loads(m.group(0))
        except: pass
    vm = re.search(r'"verdict"\s*:\s*"(CORRECT|PARTIAL|INCORRECT)"', s, re.IGNORECASE)
    if vm:
        return {"verdict": vm.group(1).upper(), "reason": "[recovered]"}
    return None


def judge_one(client, model_name, img_path, question, gold, prediction,
              max_tokens=4096, max_retries=6):
    with img_path.open("rb") as f:
        image_bytes = f.read()
    prompt = JUDGE_PROMPT.replace("{question}", question or "") \
                         .replace("{gold}", gold or "") \
                         .replace("{prediction}", prediction or "")
    contents = [
        genai_types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
        genai_types.Part.from_text(text=prompt),
    ]
    last_err = None
    for attempt in range(max_retries):
        try:
            t0 = time.time()
            resp = client.models.generate_content(
                model=model_name, contents=contents,
                config=genai_types.GenerateContentConfig(
                    temperature=0.0, max_output_tokens=max_tokens,
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


def load_cache(p):
    if not p.exists(): return {}
    out = {}
    for line in p.open():
        s = line.strip().lstrip("\x00")
        if not s: continue
        try: r = json.loads(s)
        except: continue
        out[r["sample_id"]] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions",
                    default="results/mtvqa_perlang_diagnostic/per_pair_base.jsonl")
    ap.add_argument("--judge-model", default="gemini-2.5-flash")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "judgements.jsonl"
    cache = load_cache(cache_path)
    print(f"[cache] {len(cache)} existing", flush=True)

    client = genai.Client(vertexai=True,
        project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
        location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"))

    rows = []
    for line in Path(args.predictions).open():
        s = line.strip().lstrip("\x00")
        if not s: continue
        try: r = json.loads(s)
        except: continue
        rows.append(r)
    if args.limit:
        rows = rows[:args.limit]
    print(f"[load] {len(rows)} rows from {args.predictions}", flush=True)

    queue = [r for r in rows
             if r["sample_id"] not in cache or cache[r["sample_id"]].get("verdict") == "?"]
    print(f"[queue] {len(queue)} pending ({args.workers} workers)", flush=True)

    lock = threading.Lock()
    fh = cache_path.open("a", encoding="utf-8")
    done = [0]; errors = [0]
    t0 = time.time()

    def worker(item):
        try:
            img = reconstruct_image_path(item["image_uid"])
            if not img.exists():
                rec = {"sample_id": item["sample_id"], "language": item["language"],
                       "image_uid": item["image_uid"], "verdict": "INCORRECT",
                       "reason": "image-missing", "raw": "",
                       "gold": item.get("gold", ""), "pred": item.get("pred", ""),
                       "question": item.get("question", ""),
                       "strict_em": bool(item.get("em", False)),
                       "elapsed_s": 0.0, "judged_at": time.time()}
            else:
                j = judge_one(client, args.judge_model, img,
                              item.get("question", ""), item.get("gold", ""),
                              item.get("pred", ""))
                p = j["parsed"] or {}
                rec = {
                    "sample_id": item["sample_id"], "language": item["language"],
                    "image_uid": item["image_uid"],
                    "question": item.get("question", ""),
                    "gold": item.get("gold", ""), "pred": item.get("pred", ""),
                    "strict_em": bool(item.get("em", False)),
                    "verdict": (p.get("verdict") or "?").upper(),
                    "reason": p.get("reason", "")[:300],
                    "raw": j["raw"][:500], "elapsed_s": j["elapsed_s"],
                    "judged_at": time.time(),
                }
            with lock:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
                cache[item["sample_id"]] = rec
                done[0] += 1
                if done[0] % 50 == 0 or done[0] == len(queue):
                    el = time.time() - t0
                    rate = done[0] / el if el > 0 else 0
                    eta = (len(queue) - done[0]) / rate if rate > 0 else 0
                    print(f"  {done[0]}/{len(queue)}  rate={rate:.2f}/s  ETA={eta/60:.1f}min",
                          flush=True)
        except Exception as e:
            with lock:
                done[0] += 1; errors[0] += 1
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
    print(f"[summary] MTVQA-full (n={n}, errors={errors[0]}):")
    for k, v in by_v.most_common():
        print(f"  {k}={v} ({v/n*100:.1f}%)")


if __name__ == "__main__":
    main()
