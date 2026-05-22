#!/usr/bin/env python3
"""Day-21: Cross-FAMILY judge using Claude Sonnet 4 via OpenRouter.

Addresses reviewer-defense gap R3 from the hostile-review pass:
the existing Gemini-2.5-Flash vs Gemini-2.5-Pro cross-judge is INTRA-family
(both Gemini, same RLHF lineage). Running Claude on the same 716 (Qwen +
InternVL) held-out tuples gives a genuine cross-FAMILY check.

Uses the identical Appendix-A prompt and identical (image, question, gold,
prediction) inputs as the Gemini runs. Writes verdicts row-by-row to disk so
it's resumable on interruption.

Inputs:
  - results/day19_cross_judge_pro/judgements.jsonl  (716 tuples; we re-use the
    pred/gold/question/image_uid fields, drop the Gemini verdict)
  - MTVQA train parquet shards in the local HF hub cache (already downloaded
    by scripts/day21_extract_human_validation_images.py)
  - OPENROUTER_API_KEY in <repo>/../secrets/.env

Output:
  results/day21_cross_judge_claude/judgements.jsonl

Usage:
  python scripts/day21_judge_claude_crossfamily.py \
      --model anthropic/claude-sonnet-4 \
      --concurrency 2
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq
from dotenv import load_dotenv
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
SECRETS_ENV = REPO.parent / "secrets" / ".env"
INPUT_JSONL = REPO / "results/day19_cross_judge_pro/judgements.jsonl"
OUT_DIR = REPO / "results/day21_cross_judge_claude"
OUT_JSONL = OUT_DIR / "judgements.jsonl"
HUB = Path.home() / ".cache/huggingface/hub/datasets--ByteDance--MTVQA"

UID_RE = re.compile(r"^(?P<split>train|val|test)__(?P<idx>\d{5})__(?P<src>.+)$")

PROMPT = (
    "You are evaluating a Visual Question Answering (VQA) model's prediction "
    "against a gold-standard answer.\n\n"
    "Given the document image, the question, the gold answer, and the model's "
    "prediction, decide whether the prediction is:\n"
    "- CORRECT: semantically equivalent to gold (synonyms, whitespace, "
    "diacritics, alternate phrasing, number-vs-numeral, valid translation, "
    "lossless reformat -- same meaning).\n"
    "- PARTIAL: contains correct information but is incomplete or less specific "
    "than gold (e.g., gives surname without first name; gives one item from a list).\n"
    "- INCORRECT: states something different from gold, fabricates information "
    "not supported by the image, or is unrelated.\n\n"
    "You MUST look at the image. If the gold answer itself is wrong and the "
    "model's prediction matches the image better, judge against the IMAGE "
    "evidence, not the gold.\n\n"
    'Output FORMAT (strict): respond with ONLY a single-line JSON object, no '
    'preamble, no reasoning, no markdown fence. Exactly:\n'
    '{"verdict":"CORRECT"|"PARTIAL"|"INCORRECT","reason":"<one short sentence>"}\n\n'
)


# ---------------------------------------------------------------------------
# Image loading from local MTVQA parquet shards
# ---------------------------------------------------------------------------

def _shard_cum_counts(split: str) -> tuple[list[Path], list[int]]:
    """Return (shards, cum_counts) for split. cum_counts[i] = total rows before shard i."""
    snap = next((HUB / "snapshots").iterdir())
    shards = sorted((snap / "data").glob(f"{split}-*.parquet"))
    cum = [0]
    for sh in shards:
        cum.append(cum[-1] + pq.read_metadata(str(sh)).num_rows)
    return shards, cum


class ImageLoader:
    def __init__(self) -> None:
        self.shards, self.cum = _shard_cum_counts("train")
        self._cache: dict[int, "pq.Table"] = {}

    def get(self, uid: str) -> Image.Image:
        m = UID_RE.match(uid)
        if not m:
            raise ValueError(f"bad image_uid: {uid}")
        gidx = int(m.group("idx"))
        # find shard
        shard_i = 0
        while shard_i + 1 < len(self.cum) and self.cum[shard_i + 1] <= gidx:
            shard_i += 1
        local = gidx - self.cum[shard_i]
        if shard_i not in self._cache:
            self._cache[shard_i] = pq.read_table(str(self.shards[shard_i]), columns=["image"])
        cell = self._cache[shard_i].column("image")[local].as_py()
        if isinstance(cell, dict) and cell.get("bytes"):
            return Image.open(io.BytesIO(cell["bytes"])).convert("RGB")
        if isinstance(cell, dict) and cell.get("path"):
            return Image.open(cell["path"]).convert("RGB")
        raise ValueError(f"unknown image format at gidx={gidx}: {type(cell).__name__}")


# ---------------------------------------------------------------------------
# Claude judge
# ---------------------------------------------------------------------------

def _b64_jpeg(img: Image.Image, quality: int = 85) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _parse_verdict(raw: str) -> tuple[str, str]:
    """Robust verdict-recovery from Claude's response.

    Strategy:
      1. Try strict JSON parse (handles markdown code fences and trailing junk).
      2. Fall back to prose-extraction: scan for CORRECT/PARTIAL/INCORRECT
         tokens in the text, prefer the LAST occurrence as the final verdict
         (Claude's reasoning often considers multiple options before committing).
    """
    s = raw.strip()
    # ---- (1) strict JSON path ----
    s_json = s
    if s_json.startswith("```"):
        s_json = re.sub(r"^```(?:json)?\s*", "", s_json)
        s_json = re.sub(r"\s*```\s*$", "", s_json)
    i = s_json.find("{")
    j = s_json.rfind("}")
    if i >= 0 and j > i:
        try:
            d = json.loads(s_json[i : j + 1])
            v = (d.get("verdict") or "").strip().upper()
            if v in ("CORRECT", "PARTIAL", "INCORRECT"):
                return (v, str(d.get("reason", "")).strip())
        except json.JSONDecodeError:
            pass

    # ---- (2) prose fallback: pick the last unambiguous verdict word ----
    # Match the bare words as whole tokens (avoid matching e.g. "incorrectly" → INCORRECT).
    matches = list(re.finditer(r"\b(CORRECT|PARTIAL|INCORRECT)\b", s, flags=re.IGNORECASE))
    if matches:
        m = matches[-1]
        v = m.group(1).upper()
        # Pull a short reason from the surrounding sentence (~80 chars window).
        start = max(0, m.start() - 80)
        end = min(len(s), m.end() + 80)
        snippet = s[start:end].replace("\n", " ").strip()
        return (v, f"prose-recovered: {snippet[:200]}")

    return ("MISSING", f"unparseable: {raw[:160]}")


def _call_openrouter(client, model: str, b64: str, user_text: str):
    """OpenRouter via openai SDK. Returns (raw_text, prompt_tokens, completion_tokens)."""
    resp = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": user_text},
            ],
        }],
        max_tokens=256,
        temperature=0,
    )
    return (
        resp.choices[0].message.content or "",
        resp.usage.prompt_tokens if resp.usage else None,
        resp.usage.completion_tokens if resp.usage else None,
    )


def _call_anthropic(client, model: str, b64: str, user_text: str):
    """Anthropic-direct SDK. Returns (raw_text, prompt_tokens, completion_tokens)."""
    resp = client.messages.create(
        model=model,
        max_tokens=256,
        temperature=0,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": user_text},
            ],
        }],
    )
    raw = "".join(b.text for b in resp.content if hasattr(b, "text"))
    return (raw, resp.usage.input_tokens, resp.usage.output_tokens)


def judge_one(
    client,
    provider: str,
    model: str,
    row: dict,
    loader: ImageLoader,
    max_retries: int = 6,
) -> dict:
    img = loader.get(row["image_uid"])
    b64 = _b64_jpeg(img)
    user_text = PROMPT + (
        f"Question: {row['question']}\n"
        f"Gold: {row['gold']}\n"
        f"Prediction: {row['pred']}\n"
    )

    call = _call_openrouter if provider == "openrouter" else _call_anthropic
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            t0 = time.time()
            raw, ptok, ctok = call(client, model, b64, user_text)
            verdict, reason = _parse_verdict(raw)
            elapsed = time.time() - t0
            return {
                "sample_id":   row["sample_id"],
                "vlm_model":   row["vlm_model"],
                "language":    row["language"],
                "image_uid":   row["image_uid"],
                "question":    row["question"],
                "gold":        row["gold"],
                "pred":        row["pred"],
                "verdict":     verdict,
                "reason":      reason,
                "raw":         raw,
                "judge_model": f"{provider}:{model}",
                "elapsed_s":   round(elapsed, 2),
                "judged_at":   time.time(),
                "prompt_tokens":     ptok,
                "completion_tokens": ctok,
            }
        except Exception as e:
            last_err = e
            wait = 2 ** attempt
            print(f"  [retry {attempt+1}/{max_retries}] {row['sample_id']}: {type(e).__name__}: {str(e)[:100]} -- sleeping {wait}s", flush=True)
            time.sleep(wait)

    # Return None to signal retry-exhaustion. main() will NOT persist None
    # results, so this sample_id stays pending and is automatically re-tried
    # on the next run. This is safer than writing MISSING rows which would
    # then be considered "done" and skipped forever.
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=("openrouter", "anthropic"), default="openrouter",
                    help="API provider: openrouter (uses openai SDK + OPENROUTER_API_KEY) "
                         "or anthropic (uses anthropic SDK + ANTHROPIC_API_KEY).")
    ap.add_argument("--model", default=None,
                    help="Judge model id. Defaults: anthropic/claude-sonnet-4 (openrouter) "
                         "or claude-sonnet-4-20250514 (anthropic).")
    ap.add_argument("--concurrency", type=int, default=2,
                    help="Parallel requests (default 2)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N rows (for smoke testing)")
    args = ap.parse_args()

    load_dotenv(SECRETS_ENV, override=True)

    if args.provider == "openrouter":
        from openai import OpenAI
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            sys.exit(f"OPENROUTER_API_KEY not found in {SECRETS_ENV}")
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        model = args.model or "anthropic/claude-sonnet-4"
    else:
        try:
            from anthropic import Anthropic
        except ImportError:
            sys.exit("anthropic SDK not installed. Run: pip install anthropic")
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            sys.exit(f"ANTHROPIC_API_KEY not found in {SECRETS_ENV}")
        client = Anthropic(api_key=api_key)
        # claude-sonnet-4-5 (released 2025-09-29) is current and not deprecated;
        # claude-sonnet-4-20250514 (the OpenRouter alias) was deprecated EOL 2026-06-15.
        model = args.model or "claude-sonnet-4-5-20250929"

    print(f"[provider] {args.provider}  model={model}", flush=True)
    args.model = model  # so the "[plan] ... with {args.model}" print below shows the resolved name
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    loader = ImageLoader()

    # Load all 716 input rows.
    rows = []
    with INPUT_JSONL.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"[load] {len(rows)} input rows from {INPUT_JSONL.name}", flush=True)

    # Resume: skip sample_id+vlm_model pairs already judged.
    done: set[tuple[str, str]] = set()
    if OUT_JSONL.exists():
        with OUT_JSONL.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    done.add((d["sample_id"], d["vlm_model"]))
        print(f"[resume] {len(done)} rows already judged in {OUT_JSONL.name}", flush=True)

    pending = [r for r in rows if (r["sample_id"], r["vlm_model"]) not in done]
    if args.limit:
        pending = pending[: args.limit]
    print(f"[plan] {len(pending)} rows to judge with {args.model} @ concurrency={args.concurrency}", flush=True)

    if not pending:
        print("nothing to do.")
        return

    t_start = time.time()
    done_count = 0
    tokens_in = 0
    tokens_out = 0
    skipped_count = 0
    with OUT_JSONL.open("a", encoding="utf-8") as out, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(judge_one, client, args.provider, model, r, loader): r for r in pending}
        for fut in as_completed(futures):
            r = futures[fut]
            try:
                rec = fut.result()
            except Exception as e:
                print(f"[error] {r['sample_id']}: {type(e).__name__}: {e}", flush=True)
                continue
            if rec is None:
                # retry-exhaustion: do NOT persist, so this row stays pending for next run
                skipped_count += 1
                continue
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            done_count += 1
            tokens_in  += rec.get("prompt_tokens") or 0
            tokens_out += rec.get("completion_tokens") or 0
            if done_count % 20 == 0 or done_count == len(pending):
                elapsed = time.time() - t_start
                rate = done_count / elapsed if elapsed > 0 else 0
                eta_min = (len(pending) - done_count) / rate / 60 if rate > 0 else 0
                print(f"  [{done_count:>4}/{len(pending)}] {rec['vlm_model']:18s} {rec['language']:3s} -> {rec['verdict']:9s}  "
                      f"rate={rate:.2f}/s  eta={eta_min:.1f}min  tokens={tokens_in+tokens_out:,}",
                      flush=True)

    elapsed = time.time() - t_start
    print(f"\n[done] {done_count} rows judged in {elapsed/60:.1f} min", flush=True)
    if skipped_count:
        print(f"  [pending] {skipped_count} rows exhausted retries this run -- they stay pending and will be re-tried on the next resume.", flush=True)
    print(f"  tokens: prompt={tokens_in:,}  completion={tokens_out:,}", flush=True)
    # rough cost at Sonnet-4 rates ($3/1M in, $15/1M out)
    cost = (tokens_in or 0) / 1_000_000 * 3.0 + (tokens_out or 0) / 1_000_000 * 15.0
    print(f"  approx cost: ${cost:.2f}", flush=True)
    print(f"\n  output: {OUT_JSONL}")


if __name__ == "__main__":
    main()
