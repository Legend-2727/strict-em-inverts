#!/usr/bin/env python3
"""Day-18: Build a stratified sample of base XFUND predictions (full test set)
for the semantic-EM judge. 200/lang × 7 = 1400 rows.

Seed 42, language-balanced. Output: results/day18_xfund_semantic/sample.jsonl
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "results/xfund_qa/pa_qlora_s42_layer20_lam0p05/evaluations/test/predictions_base.jsonl"
OUT = REPO / "results/day18_xfund_semantic/sample.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--per-lang", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    rng = random.Random(args.seed)

    by_lang = {}
    for line in Path(args.src).open():
        s = line.strip().lstrip("\x00")
        if not s:
            continue
        try:
            r = json.loads(s)
        except json.JSONDecodeError:
            continue
        by_lang.setdefault(r["language"], []).append(r)

    print(f"[load] {sum(len(v) for v in by_lang.values())} rows across {len(by_lang)} langs")
    for lang in sorted(by_lang):
        print(f"  {lang}: {len(by_lang[lang])}")

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    sample = []
    for lang in sorted(by_lang):
        rows = by_lang[lang]
        rng.shuffle(rows)
        sample.extend(rows[: args.per_lang])

    with out_p.open("w", encoding="utf-8") as f:
        for r in sample:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[done] wrote {len(sample)} rows ({args.per_lang}/lang) to {out_p}")


if __name__ == "__main__":
    main()
