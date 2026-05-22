#!/usr/bin/env python3
"""Day-18: Build a stratified 50-row sample for human validation of the
Gemini semantic-EM judge.

Sampling: 5 verdicts (C/P/I) × 7 languages × constraints:
  - At least 1 of each verdict per language (where possible)
  - Mix base + d16 model predictions
  - Skip rows where pred is empty or gold is empty

Output:
  results/day18_human_validation/sample.tsv (one row per item, blank verdict
    column for the human to fill in)
  results/day18_human_validation/sample.jsonl (full machine-readable form)
  results/day18_human_validation/INSTRUCTIONS.md (how to do the annotation)
"""
from __future__ import annotations

import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
JUDGE = REPO / "results/day17_forensic/semantic_judgements.jsonl"
IMG_DIR = REPO / "data/processed/mtvqa/images"
OUT_DIR = REPO / "results/day18_human_validation"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)

    # Pool judgments
    rows = []
    for line in JUDGE.open():
        s = line.strip().lstrip("\x00")
        if not s:
            continue
        try:
            r = json.loads(s)
        except json.JSONDecodeError:
            continue
        if r.get("verdict") not in ("CORRECT", "PARTIAL", "INCORRECT"):
            continue
        if not r.get("gold") or not r.get("pred"):
            continue
        rows.append(r)

    print(f"[load] {len(rows)} judged rows")

    # Stratify: lang × verdict
    pool = defaultdict(list)
    for r in rows:
        pool[(r["language"], r["verdict"])].append(r)

    # Pull: 2 per cell (lang × verdict) where available; oversample where sparse
    sample = []
    target_per_cell = 2  # 9 langs × 3 verdicts × 2 = 54 (clip to 50)
    langs = sorted({r["language"] for r in rows})
    for lang in langs:
        for verdict in ("CORRECT", "PARTIAL", "INCORRECT"):
            cell = pool[(lang, verdict)]
            rng.shuffle(cell)
            take = min(target_per_cell, len(cell))
            sample.extend(cell[:take])

    rng.shuffle(sample)
    sample = sample[:50]
    print(f"[sample] {len(sample)} rows; per-verdict: ", end="")
    from collections import Counter
    print(dict(Counter(r["verdict"] for r in sample)))

    # Copy images to a flat dir for easy annotation
    img_out_dir = OUT_DIR / "images"
    img_out_dir.mkdir(exist_ok=True)
    for i, r in enumerate(sample):
        uid = r["image_uid"]
        if uid.startswith("train__"):
            split = "train"
        elif uid.startswith("test__"):
            split = "test"
        else:
            split = "val"
        src = IMG_DIR / split / f"{uid}.png"
        if src.exists():
            dst = img_out_dir / f"{i:02d}_{uid}.png"
            if not dst.exists():
                shutil.copy(src, dst)
        r["_anno_idx"] = i
        r["_image_local"] = f"images/{i:02d}_{uid}.png"

    # Write TSV (for human to annotate) and JSONL (machine-readable)
    tsv_path = OUT_DIR / "sample.tsv"
    with tsv_path.open("w", encoding="utf-8") as f:
        f.write("idx\tlanguage\tmodel\timage\tquestion\tgold\tpred\tgemini_verdict\tHUMAN_VERDICT\tHUMAN_NOTES\n")
        for r in sample:
            cells = [
                str(r["_anno_idx"]),
                r["language"],
                r["model"],
                r["_image_local"],
                (r.get("question") or "").replace("\t", " ").replace("\n", " "),
                (r.get("gold") or "").replace("\t", " ").replace("\n", " "),
                (r.get("pred") or "").replace("\t", " ").replace("\n", " "),
                r["verdict"],
                "",  # HUMAN_VERDICT (blank — to fill in: CORRECT / PARTIAL / INCORRECT)
                "",  # HUMAN_NOTES (blank — optional reason)
            ]
            f.write("\t".join(cells) + "\n")

    jsonl_path = OUT_DIR / "sample.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in sample:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Instructions
    instructions = """# Human validation of the Gemini-Flash semantic-EM judge

50 rows stratified across {langs} × {{CORRECT, PARTIAL, INCORRECT}} from the
Day-17 semantic_judgements.jsonl.

## How to annotate

1. Open `sample.tsv` in a spreadsheet (Google Sheets, Excel, LibreOffice).
2. For each row:
   - Open the image at `images/{{idx}}_{{image_uid}}.png`
   - Read the question, gold, and pred columns
   - **Do NOT look at the `gemini_verdict` column until you've decided** —
     fill in `HUMAN_VERDICT` independently with one of:
       - CORRECT  : prediction is semantically equivalent to gold (synonyms,
         whitespace, diacritics, alternate phrasing, valid translation, lossless
         reformat — same meaning)
       - PARTIAL  : prediction contains correct info but is incomplete or less
         specific than gold (e.g., distance without direction; 1 of 3 items)
       - INCORRECT: prediction states something different from gold, or
         fabricates info not in the image, or is unrelated
   - If the gold answer itself is wrong and the model's prediction matches the
     image better, mark CORRECT (judge against image evidence, not gold)
   - Optionally add a one-line note in `HUMAN_NOTES`

## How to score concordance

Once filled, run:
```
python scripts/day18_human_validation_scoring.py
```

This computes Cohen's kappa and percent-agreement between human and Gemini
verdicts, plus a confusion matrix and the rows where they disagree.

## What we need

Cohen's κ ≥ 0.6 OR percent-agreement ≥ 80%: judge is reliable enough to
publish semantic-EM measurements with.

If lower than that, the judge needs prompt-tuning before we can claim
semantic-EM in the paper.
""".format(langs="ar/de/fr/it/ja/kr/ru/th/vi")
    (OUT_DIR / "INSTRUCTIONS.md").write_text(instructions, encoding="utf-8")

    print(f"\n[done]")
    print(f"  sample.tsv     : {tsv_path}")
    print(f"  sample.jsonl   : {jsonl_path}")
    print(f"  INSTRUCTIONS.md: {OUT_DIR / 'INSTRUCTIONS.md'}")
    print(f"  images/        : {img_out_dir} ({len(list(img_out_dir.iterdir()))} files)")


if __name__ == "__main__":
    main()
