# Human validation of the Gemini-Flash semantic-EM judge

50 rows stratified across ar/de/fr/it/ja/kr/ru/th/vi × {CORRECT, PARTIAL, INCORRECT} from the
Day-17 semantic_judgements.jsonl.

## How to annotate

1. Open `sample.tsv` in a spreadsheet (Google Sheets, Excel, LibreOffice).
2. For each row:
   - Open the image at `images/{idx}_{image_uid}.png`
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
