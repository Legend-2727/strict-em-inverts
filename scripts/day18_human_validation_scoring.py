#!/usr/bin/env python3
"""Day-18: Score human-vs-Gemini concordance on the 50-row validation sample.

Reads results/day18_human_validation/sample.tsv (with HUMAN_VERDICT filled in)
and computes:
  - % agreement
  - Cohen's kappa
  - 3×3 confusion matrix
  - Rows where Gemini and human disagree (for error analysis)
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TSV = REPO / "results/day18_human_validation/sample.tsv"
OUT = REPO / "results/day18_human_validation/concordance_report.md"

VALID = ("CORRECT", "PARTIAL", "INCORRECT")


def cohens_kappa(observed, expected):
    """observed = P(agree), expected = P(agree by chance)"""
    return (observed - expected) / (1 - expected) if (1 - expected) > 0 else 0.0


def main():
    rows = []
    with TSV.open() as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for r in rdr:
            r["GEMINI"] = r["gemini_verdict"].strip().upper()
            r["HUMAN"] = r["HUMAN_VERDICT"].strip().upper()
            rows.append(r)

    annotated = [r for r in rows if r["HUMAN"] in VALID]
    unannotated = [r for r in rows if r["HUMAN"] not in VALID]
    n = len(annotated)
    if n == 0:
        print(f"[error] no rows have HUMAN_VERDICT filled in (of {len(rows)} total).")
        print(f"        Open {TSV} and fill HUMAN_VERDICT column.")
        return

    print(f"[load] {n} annotated rows ({len(unannotated)} unannotated remaining)")
    if unannotated:
        for r in unannotated[:5]:
            print(f"  unannotated idx={r['idx']}")
        if len(unannotated) > 5:
            print(f"  ... and {len(unannotated)-5} more")

    # Confusion matrix: gemini × human
    matrix = {(g, h): 0 for g in VALID for h in VALID}
    agree = 0
    for r in annotated:
        g, h = r["GEMINI"], r["HUMAN"]
        if g not in VALID:
            continue
        matrix[(g, h)] = matrix.get((g, h), 0) + 1
        if g == h:
            agree += 1

    observed = agree / n
    # P(chance agreement) = sum over verdicts of (P_g[v] * P_h[v])
    p_g = Counter(r["GEMINI"] for r in annotated if r["GEMINI"] in VALID)
    p_h = Counter(r["HUMAN"] for r in annotated if r["HUMAN"] in VALID)
    expected = sum(p_g[v] / n * p_h[v] / n for v in VALID)
    kappa = cohens_kappa(observed, expected)

    # Strict-binary view: CORRECT vs (PARTIAL/INCORRECT)
    binary_agree = sum(
        1 for r in annotated
        if (r["GEMINI"] == "CORRECT") == (r["HUMAN"] == "CORRECT")
    )
    binary_acc = binary_agree / n

    # Output
    L = [
        f"# Human vs Gemini-Flash semantic-EM judge concordance (n={n})",
        "",
        f"- **Percent agreement** (3-way): **{observed*100:.1f}%** ({agree}/{n})",
        f"- **Cohen's κ** (3-way): **{kappa:.3f}**",
        f"- **Binary CORRECT vs not**: **{binary_acc*100:.1f}%** ({binary_agree}/{n})",
        "",
        f"Acceptance bar (paper): κ ≥ 0.6 or % agreement ≥ 80%.",
        f"  → {'✅ PASS' if kappa >= 0.6 or observed >= 0.8 else '❌ NEEDS prompt tuning or annotation review'}",
        "",
        "## Confusion matrix (rows = Gemini, columns = Human)",
        "",
        "| | CORRECT | PARTIAL | INCORRECT |",
        "|---|---:|---:|---:|",
    ]
    for g in VALID:
        cells = [str(matrix.get((g, h), 0)) for h in VALID]
        L.append(f"| {g} | " + " | ".join(cells) + " |")

    L.extend(["", "## Disagreements"])
    for r in annotated:
        if r["GEMINI"] != r["HUMAN"]:
            L.append(f"- idx={r['idx']} [{r['language']}] gemini={r['GEMINI']} / human={r['HUMAN']}  "
                     f"gold=`{r['gold'][:50]}`  pred=`{r['pred'][:50]}`")
            if r.get("HUMAN_NOTES", "").strip():
                L.append(f"  note: {r['HUMAN_NOTES'][:200]}")

    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"\n[done] {OUT}")
    print(f"  agreement = {observed*100:.1f}%   κ = {kappa:.3f}   binary = {binary_acc*100:.1f}%")


if __name__ == "__main__":
    main()
