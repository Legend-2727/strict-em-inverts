#!/usr/bin/env python3
"""Day-21: After the human-validation pass is finished, compute kappa /
percent-agreement / confusion matrix from the streamlit-app output and patch
the values into the EMNLP paper, replacing the red `\hjkappa` / `\hjpct` /
`\hjpending` placeholders with the real numbers.

Run this once after the 50-row annotation is complete:

  python scripts/day18_human_validation_scoring.py    # writes concordance_report.md
  python scripts/day21_patch_kappa_into_paper.py      # patches main.tex + 11_appendix.tex

Then recompile the paper.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LABELS = REPO / "results/day18_human_validation/human_labels.json"
TSV = REPO / "results/day18_human_validation/sample.tsv"
SAMPLE = REPO / "results/day18_human_validation/sample.jsonl"
MAIN_TEX = REPO / "paper_emnlp_main_track/main.tex"
APP_TEX = REPO / "paper_emnlp_main_track/sections/11_appendix.tex"

VALID = ("CORRECT", "PARTIAL", "INCORRECT")


def cohens_kappa(observed: float, expected: float) -> float:
    return (observed - expected) / (1 - expected) if (1 - expected) > 0 else 0.0


def load_pairs() -> list[tuple[str, str]]:
    """Returns [(gemini_verdict, human_verdict), ...] for annotated rows."""
    if not LABELS.exists():
        raise SystemExit(
            f"human_labels.json not found at {LABELS}\n"
            "Annotate at http://localhost:8501 first."
        )
    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    sample = []
    with SAMPLE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sample.append(json.loads(line))
    pairs: list[tuple[str, str]] = []
    for row in sample:
        key = str(row["_anno_idx"])
        lab = labels.get(key)
        if not lab:
            continue
        h = (lab.get("verdict") or "").upper()
        g = (row.get("verdict") or "").upper()
        if h in VALID and g in VALID:
            pairs.append((g, h))
    return pairs


def compute_stats(pairs: list[tuple[str, str]]) -> dict:
    n = len(pairs)
    if n == 0:
        raise SystemExit("no annotated rows; annotate via streamlit first.")
    agree = sum(1 for g, h in pairs if g == h)
    p_g = Counter(g for g, _ in pairs)
    p_h = Counter(h for _, h in pairs)
    observed = agree / n
    expected = sum((p_g[v] / n) * (p_h[v] / n) for v in VALID)
    kappa = cohens_kappa(observed, expected)

    binary_agree = sum(1 for g, h in pairs if (g == "CORRECT") == (h == "CORRECT"))
    binary_pct = binary_agree / n

    matrix = {(g, h): 0 for g in VALID for h in VALID}
    for g, h in pairs:
        matrix[(g, h)] = matrix.get((g, h), 0) + 1

    return {
        "n": n,
        "agree": agree,
        "kappa": kappa,
        "observed_pct": observed * 100,
        "binary_pct": binary_pct * 100,
        "binary_agree": binary_agree,
        "matrix": matrix,
    }


def patch_main_tex(stats: dict) -> None:
    """Replace the red \hjkappa / \hjpct / \hjpending macro definitions with
    real numbers (formatted) and remove the red color so they render normally."""
    src = MAIN_TEX.read_text(encoding="utf-8")
    new_macros = (
        f"% Human-judge agreement statistics from the {stats['n']}-row blind human-validation\n"
        "% pass (scripts/day18_human_validation_scoring.py + day21_patch_kappa_into_paper.py).\n"
        f"\\newcommand{{\\hjkappa}}{{{stats['kappa']:.2f}}}\n"
        f"\\newcommand{{\\hjpct}}{{{stats['observed_pct']:.1f}}}\n"
        f"\\newcommand{{\\hjbinpct}}{{{stats['binary_pct']:.1f}}}"
    )
    # Match the existing macro block. Handle both the original red-placeholder
    # version (with \hjpending) and a previously-patched version.
    pattern = re.compile(
        r"% Human-judge agreement statistics[\s\S]*?"
        r"(?:\\newcommand\{\\hjpending\}\{(?:[^{}]|\{[^{}]*\})*\}"
        r"|\\newcommand\{\\hjbinpct\}\{[^}]*\})",
        re.DOTALL,
    )
    if not pattern.search(src):
        raise SystemExit("could not locate \\hjkappa / \\hjpct / \\hjpending block in main.tex")
    # use a callable to bypass re.sub's interpretation of backslashes in the replacement string
    new_src = pattern.sub(lambda _m: new_macros.rstrip(), src)
    MAIN_TEX.write_text(new_src, encoding="utf-8")
    print(f"[patch] main.tex: \\hjkappa = {stats['kappa']:.2f}, "
          f"\\hjpct = {stats['observed_pct']:.1f}")


def patch_appendix_tex(stats: dict) -> None:
    """Replace each \hjpending cell in 11_appendix.tex's 3x3 confusion matrix
    with the corresponding count from stats['matrix']."""
    src = APP_TEX.read_text(encoding="utf-8")
    m = stats["matrix"]
    new_rows = [
        f"Gemini-C & {m[('CORRECT','CORRECT')]} & {m[('CORRECT','PARTIAL')]} & {m[('CORRECT','INCORRECT')]} \\\\",
        f"Gemini-P & {m[('PARTIAL','CORRECT')]} & {m[('PARTIAL','PARTIAL')]} & {m[('PARTIAL','INCORRECT')]} \\\\",
        f"Gemini-I & {m[('INCORRECT','CORRECT')]} & {m[('INCORRECT','PARTIAL')]} & {m[('INCORRECT','INCORRECT')]} \\\\",
    ]
    old_rows = (
        "Gemini-C & \\hjpending & \\hjpending & \\hjpending \\\\\n"
        "Gemini-P & \\hjpending & \\hjpending & \\hjpending \\\\\n"
        "Gemini-I & \\hjpending & \\hjpending & \\hjpending \\\\"
    )
    if old_rows not in src:
        raise SystemExit("could not locate the 3x3 confusion-matrix block in 11_appendix.tex")
    new_src = src.replace(old_rows, "\n".join(new_rows))
    APP_TEX.write_text(new_src, encoding="utf-8")
    print(f"[patch] 11_appendix.tex: 3x3 confusion matrix populated")


def main() -> None:
    pairs = load_pairs()
    stats = compute_stats(pairs)
    print(f"\n=== Human-vs-Gemini concordance (n={stats['n']}) ===")
    print(f"  3-way agreement: {stats['observed_pct']:.1f}% ({stats['agree']}/{stats['n']})")
    print(f"  Cohen's kappa:   {stats['kappa']:.3f}")
    print(f"  binary C-vs-not: {stats['binary_pct']:.1f}% ({stats['binary_agree']}/{stats['n']})")
    threshold_pass = stats["kappa"] >= 0.6 or stats["observed_pct"] >= 80
    print(f"  threshold ({'PASS' if threshold_pass else 'FAIL'}): "
          f"kappa >= 0.6 or %agree >= 80%")
    print(f"\n  matrix (rows = Gemini, cols = Human):")
    print(f"           CORRECT  PARTIAL  INCORRECT")
    for g in VALID:
        cells = "  ".join(f"{stats['matrix'][(g, h)]:>7}" for h in VALID)
        print(f"    {g[:7]:7s}  {cells}")

    if stats["n"] < 50:
        print(f"\n[warn] only {stats['n']}/50 rows annotated -- patching anyway with what we have")

    patch_main_tex(stats)
    patch_appendix_tex(stats)
    print(f"\n[done] paper macros patched. Recompile:")
    print(f"  cd paper_emnlp_main_track && latexmk -pdf main.tex")


if __name__ == "__main__":
    main()
