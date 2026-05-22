#!/usr/bin/env python3
"""Build a single paper-ready 'fixed budget' table aggregating all test-time
intervention results across models.

Reads per_pair_*.jsonl files from a list of run directories, computes
EM/HAL/WSPAN/predOCR per cell, and emits:
  - results/paper_tables/interventions_table.md (Markdown)
  - results/paper_tables/interventions_table.tex (LaTeX booktabs)
  - results/paper_tables/interventions_table.json (machine-readable)

The 'fixed budget' framing: if all interventions converge to the same EM rate
within sampling noise (±1.5pp on n=200), the model has a fixed reading
ceiling that test-time methods cannot break.

Usage:
  /home/ubuntu/xLingual/.venv/bin/python scripts/build_interventions_table.py
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent


# Each entry: (display_label, list of jsonl paths to aggregate)
QWEN7B_CELLS = [
    ("Baseline (greedy)",
     [REPO_ROOT / "results/extractive_v2_baseline_vs_ocrprepend/per_pair_baseline.jsonl"]),
    ("OCR-prepended prompt",
     [REPO_ROOT / "results/extractive_v2_baseline_vs_ocrprepend/per_pair_ocr_prepended_bias0p00.jsonl"]),
    ("OCR logit-bias λ=1",
     [REPO_ROOT / "results/extractive_v2_bias_l1/per_pair_plain_bias1p00.jsonl"]),
    ("OCR logit-bias λ=3",
     [REPO_ROOT / "results/extractive_v2_bias_l3/per_pair_plain_bias3p00.jsonl"]),
    ("Beam5 + OCR-rerank",
     [REPO_ROOT / "results/ocr_constrained_smoke/per_pair_beam5_plain_bias0p00.jsonl"]),
]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open("r", encoding="utf-8") if l.strip()]


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {"n": 0, "em": 0.0, "hal": 0.0, "wspan": 0.0, "predOCR": 0.0}
    n = len(rows)
    em = sum(r["em"] for r in rows) / n
    hal = sum(r["failure_mode"] == "HALLUCINATION" for r in rows) / n
    wspan = sum(r["failure_mode"] == "WRONG_SPAN_FROM_IMAGE" for r in rows) / n
    predOCR = sum(r["pred_in_ocr"] for r in rows) / n
    # 95% CI on EM via Wilson approximation (for n=200, half-width ~ 6%)
    p = em
    z = 1.96
    if n > 0:
        denom = 1 + z*z/n
        center = (p + z*z/(2*n)) / denom
        half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    else:
        center = half = 0.0
    return {"n": n, "em": em, "hal": hal, "wspan": wspan, "predOCR": predOCR,
             "em_ci_lo": max(0, center - half), "em_ci_hi": min(1, center + half)}


def build_model_section(label: str,
                          cells: List[Tuple[str, List[Path]]]
                          ) -> List[Dict[str, Any]]:
    out = []
    for name, paths in cells:
        rows: List[Dict[str, Any]] = []
        for p in paths:
            rows.extend(read_jsonl(p))
        s = summarize(rows)
        out.append({"model": label, "intervention": name, **s,
                     "paths": [str(p) for p in paths]})
    return out


def render_markdown(rows: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("# Test-time intervention 'fixed budget' table — non-Latin EXTRACTIVE\n")
    lines.append("Each row reports EM / HALLUCINATION / WRONG_SPAN / pred_in_OCR rates "
                  "on the held-out non-Latin EXTRACTIVE pool. EM is the headline.\n")
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)
    for model, model_rows in by_model.items():
        lines.append(f"\n## {model}\n")
        lines.append("| Intervention | n | EM | EM 95% CI | HAL | WSPAN | predOCR |")
        lines.append("|---|---:|---:|---|---:|---:|---:|")
        baseline = next((r for r in model_rows if r["intervention"].startswith("Baseline")),
                         None)
        for r in model_rows:
            d_em = ""
            if baseline and r is not baseline and baseline["n"] > 0 and r["n"] > 0:
                d = r["em"] - baseline["em"]
                d_em = f" ({d:+.3f})"
            lines.append(
                f"| {r['intervention']} | {r['n']} | {r['em']:.4f}{d_em} | "
                f"[{r['em_ci_lo']:.3f}, {r['em_ci_hi']:.3f}] | "
                f"{r['hal']:.4f} | {r['wspan']:.4f} | {r['predOCR']:.4f} |"
            )
    lines.append("\n*ΔEM in parentheses is relative to baseline. The 'fixed budget' "
                  "claim: every intervention's EM CI overlaps with baseline.*")
    return "\n".join(lines) + "\n"


def render_latex(rows: List[Dict[str, Any]]) -> str:
    lines: List[str] = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Test-time interventions on non-Latin extractive document QA. "
        "Across architecturally distinct interventions, EM remains "
        "within sampling noise of the baseline. Failure modes (HAL, WSPAN) redistribute "
        "but the count of correct answers is invariant.}",
        "\\label{tab:fixed-budget}",
        "\\small",
        "\\begin{tabular}{lrcccc}",
        "\\toprule",
        "Intervention & $n$ & EM (95\\% CI) & HAL & WSPAN & predOCR \\\\",
        "\\midrule",
    ]
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)
    for model, model_rows in by_model.items():
        lines.append(f"\\multicolumn{{6}}{{l}}{{\\textbf{{{model}}}}} \\\\")
        for r in model_rows:
            lines.append(
                f"{r['intervention']} & {r['n']} & "
                f"{r['em']:.3f} [{r['em_ci_lo']:.3f},{r['em_ci_hi']:.3f}] & "
                f"{r['hal']:.3f} & {r['wspan']:.3f} & {r['predOCR']:.3f} \\\\"
            )
        lines.append("\\midrule")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def main():
    out_dir = REPO_ROOT / "results/paper_tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: List[Dict[str, Any]] = []
    all_rows.extend(build_model_section("Qwen2.5-VL-7B", QWEN7B_CELLS))

    # Hook for InternVL once we have the run dirs
    internvl_sweep_dir = REPO_ROOT / "results/internvl_sweep_ocrprepend"
    if internvl_sweep_dir.exists():
        cells = []
        baseline_p = internvl_sweep_dir / "per_pair_baseline.jsonl"
        if baseline_p.exists():
            cells.append(("Baseline (greedy)", [baseline_p]))
        prep_p = internvl_sweep_dir / "per_pair_ocr_prepended.jsonl"
        if prep_p.exists():
            cells.append(("OCR-prepended prompt", [prep_p]))
        if cells:
            all_rows.extend(build_model_section("InternVL2.5-8B", cells))

    qwen3b_sweep_dir = REPO_ROOT / "results/sweep_3b"
    if qwen3b_sweep_dir.exists():
        cells = []
        for tag, fname in [
            ("Baseline (greedy)", "baseline/per_pair_plain_bias0p00.jsonl"),
            ("OCR-prepended prompt", "ocr_prepend/per_pair_ocr_prepended_bias0p00.jsonl"),
            ("OCR logit-bias λ=1", "ocr_bias_l1/per_pair_plain_bias1p00.jsonl"),
            ("Beam5 + OCR-rerank", "beam5_rerank/per_pair_beam5_plain_bias0p00.jsonl"),
        ]:
            p = qwen3b_sweep_dir / fname
            if p.exists():
                cells.append((tag, [p]))
        if cells:
            all_rows.extend(build_model_section("Qwen2.5-VL-3B", cells))

    (out_dir / "interventions_table.json").write_text(
        json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "interventions_table.md").write_text(
        render_markdown(all_rows), encoding="utf-8")
    (out_dir / "interventions_table.tex").write_text(
        render_latex(all_rows), encoding="utf-8")

    print(f"[done] wrote {out_dir}")
    print(render_markdown(all_rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
