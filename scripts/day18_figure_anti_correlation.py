#!/usr/bin/env python3
"""Day-18: Per-language Δstrict-EM vs Δsemantic-EM-C under SFT (d16 vs base).
Visualizes the anti-correlation on non-Latin scripts.

Output: paper_emnlp_main_track/figures/fig_sft_anticorrelation.pdf
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "paper_emnlp_main_track/figures/fig_sft_anticorrelation.pdf"

LATIN = {"de","fr","it","vi"}

def main():
    t = json.load((REPO/"results/day17_forensic/semantic_em_table.json").open())
    langs = sorted(t["by_language"].keys())

    dstrict = []
    dsem = []
    annot = []
    is_latin = []
    sizes = []
    for lang in langs:
        v = t["by_language"][lang]
        if lang == "th":
            continue  # n=3 noise
        ds = (v["d16_strict"] - v["base_strict"]) * 100
        dc = (v["d16_sem_correct_only"] - v["base_sem_correct_only"]) * 100
        dstrict.append(ds)
        dsem.append(dc)
        annot.append(lang)
        is_latin.append(lang in LATIN)
        sizes.append(v["n"] * 8)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for ds, dc, lab, lat, sz in zip(dstrict, dsem, annot, is_latin, sizes):
        color = "#1f77b4" if lat else "#d62728"
        ax.scatter(ds, dc, s=sz, color=color, edgecolor="black", alpha=0.75,
                   label="Latin" if lat and lab=="fr" else ("non-Latin" if (not lat) and lab=="ar" else None))
        ax.annotate(lab, (ds, dc), xytext=(7, 4), textcoords="offset points", fontsize=11)

    # zero lines
    ax.axhline(0, color="gray", linewidth=0.7, linestyle="--", alpha=0.6)
    ax.axvline(0, color="gray", linewidth=0.7, linestyle="--", alpha=0.6)
    # quadrant labels
    ax.text(11, 4, "lift on both\n(true capability gain)", ha="center", fontsize=9, color="green", alpha=0.6)
    ax.text(11, -10, "strict lift only\n(style-only gain)", ha="center", fontsize=9, color="orange", alpha=0.7)
    ax.text(-6, 4, "semantic lift\nwithout strict",  ha="center", fontsize=9, color="gray", alpha=0.6)
    ax.text(-6, -10, "lose on both", ha="center", fontsize=9, color="red", alpha=0.7)

    ax.set_xlabel(r"$\Delta$ strict-EM (pp)", fontsize=12)
    ax.set_ylabel(r"$\Delta$ semantic-EM-C (pp)", fontsize=12)
    ax.set_title("Per-language SFT (d16, 6054 pairs) lift decomposition\nQwen2.5-VL-7B base $\\to$ d16, MTVQA 358 held-out",
                 fontsize=11)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.grid(alpha=0.3)
    ax.set_xlim(-7, 17)
    ax.set_ylim(-18, 6)
    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, bbox_inches="tight")
    plt.savefig(OUT.with_suffix(".png"), bbox_inches="tight", dpi=150)
    print(f"[done] wrote {OUT}")


if __name__ == "__main__":
    main()
