#!/usr/bin/env python3
"""Day-19: Three-VLM artifact-share bar figure.

Bars: Qwen2.5-VL-7B (+66.4%), InternVL-2.5-8B (+7.7%), Phi-3.5-vision (-57.4%).
Error bars from bootstrap CIs. Horizontal zero line annotated "no measurement bias".

Output: paper_emnlp_main_track/figures/fig_three_vlm_artifact.{pdf,png}
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_PDF = Path("paper_emnlp_main_track/figures/fig_three_vlm_artifact.pdf")
OUT_PNG = Path("paper_emnlp_main_track/figures/fig_three_vlm_artifact.png")

# Point estimates and bootstrap 95% CIs (from results/day19_bootstrap/cis.md)
models = ["Qwen2.5-VL-7B", "InternVL-2.5-8B", "Phi-3.5-vision"]
points = [66.4, 7.7, -57.4]
lo =     [-20.0, -158.9, -156.5]
hi =     [182.8, 112.4, -4.8]
err_lo = [points[i] - lo[i] for i in range(3)]
err_hi = [hi[i] - points[i] for i in range(3)]

colors = ["#1f77b4", "#2ca02c", "#d62728"]

fig, ax = plt.subplots(figsize=(6.0, 3.6))
xpos = list(range(3))
bars = ax.bar(xpos, points, color=colors, alpha=0.85,
              edgecolor="black", linewidth=0.8, width=0.65)
ax.errorbar(xpos, points, yerr=[err_lo, err_hi], fmt="none",
            ecolor="black", capsize=4, capthick=1.0, elinewidth=0.8)

ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.8)
ax.annotate("no measurement bias", xy=(2.45, 5), xytext=(2.45, 25),
            fontsize=8, color="dimgray", ha="right",
            arrowprops=dict(arrowstyle="-", color="dimgray", lw=0.6))

for x, v in zip(xpos, points):
    label = f"{v:+.1f}%"
    yoff = 14 if v >= 0 else -22
    ax.annotate(label, (x, v), ha="center", va="center",
                xytext=(0, yoff), textcoords="offset points",
                fontsize=10, fontweight="bold")

ax.set_xticks(xpos)
ax.set_xticklabels([m + f"\n(n={n})" for m, n in
                     zip(models, [357, 354, 341])], fontsize=9)
ax.set_ylabel("Latin/non-Latin artifact share (%)", fontsize=10)
ax.set_title("Strict-EM measurement bias spans 121 pp across three VLM architectures\n"
             "on identical MTVQA inputs", fontsize=10.5)
ax.set_ylim(-200, 220)
ax.grid(axis="y", alpha=0.25, linestyle=":")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Annotate Phi region: "strict-EM understates"
ax.text(2, -180, "strict-EM\nunderstates gap", ha="center", va="center",
        fontsize=8, color="darkred", style="italic")
ax.text(0, 200, "strict-EM\noverstates gap", ha="center", va="center",
        fontsize=8, color="darkblue", style="italic")

fig.tight_layout()
OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, bbox_inches="tight", dpi=160)
print(f"[done] wrote {OUT_PDF} and {OUT_PNG}")
