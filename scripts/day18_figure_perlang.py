#!/usr/bin/env python3
"""Day-18: Per-language strict-EM vs semantic-EM-C for Qwen2.5-VL-7B + InternVL-2.5-8B.

Output: paper_emnlp_main_track/figures/fig_perlang_strict_vs_semantic.pdf
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "paper_emnlp_main_track/figures/fig_perlang_strict_vs_semantic.pdf"

LANG_ORDER = ["fr", "vi", "it", "de", "ja", "kr", "ru", "ar"]  # Latin then non-Latin
LATIN = {"de","fr","it","vi"}


def load_qwen_base():
    judg = {}
    for line in (REPO/"results/day17_forensic/semantic_judgements.jsonl").open():
        s = line.strip().lstrip("\x00")
        if not s: continue
        try: r = json.loads(s)
        except: continue
        if r.get("model") == "base" and r.get("verdict") in ("CORRECT","PARTIAL","INCORRECT"):
            judg[r["sample_id"]] = r
    pred = {}
    for line in (REPO/"results/mtvqa_perlang_diagnostic/per_pair_base.jsonl").open():
        s = line.strip().lstrip("\x00")
        if not s: continue
        try: r = json.loads(s)
        except: continue
        pred[r["sample_id"]] = r
    rows = []
    for sid, b in pred.items():
        j = judg.get(sid)
        if not j: continue
        rows.append({"lang": b["language"], "strict": bool(b.get("em", False)),
                     "v": j["verdict"]})
    return rows


def load_intern():
    judg = {}
    for line in (REPO/"results/day18_internvl_semantic/judgements.jsonl").open():
        s = line.strip().lstrip("\x00")
        if not s: continue
        try: r = json.loads(s)
        except: continue
        if r.get("verdict") in ("CORRECT","PARTIAL","INCORRECT"):
            judg[r["sample_id"]] = r
    eval_imgs = set(json.loads((REPO/"data/processed/dpo_pairs_real_failures/eval_image_uids.json").read_text()))
    pred = {}
    for line in (REPO/"results/mtvqa_perlang_internvl/per_pair_base.jsonl").open():
        s = line.strip().lstrip("\x00")
        if not s: continue
        try: r = json.loads(s)
        except: continue
        if r.get("image_uid") in eval_imgs:
            pred[r["sample_id"]] = r
    rows = []
    for sid, b in pred.items():
        j = judg.get(sid)
        if not j: continue
        rows.append({"lang": b["language"], "strict": bool(b.get("em", False)),
                     "v": j["verdict"]})
    return rows


def per_lang(rows):
    out = defaultdict(lambda: {"n":0,"strict":0,"semc":0})
    for r in rows:
        d = out[r["lang"]]
        d["n"] += 1
        d["strict"] += int(r["strict"])
        d["semc"] += int(r["v"] == "CORRECT")
    return out


def main():
    q = per_lang(load_qwen_base())
    i = per_lang(load_intern())

    fig, axes = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
    x = np.arange(len(LANG_ORDER))
    width = 0.4

    for ax, data, name, color in [(axes[0], q, "Qwen2.5-VL-7B", "#1f77b4"),
                                   (axes[1], i, "InternVL-2.5-8B", "#ff7f0e")]:
        strict = [data[l]["strict"]/data[l]["n"]*100 if data[l]["n"] else 0 for l in LANG_ORDER]
        semc   = [data[l]["semc"]/data[l]["n"]*100 if data[l]["n"] else 0 for l in LANG_ORDER]
        ax.bar(x - width/2, strict, width, label="strict-EM", color=color, alpha=0.55, edgecolor="black")
        ax.bar(x + width/2, semc,   width, label="semantic-EM-C", color=color, alpha=1.0, edgecolor="black")
        for i_, (sv, cv) in enumerate(zip(strict, semc)):
            ax.text(i_ - width/2, sv + 1, f"{sv:.0f}", ha="center", fontsize=7)
            ax.text(i_ + width/2, cv + 1, f"{cv:.0f}", ha="center", fontsize=7)
        ax.set_title(name, fontsize=11, loc="left")
        ax.set_ylim(0, 95)
        ax.set_ylabel("EM (%)", fontsize=10)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
        ax.grid(axis="y", alpha=0.3)
        # Dotted vline between Latin and non-Latin
        n_latin = sum(1 for l in LANG_ORDER if l in LATIN)
        ax.axvline(n_latin - 0.5, color="gray", linestyle="--", alpha=0.5)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(LANG_ORDER, fontsize=10)
    axes[1].text(1.5, -22, "Latin scripts", ha="center", fontsize=10, fontstyle="italic")
    axes[1].text(5.5, -22, "non-Latin scripts", ha="center", fontsize=10, fontstyle="italic")
    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, bbox_inches="tight")
    plt.savefig(OUT.with_suffix(".png"), bbox_inches="tight", dpi=150)
    print(f"[done] wrote {OUT}")

    # Print numbers
    print("\nQwen per-lang:")
    for l in LANG_ORDER:
        d = q[l]; n = d["n"]
        if n: print(f"  {l}: n={n} strict={d['strict']/n*100:.1f} semC={d['semc']/n*100:.1f}")
    print("\nInternVL per-lang:")
    for l in LANG_ORDER:
        d = i[l]; n = d["n"]
        if n: print(f"  {l}: n={n} strict={d['strict']/n*100:.1f} semC={d['semc']/n*100:.1f}")


if __name__ == "__main__":
    main()
