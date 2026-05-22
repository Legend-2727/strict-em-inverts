#!/usr/bin/env python3
"""Day-18: Generate the headline figure for the main-track paper:
   strict-EM vs semantic-EM on Qwen2.5-VL-7B and InternVL-2.5-8B,
   overall + per-script (Latin / non-Latin).

Output: paper_emnlp_main_track/figures/fig_headline_strict_vs_semantic.pdf
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
OUT = REPO / "paper_emnlp_main_track/figures/fig_headline_strict_vs_semantic.pdf"

LATIN = {"de", "fr", "it", "vi"}
NON_LATIN = {"ar", "ja", "kr", "ru", "th"}


def load(judge_path, pred_path, eval_imgs_path=None, pred_strict_key="em"):
    judg = {}
    for line in Path(judge_path).open():
        s = line.strip().lstrip("\x00")
        if not s: continue
        try: r = json.loads(s)
        except: continue
        if r.get("verdict") in ("CORRECT","PARTIAL","INCORRECT"):
            judg[r["sample_id"]] = r

    base = {}
    for line in Path(pred_path).open():
        s = line.strip().lstrip("\x00")
        if not s: continue
        try: r = json.loads(s)
        except: continue
        base[r["sample_id"]] = r

    if eval_imgs_path:
        eval_imgs = set(json.loads(Path(eval_imgs_path).read_text()))
        base = {sid: b for sid, b in base.items() if b.get("image_uid") in eval_imgs}

    rows = []
    for sid, b in base.items():
        j = judg.get(sid)
        if not j:
            continue
        rows.append({
            "sid": sid,
            "lang": b["language"],
            "strict": bool(b.get(pred_strict_key, False)),
            "v": j["verdict"],
        })
    return rows


def scores(rows):
    by_lang = defaultdict(lambda: {"n":0,"strict":0,"semc":0,"semcp":0})
    for r in rows:
        d = by_lang[r["lang"]]
        d["n"] += 1
        d["strict"] += int(r["strict"])
        d["semc"] += int(r["v"] == "CORRECT")
        d["semcp"] += int(r["v"] in ("CORRECT","PARTIAL"))
    def avg(langs, k):
        n = sum(by_lang[l]["n"] for l in langs if l in by_lang)
        s = sum(by_lang[l][k] for l in langs if l in by_lang)
        return s / n * 100 if n else 0
    return {
        "all_strict": avg(by_lang.keys(), "strict"),
        "all_semc":   avg(by_lang.keys(), "semc"),
        "lat_strict": avg(LATIN, "strict"),
        "lat_semc":   avg(LATIN, "semc"),
        "nl_strict":  avg(NON_LATIN, "strict"),
        "nl_semc":    avg(NON_LATIN, "semc"),
        "n": sum(d["n"] for d in by_lang.values()),
    }


def main():
    qwen = load(
        REPO/"results/day17_forensic/semantic_judgements.jsonl",
        REPO/"results/mtvqa_perlang_diagnostic/per_pair_base.jsonl",
    )
    qwen_base = [r for r in qwen if True]
    intern = load(
        REPO/"results/day18_internvl_semantic/judgements.jsonl",
        REPO/"results/mtvqa_perlang_internvl/per_pair_base.jsonl",
        REPO/"data/processed/dpo_pairs_real_failures/eval_image_uids.json",
    )

    # filter qwen to base-model rows
    qwen_base_path = REPO/"results/day17_forensic/semantic_judgements.jsonl"
    # The combined judgement file has both base and d16 rows; we need base only.
    judg_base = {}
    for line in qwen_base_path.open():
        s = line.strip().lstrip("\x00")
        if not s: continue
        try: r = json.loads(s)
        except: continue
        if r.get("model") == "base" and r.get("verdict") in ("CORRECT","PARTIAL","INCORRECT"):
            judg_base[r["sample_id"]] = r
    pred_q = {}
    for line in (REPO/"results/mtvqa_perlang_diagnostic/per_pair_base.jsonl").open():
        s = line.strip().lstrip("\x00")
        if not s: continue
        try: r = json.loads(s)
        except: continue
        pred_q[r["sample_id"]] = r
    qrows = []
    for sid, b in pred_q.items():
        j = judg_base.get(sid)
        if not j: continue
        qrows.append({"sid": sid, "lang": b["language"],
                      "strict": bool(b.get("em", False)), "v": j["verdict"]})

    qs = scores(qrows)
    iv = scores(intern)
    print(f"Qwen: n={qs['n']}, all strict={qs['all_strict']:.1f}, all sem-C={qs['all_semc']:.1f}")
    print(f"  Latin strict={qs['lat_strict']:.1f}, sem-C={qs['lat_semc']:.1f}")
    print(f"  nonLatin strict={qs['nl_strict']:.1f}, sem-C={qs['nl_semc']:.1f}")
    print(f"InternVL: n={iv['n']}, all strict={iv['all_strict']:.1f}, all sem-C={iv['all_semc']:.1f}")
    print(f"  Latin strict={iv['lat_strict']:.1f}, sem-C={iv['lat_semc']:.1f}")
    print(f"  nonLatin strict={iv['nl_strict']:.1f}, sem-C={iv['nl_semc']:.1f}")

    # Plot: 3 group panels (Overall / Latin / non-Latin) × (Strict, Semantic-C) × (Qwen, InternVL)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), sharey=True)
    panels = [
        ("Overall (n=358)", "all_strict", "all_semc"),
        ("Latin (de/fr/it/vi)", "lat_strict", "lat_semc"),
        ("non-Latin (ar/ja/kr/ru/th)", "nl_strict", "nl_semc"),
    ]
    x = np.arange(2)
    width = 0.35
    for ax, (title, sk, ck) in zip(axes, panels):
        q_vals = [qs[sk], qs[ck]]
        i_vals = [iv[sk], iv[ck]]
        ax.bar(x - width/2, q_vals, width, label="Qwen2.5-VL-7B", color="#1f77b4")
        ax.bar(x + width/2, i_vals, width, label="InternVL-2.5-8B", color="#ff7f0e")
        for i, (qv, iv_) in enumerate(zip(q_vals, i_vals)):
            ax.text(i - width/2, qv + 1, f"{qv:.0f}", ha="center", fontsize=8)
            ax.text(i + width/2, iv_ + 1, f"{iv_:.0f}", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(["strict-EM", "semantic-EM-C"], fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, 85)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("EM (%)", fontsize=11)
    axes[2].legend(loc="upper right", fontsize=9, framealpha=0.95)
    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, bbox_inches="tight")
    plt.savefig(OUT.with_suffix(".png"), bbox_inches="tight", dpi=150)
    print(f"[done] wrote {OUT}")


if __name__ == "__main__":
    main()
