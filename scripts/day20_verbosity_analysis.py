"""Verbosity (mean output character count) per VLM x script on the joint
3-VLM MTVQA held-out and the Qwen/InternVL joint full-val.

Loads four judgement files:
  - Qwen2.5-VL-7B   : results/day19_mtvqa_full_semantic/judgements.jsonl
  - InternVL-2.5-8B : results/day19_internvl_full_semantic/judgements.jsonl
  - Phi-3.5-vision  : results/day19_phi35vision_semantic/judgements.jsonl
  - InternVL held-out: results/day18_internvl_semantic/judgements.jsonl  (unused; full-val supersedes)

Computes mean character count of model predictions, partitioned by
Latin (de, fr, it, vi) vs non-Latin (ar, ja, kr, ru, th) MTVQA languages,
and the non-Latin/Latin ratio. This ratio rank-orders the three VLMs
identically to their Latin/non-Latin artifact share -- evidence that
output-style verbosity is the mechanism behind the architecture-specific
artifact share.

Writes:
  results/day20_verbosity/summary.md
"""

import json
from collections import defaultdict
from pathlib import Path

LATIN = {"de", "fr", "it", "vi"}
NONLATIN = {"ar", "ja", "kr", "ru", "th"}

ROOT = Path(__file__).resolve().parent.parent
FILES = {
    "Qwen2.5-VL-7B": ROOT / "results/day19_mtvqa_full_semantic/judgements.jsonl",
    "InternVL-2.5-8B": ROOT / "results/day19_internvl_full_semantic/judgements.jsonl",
    "Phi-3.5-vision": ROOT / "results/day19_phi35vision_semantic/judgements.jsonl",
}


def load(path):
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                rows[d["sample_id"]] = d
            except json.JSONDecodeError:
                pass
    return rows


def script_of(lang):
    if lang in LATIN:
        return "Latin"
    if lang in NONLATIN:
        return "non-Latin"
    return "other"


def stats(rows, sample_ids):
    by_script = defaultdict(list)
    for sid in sample_ids:
        d = rows[sid]
        s = script_of(d["language"])
        pred = (d.get("pred") or "").strip()
        by_script[s].append(len(pred))
    out = {}
    for s in ("Latin", "non-Latin"):
        v = by_script[s]
        out[s] = (len(v), sum(v) / len(v) if v else 0.0)
    if out["Latin"][1] > 0:
        out["ratio"] = out["non-Latin"][1] / out["Latin"][1]
    else:
        out["ratio"] = float("nan")
    return out


def main():
    all_rows = {name: load(p) for name, p in FILES.items()}
    joint_held = set.intersection(*[set(r) for r in all_rows.values()])
    joint_full2 = set(all_rows["Qwen2.5-VL-7B"]) & set(all_rows["InternVL-2.5-8B"])

    out_dir = ROOT / "results/day20_verbosity"
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Verbosity analysis: mean output char count per VLM x script\n")
    lines.append(f"3-VLM joint held-out: n={len(joint_held)}\n")
    lines.append(f"2-VLM joint full-val: n={len(joint_full2)}\n\n")

    lines.append("## 3-VLM joint held-out\n\n")
    lines.append("| VLM | Latin n | Latin mean | non-Latin n | non-Latin mean | ratio |\n")
    lines.append("|-----|--------:|-----------:|------------:|---------------:|------:|\n")
    for name in all_rows:
        s = stats(all_rows[name], joint_held)
        lines.append(
            f"| {name} | {s['Latin'][0]} | {s['Latin'][1]:.1f} | "
            f"{s['non-Latin'][0]} | {s['non-Latin'][1]:.1f} | {s['ratio']:.2f} |\n"
        )

    lines.append("\n## 2-VLM joint full MTVQA val\n\n")
    lines.append("| VLM | Latin n | Latin mean | non-Latin n | non-Latin mean | ratio |\n")
    lines.append("|-----|--------:|-----------:|------------:|---------------:|------:|\n")
    for name in ("Qwen2.5-VL-7B", "InternVL-2.5-8B"):
        s = stats(all_rows[name], joint_full2)
        lines.append(
            f"| {name} | {s['Latin'][0]} | {s['Latin'][1]:.1f} | "
            f"{s['non-Latin'][0]} | {s['non-Latin'][1]:.1f} | {s['ratio']:.2f} |\n"
        )

    (out_dir / "summary.md").write_text("".join(lines), encoding="utf-8")
    print(f"wrote {out_dir / 'summary.md'}")
    print("".join(lines))


if __name__ == "__main__":
    main()
