#!/usr/bin/env python3
"""Day-18: Analyze InternVL semantic-EM results vs Qwen.

Computes:
  - Strict-EM (from per_pair_base) and Semantic-EM (from judge) per-language
  - Latin/non-Latin gap under each metric
  - Artifact share of the cross-lingual gap
  - Comparison to Qwen findings
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
JUDGE = REPO / "results/day18_internvl_semantic/judgements.jsonl"
PRED = REPO / "results/mtvqa_perlang_internvl/per_pair_base.jsonl"
EVAL_IMGS = REPO / "data/processed/dpo_pairs_real_failures/eval_image_uids.json"
OUT = REPO / "results/day18_internvl_semantic/summary.md"
QWEN_SUMMARY = REPO / "results/day17_forensic/semantic_em_table.json"

LATIN = {"de", "fr", "it", "vi"}
NON_LATIN = {"ar", "ja", "kr", "ru", "th"}


def main():
    eval_imgs = set(json.loads(EVAL_IMGS.read_text()))

    # Load InternVL predictions, filter to held-out 358
    base = {}
    for line in PRED.open():
        s = line.strip().lstrip("\x00")
        if not s:
            continue
        try:
            r = json.loads(s)
        except Exception:
            continue
        if r["image_uid"] in eval_imgs:
            base[r["sample_id"]] = r

    judg = {}
    for line in JUDGE.open():
        s = line.strip().lstrip("\x00")
        if not s:
            continue
        try:
            r = json.loads(s)
        except Exception:
            continue
        judg[r["sample_id"]] = r

    # 3-way joined rows: have base prediction AND a judge verdict
    rows = []
    for sid, b in base.items():
        j = judg.get(sid)
        if not j:
            continue
        rows.append({
            "sid": sid,
            "lang": b["language"],
            "strict_em": bool(b.get("em", False)),
            "verdict": j["verdict"],
        })

    n = len(rows)
    n_total_holdout = len(base)
    print(f"InternVL held-out rows: {n_total_holdout}, judged: {n}")

    def sem(v, partial=False):
        if v == "CORRECT":
            return 1
        if partial and v == "PARTIAL":
            return 1
        return 0

    def script_stats(model_strict_key, partial=False):
        l_n = l_strict = l_sem = 0
        nl_n = nl_strict = nl_sem = 0
        for r in rows:
            if r["lang"] in LATIN:
                l_n += 1
                l_strict += int(r["strict_em"])
                l_sem += sem(r["verdict"], partial)
            elif r["lang"] in NON_LATIN:
                nl_n += 1
                nl_strict += int(r["strict_em"])
                nl_sem += sem(r["verdict"], partial)
        return {
            "latin": (l_strict, l_sem, l_n),
            "nonlatin": (nl_strict, nl_sem, nl_n),
            "latin_strict": l_strict / l_n if l_n else 0,
            "latin_sem": l_sem / l_n if l_n else 0,
            "nl_strict": nl_strict / nl_n if nl_n else 0,
            "nl_sem": nl_sem / nl_n if nl_n else 0,
            "gap_strict": (l_strict / l_n - nl_strict / nl_n) if l_n and nl_n else 0,
            "gap_sem": (l_sem / l_n - nl_sem / nl_n) if l_n and nl_n else 0,
        }

    overall_strict = sum(int(r["strict_em"]) for r in rows) / n
    overall_sem_c = sum(sem(r["verdict"]) for r in rows) / n
    overall_sem_cp = sum(sem(r["verdict"], partial=True) for r in rows) / n

    # Per-language
    by_lang = defaultdict(lambda: {"n": 0, "strict": 0, "sem_c": 0, "sem_cp": 0})
    for r in rows:
        by_lang[r["lang"]]["n"] += 1
        by_lang[r["lang"]]["strict"] += int(r["strict_em"])
        by_lang[r["lang"]]["sem_c"] += sem(r["verdict"])
        by_lang[r["lang"]]["sem_cp"] += sem(r["verdict"], partial=True)

    script_c = script_stats("strict_em", partial=False)
    script_cp = script_stats("strict_em", partial=True)

    # Load Qwen comparison
    qwen = json.loads(QWEN_SUMMARY.read_text())
    qsg = qwen["script_gap"]

    artifact_share_c = (1 - script_c["gap_sem"] / script_c["gap_strict"]) * 100 if script_c["gap_strict"] else 0
    artifact_share_cp = (1 - script_cp["gap_sem"] / script_cp["gap_strict"]) * 100 if script_cp["gap_strict"] else 0
    qwen_artifact_c = (1 - qsg["semantic_correct_only_base"]["gap"] / qsg["strict_base"]["gap"]) * 100 if qsg["strict_base"]["gap"] else 0
    qwen_artifact_cp = (1 - qsg["semantic_correct_or_partial_base"]["gap"] / qsg["strict_base"]["gap"]) * 100 if qsg["strict_base"]["gap"] else 0

    L = [
        f"# Day-18: InternVL-2.5-8B semantic-EM replicate (n={n} of {n_total_holdout} held-out)",
        "",
        "## Overall EM under three metrics",
        "",
        "| metric | InternVL | (Qwen base for reference) |",
        "|---|---:|---:|",
        f"| Strict-EM | {overall_strict*100:.2f}% | (25.70%) |",
        f"| Semantic-EM (CORRECT only) | {overall_sem_c*100:.2f}% | (54.47%) |",
        f"| Semantic-EM (CORRECT+PARTIAL) | {overall_sem_cp*100:.2f}% | (71.79%) |",
        "",
        "## Latin vs non-Latin gap on InternVL",
        "",
        "| metric | Latin | non-Latin | gap |",
        "|---|---:|---:|---:|",
        f"| Strict-EM | {script_c['latin_strict']*100:.2f}% | {script_c['nl_strict']*100:.2f}% | "
        f"{script_c['gap_strict']*100:+.2f}pp |",
        f"| Semantic-EM-C | {script_c['latin_sem']*100:.2f}% | {script_c['nl_sem']*100:.2f}% | "
        f"{script_c['gap_sem']*100:+.2f}pp |",
        f"| Semantic-EM-C+P | {script_cp['latin_sem']*100:.2f}% | {script_cp['nl_sem']*100:.2f}% | "
        f"{script_cp['gap_sem']*100:+.2f}pp |",
        "",
        "## Cross-VLM artifact-share comparison",
        "",
        "| VLM | strict gap | sem-C gap | artifact share (C) | sem-C+P gap | artifact share (C+P) |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Qwen base | {qsg['strict_base']['gap']*100:+.2f}pp | "
        f"{qsg['semantic_correct_only_base']['gap']*100:+.2f}pp | "
        f"{qwen_artifact_c:.1f}% | "
        f"{qsg['semantic_correct_or_partial_base']['gap']*100:+.2f}pp | "
        f"{qwen_artifact_cp:.1f}% |",
        f"| InternVL-2.5-8B | {script_c['gap_strict']*100:+.2f}pp | "
        f"{script_c['gap_sem']*100:+.2f}pp | "
        f"{artifact_share_c:.1f}% | "
        f"{script_cp['gap_sem']*100:+.2f}pp | "
        f"{artifact_share_cp:.1f}% |",
        "",
        "## Per-language InternVL EM",
        "",
        "| lang | n | strict | sem-C | sem-C+P |",
        "|---|---:|---:|---:|---:|",
    ]
    for lang in sorted(by_lang.keys()):
        v = by_lang[lang]
        ni = v["n"]
        L.append(f"| {lang} | {ni} | {v['strict']/ni*100:.1f}% | "
                 f"{v['sem_c']/ni*100:.1f}% | {v['sem_cp']/ni*100:.1f}% |")

    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"[done] wrote {OUT}")
    print()
    print("=== Cross-VLM headline ===")
    print(f"  Qwen strict gap → sem-C gap: {qsg['strict_base']['gap']*100:+.2f}pp → "
          f"{qsg['semantic_correct_only_base']['gap']*100:+.2f}pp "
          f"(artifact share {qwen_artifact_c:.0f}%)")
    print(f"  InternVL strict gap → sem-C gap: {script_c['gap_strict']*100:+.2f}pp → "
          f"{script_c['gap_sem']*100:+.2f}pp "
          f"(artifact share {artifact_share_c:.0f}%)")


if __name__ == "__main__":
    main()
