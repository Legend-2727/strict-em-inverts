#!/usr/bin/env python3
"""Day-19 / P0-B: Analyze Phi-3.5-vision-instruct semantic-EM results.

Mirrors day18_internvl_analysis.py. Phi judge occasionally returns Spanish
typo verdicts ("INCERRECT", "INCUMPLIDO"); these are coerced to INCORRECT.

Output: results/day19_phi35vision_semantic/summary.md
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PRED = REPO / "results/day19_phi35vision_heldout/per_pair_base.jsonl"
JUDGE = REPO / "results/day19_phi35vision_semantic/judgements.jsonl"
OUT = REPO / "results/day19_phi35vision_semantic/summary.md"

LATIN = {"de", "fr", "it", "vi"}
NON_LATIN = {"ar", "ja", "kr", "ru", "th"}


def coerce_verdict(v: str) -> str:
    v = (v or "").strip().upper()
    if v in {"CORRECT", "PARTIAL", "INCORRECT"}:
        return v
    return "INCORRECT"  # treat typos / Spanish ("INCERRECT", "INCUMPLIDO", "?") as INCORRECT


def main():
    base = {}
    for line in PRED.open():
        s = line.strip().lstrip("\x00")
        if not s:
            continue
        try:
            r = json.loads(s)
        except Exception:
            continue
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

    rows = []
    for sid, b in base.items():
        j = judg.get(sid)
        if not j:
            continue
        rows.append({
            "sid": sid,
            "lang": b["language"],
            "strict_em": bool(b.get("em", False)),
            "verdict": coerce_verdict(j.get("verdict", "")),
        })

    n = len(rows)
    n_total = len(base)
    print(f"Phi-3.5-vision held-out rows: {n_total}, judged: {n}")

    def sem(v, partial=False):
        if v == "CORRECT":
            return 1
        if partial and v == "PARTIAL":
            return 1
        return 0

    def script_stats(partial=False):
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
            "latin_strict": l_strict / l_n if l_n else 0,
            "latin_sem": l_sem / l_n if l_n else 0,
            "nl_strict": nl_strict / nl_n if nl_n else 0,
            "nl_sem": nl_sem / nl_n if nl_n else 0,
            "gap_strict": (l_strict / l_n - nl_strict / nl_n) if l_n and nl_n else 0,
            "gap_sem": (l_sem / l_n - nl_sem / nl_n) if l_n and nl_n else 0,
            "l_n": l_n, "nl_n": nl_n,
        }

    overall_strict = sum(int(r["strict_em"]) for r in rows) / n
    overall_sem_c = sum(sem(r["verdict"]) for r in rows) / n
    overall_sem_cp = sum(sem(r["verdict"], partial=True) for r in rows) / n

    by_lang = defaultdict(lambda: {"n": 0, "strict": 0, "sem_c": 0, "sem_cp": 0})
    for r in rows:
        by_lang[r["lang"]]["n"] += 1
        by_lang[r["lang"]]["strict"] += int(r["strict_em"])
        by_lang[r["lang"]]["sem_c"] += sem(r["verdict"])
        by_lang[r["lang"]]["sem_cp"] += sem(r["verdict"], partial=True)

    sc = script_stats(partial=False)
    scp = script_stats(partial=True)
    art_c = (1 - sc["gap_sem"] / sc["gap_strict"]) * 100 if sc["gap_strict"] else 0
    art_cp = (1 - scp["gap_sem"] / scp["gap_strict"]) * 100 if scp["gap_strict"] else 0

    L = [
        f"# Day-19 / P0-B: Phi-3.5-vision-instruct semantic-EM (n={n} of {n_total} held-out)",
        "",
        "Verdict coercion: 5 rows judged \"INCERRECT\" + 1 \"INCUMPLIDO\" + 1 \"?\" treated as INCORRECT.",
        "",
        "## Overall EM under three metrics",
        "",
        "| metric | Phi-3.5-vision (4.1B) | Qwen base (7B, ref) | InternVL base (8B, ref) |",
        "|---|---:|---:|---:|",
        f"| Strict-EM | {overall_strict*100:.2f}% | 25.70% | 34.15% |",
        f"| Semantic-EM (CORRECT) | {overall_sem_c*100:.2f}% | 54.47% | 48.86% |",
        f"| Semantic-EM (CORRECT+PARTIAL) | {overall_sem_cp*100:.2f}% | 71.79% | 65.05% |",
        "",
        "## Latin vs non-Latin gap on Phi-3.5-vision",
        "",
        f"Latin n={sc['l_n']}; non-Latin n={sc['nl_n']}",
        "",
        "| metric | Latin | non-Latin | gap |",
        "|---|---:|---:|---:|",
        f"| Strict-EM | {sc['latin_strict']*100:.2f}% | {sc['nl_strict']*100:.2f}% | "
        f"{sc['gap_strict']*100:+.2f}pp |",
        f"| Semantic-EM-C | {sc['latin_sem']*100:.2f}% | {sc['nl_sem']*100:.2f}% | "
        f"{sc['gap_sem']*100:+.2f}pp |",
        f"| Semantic-EM-C+P | {scp['latin_sem']*100:.2f}% | {scp['nl_sem']*100:.2f}% | "
        f"{scp['gap_sem']*100:+.2f}pp |",
        "",
        "## Artifact share (Phi-3.5-vision)",
        "",
        f"- semantic-EM-C: **{art_c:.1f}%**",
        f"- semantic-EM-C+P: **{art_cp:.1f}%**",
        "",
        "## Three-VLM comparison (artifact share under semantic-EM-C, Gemini-Flash judge)",
        "",
        "| VLM | params | strict gap | sem-C gap | artifact share |",
        "|---|---:|---:|---:|---:|",
        f"| Qwen2.5-VL-7B   | 7B   | +12.92pp | +4.59pp  | 64.5% |",
        f"| InternVL-2.5-8B | 8B   | +10.68pp | +9.83pp  |  8.0% |",
        f"| Phi-3.5-vision  | 4.1B | {sc['gap_strict']*100:+.2f}pp | {sc['gap_sem']*100:+.2f}pp | {art_c:.1f}% |",
        "",
        "## Per-language Phi-3.5-vision EM",
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
    print("=== Phi-3.5-vision headline ===")
    print(f"  strict gap → sem-C gap: {sc['gap_strict']*100:+.2f}pp → "
          f"{sc['gap_sem']*100:+.2f}pp (artifact share {art_c:.0f}%)")


if __name__ == "__main__":
    main()
