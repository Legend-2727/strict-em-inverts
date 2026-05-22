#!/usr/bin/env python3
"""Day-18: Analyze XFUND semantic-EM judgments for within-CJK & cross-script gaps.

Headline question: does the within-CJK Japanese-vs-Chinese ~20pp strict-EM gap
(reported in the Findings draft) survive semantic-EM evaluation? And does the
Latin-vs-CJK script comparison show the same artifact-share pattern as MTVQA?

Inputs:
  results/day18_xfund_semantic/judgements.jsonl
  results/day18_xfund_semantic/sample.jsonl  (the original 1400-row sample)

Output:
  results/day18_xfund_semantic/summary.md
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
JUDGE = REPO / "results/day18_xfund_semantic/judgements.jsonl"
SAMPLE = REPO / "results/day18_xfund_semantic/sample.jsonl"
OUT = REPO / "results/day18_xfund_semantic/summary.md"

LATIN = {"de", "es", "fr", "it", "pt"}
CJK = {"ja", "zh"}


def load_jsonl(p: Path):
    rows = []
    for line in p.open():
        s = line.strip().lstrip("\x00")
        if not s:
            continue
        try:
            rows.append(json.loads(s))
        except json.JSONDecodeError:
            continue
    return rows


def main():
    sample = load_jsonl(SAMPLE)
    judg = load_jsonl(JUDGE)
    judg_by_sid = {r["sample_id"]: r for r in judg}

    rows = []
    for s in sample:
        sid = s["sample_id"]
        j = judg_by_sid.get(sid)
        if not j:
            continue
        if j.get("verdict") not in ("CORRECT", "PARTIAL", "INCORRECT"):
            continue
        if j.get("reason") == "image-missing":
            continue
        rows.append({
            "sid": sid,
            "lang": s["language"],
            "strict_em": bool(s.get("strict_em", False)),
            "normalized_em": bool(s.get("normalized_em", False)),
            "verdict": j["verdict"],
        })

    n_sample = len(sample)
    n_judged = len(judg)
    n_clean = len(rows)
    print(f"sample={n_sample}  judged={n_judged}  clean-joined={n_clean}")

    def is_c(v):
        return v == "CORRECT"

    def is_cp(v):
        return v in ("CORRECT", "PARTIAL")

    by_lang = defaultdict(lambda: {"n": 0, "strict": 0, "norm": 0, "sem_c": 0, "sem_cp": 0})
    for r in rows:
        d = by_lang[r["lang"]]
        d["n"] += 1
        d["strict"] += int(r["strict_em"])
        d["norm"] += int(r["normalized_em"])
        d["sem_c"] += int(is_c(r["verdict"]))
        d["sem_cp"] += int(is_cp(r["verdict"]))

    def script_avg(langs, key):
        nums = [by_lang[l][key] for l in langs if l in by_lang]
        ns = [by_lang[l]["n"] for l in langs if l in by_lang]
        return sum(nums) / sum(ns) if sum(ns) > 0 else 0

    # Overall
    n_all = sum(by_lang[l]["n"] for l in by_lang)
    overall = {
        "strict": sum(by_lang[l]["strict"] for l in by_lang) / n_all,
        "norm": sum(by_lang[l]["norm"] for l in by_lang) / n_all,
        "sem_c": sum(by_lang[l]["sem_c"] for l in by_lang) / n_all,
        "sem_cp": sum(by_lang[l]["sem_cp"] for l in by_lang) / n_all,
    }

    # Script-level
    script_rows = []
    for name, langs in [("Latin (de/es/fr/it/pt)", LATIN), ("CJK (ja/zh)", CJK)]:
        present = [l for l in langs if l in by_lang]
        if not present:
            continue
        script_rows.append({
            "name": name,
            "langs": present,
            "n": sum(by_lang[l]["n"] for l in present),
            "strict": script_avg(present, "strict"),
            "norm": script_avg(present, "norm"),
            "sem_c": script_avg(present, "sem_c"),
            "sem_cp": script_avg(present, "sem_cp"),
        })

    # Latin-vs-CJK gap
    latin_n = sum(by_lang[l]["n"] for l in LATIN if l in by_lang)
    cjk_n = sum(by_lang[l]["n"] for l in CJK if l in by_lang)

    def gap(metric):
        l = script_avg(LATIN, metric)
        c = script_avg(CJK, metric)
        return l - c

    script_gap_strict = gap("strict")
    script_gap_norm = gap("norm")
    script_gap_sem_c = gap("sem_c")
    script_gap_sem_cp = gap("sem_cp")

    artifact_c = (1 - script_gap_sem_c / script_gap_strict) * 100 if script_gap_strict else 0
    artifact_cp = (1 - script_gap_sem_cp / script_gap_strict) * 100 if script_gap_strict else 0

    # Within-CJK ja-vs-zh
    ja = by_lang.get("ja", {"n": 0, "strict": 0, "norm": 0, "sem_c": 0, "sem_cp": 0})
    zh = by_lang.get("zh", {"n": 0, "strict": 0, "norm": 0, "sem_c": 0, "sem_cp": 0})

    def pct(d, key):
        return d[key] / d["n"] * 100 if d["n"] else 0

    jazh_strict_gap = pct(zh, "strict") - pct(ja, "strict")
    jazh_norm_gap = pct(zh, "norm") - pct(ja, "norm")
    jazh_sem_c_gap = pct(zh, "sem_c") - pct(ja, "sem_c")
    jazh_sem_cp_gap = pct(zh, "sem_cp") - pct(ja, "sem_cp")

    jazh_artifact_c = (
        (1 - jazh_sem_c_gap / jazh_strict_gap) * 100 if jazh_strict_gap else 0
    )
    jazh_artifact_cp = (
        (1 - jazh_sem_cp_gap / jazh_strict_gap) * 100 if jazh_strict_gap else 0
    )

    L = [
        f"# Day-18: XFUND semantic-EM (n={n_clean} clean, of {n_sample} sampled, {n_judged} judged)",
        "",
        "## Overall (all 7 languages)",
        "",
        "| metric | EM |",
        "|---|---:|",
        f"| Strict-EM | {overall['strict']*100:.2f}% |",
        f"| Normalized-EM | {overall['norm']*100:.2f}% |",
        f"| Semantic-EM (CORRECT) | {overall['sem_c']*100:.2f}% |",
        f"| Semantic-EM (CORRECT+PARTIAL) | {overall['sem_cp']*100:.2f}% |",
        "",
        "## Script-level (Latin vs CJK)",
        "",
        "| script | n | strict | norm | sem-C | sem-C+P |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for sr in script_rows:
        L.append(
            f"| {sr['name']} | {sr['n']} | {sr['strict']*100:.2f}% | "
            f"{sr['norm']*100:.2f}% | {sr['sem_c']*100:.2f}% | {sr['sem_cp']*100:.2f}% |"
        )

    L += [
        "",
        f"**Latin − CJK gap**:",
        "",
        "| metric | gap | vs strict |",
        "|---|---:|---:|",
        f"| strict-EM | {script_gap_strict*100:+.2f}pp | (baseline) |",
        f"| normalized-EM | {script_gap_norm*100:+.2f}pp | |",
        f"| semantic-EM-C | {script_gap_sem_c*100:+.2f}pp | artifact share = {artifact_c:.1f}% |",
        f"| semantic-EM-C+P | {script_gap_sem_cp*100:+.2f}pp | artifact share = {artifact_cp:.1f}% |",
        "",
        "## Within-CJK: Chinese − Japanese (the existing Findings claim)",
        "",
        f"Findings paper claim: JA−ZH strict-EM gap is **~19.9–21.6pp** across six methods (B3 translate-train minimum: 19.8553pp). On the held-out stratified sample of XFUND test:",
        "",
        "| metric | ja | zh | zh − ja gap |",
        "|---|---:|---:|---:|",
        f"| strict-EM | {pct(ja,'strict'):.2f}% | {pct(zh,'strict'):.2f}% | {jazh_strict_gap:+.2f}pp |",
        f"| normalized-EM | {pct(ja,'norm'):.2f}% | {pct(zh,'norm'):.2f}% | {jazh_norm_gap:+.2f}pp |",
        f"| semantic-EM-C | {pct(ja,'sem_c'):.2f}% | {pct(zh,'sem_c'):.2f}% | {jazh_sem_c_gap:+.2f}pp |",
        f"| semantic-EM-C+P | {pct(ja,'sem_cp'):.2f}% | {pct(zh,'sem_cp'):.2f}% | {jazh_sem_cp_gap:+.2f}pp |",
        "",
        f"**JA→ZH artifact share** under sem-C: **{jazh_artifact_c:.1f}%**  /  under sem-C+P: **{jazh_artifact_cp:.1f}%**",
        "",
        "## Per-language EM (all 7)",
        "",
        "| lang | n | strict | norm | sem-C | sem-C+P |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for lang in sorted(by_lang.keys()):
        d = by_lang[lang]
        L.append(
            f"| {lang} | {d['n']} | {pct(d,'strict'):.2f}% | "
            f"{pct(d,'norm'):.2f}% | {pct(d,'sem_c'):.2f}% | {pct(d,'sem_cp'):.2f}% |"
        )

    # Per-verdict counts for QA on the run
    v_counts = Counter(r["verdict"] for r in rows)
    L += [
        "",
        "## Verdict distribution",
        "",
        "| verdict | n | pct |",
        "|---|---:|---:|",
    ]
    for v, c in v_counts.most_common():
        L.append(f"| {v} | {c} | {c/n_clean*100:.1f}% |")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"[done] wrote {OUT}")
    print()
    print("=== HEADLINE ===")
    print(f"Latin−CJK script gap: strict={script_gap_strict*100:+.2f}pp  "
          f"sem-C={script_gap_sem_c*100:+.2f}pp  artifact={artifact_c:.0f}%")
    print(f"ZH−JA within-CJK gap: strict={jazh_strict_gap:+.2f}pp  "
          f"sem-C={jazh_sem_c_gap:+.2f}pp  artifact={jazh_artifact_c:.0f}%")


if __name__ == "__main__":
    main()
