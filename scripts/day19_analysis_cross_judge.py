#!/usr/bin/env python3
"""Day-19 / P0-A analysis: Cross-judge agreement between Gemini-Flash and Gemini-Pro
on the 358-row MTVQA held-out, for both Qwen2.5-VL-7B and InternVL-2.5-8B.

Tests whether the artifact-share claim (Qwen 65% / InternVL 8%) is robust to
judge identity.

Inputs:
  results/day17_forensic/semantic_judgements.jsonl       (Flash, Qwen base + d16)
  results/day18_internvl_semantic/judgements.jsonl        (Flash, InternVL base)
  results/day19_cross_judge_pro/judgements.jsonl         (Pro, both models)

Output: results/day19_cross_judge_pro/agreement_report.md
"""
from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LATIN = {"de","fr","it","vi"}
NONLATIN = {"ar","ja","kr","ru","th"}


def load_jsonl(p, key="sample_id"):
    out = {}
    for line in p.open():
        s = line.strip().lstrip("\x00")
        if not s: continue
        try: r = json.loads(s)
        except: continue
        if r.get("verdict") in ("CORRECT","PARTIAL","INCORRECT"):
            out[r[key]] = r
    return out


def cohens_kappa(rows_a, rows_b):
    """3-class Cohen's kappa given two parallel verdict lists."""
    if not rows_a:
        return 0.0, 0.0
    n = len(rows_a)
    agree = sum(1 for a, b in zip(rows_a, rows_b) if a == b)
    p_obs = agree / n
    cats = ("CORRECT","PARTIAL","INCORRECT")
    pa = Counter(rows_a); pb = Counter(rows_b)
    p_exp = sum((pa[c]/n)*(pb[c]/n) for c in cats)
    if (1 - p_exp) < 1e-9: return p_obs, 0.0
    return p_obs, (p_obs - p_exp) / (1 - p_exp)


def artifact_share(rows_by_lang, strict_lookup):
    """Compute Latin-vs-non-Latin gap under strict & sem-C, then artifact share."""
    L = []; NL = []
    for sid, j in rows_by_lang.items():
        lang = j["language"]
        s = strict_lookup.get(sid, False)
        c = j["verdict"] == "CORRECT"
        if lang in LATIN:
            L.append((s, c))
        elif lang in NONLATIN:
            NL.append((s, c))
    if not L or not NL: return None
    nl = len(L); nnl = len(NL)
    strict_L = sum(x[0] for x in L)/nl
    strict_NL = sum(x[0] for x in NL)/nnl
    sem_L = sum(x[1] for x in L)/nl
    sem_NL = sum(x[1] for x in NL)/nnl
    gap_s = strict_L - strict_NL
    gap_c = sem_L - sem_NL
    art = 1 - gap_c/gap_s if gap_s != 0 else 0.0
    return {"nL": nl, "nNL": nnl,
            "strict_L": strict_L, "strict_NL": strict_NL, "gap_strict": gap_s,
            "sem_L": sem_L, "sem_NL": sem_NL, "gap_sem": gap_c,
            "artifact": art*100}


def main():
    # Flash judgments — Qwen base
    flash_qwen = {}
    for line in (REPO/"results/day17_forensic/semantic_judgements.jsonl").open():
        s = line.strip().lstrip("\x00")
        if not s: continue
        try: r = json.loads(s)
        except: continue
        if r.get("model") == "base" and r.get("verdict") in ("CORRECT","PARTIAL","INCORRECT"):
            flash_qwen[r["sample_id"]] = r

    # Flash judgments — InternVL base
    flash_intern = load_jsonl(REPO/"results/day18_internvl_semantic/judgements.jsonl")

    # Pro judgments — both, key by (sid, vlm_model)
    pro_all = {}
    for line in (REPO/"results/day19_cross_judge_pro/judgements.jsonl").open():
        s = line.strip().lstrip("\x00")
        if not s: continue
        try: r = json.loads(s)
        except: continue
        if r.get("verdict") in ("CORRECT","PARTIAL","INCORRECT"):
            pro_all[(r["sample_id"], r["vlm_model"])] = r

    # Strict EM lookups
    qwen_strict = {}
    for line in (REPO/"results/mtvqa_perlang_diagnostic/per_pair_base.jsonl").open():
        s = line.strip().lstrip("\x00")
        if not s: continue
        try: r = json.loads(s)
        except: continue
        qwen_strict[r["sample_id"]] = bool(r.get("em", False))
    intern_strict = {}
    for line in (REPO/"results/mtvqa_perlang_internvl/per_pair_base.jsonl").open():
        s = line.strip().lstrip("\x00")
        if not s: continue
        try: r = json.loads(s)
        except: continue
        intern_strict[r["sample_id"]] = bool(r.get("em", False))

    # Restrict to rows judged by BOTH Flash and Pro (per VLM)
    qwen_sids_both = [sid for sid in flash_qwen
                       if (sid, "qwen25vl_7b") in pro_all]
    intern_sids_both = [sid for sid in flash_intern
                         if (sid, "internvl25_8b") in pro_all]

    # Per-VLM agreement
    L = ["# Day-19 P0-A: Cross-judge agreement (Flash vs Pro)\n"]

    for name, sids, flash_dict, pro_key in [
        ("Qwen2.5-VL-7B base", qwen_sids_both, flash_qwen, "qwen25vl_7b"),
        ("InternVL-2.5-8B base", intern_sids_both, flash_intern, "internvl25_8b"),
    ]:
        if not sids:
            L.append(f"\n## {name}: no rows jointly judged yet.\n")
            continue
        a = [flash_dict[s]["verdict"] for s in sids]
        b = [pro_all[(s, pro_key)]["verdict"] for s in sids]
        p_obs, kappa = cohens_kappa(a, b)
        n = len(sids)
        flash_c = sum(1 for v in a if v == "CORRECT")
        pro_c = sum(1 for v in b if v == "CORRECT")
        L.append(f"\n## {name} (n={n})\n")
        L.append(f"- 3-way agreement: **{p_obs*100:.1f}%**")
        L.append(f"- Cohen's κ: **{kappa:.3f}**")
        L.append(f"- semantic-EM-C (Flash): **{flash_c/n*100:.2f}%**  ({flash_c}/{n})")
        L.append(f"- semantic-EM-C (Pro):   **{pro_c/n*100:.2f}%**  ({pro_c}/{n})")
        L.append(f"- absolute |Flash − Pro|: **{abs(flash_c - pro_c)/n*100:.2f} pp**\n")
        # Confusion
        cats = ("CORRECT","PARTIAL","INCORRECT")
        conf = {(x,y): 0 for x in cats for y in cats}
        for av, bv in zip(a, b):
            conf[(av, bv)] += 1
        L.append("Confusion (rows = Flash, cols = Pro):\n")
        L.append("| | " + " | ".join(cats) + " |")
        L.append("|---|" + "---|"*len(cats))
        for cx in cats:
            row = "| " + cx + " | " + " | ".join(str(conf[(cx,cy)]) for cy in cats) + " |"
            L.append(row)

    # Artifact share comparison: Flash vs Pro for each VLM
    L.append("\n## Artifact-share replication\n")
    L.append("Latin (de/fr/it/vi) vs non-Latin (ar/ja/kr/ru/th) script gap, "
             "decomposed under each judge.\n")
    L.append("| VLM | judge | n_L | n_NL | gap strict | gap sem-C | artifact share |")
    L.append("|---|---|---:|---:|---:|---:|---:|")
    # Qwen Flash artifact
    q_flash_subset = {s: flash_qwen[s] for s in qwen_sids_both}
    q_pro_subset = {s: pro_all[(s, "qwen25vl_7b")] for s in qwen_sids_both}
    # Need to inject language on q_pro_subset — already in pro records
    a_q_flash = artifact_share(q_flash_subset, qwen_strict)
    a_q_pro = artifact_share(q_pro_subset, qwen_strict)
    i_flash_subset = {s: flash_intern[s] for s in intern_sids_both}
    i_pro_subset = {s: pro_all[(s, "internvl25_8b")] for s in intern_sids_both}
    a_i_flash = artifact_share(i_flash_subset, intern_strict)
    a_i_pro = artifact_share(i_pro_subset, intern_strict)

    for vlm, judge, art in [
        ("Qwen2.5-VL-7B", "Gemini-Flash", a_q_flash),
        ("Qwen2.5-VL-7B", "Gemini-Pro",  a_q_pro),
        ("InternVL-2.5-8B", "Gemini-Flash", a_i_flash),
        ("InternVL-2.5-8B", "Gemini-Pro", a_i_pro),
    ]:
        if art is None:
            L.append(f"| {vlm} | {judge} | -- | -- | -- | -- | -- |")
            continue
        L.append(f"| {vlm} | {judge} | {art['nL']} | {art['nNL']} | "
                 f"{art['gap_strict']*100:+.2f}pp | {art['gap_sem']*100:+.2f}pp | "
                 f"{art['artifact']:.1f}% |")

    # Save
    out = REPO/"results/day19_cross_judge_pro/agreement_report.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"[done] {out}")
    print("\n".join(L[-20:]))


if __name__ == "__main__":
    main()
