#!/usr/bin/env python3
"""Day-19: Bootstrap 95% CIs on the paper's headline numbers.

Computes 2000-iteration bootstrap CIs for:
  (a) Qwen vs InternVL strict-EM and sem-C aggregate, plus the head-to-head
      difference, on the 351-row joint held-out
  (b) Cross-script Latin/non-Latin gap, and artifact share, for each of the
      three VLMs (Qwen, InternVL, Phi)
  (c) Difference of artifact shares between VLM pairs (to support the
      "spread across 121 pp" claim)

Output: results/day19_bootstrap/cis.md
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results/day19_bootstrap/cis.md"
N_BOOT = 2000
SEED = 42

LATIN = {"de", "fr", "it", "vi"}
NON_LATIN = {"ar", "ja", "kr", "ru", "th"}

EVAL_IMGS = REPO / "data/processed/dpo_pairs_real_failures/eval_image_uids.json"


def read_jsonl(p):
    out = []
    for line in p.open():
        s = line.strip().lstrip("\x00")
        if not s:
            continue
        try:
            out.append(json.loads(s))
        except Exception:
            pass
    return out


def coerce(v):
    v = (v or "").strip().upper()
    return v if v in {"CORRECT", "PARTIAL", "INCORRECT"} else "INCORRECT"


def load_qwen_holdout():
    base = {r["sample_id"]: r for r in read_jsonl(REPO / "results/mtvqa_perlang_diagnostic/per_pair_base.jsonl")}
    judg_sem = {}
    for line in (REPO / "results/day17_forensic/semantic_judgements.jsonl").open():
        s = line.strip().lstrip("\x00")
        if not s:
            continue
        try:
            r = json.loads(s)
        except Exception:
            continue
        if r.get("model") == "base":
            judg_sem[r["sample_id"]] = r
    eval_imgs = set(json.loads(EVAL_IMGS.read_text()))
    rows = []
    for sid, b in base.items():
        if b["image_uid"] not in eval_imgs:
            continue
        j = judg_sem.get(sid)
        if not j:
            continue
        rows.append({
            "sid": sid, "lang": b["language"],
            "em": bool(b.get("em", False)),
            "v": coerce(j.get("verdict", "")),
        })
    return rows


def load_internvl_holdout():
    base = {r["sample_id"]: r for r in read_jsonl(REPO / "results/mtvqa_perlang_internvl/per_pair_base.jsonl")}
    judg = {r["sample_id"]: r for r in read_jsonl(REPO / "results/day18_internvl_semantic/judgements.jsonl")}
    eval_imgs = set(json.loads(EVAL_IMGS.read_text()))
    rows = []
    for sid, b in base.items():
        if b["image_uid"] not in eval_imgs:
            continue
        j = judg.get(sid)
        if not j:
            continue
        rows.append({
            "sid": sid, "lang": b["language"],
            "em": bool(b.get("em", False)),
            "v": coerce(j.get("verdict", "")),
        })
    return rows


def load_phi_holdout():
    base = {r["sample_id"]: r for r in read_jsonl(REPO / "results/day19_phi35vision_heldout/per_pair_base.jsonl")}
    judg = {r["sample_id"]: r for r in read_jsonl(REPO / "results/day19_phi35vision_semantic/judgements.jsonl")}
    rows = []
    for sid, b in base.items():
        j = judg.get(sid)
        if not j:
            continue
        rows.append({
            "sid": sid, "lang": b["language"],
            "em": bool(b.get("em", False)),
            "v": coerce(j.get("verdict", "")),
        })
    return rows


def strict(r): return int(r["em"])
def sem_c(r):  return int(r["v"] == "CORRECT")


def script_gap(rows, value_fn):
    l_n = l_v = 0
    nl_n = nl_v = 0
    for r in rows:
        if r["lang"] in LATIN:
            l_n += 1
            l_v += value_fn(r)
        elif r["lang"] in NON_LATIN:
            nl_n += 1
            nl_v += value_fn(r)
    if not l_n or not nl_n:
        return 0.0
    return l_v / l_n - nl_v / nl_n


def artifact_share(rows):
    g_strict = script_gap(rows, strict)
    g_sem = script_gap(rows, sem_c)
    if g_strict == 0:
        return float("nan")
    return (1 - g_sem / g_strict) * 100


def aggregate(rows, value_fn):
    if not rows:
        return 0.0
    return sum(value_fn(r) for r in rows) / len(rows) * 100


def bootstrap(rows, stat_fn, n=N_BOOT, seed=SEED):
    rng = random.Random(seed)
    N = len(rows)
    vals = []
    for _ in range(n):
        sample = [rows[rng.randint(0, N - 1)] for _ in range(N)]
        v = stat_fn(sample)
        if v == v:  # filter nan
            vals.append(v)
    vals.sort()
    lo = vals[int(0.025 * len(vals))]
    hi = vals[int(0.975 * len(vals))]
    return lo, hi


def joint_rows(qwen_rows, intvl_rows):
    """Return aligned (qwen, intvl) rows on the joint sample_ids."""
    q = {r["sid"]: r for r in qwen_rows}
    i = {r["sid"]: r for r in intvl_rows}
    common = sorted(set(q) & set(i))
    return [(q[s], i[s]) for s in common]


def main():
    qwen = load_qwen_holdout()
    intvl = load_internvl_holdout()
    phi = load_phi_holdout()
    print(f"loaded Qwen={len(qwen)} InternVL={len(intvl)} Phi={len(phi)}")

    joint = joint_rows(qwen, intvl)
    joint_q = [pair[0] for pair in joint]
    joint_i = [pair[1] for pair in joint]
    print(f"Qwen+InternVL joint sample_ids: {len(joint)}")

    L = ["# Day-19: Bootstrap 95% CIs on headline paper numbers",
         "",
         f"Bootstrap iterations: {N_BOOT}; resample seed: {SEED}; "
         "percentile method; CIs are reported as `point [lo, hi]`.",
         ""]

    # (a) Joint Qwen vs InternVL on held-out
    L += ["## (a) Cross-VLM head-to-head, joint held-out", "",
          f"n = {len(joint)} sample_ids common to Qwen and InternVL judged sets.", ""]
    L += ["| metric | Qwen | InternVL | gap (Qwen − InternVL) |", "|---|---|---|---|"]

    for label, vfn in [("strict-EM", strict), ("sem-EM-C", sem_c)]:
        q_pt = aggregate(joint_q, vfn)
        i_pt = aggregate(joint_i, vfn)
        gap_pt = q_pt - i_pt
        q_lo, q_hi = bootstrap(joint_q, lambda rs: aggregate(rs, vfn))
        i_lo, i_hi = bootstrap(joint_i, lambda rs: aggregate(rs, vfn))
        # paired bootstrap for gap (resample pairs)
        rng = random.Random(SEED + 1)
        N = len(joint)
        gaps = []
        for _ in range(N_BOOT):
            idx = [rng.randint(0, N - 1) for _ in range(N)]
            sq = [joint_q[k] for k in idx]
            si = [joint_i[k] for k in idx]
            gaps.append(aggregate(sq, vfn) - aggregate(si, vfn))
        gaps.sort()
        g_lo = gaps[int(0.025 * N_BOOT)]
        g_hi = gaps[int(0.975 * N_BOOT)]
        L.append(f"| {label} | {q_pt:.2f}% [{q_lo:.2f}, {q_hi:.2f}] | "
                 f"{i_pt:.2f}% [{i_lo:.2f}, {i_hi:.2f}] | "
                 f"{gap_pt:+.2f}pp [{g_lo:+.2f}, {g_hi:+.2f}] |")
    L.append("")

    # (b) Per-VLM artifact share with CIs
    L += ["## (b) Per-VLM Latin/non-Latin artifact share, with CIs", ""]
    L += ["| VLM | n | strict gap (pp) | sem-C gap (pp) | artifact share (%) |",
          "|---|---:|---|---|---|"]
    for name, rows in [("Qwen2.5-VL-7B", qwen), ("InternVL-2.5-8B", intvl),
                       ("Phi-3.5-vision", phi)]:
        gs_pt = script_gap(rows, strict) * 100
        gc_pt = script_gap(rows, sem_c) * 100
        as_pt = artifact_share(rows)
        gs_lo, gs_hi = bootstrap(rows, lambda rs: script_gap(rs, strict) * 100)
        gc_lo, gc_hi = bootstrap(rows, lambda rs: script_gap(rs, sem_c) * 100)
        as_lo, as_hi = bootstrap(rows, artifact_share)
        L.append(f"| {name} | {len(rows)} | "
                 f"{gs_pt:+.2f} [{gs_lo:+.2f}, {gs_hi:+.2f}] | "
                 f"{gc_pt:+.2f} [{gc_lo:+.2f}, {gc_hi:+.2f}] | "
                 f"{as_pt:+.1f} [{as_lo:+.1f}, {as_hi:+.1f}] |")
    L.append("")

    # (c) Pairwise difference of artifact shares
    L += ["## (c) Pairwise differences of artifact share (point + 95% CI)", "",
          "Each VLM is an independent sample; differences are computed by "
          "resampling each rowset independently.", ""]
    pairs = [
        ("Qwen", "InternVL", qwen, intvl),
        ("Qwen", "Phi", qwen, phi),
        ("InternVL", "Phi", intvl, phi),
    ]
    L += ["| pair | Δ artifact share (pp) |", "|---|---|"]
    for a_name, b_name, a_rows, b_rows in pairs:
        a_pt = artifact_share(a_rows)
        b_pt = artifact_share(b_rows)
        diffs = []
        rng = random.Random(SEED + 2)
        Na, Nb = len(a_rows), len(b_rows)
        for _ in range(N_BOOT):
            sa = [a_rows[rng.randint(0, Na - 1)] for _ in range(Na)]
            sb = [b_rows[rng.randint(0, Nb - 1)] for _ in range(Nb)]
            d = artifact_share(sa) - artifact_share(sb)
            if d == d:
                diffs.append(d)
        diffs.sort()
        d_lo = diffs[int(0.025 * len(diffs))]
        d_hi = diffs[int(0.975 * len(diffs))]
        d_pt = a_pt - b_pt
        L.append(f"| {a_name} − {b_name} | "
                 f"{d_pt:+.1f} [{d_lo:+.1f}, {d_hi:+.1f}] |")
    L.append("")

    L += ["## (d) Ranking-inversion significance test", "",
          "We test whether (qwen sem-C > intvl sem-C) AND (qwen strict < intvl strict) "
          "holds in each bootstrap resample of the joint 351-row set. The fraction "
          "of resamples where the joint inversion holds is reported.", ""]
    rng = random.Random(SEED + 3)
    N = len(joint)
    inv_count = 0
    qsem_count = 0
    istrict_count = 0
    for _ in range(N_BOOT):
        idx = [rng.randint(0, N - 1) for _ in range(N)]
        sq = [joint_q[k] for k in idx]
        si = [joint_i[k] for k in idx]
        q_sem = aggregate(sq, sem_c)
        i_sem = aggregate(si, sem_c)
        q_str = aggregate(sq, strict)
        i_str = aggregate(si, strict)
        if q_sem > i_sem:
            qsem_count += 1
        if i_str > q_str:
            istrict_count += 1
        if q_sem > i_sem and i_str > q_str:
            inv_count += 1
    L.append(f"- P(Qwen sem-C > InternVL sem-C) = **{qsem_count/N_BOOT:.3f}**")
    L.append(f"- P(InternVL strict > Qwen strict) = **{istrict_count/N_BOOT:.3f}**")
    L.append(f"- P(joint ranking inversion) = **{inv_count/N_BOOT:.3f}**")
    L.append("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"[done] wrote {OUT}")


if __name__ == "__main__":
    main()
