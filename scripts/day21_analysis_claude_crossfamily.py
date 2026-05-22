#!/usr/bin/env python3
"""Day-21 analysis: Claude-vs-Gemini-Flash cross-FAMILY judge agreement on the
358-row MTVQA held-out (Qwen-7B base + InternVL-8B base).

Addresses reviewer-defense gap R3: Gemini-Flash vs Gemini-Pro is intra-family;
this script computes the *genuine* cross-family inter-judge agreement using
the Claude-Sonnet-4 verdicts written by scripts/day21_judge_claude_crossfamily.py.

Inputs:
  results/day17_forensic/semantic_judgements.jsonl     (Flash, Qwen base)
  results/day18_internvl_semantic/judgements.jsonl     (Flash, InternVL base)
  results/day21_cross_judge_claude/judgements.jsonl    (Claude, both VLMs)

Output:
  results/day21_cross_judge_claude/agreement_report.md
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FLASH_QWEN     = REPO / "results/day17_forensic/semantic_judgements.jsonl"
FLASH_INTERNVL = REPO / "results/day18_internvl_semantic/judgements.jsonl"
CLAUDE         = REPO / "results/day21_cross_judge_claude/judgements.jsonl"
OUT_MD         = REPO / "results/day21_cross_judge_claude/agreement_report.md"

VALID = ("CORRECT", "PARTIAL", "INCORRECT")
LATIN = {"de", "fr", "it", "vi"}
NONLATIN = {"ar", "ja", "kr", "ru", "th"}


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip().lstrip("\x00")
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def cohens_kappa(observed: float, expected: float) -> float:
    return (observed - expected) / (1 - expected) if (1 - expected) > 0 else 0.0


def _kappa_table(pairs: list[tuple[str, str]]) -> dict:
    """Compute 3-way agreement + Cohen's kappa on (a, b) pairs."""
    n = len(pairs)
    if n == 0:
        return {"n": 0, "agree": 0, "pct": 0.0, "kappa": 0.0}
    agree = sum(1 for a, b in pairs if a == b)
    pa = Counter(a for a, _ in pairs)
    pb = Counter(b for _, b in pairs)
    observed = agree / n
    expected = sum((pa[v] / n) * (pb[v] / n) for v in VALID)
    return {
        "n": n,
        "agree": agree,
        "pct": observed * 100,
        "kappa": cohens_kappa(observed, expected),
    }


def _confusion(pairs: list[tuple[str, str]]) -> dict[tuple[str, str], int]:
    m = {(a, b): 0 for a in VALID for b in VALID}
    for a, b in pairs:
        if a in VALID and b in VALID:
            m[(a, b)] = m.get((a, b), 0) + 1
    return m


def main() -> None:
    # ---- Load Flash verdicts (Qwen base + InternVL base) ----
    # Schema notes:
    # - day17_forensic: 'model' field with values 'base' (Qwen-VL-7B base predictions)
    #   or 'd13'/'d16' (SFT checkpoint predictions). We want 'base' for the cross-judge.
    # - day18_internvl: no VLM field at all; all rows are InternVL-2.5-8B base.
    # - day21_claude:   'vlm_model' field with 'qwen25vl_7b' or 'internvl25_8b'.
    # We canonicalise to the day21 names so the join key matches Claude rows.
    flash: dict[tuple[str, str], dict] = {}
    for r in load_jsonl(FLASH_QWEN):
        if r.get("model") != "base":
            continue
        if r.get("verdict") not in VALID:
            continue
        flash[(r["sample_id"], "qwen25vl_7b")] = r
    for r in load_jsonl(FLASH_INTERNVL):
        if r.get("verdict") not in VALID:
            continue
        flash[(r["sample_id"], "internvl25_8b")] = r
    print(f"[load] Flash verdicts: {len(flash)} ({FLASH_QWEN.name} + {FLASH_INTERNVL.name})")

    # ---- Load Claude verdicts ----
    claude_rows = [r for r in load_jsonl(CLAUDE) if r.get("verdict") in VALID]
    print(f"[load] Claude verdicts: {len(claude_rows)}  ({CLAUDE.name})")

    # ---- Match by (sample_id, vlm_model) ----
    pairs_all: list[tuple[str, str]] = []
    pairs_by_vlm: dict[str, list] = defaultdict(list)
    pairs_by_lang: dict[str, list] = defaultdict(list)
    pairs_by_script: dict[str, list] = defaultdict(list)

    matched = 0
    unmatched = 0
    for c in claude_rows:
        key = (c["sample_id"], c["vlm_model"])
        # canonicalise InternVL key — day18 file uses 'internvl_8b'; Pro file uses same
        f = flash.get(key)
        if not f:
            unmatched += 1
            continue
        matched += 1
        pair = (f["verdict"], c["verdict"])  # (Flash, Claude)
        pairs_all.append(pair)
        pairs_by_vlm[c["vlm_model"]].append(pair)
        pairs_by_lang[c["language"]].append(pair)
        s = "Latin" if c["language"] in LATIN else ("non-Latin" if c["language"] in NONLATIN else "other")
        pairs_by_script[s].append(pair)

    print(f"[match] {matched} matched, {unmatched} unmatched")

    # ---- Overall stats ----
    overall = _kappa_table(pairs_all)
    print(f"\n=== Overall Flash-vs-Claude (n={overall['n']}) ===")
    print(f"  3-way agreement: {overall['pct']:.1f}% ({overall['agree']}/{overall['n']})")
    print(f"  Cohen's kappa:   {overall['kappa']:.3f}")

    # Binary CORRECT vs not
    bin_pairs = [((a == "CORRECT"), (b == "CORRECT")) for a, b in pairs_all]
    bin_agree = sum(1 for a, b in bin_pairs if a == b)
    bin_pct = bin_agree / len(bin_pairs) * 100
    print(f"  Binary C-vs-not: {bin_pct:.1f}% ({bin_agree}/{len(bin_pairs)})")

    # ---- Confusion matrix ----
    conf = _confusion(pairs_all)
    print(f"\n  confusion (rows = Flash, cols = Claude):")
    print(f"           CORRECT  PARTIAL  INCORRECT")
    for a in VALID:
        cells = "  ".join(f"{conf[(a, b)]:>7}" for b in VALID)
        print(f"    {a[:7]:7s}  {cells}")

    # ---- Per-VLM ----
    print(f"\n=== Per-VLM ===")
    per_vlm = {}
    for vlm in sorted(pairs_by_vlm):
        s = _kappa_table(pairs_by_vlm[vlm])
        per_vlm[vlm] = s
        print(f"  {vlm:20s}  n={s['n']:>3}  agree={s['pct']:>5.1f}%   kappa={s['kappa']:.3f}")

    # ---- Per-script ----
    print(f"\n=== Per-script ===")
    per_script = {}
    for sc in ("Latin", "non-Latin"):
        s = _kappa_table(pairs_by_script[sc])
        per_script[sc] = s
        print(f"  {sc:10s}  n={s['n']:>3}  agree={s['pct']:>5.1f}%   kappa={s['kappa']:.3f}")

    # ---- Per-language ----
    per_lang = {}
    for lang in sorted(pairs_by_lang):
        per_lang[lang] = _kappa_table(pairs_by_lang[lang])

    # ---- Write report ----
    L = [
        "# Claude-Sonnet-4 vs Gemini-2.5-Flash cross-family judge agreement\n",
        f"On the {overall['n']} matched (sample_id, vlm_model) tuples shared between the "
        f"existing Gemini-Flash held-out judging and the new Claude-Sonnet-4 run "
        f"(`{CLAUDE.parent.name}/`). This is the **genuine cross-family** check "
        f"(R4 in the paper's protocol).\n",
        "## Headline\n",
        f"- **3-way agreement**: **{overall['pct']:.1f}%** ({overall['agree']}/{overall['n']})\n",
        f"- **Cohen's κ**: **{overall['kappa']:.3f}**\n",
        f"- **Binary CORRECT-vs-not**: {bin_pct:.1f}% ({bin_agree}/{len(bin_pairs)})\n",
        "\nCompare to the intra-family Gemini-Flash vs Gemini-Pro number: $89.2\\%$ / $\\kappa \\approx 0.82$. "
        "A cross-family kappa lower than the intra-family one is expected (different model families have "
        "different decision boundaries on PARTIAL); the headline is whether agreement remains in the "
        "'substantial' band ($\\kappa \\geq 0.6$) under Landis & Koch (1977).\n",
        "## 3×3 confusion (rows = Flash, cols = Claude)\n",
        "| | CORRECT | PARTIAL | INCORRECT |",
        "|---|---:|---:|---:|",
    ]
    for a in VALID:
        cells = " | ".join(str(conf[(a, b)]) for b in VALID)
        L.append(f"| Flash-{a[0]} | {cells} |")

    L.append("\n## Per-VLM\n")
    L.append("| VLM | n | 3-way agreement | κ |")
    L.append("|---|---:|---:|---:|")
    for vlm, s in per_vlm.items():
        L.append(f"| {vlm} | {s['n']} | {s['pct']:.1f}% | {s['kappa']:.3f} |")

    L.append("\n## Per-script\n")
    L.append("| script | n | 3-way agreement | κ |")
    L.append("|---|---:|---:|---:|")
    for sc, s in per_script.items():
        L.append(f"| {sc} | {s['n']} | {s['pct']:.1f}% | {s['kappa']:.3f} |")

    L.append("\n## Per-language\n")
    L.append("| lang | n | 3-way agreement | κ |")
    L.append("|---|---:|---:|---:|")
    for lang, s in per_lang.items():
        L.append(f"| {lang} | {s['n']} | {s['pct']:.1f}% | {s['kappa']:.3f} |")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print(f"\n[done] report: {OUT_MD}")


if __name__ == "__main__":
    main()
