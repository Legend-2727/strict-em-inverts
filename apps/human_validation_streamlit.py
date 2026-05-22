#!/usr/bin/env python3
"""Localhost Streamlit annotation app for the 50-row human-validation pass
against the Gemini-2.5-Flash semantic-EM judge.

Run:
  streamlit run apps/human_validation_streamlit.py

Layout per row:
  - document image (the question is grounded against this)
  - question (rendered in the source language)
  - gold answer (string)
  - model prediction (string)
  - your three buttons: CORRECT / PARTIAL / INCORRECT
  - optional one-line note

The Gemini verdict + reason are HIDDEN until after you have submitted your
own verdict, so the judgement is blind (the kappa pass requires this).

Labels persist after every submission to
  results/day18_human_validation/human_labels.json
  results/day18_human_validation/sample.tsv  (HUMAN_VERDICT column filled in)

Refresh-safe: on page reload, the app re-loads labels from JSON and resumes
on the first unannotated row (or wherever you last navigated to).
"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

# ---------- paths ----------

REPO = Path(__file__).resolve().parent.parent
DAY18 = REPO / "results/day18_human_validation"
SAMPLE = DAY18 / "sample.jsonl"
IMG_DIR = DAY18 / "images"
LABELS_JSON = DAY18 / "human_labels.json"
SAMPLE_TSV = DAY18 / "sample.tsv"

VERDICTS = ("CORRECT", "PARTIAL", "INCORRECT")


# ---------- data ----------

@st.cache_data
def load_sample() -> list[dict]:
    rows = []
    with SAMPLE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: r["_anno_idx"])
    return rows


def load_labels() -> dict[str, dict]:
    if not LABELS_JSON.exists():
        return {}
    return json.loads(LABELS_JSON.read_text(encoding="utf-8"))


def save_labels(labels: dict[str, dict]) -> None:
    LABELS_JSON.write_text(json.dumps(labels, ensure_ascii=False, indent=2),
                           encoding="utf-8")


def update_tsv(rows: list[dict], labels: dict[str, dict]) -> None:
    """Rewrite sample.tsv with HUMAN_VERDICT and HUMAN_NOTES columns filled
    from the current labels dict. The scoring script reads this file."""
    lines = ["idx\tlanguage\tmodel\timage\tquestion\tgold\tpred\tgemini_verdict\tHUMAN_VERDICT\tHUMAN_NOTES"]
    for r in rows:
        key = str(r["_anno_idx"])
        lab = labels.get(key, {})
        human = lab.get("verdict", "")
        notes = lab.get("notes", "")
        cells = [
            str(r["_anno_idx"]),
            r["language"],
            r["model"],
            r["_image_local"],
            (r.get("question") or "").replace("\t", " ").replace("\n", " "),
            (r.get("gold") or "").replace("\t", " ").replace("\n", " "),
            (r.get("pred") or "").replace("\t", " ").replace("\n", " "),
            r["verdict"],
            human,
            notes.replace("\t", " ").replace("\n", " "),
        ]
        lines.append("\t".join(cells))
    SAMPLE_TSV.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------- app ----------

st.set_page_config(page_title="Judge κ validation", layout="wide")

rows = load_sample()
labels = load_labels()
n_total = len(rows)
n_done = len(labels)

# session state for current idx
if "idx" not in st.session_state:
    # resume at first unannotated row
    for r in rows:
        key = str(r["_anno_idx"])
        if key not in labels:
            st.session_state.idx = r["_anno_idx"]
            break
    else:
        st.session_state.idx = 0

if "reveal" not in st.session_state:
    st.session_state.reveal = False


# ---------- sidebar: progress + nav ----------

with st.sidebar:
    st.title("Judge κ validation")
    st.metric("Annotated", f"{n_done} / {n_total}")
    st.progress(n_done / max(n_total, 1))

    st.divider()
    st.subheader("Navigate")

    # jump dropdown
    options = []
    for r in rows:
        key = str(r["_anno_idx"])
        mark = "✓" if key in labels else "·"
        options.append(f"{mark} #{r['_anno_idx']:02d}  [{r['language']}]")
    sel = st.selectbox("Jump to row", options=options,
                       index=st.session_state.idx,
                       label_visibility="collapsed")
    new_idx = int(sel.split("#")[1].split(" ")[0])
    if new_idx != st.session_state.idx:
        st.session_state.idx = new_idx
        st.session_state.reveal = False
        st.rerun()

    col_prev, col_next = st.columns(2)
    with col_prev:
        if st.button("← Prev", use_container_width=True,
                     disabled=st.session_state.idx == 0):
            st.session_state.idx -= 1
            st.session_state.reveal = False
            st.rerun()
    with col_next:
        if st.button("Next →", use_container_width=True,
                     disabled=st.session_state.idx >= n_total - 1):
            st.session_state.idx += 1
            st.session_state.reveal = False
            st.rerun()

    st.divider()
    st.subheader("Rubric")
    st.markdown(
        "- **CORRECT** — semantically equivalent to gold (synonyms, "
        "whitespace, diacritics, alternate phrasing, valid translation, "
        "lossless reformat — same meaning).\n"
        "- **PARTIAL** — contains correct info but is incomplete or less "
        "specific (e.g., distance without direction; 1 of 3 items).\n"
        "- **INCORRECT** — states something different, fabricates info "
        "not in the image, or is unrelated.\n\n"
        "If the gold itself is wrong and the prediction matches the "
        "image better, mark **CORRECT** (judge against image, not gold)."
    )

    st.divider()
    if n_done == n_total:
        st.success("All 50 rows annotated. Run:\n\n"
                   "`python scripts/day18_human_validation_scoring.py`")


# ---------- main: current row ----------

cur = rows[st.session_state.idx]
cur_key = str(cur["_anno_idx"])
cur_lab = labels.get(cur_key, {})

st.markdown(
    f"### Row #{cur['_anno_idx']:02d}  &nbsp; "
    f"`{cur['language']}`  &nbsp; "
    f"model=`{cur['model']}`",
    unsafe_allow_html=True,
)

col_img, col_qa = st.columns([1, 1])

with col_img:
    img_path = IMG_DIR / f"{cur['_anno_idx']:02d}_{cur['image_uid']}.png"
    if img_path.exists():
        st.image(str(img_path), use_container_width=True)
    else:
        st.warning(f"Image not found: {img_path.name}\n\n"
                   "Run `python scripts/day21_extract_human_validation_images.py` "
                   "to fetch the 50 images from HF MTVQA.")

with col_qa:
    st.markdown("**Question**")
    st.markdown(f"<div style='font-size:1.1em'>{cur.get('question','')}</div>",
                unsafe_allow_html=True)

    st.markdown("**Gold**")
    st.code(cur.get("gold", ""), language=None)

    st.markdown("**Model prediction**")
    st.code(cur.get("pred", ""), language=None)

    st.markdown("---")

    # Verdict buttons
    st.markdown("**Your verdict** (blind to Gemini)")
    current_verdict = cur_lab.get("verdict", "")
    c1, c2, c3 = st.columns(3)
    pick = None
    with c1:
        if st.button("✅ CORRECT",
                     type="primary" if current_verdict == "CORRECT" else "secondary",
                     use_container_width=True):
            pick = "CORRECT"
    with c2:
        if st.button("➖ PARTIAL",
                     type="primary" if current_verdict == "PARTIAL" else "secondary",
                     use_container_width=True):
            pick = "PARTIAL"
    with c3:
        if st.button("❌ INCORRECT",
                     type="primary" if current_verdict == "INCORRECT" else "secondary",
                     use_container_width=True):
            pick = "INCORRECT"

    note = st.text_input("Optional note", value=cur_lab.get("notes", ""),
                         key=f"note_{cur_key}")

    if pick is not None:
        labels[cur_key] = {
            "verdict": pick,
            "notes": note,
            "language": cur["language"],
            "image_uid": cur["image_uid"],
            "gemini_verdict": cur["verdict"],
        }
        save_labels(labels)
        update_tsv(rows, labels)
        # auto-advance
        if st.session_state.idx < n_total - 1:
            st.session_state.idx += 1
            st.session_state.reveal = False
        st.rerun()

    # Save note even without a verdict change
    if note != cur_lab.get("notes", "") and cur_key in labels:
        labels[cur_key]["notes"] = note
        save_labels(labels)
        update_tsv(rows, labels)

    st.markdown("---")

    # Reveal Gemini verdict (after annotation, for review)
    if cur_key in labels:
        if not st.session_state.reveal:
            if st.button("👁  Reveal Gemini verdict (post-annotation)",
                         use_container_width=True):
                st.session_state.reveal = True
                st.rerun()
        else:
            gem = cur["verdict"]
            human = labels[cur_key]["verdict"]
            agree = "✓ AGREE" if gem == human else "✗ DISAGREE"
            color = "green" if gem == human else "red"
            st.markdown(
                f"**Gemini said:** `{gem}` &nbsp;&nbsp; "
                f"<span style='color:{color}'>**{agree}**</span>",
                unsafe_allow_html=True,
            )
            st.caption(f"Gemini's reason: {cur.get('reason','')}")
    else:
        st.caption("Pick a verdict above; the Gemini verdict stays hidden "
                   "until you submit yours.")
