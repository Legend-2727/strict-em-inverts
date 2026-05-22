# Strict-EM Inverts Cross-Lingual VLM Rankings: An Architecture-Specific Measurement Artifact

Source code, evaluation tuples, and reproduction recipe for the EMNLP 2026
submission *"Strict-EM Inverts Cross-Lingual VLM Rankings: An
Architecture-Specific Measurement Artifact"*.

**Headline claims this repository reproduces:**

1. On the full MTVQA 9-language validation set (`n=2142` jointly judged rows),
   strict-EM and a vision-grounded semantic-EM judge **reverse the
   Qwen-vs-InternVL cross-VLM ranking in 100% of 2000 paired bootstrap
   resamples** (`P(joint inversion) = 1.000`).
2. The Latin/non-Latin "artifact share" — the fraction of the script gap
   explained by surface-form mismatch rather than capability — **spans 121
   percentage points across three open VLMs** (Qwen +66%, InternVL +8%,
   Phi-3.5-vision −57%).
3. 13 plausible cross-lingual interventions on Qwen-7B move strict-EM by up
   to ±4.2 pp while leaving aggregate semantic-EM-C statistically null.
4. The Gemini-2.5-Flash semantic-EM judge has 89.2% 3-way agreement
   (κ ≈ 0.82) with Gemini-2.5-Pro and 74.0% (κ = 0.61) with a 50-row blind
   human pass.

## Repository layout

```
strict-em-inverts/
├── README.md                       # this file
├── requirements.txt                # pinned deps
├── setup.py                        # `pip install -e .`
├── LICENSE                         # MIT
├── .gitignore
│
├── paper/                          # camera-ready sources
│   ├── main.tex                    # ACL template
│   ├── main.pdf                    # 15-page compiled PDF
│   ├── sections/                   # §1–§Appendix
│   ├── figures/                    # PDF + PNG figures
│   ├── references.bib
│   └── acl.sty, acl_natbib.bst     # official ACL style files
│
├── src/                            # python package (`import strict_em_inverts.*`)
│   ├── data/                       # MTVQA / XFUND dataset loaders + prompts
│   ├── evaluation/                 # paired EM, judge eval, metric utilities
│   ├── models/                     # Qwen2.5-VL / InternVL / Phi wrappers
│   ├── steer/                      # KV-cache steering, VAR, logit-lens (used by §6)
│   ├── activation/                 # hidden-state extraction
│   └── utils/                      # logging, qwen-vl helpers, evidence parsing
│
├── scripts/                        # experiment + analysis entrypoints
│   # ---- inference (GPU-bound) ----
│   ├── day19_phi_vision_inference.py    # Phi-3.5-vision on 358-row held-out
│   ├── day19_judge_mtvqa_full.py        # Gemini-Flash judge on Qwen full-val
│   ├── day19_judge_phi.py               # Gemini-Flash judge on Phi held-out
│   ├── day19_judge_crossvalidation.py   # Gemini-Pro cross-judge on 716 tuples
│   ├── day18_judge_internvl_heldout.py  # judge InternVL held-out preds
│   ├── day18_judge_xfund.py             # Gemini-Flash judge on XFUND
│   # ---- analysis (CPU-only, fast) ----
│   ├── day19_phi_analysis.py            # Phi artifact share + figure
│   ├── day19_analysis_cross_judge.py    # Flash-vs-Pro agreement + κ
│   ├── day19_bootstrap_cis.py           # 2000-iter paired bootstrap
│   ├── day19_sft_perlang_cis.py         # paired-mean per-language CIs
│   ├── day19_sensitivity_dropped.py     # 3-coercion sensitivity
│   ├── day19_figure_three_vlm.py        # Fig 2 (3-VLM artifact bar)
│   ├── day20_verbosity_analysis.py      # Table 3 (output char ratios)
│   ├── day18_human_validation_scoring.py    # κ / agreement scoring
│   ├── day21_extract_human_validation_images.py
│   ├── day21_patch_kappa_into_paper.py
│   ├── day18_figure_headline.py             # Fig 1
│   ├── day18_figure_perlang.py
│   ├── day18_figure_anti_correlation.py     # Fig 3
│   ├── day18_xfund_analysis.py
│   ├── day18_xfund_stratified_sample.py
│   ├── day18_internvl_analysis.py
│   └── day18_build_human_validation_sample.py
│
├── apps/
│   └── human_validation_streamlit.py   # localhost blind-annotation app
│
├── notebooks/
│   └── colab_runner.ipynb              # one-shot end-to-end Colab runner
│
└── results/                            # released evaluation tuples (R9 of protocol)
    ├── day18_human_validation/         # 50-row blind sample + labels
    │   ├── sample.jsonl                # the 50 (image_uid, gold, pred, gemini_verdict) tuples
    │   ├── sample.tsv                  # human-friendly view with HUMAN_VERDICT column
    │   ├── human_labels.json           # per-row human verdicts from the Streamlit app
    │   └── INSTRUCTIONS.md
    ├── day19_mtvqa_full_semantic/      # Qwen full-val Gemini-Flash verdicts (n=2163)
    │   ├── judgements.jsonl
    │   └── summary.md
    ├── day19_internvl_full_semantic/   # InternVL full-val (n=2182)
    │   ├── judgements.jsonl
    │   └── joint_analysis.md           # joint Qwen-vs-InternVL aggregates (n=2142)
    ├── day19_phi35vision_semantic/     # Phi-3.5-vision held-out (n=341)
    │   ├── judgements.jsonl
    │   └── summary.md
    ├── day19_phi35vision_heldout/      # raw Phi predictions (pre-judge)
    │   ├── per_pair_base.jsonl
    │   └── summary_per_language.json
    ├── day19_cross_judge_pro/          # Gemini-Pro cross-judge run (358 rows × 2 VLMs)
    │   ├── judgements.jsonl
    │   └── agreement_report.md         # 89.2% / κ=0.819, 0.827
    ├── day19_bootstrap/cis.md          # 2000-iter paired bootstrap CIs on every headline
    ├── day19_sensitivity/dropped.md    # drop / all-CORRECT / all-INCORRECT sensitivity
    ├── day19_sft_cis/perlang.md        # per-language Δstrict / Δsem-C with CIs
    └── day20_verbosity/summary.md      # Table 3 numbers (non-Latin/Latin char ratios)
```

## Reproduction

There are three reproduction modes, in order of increasing cost.

### A. Reproduce the analysis from the released tuples (no GPU, ~5 min, free)

Every aggregate cell in the paper is computed by a CPU-only script from the
released `judgements.jsonl` files in `results/`. To reproduce the headline
numbers:

```bash
pip install -e .
pip install numpy pandas matplotlib

# 2000-iter paired bootstrap on the inversion + artifact-share split:
python scripts/day19_bootstrap_cis.py
# -> writes results/day19_bootstrap/cis.md

# Per-VLM artifact share + cross-VLM differences:
python scripts/day19_phi_analysis.py

# Verbosity (non-Latin/Latin char ratio, Table 3):
python scripts/day20_verbosity_analysis.py

# Gemini-Flash vs Gemini-Pro cross-judge agreement (κ, % agree):
python scripts/day19_analysis_cross_judge.py

# Human-vs-Gemini agreement on the 50-row blind sample:
python scripts/day18_human_validation_scoring.py
```

### B. Reproduce the VLM inference + judge run (GPU, ~6–10 h, ~$50 API)

Use the Colab notebook at `notebooks/colab_runner.ipynb`. It clones this
repo into a Colab runtime, installs deps, authenticates Vertex AI (you
provide a service-account JSON via Colab Secret), runs every GPU-bound
experiment that produces a paper number end-to-end (Qwen-7B / InternVL-8B /
Phi-3.5-vision inference + Gemini-2.5-Flash judging), runs the analysis
scripts, regenerates the figures, and pushes the resulting artifacts back
to GitHub.

Required Colab Secrets:
- `REPO_ACCESS_TOKEN` — a GitHub PAT with `repo` scope on this repository
- `VERTEX_SA_JSON` — base64-encoded contents of a Vertex AI service-account
  JSON with `roles/aiplatform.user`
- `GOOGLE_CLOUD_PROJECT` — your GCP project ID

### C. Re-do the human validation pass (~2 h annotator time, free)

```bash
pip install streamlit datasets pyarrow
python scripts/day21_extract_human_validation_images.py   # ~5 min, ~3 GB HF download
streamlit run apps/human_validation_streamlit.py
# open http://localhost:8501, label the 50 rows blind to Gemini's verdict
python scripts/day18_human_validation_scoring.py          # writes concordance_report.md
python scripts/day21_patch_kappa_into_paper.py            # patches paper macros
```

## Datasets

- **MTVQA** (`ByteDance/MTVQA` on HuggingFace) — multilingual document VQA,
  9 languages, 2203-row validation split.
- **XFUND** — multilingual form understanding, 7 languages, 10k-row test
  split (we stratified-sample 200 rows per language; n=1400).

Both datasets are downloaded automatically by their respective HuggingFace
loaders; no local data preparation required.

## Models evaluated

| Model | Params | Source | Used for |
|---|---:|---|---|
| Qwen2.5-VL-7B-Instruct | 7 B | `Qwen/Qwen2.5-VL-7B-Instruct` | primary subject, §4–§7 |
| Qwen2.5-VL-3B-Instruct | 3 B | `Qwen/Qwen2.5-VL-3B-Instruct` | smaller-model sanity check |
| InternVL-2.5-8B | 8 B | `OpenGVLab/InternVL2_5-8B` | cross-architecture replicate, §5 |
| Phi-3.5-vision-instruct | 4.1 B | `microsoft/Phi-3.5-vision-instruct` | third architecture, §5 |

Judges:
- **Gemini-2.5-Flash** (`gemini-2.5-flash` via Vertex AI) — headline judge.
- **Gemini-2.5-Pro** (`gemini-2.5-pro` via Vertex AI) — cross-judge for
  Appendix B.

## License

Released under the MIT License (see `LICENSE`). Model weights are subject to
their respective licenses (Qwen, InternVL, Phi-3.5 are Apache-2 /
permissive; check each model card before commercial use).

## Citation

Will be added at publication.
