# Day-21: Bootstrap 95% CIs, FOUR-VLM artifact-share comparison (R1 closure)

Bootstrap iterations: 2000; resample seed: 42; percentile method.
Llama-3.2-Vision is the 4th architecture, joining Qwen, InternVL, Phi.
All four VLMs judged by Gemini-2.5-Flash with identical prompt.

## Per-VLM Latin/non-Latin artifact share, with 95% CIs

| VLM | n | strict gap (pp) | sem-C gap (pp) | artifact share (%) |
|---|---:|---|---|---|
| Qwen2.5-VL-7B | 357 | +12.95 [+4.14, +21.76] | +4.34 [-6.02, +14.75] | +66.4 [-20.0, +182.8] |
| InternVL-2.5-8B | 354 | +10.69 [+0.71, +20.61] | +9.87 [-0.08, +20.56] | +7.7 [-158.9, +112.4] |
| Phi-3.5-vision | 341 | +12.69 [+6.17, +19.10] | +19.98 [+11.30, +28.51] | -57.4 [-156.5, -4.8] |
| Llama-3.2-Vision-11B | 353 | +9.61 [+3.16, +15.58] | +13.85 [+3.26, +23.86] | -44.2 [-247.0, +62.0] |

## Pairwise differences of artifact share (point + 95% CI)

Each VLM is an independent sample; differences are computed by resampling each rowset independently.

| pair | Δ artifact share (pp) |
|---|---|
| Qwen − InternVL | +58.8 [-95.1, +277.2] |
| Qwen − Phi | +123.8 [+33.0, +279.0] |
| Qwen − Llama | +110.7 [-20.2, +340.1] |
| InternVL − Phi | +65.0 [-114.7, +232.7] |
| InternVL − Llama | +51.9 [-115.1, +322.1] |
| Phi − Llama | -13.2 [-168.4, +189.4] |

## 4-VLM artifact-share spread

Range across 4 architectures: **123.8 pp** (from -57.4% to +66.4%).