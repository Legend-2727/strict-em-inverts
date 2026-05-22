# Day-19: Bootstrap 95% CIs on headline paper numbers

Bootstrap iterations: 2000; resample seed: 42; percentile method; CIs are reported as `point [lo, hi]`.

## (a) Cross-VLM head-to-head, joint held-out

n = 353 sample_ids common to Qwen and InternVL judged sets.

| metric | Qwen | InternVL | gap (Qwen − InternVL) |
|---|---|---|---|
| strict-EM | 24.36% [19.83, 28.90] | 33.99% [29.18, 39.09] | -9.63pp [-14.16, -5.38] |
| sem-EM-C | 55.24% [50.14, 60.62] | 48.44% [43.34, 54.11] | +6.80pp [+1.13, +12.18] |

## (b) Per-VLM Latin/non-Latin artifact share, with CIs

| VLM | n | strict gap (pp) | sem-C gap (pp) | artifact share (%) |
|---|---:|---|---|---|
| Qwen2.5-VL-7B | 357 | +12.95 [+4.14, +21.76] | +4.34 [-6.02, +14.75] | +66.4 [-20.0, +182.8] |
| InternVL-2.5-8B | 354 | +10.69 [+0.71, +20.61] | +9.87 [-0.08, +20.56] | +7.7 [-158.9, +112.4] |
| Phi-3.5-vision | 341 | +12.69 [+6.17, +19.10] | +19.98 [+11.30, +28.51] | -57.4 [-156.5, -4.8] |

## (c) Pairwise differences of artifact share (point + 95% CI)

Each VLM is an independent sample; differences are computed by resampling each rowset independently.

| pair | Δ artifact share (pp) |
|---|---|
| Qwen − InternVL | +58.8 [-95.1, +277.2] |
| Qwen − Phi | +123.8 [+33.0, +279.0] |
| InternVL − Phi | +65.0 [-114.7, +232.7] |

## (d) Ranking-inversion significance test

We test whether (qwen sem-C > intvl sem-C) AND (qwen strict < intvl strict) holds in each bootstrap resample of the joint 351-row set. The fraction of resamples where the joint inversion holds is reported.

- P(Qwen sem-C > InternVL sem-C) = **0.991**
- P(InternVL strict > Qwen strict) = **1.000**
- P(joint ranking inversion) = **0.991**
