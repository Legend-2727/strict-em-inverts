# Claude-Sonnet-4 vs Gemini-2.5-Flash cross-family judge agreement

On the 671 matched (sample_id, vlm_model) tuples shared between the existing Gemini-Flash held-out judging and the new Claude-Sonnet-4 run (`day21_cross_judge_claude/`). This is the **genuine cross-family** check (R4 in the paper's protocol).

## Headline

- **3-way agreement**: **83.0%** (557/671)

- **Cohen's κ**: **0.723**

- **Binary CORRECT-vs-not**: 87.5% (587/671)


Compare to the intra-family Gemini-Flash vs Gemini-Pro number: $89.2\%$ / $\kappa \approx 0.82$. A cross-family kappa lower than the intra-family one is expected (different model families have different decision boundaries on PARTIAL); the headline is whether agreement remains in the 'substantial' band ($\kappa \geq 0.6$) under Landis & Koch (1977).

## 3×3 confusion (rows = Flash, cols = Claude)

| | CORRECT | PARTIAL | INCORRECT |
|---|---:|---:|---:|
| Flash-C | 304 | 22 | 32 |
| Flash-P | 9 | 105 | 10 |
| Flash-I | 21 | 20 | 148 |

## Per-VLM

| VLM | n | 3-way agreement | κ |
|---|---:|---:|---:|
| internvl25_8b | 331 | 85.8% | 0.772 |
| qwen25vl_7b | 340 | 80.3% | 0.675 |

## Per-script

| script | n | 3-way agreement | κ |
|---|---:|---:|---:|
| Latin | 397 | 86.1% | 0.763 |
| non-Latin | 274 | 78.5% | 0.662 |

## Per-language

| lang | n | 3-way agreement | κ |
|---|---:|---:|---:|
| ar | 63 | 77.8% | 0.609 |
| de | 115 | 88.7% | 0.822 |
| fr | 78 | 85.9% | 0.729 |
| it | 104 | 78.8% | 0.629 |
| ja | 112 | 78.6% | 0.642 |
| kr | 56 | 78.6% | 0.671 |
| ru | 38 | 76.3% | 0.589 |
| th | 5 | 100.0% | 1.000 |
| vi | 100 | 91.0% | 0.837 |