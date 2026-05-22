# Day-19 / P0-C: Full MTVQA val semantic-EM (Qwen2.5-VL-7B base, n=2163 of 2203)

Input: `results/mtvqa_perlang_diagnostic/per_pair_base.jsonl` (Qwen2.5-VL-7B base predictions on full MTVQA val 9-language n=2203). Judge: Gemini-2.5-Flash via Vertex AI, identical prompt to held-out runs. 40 rows lost to retry exhaustion (~1.8%).

## Aggregate (full val, n=2163)

| metric | value |
|---|---:|
| Strict-EM | **29.17%** |
| Semantic-EM-C | **60.56%** |
| Semantic-EM-C+P | **75.68%** |

## Per-language

| lang | n | strict | sem-C | sem-C+P |
|---|---:|---:|---:|---:|
| ar | 178 | 12.4% | 41.0% | 47.8% |
| de | 411 | 32.6% | 57.7% | 73.0% |
| fr | 267 | 45.3% | 75.3% | 88.4% |
| it | 222 | 29.3% | 69.4% | 81.5% |
| ja | 318 | 21.1% | 58.5% | 73.6% |
| kr | 134 | 21.6% | 56.7% | 83.6% |
| ru | 173 | 13.9% | 58.4% | 71.1% |
| th | 58 | 6.9%  | 27.6% | 62.1% |
| vi | 402 | 41.0% | 66.2% | 82.1% |

## Latin / non-Latin (full val)

| script | n | strict | sem-C |
|---|---:|---:|---:|
| Latin (de/fr/it/vi)       | 1302 | 37.25% | 65.90% |
| non-Latin (ar/ja/kr/ru/th)|  861 | 16.96% | 52.50% |
| **gap**                   |      | **+20.29pp** | **+13.40pp** |
| artifact share            |      |        | **34.0%** |

## Comparison to 358-row held-out

| split | n | strict | sem-C | gap-strict | gap-sem-C | artifact share |
|---|---:|---:|---:|---:|---:|---:|
| full val           | 2163 | 29.17% | 60.56% | +20.29pp | +13.40pp | **34.0%** |
| held-out (paper §5)| 351  | 24.5%  | 55.1%  | +12.92pp | +4.59pp  | **64.5%** |

The held-out sample was constructed as the DPO-pair eligible eval set (rows where Qwen produces real failures of specific types); this selection enriches for the high-artifact-share population. At the population level (full val) the artifact share is 34%, still substantial but smaller than the 65% on the held-out. Both numbers are non-zero and significant; the held-out is where the rank-inversion analysis is performed because it is the locked, replicated set across all three VLMs.
