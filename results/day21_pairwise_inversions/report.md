# Pairwise ranking-inversion test (R3 closure)

For each of the 6 VLM pairs from {Qwen-7B, InternVL-8B, Phi-3.5-vision, Llama-3.2-V-11B}, we test whether strict-EM and semantic-EM-C disagree on the direction of the head-to-head gap. Each pair uses the joint set of sample_ids judged by both VLMs.

Paired-bootstrap p(inversion) is the fraction of 2000 resamples on which $(\text{strict-EM}_A > \text{strict-EM}_B)$ XOR $(\text{sem-C}_A > \text{sem-C}_B)$.

| pair (A vs B) | n_joint | A strict | B strict | gap (A-B) strict | A sem-C | B sem-C | gap (A-B) sem-C | inversion? | P(inv) |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|---:|
| Qwen-7B vs InternVL-8B | 353 | 24.36% | 33.99% | -9.63 [-13.88, -5.67] | 55.24% | 48.44% | +6.80 [+0.85, +12.18] | **YES** | **0.987** |
| Qwen-7B vs Phi-3.5 | 341 | 23.46% | 13.20% | +10.26 [+6.16, +14.37] | 53.96% | 24.63% | +29.33 [+22.87, +35.48] | no | **0.000** |
| Qwen-7B vs Llama-3.2-V | 352 | 24.15% | 11.08% | +13.07 [+9.09, +17.05] | 54.83% | 39.49% | +15.34 [+10.23, +20.74] | no | **0.000** |
| InternVL-8B vs Phi-3.5 | 337 | 33.83% | 13.35% | +20.47 [+16.02, +25.22] | 48.07% | 24.93% | +23.15 [+17.51, +28.49] | no | **0.000** |
| InternVL-8B vs Llama-3.2-V | 349 | 34.10% | 11.17% | +22.92 [+18.34, +27.51] | 48.42% | 39.83% | +8.60 [+3.44, +13.75] | no | **0.001** |
| Phi-3.5 vs Llama-3.2-V | 336 | 13.10% | 10.71% | +2.38 [-1.49, +5.95] | 24.40% | 38.99% | -14.58 [-19.94, -9.23] | **YES** | **0.887** |

## Summary

- **2/6 pairs invert** at the point estimate (strict and sem-C disagree on direction).

This closes the reviewer concern that 'Strict-EM Inverts Cross-Lingual VLM Rankings' only holds for one cherry-picked pair: the inversion pattern is broader across the open-VLM landscape, not a Qwen-vs-InternVL singleton.