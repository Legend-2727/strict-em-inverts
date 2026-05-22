# Day-19: Qwen vs InternVL full MTVQA val joint analysis (n=2142)

Joint = sample_ids with clean Gemini-Flash verdicts on both Qwen-7B and InternVL-2.5-8B base predictions.

## Aggregate (joint n=2142)

| metric | Qwen2.5-VL-7B | InternVL-2.5-8B | gap (Q-I) | 95% CI |
|---|---:|---:|---:|---:|
| strict-EM | 29.18% | 34.69% | $-5.52$ pp | $[-7.47, -3.50]$ |
| sem-EM-C  | 60.55% | 48.60% | $+11.95$ pp | $[+9.57, +14.24]$ |

**P(joint ranking inversion across 2000 paired bootstrap resamples) = 1.000.**

## Per-language (joint n=2142)

| lang | n | Qwen strict | InternVL strict | Qwen sem-C | InternVL sem-C |
|---|---:|---:|---:|---:|---:|
| ar | 176 | 12.5% | 19.9% | 40.3% | 27.3% |
| de | 406 | 32.5% | 39.7% | 57.9% | 52.0% |
| fr | 262 | 45.0% | 50.8% | 74.8% | 68.7% |
| it | 221 | 29.4% | 32.6% | 69.2% | 50.7% |
| ja | 316 | 21.2% | 31.0% | 58.9% | 54.4% |
| kr | 133 | 21.8% | 21.8% | 56.4% | 39.8% |
| ru | 172 | 14.0% | 18.0% | 58.7% | 23.3% |
| th |  57 |  7.0% | 15.8% | 28.1% | 24.6% |
| vi | 399 | 41.1% | 43.9% | 66.2% | 52.9% |

InternVL is uniformly stronger or tied on strict-EM. Qwen is uniformly stronger on sem-EM-C, in some languages by very large margins (Russian +35.4 pp, Italian +18.5 pp, Korean +16.6 pp).

## Per-script artifact share (joint n=2142, full val)

| VLM | Latin strict | non-Latin strict | strict gap | Latin sem-C | non-Latin sem-C | sem-C gap | artifact share | 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen | 37.19% | 17.10% | $+20.09$ pp | 65.84% | 52.58% | $+13.26$ pp | $+33.7\%$ | $[+13.8, +52.8]$ |
| InternVL | 42.00% | 23.65% | $+18.35$ pp | 55.43% | 38.29% | $+17.14$ pp | $+6.3\%$ | $[-12.4, +24.2]$ |

**Δ artifact share (Qwen − InternVL) = $+27.4$ pp, 95% CI $[+4.9, +51.0]$** — excludes zero. At population scale the Qwen-vs-InternVL artifact-share difference is bootstrap-significant; the held-out $n=353$ point estimate of $+59$ pp had CI $[-95, +277]$ that included zero, but the qualitative direction now confirms at $n=2142$.

## Comparison to held-out

| split | $n$ | P(inversion) | Δ artifact share Q-I | CI |
|---|---:|---:|---:|---:|
| held-out  |  353 | 0.991 | $+58.8$ pp | $[-95.1, +277.2]$ (incl. 0) |
| full val  | 2142 | 1.000 | $+27.4$ pp | $[+4.9, +51.0]$ (excl. 0) |

The held-out enriched for high-artifact-share rows, inflating Qwen's point estimate; the population-level estimate is half as large but cleanly significant.
