# Day-19: Sensitivity to judge-dropped rows

Each split is judged with three coercion policies for rows that exhausted retries on the Gemini-Flash judge (HTTP 429 / RESOURCE\_EXHAUSTED): (drop) baseline, (all CORRECT) most-generous upper bound, (all INCORRECT) least-generous lower bound. We report artifact share under each policy.

## Qwen held-out

Base predictions n=358; judged n=357; dropped n=1.

| coercion | n | strict gap | sem-C gap | artifact share |
|---|---:|---:|---:|---:|
| drop (baseline) | 357 | +12.95pp | +4.34pp | +66.4% |
| all CORRECT | 358 | +13.06pp | +4.02pp | +69.2% |
| all INCORRECT | 358 | +13.06pp | +4.70pp | +64.0% |

## InternVL held-out

Base predictions n=358; judged n=354; dropped n=4.

| coercion | n | strict gap | sem-C gap | artifact share |
|---|---:|---:|---:|---:|
| drop (baseline) | 354 | +10.69pp | +9.87pp | +7.7% |
| all CORRECT | 358 | +10.70pp | +9.54pp | +10.8% |
| all INCORRECT | 358 | +10.70pp | +9.96pp | +7.0% |

## Phi held-out

Base predictions n=358; judged n=341; dropped n=17.

| coercion | n | strict gap | sem-C gap | artifact share |
|---|---:|---:|---:|---:|
| drop (baseline) | 341 | +12.69pp | +19.98pp | -57.4% |
| all CORRECT | 358 | +12.83pp | +19.01pp | -48.1% |
| all INCORRECT | 358 | +12.83pp | +19.03pp | -48.3% |

## Qwen full-val

Base predictions n=2203; judged n=2163; dropped n=40.

| coercion | n | strict gap | sem-C gap | artifact share |
|---|---:|---:|---:|---:|
| drop (baseline) | 2163 | +20.29pp | +13.40pp | +34.0% |
| all CORRECT | 2203 | +20.70pp | +12.06pp | +41.7% |
| all INCORRECT | 2203 | +20.70pp | +14.67pp | +29.1% |

## Ranking-inversion sensitivity (Qwen vs InternVL held-out)

Tests whether (Qwen sem-C > InternVL sem-C) AND (InternVL strict > Qwen strict) holds under each coercion of dropped rows. The joint sample_id set is recomputed per coercion.

| coercion | Qwen sem-C | InternVL sem-C | Qwen strict | InternVL strict | inversion? |
|---|---:|---:|---:|---:|---:|
| drop (n=353) | 55.24% | 48.44% | 24.36% | 33.99% | YES |
| all CORRECT (n=358) | 54.75% | 49.16% | 24.02% | 33.52% | YES |
| all INCORRECT (n=358) | 54.47% | 48.04% | 24.02% | 33.52% | YES |

## Summary

The headline claims survive both extreme coercions: the ranking inversion holds under drop, all-CORRECT, and all-INCORRECT policies; the per-VLM artifact-share signs are stable across policies. MNAR concern from judge-dropped rows is not material.