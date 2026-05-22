# Day-19 / P0-B: Phi-3.5-vision-instruct semantic-EM (n=341 of 358 held-out)

Verdict coercion: 5 rows judged "INCERRECT" + 1 "INCUMPLIDO" + 1 "?" treated as INCORRECT.

## Overall EM under three metrics

| metric | Phi-3.5-vision (4.1B) | Qwen base (7B, ref) | InternVL base (8B, ref) |
|---|---:|---:|---:|
| Strict-EM | 13.20% | 25.70% | 34.15% |
| Semantic-EM (CORRECT) | 24.63% | 54.47% | 48.86% |
| Semantic-EM (CORRECT+PARTIAL) | 40.76% | 71.79% | 65.05% |

## Latin vs non-Latin gap on Phi-3.5-vision

Latin n=201; non-Latin n=140

| metric | Latin | non-Latin | gap |
|---|---:|---:|---:|
| Strict-EM | 18.41% | 5.71% | +12.69pp |
| Semantic-EM-C | 32.84% | 12.86% | +19.98pp |
| Semantic-EM-C+P | 53.23% | 22.86% | +30.38pp |

## Artifact share (Phi-3.5-vision)

- semantic-EM-C: **-57.4%**
- semantic-EM-C+P: **-139.3%**

## Three-VLM comparison (artifact share under semantic-EM-C, Gemini-Flash judge)

| VLM | params | strict gap | sem-C gap | artifact share |
|---|---:|---:|---:|---:|
| Qwen2.5-VL-7B   | 7B   | +12.92pp | +4.59pp  | 64.5% |
| InternVL-2.5-8B | 8B   | +10.68pp | +9.83pp  |  8.0% |
| Phi-3.5-vision  | 4.1B | +12.69pp | +19.98pp | -57.4% |

## Per-language Phi-3.5-vision EM

| lang | n | strict | sem-C | sem-C+P |
|---|---:|---:|---:|---:|
| ar | 33 | 3.0% | 12.1% | 21.2% |
| de | 53 | 18.9% | 35.8% | 60.4% |
| fr | 40 | 25.0% | 37.5% | 57.5% |
| it | 53 | 22.6% | 43.4% | 60.4% |
| ja | 58 | 3.4% | 10.3% | 27.6% |
| kr | 27 | 11.1% | 18.5% | 18.5% |
| ru | 19 | 5.3% | 10.5% | 15.8% |
| th | 3 | 33.3% | 33.3% | 33.3% |
| vi | 55 | 9.1% | 16.4% | 36.4% |