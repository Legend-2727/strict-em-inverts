# Day-21: Llama-3.2-Vision-11B-Instruct semantic-EM (n=353 of 355 held-out)

Fourth architecture in the artifact-share comparison (R1 closure).
Same Gemini-Flash judge, same prompt, same extract_answer + NFKC + lower + punct-strip + ws-collapse normalization as Qwen/InternVL/Phi.

## Overall EM under three metrics

| metric | Llama-3.2-Vision (11B) | Qwen base (7B, ref) | InternVL base (8B, ref) | Phi-3.5-vision (4.1B, ref) |
|---|---:|---:|---:|---:|
| Strict-EM | 11.05% | 25.70% | 34.15% | 13.20% |
| Semantic-EM (CORRECT) | 39.38% | 54.47% | 48.86% | 39.30% |
| Semantic-EM (CORRECT+PARTIAL) | 55.81% | 71.79% | 65.05% | 55.43% |

## Latin vs non-Latin gap on Llama-3.2-Vision

Latin n=206; non-Latin n=147

| metric | Latin | non-Latin | gap |
|---|---:|---:|---:|
| Strict-EM | 15.05% | 5.44% | +9.61pp |
| Semantic-EM-C | 45.15% | 31.29% | +13.85pp |
| Semantic-EM-C+P | 65.05% | 42.86% | +22.19pp |

## Artifact share (Llama-3.2-Vision)

- semantic-EM-C: **-44.2%**
- semantic-EM-C+P: **-131.0%**

## Four-VLM comparison (artifact share under semantic-EM-C, Gemini-Flash judge)

| VLM | params | strict gap | sem-C gap | artifact share |
|---|---:|---:|---:|---:|
| Qwen2.5-VL-7B       | 7B    | +12.92pp | +4.59pp  | +66% |
| InternVL-2.5-8B     | 8B    | +10.68pp | +9.83pp  |  +8% |
| Phi-3.5-vision      | 4.1B  | +12.74pp | +19.98pp | -57% |
| Llama-3.2-Vision   | 11B   | +9.61pp | +13.85pp | -44.2% |

## Per-language Llama-3.2-Vision EM

| lang | n | strict | sem-C | sem-C+P |
|---|---:|---:|---:|---:|
| ar | 35 | 8.6% | 28.6% | 31.4% |
| de | 60 | 15.0% | 43.3% | 68.3% |
| fr | 39 | 7.7% | 51.3% | 74.4% |
| it | 53 | 17.0% | 43.4% | 60.4% |
| ja | 62 | 1.6% | 37.1% | 51.6% |
| kr | 28 | 10.7% | 25.0% | 42.9% |
| ru | 19 | 5.3% | 31.6% | 42.1% |
| th | 3 | 0.0% | 0.0% | 0.0% |
| vi | 54 | 18.5% | 44.4% | 59.3% |