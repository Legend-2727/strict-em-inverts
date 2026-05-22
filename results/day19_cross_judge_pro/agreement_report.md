# Day-19 P0-A: Cross-judge agreement (Flash vs Pro)


## Qwen2.5-VL-7B base (n=352)

- 3-way agreement: **89.2%**
- Cohen's κ: **0.819**
- semantic-EM-C (Flash): **55.11%**  (194/352)
- semantic-EM-C (Pro):   **52.56%**  (185/352)
- absolute |Flash − Pro|: **2.56 pp**

Confusion (rows = Flash, cols = Pro):

| | CORRECT | PARTIAL | INCORRECT |
|---|---|---|---|
| CORRECT | 176 | 8 | 10 |
| PARTIAL | 6 | 49 | 7 |
| INCORRECT | 3 | 4 | 89 |

## InternVL-2.5-8B base (n=352)

- 3-way agreement: **89.2%**
- Cohen's κ: **0.827**
- semantic-EM-C (Flash): **48.86%**  (172/352)
- semantic-EM-C (Pro):   **48.01%**  (169/352)
- absolute |Flash − Pro|: **0.85 pp**

Confusion (rows = Flash, cols = Pro):

| | CORRECT | PARTIAL | INCORRECT |
|---|---|---|---|
| CORRECT | 158 | 4 | 10 |
| PARTIAL | 7 | 55 | 6 |
| INCORRECT | 4 | 7 | 101 |

## Artifact-share replication

Latin (de/fr/it/vi) vs non-Latin (ar/ja/kr/ru/th) script gap, decomposed under each judge.

| VLM | judge | n_L | n_NL | gap strict | gap sem-C | artifact share |
|---|---|---:|---:|---:|---:|---:|
| Qwen2.5-VL-7B | Gemini-Flash | 207 | 145 | +12.92pp | +4.59pp | 64.5% |
| Qwen2.5-VL-7B | Gemini-Pro | 207 | 145 | +12.92pp | +7.28pp | 43.6% |
| InternVL-2.5-8B | Gemini-Flash | 208 | 144 | +10.68pp | +9.83pp | 8.0% |
| InternVL-2.5-8B | Gemini-Pro | 208 | 144 | +10.68pp | +8.39pp | 21.5% |