# Verbosity analysis: mean output char count per VLM x script
3-VLM joint held-out: n=341
2-VLM joint full-val: n=2142

## 3-VLM joint held-out

| VLM | Latin n | Latin mean | non-Latin n | non-Latin mean | ratio |
|-----|--------:|-----------:|------------:|---------------:|------:|
| Qwen2.5-VL-7B | 201 | 30.9 | 140 | 27.8 | 0.90 |
| InternVL-2.5-8B | 201 | 26.6 | 140 | 18.0 | 0.68 |
| Phi-3.5-vision | 201 | 28.1 | 140 | 17.4 | 0.62 |

## 2-VLM joint full MTVQA val

| VLM | Latin n | Latin mean | non-Latin n | non-Latin mean | ratio |
|-----|--------:|-----------:|------------:|---------------:|------:|
| Qwen2.5-VL-7B | 1288 | 29.7 | 854 | 26.1 | 0.88 |
| InternVL-2.5-8B | 1288 | 26.2 | 854 | 17.3 | 0.66 |
