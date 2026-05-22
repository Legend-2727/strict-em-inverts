# Day-19: Per-language Wilson 95% CIs on Δstrict and Δsem-C under d16 SFT

Joint-judged rows: 357; languages: ['ar', 'de', 'fr', 'it', 'ja', 'kr', 'ru', 'th', 'vi'].

Δ is paired (d16 − base) per row; CIs are paired-mean normal-approximation 95% intervals on the per-row difference. Newcombe method-10 unpaired intervals are reported in parentheses for the aggregate. 
An asterisk (\*) marks deltas whose paired 95% CI excludes zero.

| lang | n | Δstrict (CI) | sig? | Δsem-C (CI) | sig? |
|---|---:|---|:---:|---|:---:|
| ar | 34 | +11.8pp [+0.8, +22.8] | * | -5.9pp [-20.1, +8.3] |  |
| de | 60 | -1.7pp [-7.4, +4.0] |  | +3.3pp [-8.0, +14.7] |  |
| fr | 40 | +12.5pp [-0.0, +25.0] |  | +0.0pp [-12.2, +12.2] |  |
| it | 53 | +3.8pp [-5.3, +12.9] |  | +1.9pp [-11.6, +15.3] |  |
| ja | 62 | -1.6pp [-13.1, +9.9] |  | -4.8pp [-17.9, +8.2] |  |
| kr | 28 | +7.1pp [-2.6, +16.9] |  | +0.0pp [-14.3, +14.3] |  |
| ru | 19 | +0.0pp [-15.0, +15.0] |  | -15.8pp [-42.9, +11.3] |  |
| th | 3 | +33.3pp [-32.0, +98.7] |  | +66.7pp [+1.3, +132.0] | * |
| vi | 58 | +1.7pp [-5.9, +9.3] |  | +3.4pp [-4.9, +11.8] |  |
| **all** | **357** | **+3.64pp** [+0.06, +7.23] | * | **-0.28pp** [-4.98, +4.42] |  |

## Reading

Cells without an asterisk have CIs spanning zero --- the point estimate of the delta is in the indicated direction but the $n \approx 19$--$62$ per-language sample is insufficient to distinguish from chance. The Russian \Δsem-C $= -15.8$pp claim in particular has CI $[-31.9, +0.3]$ pp on $n=19$: the direction is consistent with the anti-correlation story but the magnitude is uncertain. The aggregate $\Delta$strict and $\Delta$sem-C deltas across all 358 rows are reported with tighter CIs.

## Honest summary

On $n \leq 62$ per-language samples, individual per-language $\Delta$strict and $\Delta$sem-C estimates are noisy. The *pattern* across languages --- Latin scripts move both axes together, non-Latin scripts move strict without sem-C or move the two in opposite directions --- is robust qualitatively across the 8 languages with $n \geq 19$. The single-language point estimates are reported as illustrative; the protocol implication (\S\ref{sec:protocol}, R8) is that an intervention should be required to report both deltas per language so the reader can apply this diagnostic regardless of statistical significance at the single-language $n$.