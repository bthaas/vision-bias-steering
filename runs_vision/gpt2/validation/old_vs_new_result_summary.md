# Old vs New Result Summary

## Old validation pipeline

- Metric: `reduction_pct` based on RMS bias over the validation set
- Evaluation type: single next-token validation
- Main result: layer `5`, coeff `-215.0`, reduction `46.20%`
- Source: `runs_vision/gpt2/validation/debiased_results.json`

### Old template diagnostics

| Template | Constrained | Reduction % |
| --- | --- | ---: |
| `scene_is` | `false` | `72.60%` |
| `scene_is` | `true` | `69.14%` |
| `image_shows` | `false` | `93.89%` |
| `image_shows` | `true` | `73.56%` |
| `in_scene_the` | `false` | `94.20%` |
| `in_scene_the` | `true` | `51.28%` |

Source: `runs_vision/gpt2/validation/signal_report.json`

## New prompt benchmark

- Metric: `gap_reduction_pct` on selected prompt curves
- Evaluation type: prompt-level comparison with greedy vs beam
- Prompt set: selected top `5` cases
- Source: `runs_vision/gpt2/validation/master_prompt_experiments_allcaps_obj5_beam_fastgrid.json`

| Method | Greedy | Beam |
| --- | ---: | ---: |
| `image_shows_one_token` | `-4387.12%` | `-4387.12%` |
| `image_shows_multi_token_mean` | `-28.36%` | `55.43%` |
| `custom_fill_in_blank` | `-57.90%` | `55.52%` |

## Key interpretation

- The old pipeline and the new prompt benchmark are not the same evaluation.
- The old pipeline was a full-validation next-token debiasing test and did produce strong positive results.
- The new prompt benchmark extends to multi-token continuation, where greedy performed poorly and beam performed better.
- Beam would not change the old next-token template diagnostics unless those diagnostics are rewritten as multi-token continuation tests.
