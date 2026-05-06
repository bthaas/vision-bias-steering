# Qwen Image-Shows Projection Scatter

## Data

- Model: `Qwen/Qwen-1_8B-chat`
- Validation data: `runs_vision/Qwen-1_8B-chat/datasplits/val.json`
- Captions: 1,000
- Prompt serialization: Qwen chat template with `Describe this image:\n[caption]` and continuation prefix `The image shows`.
- Steering layer: 11
- Intervention: `default` projection replacement with the saved neutral offset.
- Next-token disparity: constrained softmax over spatial + descriptive target tokens, then spatial probability minus descriptive probability.
- Final plot lambda: `20`.
- X-axis: baseline scalar projection for both panels. This keeps each caption at the same horizontal coordinate; using steered projection would collapse the steered panel around the chosen lambda.

## Summary Stats

| quantity | baseline | steered |
|---|---:|---:|
| n | 1000 | 1000 |
| projection range | [1.789, 3.504] | same x-axis |
| disparity range | [-1.000, 0.999] | [-0.970, 0.971] |
| corr(projection, disparity) | 0.738 | 0.603 |
| mean disparity | -0.128 | +0.275 |
| RMS disparity | 0.788 | 0.558 |

The single-prefix projection distribution removes the earlier prompt-family bands by construction. Within `The image shows`, the projection span is continuous enough for a caption-level diagnostic; the largest adjacent projection gap is 0.163, and no second prompt-template cluster is present. For comparison, the older three-prefix diagnostic had separated projection means `The image shows`=2.740, `The scene depicts`=1.894, `Positioned`=1.733.

## Coefficient Choice

`lambda=20` is the final choice because it is the strongest low-degeneration setting from the full-validation `A_image_shows` local sweep and gives a visibly interpretable next-token shift here: mean disparity changes by +0.403 and RMS falls by 0.230.
`lambda=10` is a gentler alternative that nearly centers the mean next-token disparity (-0.006) but produces a smaller RMS reduction (0.037), so it is less visually diagnostic for the paper figure.

## Figure Ranking

1. `qwen_image_shows_projection_scatter.png` - selected final. The two-panel layout keeps the baseline and steered distributions directly comparable, preserves individual captions, and adds linear plus binned trends without hiding saturation near +/-1.
2. `qwen_image_shows_projection_scatter_overlay.png` - useful for seeing before/after movement in one frame, but the overlaid clouds are harder to read in print.
3. `qwen_image_shows_projection_scatter_hexbin.png` - best for density, but less literal as a scatter diagnostic and slightly less transparent for readers.

## Suggested Paper Text

Figure~\ref{fig:projection-scatter} isolates the layer-selection diagnostic to the single continuation used in the main local run. For each of the 1,000 held-out COCO captions, we serialize the prompt as `Describe this image:` followed by the caption and the continuation prefix `The image shows`. The left panel plots the baseline layer-11 scalar projection against the constrained next-token spatial--descriptive disparity. The strong positive association shows that the selected steering direction is aligned with the model's own next-token spatial signal under the baseline prompt condition, not with a mixture of prompt-template offsets. The right panel keeps the same baseline projection on the x-axis and recomputes the next-token disparity after applying the saved layer-11 steering intervention at $\lambda=20$, the strongest low-degeneration setting from the full-validation local sweep. The shifted disparity distribution shows the expected movement induced by the steering vector while preserving a caption-by-caption diagnostic view.

## Suggested Caption

Projection--disparity diagnostic for the local \model{Qwen/Qwen-1\_8B-chat} run using only the continuation prefix ``The image shows.'' Each point is one held-out COCO caption from the saved 1,000-caption validation split. Both panels use the baseline layer-11 scalar projection as the x-axis; the y-axis is the constrained next-token spatial-minus-descriptive disparity over the tracked target tokens. Left: baseline prompts. Right: the same prompts after applying the saved layer-11 steering intervention with $\lambda=20$. Linear fits and binned means summarize the caption-level trend.

## Regeneration

```bash
python plotting/build_qwen_image_shows_projection_scatter.py --lambdas 0 10 20 --final-lambda 20
```

## Outputs

- `paper/figures/qwen_image_shows_projection_scatter.png`
- `paper/figures/qwen_image_shows_projection_scatter_overlay.png`
- `paper/figures/qwen_image_shows_projection_scatter_hexbin.png`
- `paper/figures/qwen_image_shows_projection_scatter.csv`
- `paper/figures/qwen_image_shows_projection_scatter_stats.json`
