# Qwen Image-Shows Diagnostic Alternatives

All plots use only `The image shows`, the saved 1,000-caption validation split, layer 11, the saved steering vector/neutral offset logic, and the constrained next-token spatial-minus-descriptive disparity metric.

## Summary

| setting | mean disparity | RMS disparity | corr(projection, disparity) | notes |
|---|---:|---:|---:|---|
| baseline | -0.128 | 0.788 | 0.738 | strong baseline alignment |
| lambda=10 | -0.006 | 0.751 | 0.726 | nearly centers the mean, but only 4.7% RMS reduction |
| lambda=20 | +0.275 | 0.558 | 0.603 | clearer control effect, 29.2% RMS reduction, but mean shifts positive |

## Options

1. Option 1 baseline alignment only: `qwen_image_shows_option1_baseline_alignment.png`. This is the cleanest layer-selection diagnostic: projection and baseline disparity align strongly (`r=0.738`). It does not show intervention effects, so it should be paired in text with the coefficient-response curve.

2. Option 2 paired before/after, lambda=10: `qwen_image_shows_option2_paired_disparity_lambda10.png`. This directly compares each prompt before and after steering. It is conservative and interpretable because the mean moves from -0.128 to -0.006, but the cloud stays close to the diagonal and the RMS reduction is modest.

3. Option 2 paired before/after, lambda=20: `qwen_image_shows_option2_paired_disparity_lambda20.png`. This is the strongest single diagnostic. It directly shows per-prompt movement, includes the unchanged diagonal and zero axes, and 75.3% of prompts move closer to zero. It should be described as positive spatial steering with RMS reduction, not as perfect centering.

4. Option 3 change plot, lambda=10: `qwen_image_shows_option3_disparity_change_lambda10.png`. The change is easy to define, but visually subtle. It supports the conservative story rather than making it obvious.

5. Option 3 change plot, lambda=20: `qwen_image_shows_option3_disparity_change_lambda20.png`. It explains the mechanism well: prompts with more negative baseline disparity receive larger positive changes (`corr(base, change)=-0.797`). It is analytically useful but a little less immediately legible than the paired scatter.

6. Option 4 alignment plus distribution, lambda=10: `qwen_image_shows_option4_alignment_distribution_lambda10.png`. This combines baseline alignment with the conservative mean-centering story, but the distribution change is small.

7. Option 4 alignment plus distribution, lambda=20: `qwen_image_shows_option4_alignment_distribution_lambda20.png`. This is a good backup if the paper wants alignment and aggregate distribution in one figure. It is less prompt-specific than the paired scatter.

## Ranking

1. `qwen_image_shows_option2_paired_disparity_lambda20.png` - recommended final. It is the most reader-friendly intervention diagnostic and fixes the old two-panel ambiguity by plotting before vs. after disparity directly.
2. `qwen_image_shows_option4_alignment_distribution_lambda20.png` - best combined alignment plus aggregate shift figure.
3. `qwen_image_shows_option1_baseline_alignment.png` - best if the figure should only support layer/vector alignment and leave intervention effects to response curves.
4. `qwen_image_shows_option3_disparity_change_lambda20.png` - best mechanistic supplement, but less intuitive as the main paper figure.
5. Lambda=10 variants - useful conservative checks, but the visual effect is too subtle for the main diagnostic.

## Recommendation

Use Option 2 at `lambda=20` as the final diagnostic if the subsection is allowed to say that steering reduces RMS disparity while shifting the next-token distribution in the spatial direction. Use Option 1 only if the paper wants this figure to make a narrower claim about baseline vector alignment and leave all intervention claims to the coefficient-response curve.

## Suggested Replacement Paragraph

Figure~\ref{fig:projection-scatter} isolates the next-token diagnostic to the main local prompt condition, using only the continuation prefix ``The image shows.'' The selected layer-11 direction is first validated by its strong baseline association with the constrained spatial-minus-descriptive next-token disparity. To make the intervention effect caption-by-caption rather than template-driven, the figure plots each prompt's baseline disparity against its disparity after applying the saved layer-11 steering intervention. Points on the diagonal would be unchanged, while points in the shaded region have smaller absolute disparity after steering. At $\lambda=20$, the intervention reduces RMS disparity from 0.788 to 0.558 and moves 75.3\% of prompts closer to zero, while also shifting the mean disparity from -0.128 to +0.275. Thus the diagnostic should be read as controlled positive spatial steering with reduced next-token disparity magnitude, not as exact centering of the distribution.

## Suggested Caption

Paired next-token disparity diagnostic for the local \model{Qwen/Qwen-1\_8B-chat} run using only the continuation prefix ``The image shows.'' Each point is one held-out COCO caption from the saved 1,000-caption validation split. The x-axis shows the baseline constrained spatial-minus-descriptive next-token disparity; the y-axis shows the same quantity after applying the saved layer-11 steering intervention with $\lambda=20$. The dashed diagonal marks no change, and the zero lines separate spatial-favoring from descriptive-favoring next-token distributions. Points in the shaded regions have smaller absolute disparity after steering.

## Regeneration

```bash
python plotting/build_qwen_image_shows_diagnostic_alternatives.py
```
