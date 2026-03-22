# Qwen-1.8B Natural Template Search — v2

Model: `Qwen/Qwen-1_8B-chat`  Layer: 11  Method: default (orthogonal projection + constant)

## All Templates: Ranking

**Primary sort**: templates with BOTH spatial AND descriptive tokens in top-10 first.
**Secondary sort**: lower baseline RMS (more balanced).

| Rank | Approach | Template | Prefix | rms@λ=0 | mean_sp | mean_desc | sp_top10 | desc_top10 |
|---|---|---|---|---|---|---|---|---|
| 1 | A2 | `A2_in_the` | "[caption]. In the" | 0.9958 | 0.9979 | 0.0021 | 6 | 0 |
| 2 | A | `A_in_the` | "In the" | 0.8885 | 0.9054 | 0.0946 | 4 | 0 |
| 3 | B | `B_positioned` | "Positioned" | 0.9670 | 0.9831 | 0.0169 | 4 | 0 |
| 4 | A2 | `A2_main_subject` | "[caption]. The main subject appears" | 0.6094 | 0.7598 | 0.2402 | 1 | 0 |
| 5 | A | `A_subject_looks` | "The subject looks" | 0.6377 | 0.6914 | 0.3086 | 1 | 0 |
| 6 | B | `B_visually` | "Visually," | 0.3671 | 0.4409 | 0.5591 | 0 | 0 |
| 7 | B | `B_looking_image` | "Looking at the image," | 0.3982 | 0.4774 | 0.5226 | 0 | 0 |
| 8 | A2 | `A2_looking_more` | "[caption]. Looking more closely," | 0.4343 | 0.4946 | 0.5054 | 0 | 0 |
| 9 | A | `A_notable_detail` | "The most notable detail is" | 0.5813 | 0.4191 | 0.5809 | 0 | 0 |
| 10 | A | `A_main_subject` | "The main subject appears" | 0.5944 | 0.6667 | 0.3333 | 0 | 0 |
| 11 | B | `B_subject_is` | "The subject is" | 0.6455 | 0.3935 | 0.6065 | 0 | 0 |
| 12 | A | `A_looking_more` | "Looking more closely," | 0.7081 | 0.8440 | 0.1560 | 0 | 0 |
| 13 | B | `B_scene_depicts` | "The scene depicts" | 0.7183 | 0.3769 | 0.6231 | 0 | 0 |
| 14 | A | `A_image_shows` | "The image shows" | 0.8021 | 0.4227 | 0.5773 | 0 | 0 |
| 15 | B | `B_foreground` | "In the foreground" | 0.8341 | 0.9150 | 0.0850 | 0 | 0 |

## Top-10 Tokens at λ=0

### `A2_in_the` (A2) — caption in assistant voice then spatial/foreground
Top-10: `distance` (0.675) | `background` (0.150) | `foreground` (0.070) | `center` (0.041) | `middle` (0.011) | `corner` (0.011) | `far` (0.008) | `shadows` (0.002) | `left` (0.002) | `front` (0.001)
  🗺 **Spatial hits**: `background` (0.150), `foreground` (0.070), `center` (0.041), `middle` (0.011), `far` (0.008), `left` (0.002)

### `A_in_the` (A) — 'foreground'/'center'/'background' — foreground IS a spatial token
Top-10: `image` (0.755) | `distance` (0.063) | `middle` (0.031) | `scene` (0.028) | `center` (0.017) | `given` (0.016) | `picture` (0.014) | `background` (0.006) | `description` (0.006) | `foreground` (0.005)
  🗺 **Spatial hits**: `middle` (0.031), `center` (0.017), `background` (0.006), `foreground` (0.005)

### `B_positioned` (B) — spatial-leaning: on/beside/above/near as first token
Top-10: `in` (0.513) | `on` (0.205) | `against` (0.099) | `at` (0.088) | `near` (0.010) | `between` (0.008) | `within` (0.005) | `along` (0.005) | `next` (0.005) | `towards` (0.004)
  🗺 **Spatial hits**: `near` (0.010), `between` (0.008), `within` (0.005), `along` (0.005)

### `A2_main_subject` (A2) — caption in assistant voice then appearance descriptor
Top-10: `to` (0.956) | `in` (0.020) | `as` (0.005) | `at` (0.003) | `on` (0.002) | `within` (0.001) | `a` (0.000) | `,` (0.000) | `surrounded` (0.000) | `slightly` (0.000)
  🗺 **Spatial hits**: `within` (0.001)

### `A_subject_looks` (A) — 'looks [adj]': bright/large/old/wooden…
Top-10: `out` (0.206) | `like` (0.162) | `at` (0.153) | `to` (0.068) | `up` (0.046) | `as` (0.018) | `peaceful` (0.018) | `over` (0.016) | `focused` (0.016) | `serene` (0.015)
  🗺 **Spatial hits**: `up` (0.046)

### `B_visually` (B) — signals descriptive information; adj likely next
Top-10: `the` (0.496) | `this` (0.483) | `I` (0.013) | `it` (0.003) | `there` (0.003) | `an` (0.001) | `we` (0.000) | `a` (0.000) | `what` (0.000) | `you` (0.000)

### `B_looking_image` (B) — meta-observer; model describes what it sees
Top-10: `it` (0.677) | `I` (0.197) | `the` (0.037) | `one` (0.036) | `there` (0.030) | `we` (0.008) | `you` (0.007) | `a` (0.003) | `what` (0.002) | `my` (0.001)

### `A2_looking_more` (A2) — caption in assistant voice then zoom-in
Top-10: `you` (0.343) | `the` (0.212) | `there` (0.126) | `one` (0.105) | `it` (0.065) | `I` (0.054) | `a` (0.047) | `we` (0.022) | `some` (0.003) | `two` (0.001)

### `A_notable_detail` (A) — noun/adj phrase follows; varies by scene
Top-10: `the` (0.967) | `that` (0.016) | `likely` (0.011) | `a` (0.002) | `probably` (0.001) | `clearly` (0.001) | `undoubtedly` (0.000) | `perhaps` (0.000) | `in` (0.000) | `most` (0.000)

### `A_main_subject` (A) — predicate adj follows 'appears': large/bright/old/red…
Top-10: `to` (0.999) | `as` (0.000) | `in` (0.000) | `be` (0.000) | `at` (0.000) | `a` (0.000) | `to` (0.000) | `here` (0.000) | `primarily` (0.000) | `clearly` (0.000)

### `B_subject_is` (B) — predicate: adj or preposition follows 'is'
Top-10: `a` (0.956) | `an` (0.025) | `likely` (0.007) | `the` (0.002) | `described` (0.002) | `depicted` (0.002) | `of` (0.001) | `clearly` (0.001) | `most` (0.001) | `in` (0.001)

### `A_looking_more` (A) — natural zoom-in; first token could be spatial/descriptive adj
Top-10: `the` (0.494) | `there` (0.140) | `this` (0.108) | `it` (0.103) | `I` (0.096) | `we` (0.025) | `you` (0.013) | `one` (0.008) | `what` (0.003) | `here` (0.002)

### `B_scene_depicts` (B) — neutral; model chooses spatial or descriptive opening
Top-10: `a` (0.988) | `an` (0.011) | `the` (0.000) | `someone` (0.000) | `two` (0.000) | `something` (0.000) | `what` (0.000) | `imagery` (0.000) | `images` (0.000) | `several` (0.000)

### `A_image_shows` (A) — baseline — caption in instruction, standard prefix
Top-10: `a` (0.996) | `an` (0.004) | `the` (0.000) | `someone` (0.000) | `two` (0.000) | `what` (0.000) | `that` (0.000) | `three` (0.000) | `one` (0.000) | `how` (0.000)

### `B_foreground` (B) — 'foreground' is a spatial token; continuation places scene element
Top-10: `of` (0.607) | `,` (0.359) | `is` (0.026) | `are` (0.002) | `there` (0.001) | `stands` (0.001) | `you` (0.001) | `we` (0.001) | `lies` (0.001) | `sits` (0.000)

## RMS Bias Reduction — Top-5 Templates

Formula: `RMS = sqrt(mean((spatial_prob − descriptive_prob)²))`
Constrained softmax · 100 val examples · Layer 11 · method=default

| Rank | Template | Prefix | rms@λ=0 | best_rms | best_coeff | reduction% |
|---|---|---|---|---|---|---|
| 1 | `A2_in_the` | "[caption]. In the" | 0.9958 | 0.2422 | +100 | 75.7% |
| 2 | `A_in_the` | "In the" | 0.8885 | 0.0903 | +100 | 89.8% |
| 3 | `B_positioned` | "Positioned" | 0.9670 | 0.0563 | +100 | 94.2% |
| 4 | `A2_main_subject` | "[caption]. The main subject appears" | 0.6094 | 0.2285 | -150 | 62.5% |
| 5 | `A_subject_looks` | "The subject looks" | 0.6377 | 0.0575 | -150 | 91.0% |

### `A2_in_the` sweep
| coeff | rms | reduction% |
|---|---|---|
| -150 | 0.4324 | 56.6% |
| -100 | 0.5784 | 41.9% |
| -50 | 0.5973 | 40.0% |
| +0 | 0.9955 | 0.0% |
| +50 | 0.5493 | 44.8% |
| +100 | 0.2422 | 75.7% |
| +150 | 0.3404 | 65.8% |

### `A_in_the` sweep
| coeff | rms | reduction% |
|---|---|---|
| -150 | 0.3508 | 60.5% |
| -100 | 0.4450 | 49.9% |
| -50 | 0.5756 | 35.2% |
| +0 | 0.8821 | 0.7% |
| +50 | 0.7590 | 14.6% |
| +100 | 0.0903 | 89.8% |
| +150 | 0.3002 | 66.2% |

### `B_positioned` sweep
| coeff | rms | reduction% |
|---|---|---|
| -150 | 0.0942 | 90.3% |
| -100 | 0.1217 | 87.4% |
| -50 | 0.3167 | 67.2% |
| +0 | 0.9625 | 0.5% |
| +50 | 0.9283 | 4.0% |
| +100 | 0.0563 | 94.2% |
| +150 | 0.2309 | 76.1% |

### `A2_main_subject` sweep
| coeff | rms | reduction% |
|---|---|---|
| -150 | 0.2285 | 62.5% |
| -100 | 0.3127 | 48.7% |
| -50 | 0.5257 | 13.7% |
| +0 | 0.6216 | -2.0% |
| +50 | 0.8388 | -37.6% |
| +100 | 0.2825 | 53.6% |
| +150 | 0.4369 | 28.3% |

### `A_subject_looks` sweep
| coeff | rms | reduction% |
|---|---|---|
| -150 | 0.0575 | 91.0% |
| -100 | 0.0669 | 89.5% |
| -50 | 0.3066 | 51.9% |
| +0 | 0.6478 | -1.6% |
| +50 | 0.8488 | -33.1% |
| +100 | 0.0673 | 89.4% |
| +150 | 0.2429 | 61.9% |
