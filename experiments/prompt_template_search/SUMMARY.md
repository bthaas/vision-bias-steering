# Prompt Template Search: Summary

## Ranking (by baseline balance — lower RMS = more balanced at λ=0)

| Rank | Template | Output Prefix | baseline_rms | mean_spatial | mean_descriptive | spatial_in_top10 | desc_in_top10 |
|---|---|---|---|---|---|---|---|
| 1 | `image_shows` | "The image shows" | 0.3070 | 0.4930 | 0.5070 | 0 | 0 |
| 2 | `it_stands` | "It stands" | 0.3991 | 0.5109 | 0.4891 | 0 | 0 |
| 3 | `it_appears` | "It appears" | 0.4558 | 0.3017 | 0.6983 | 0 | 0 |
| 4 | `the_item_is` | "The item is" | 0.5249 | 0.2647 | 0.7353 | 0 | 0 |
| 5 | `it_lies` | "It lies" | 0.5319 | 0.7426 | 0.2574 | 1 | 1 |
| 6 | `it_rests` | "It rests" | 0.5526 | 0.7019 | 0.2981 | 1 | 1 |
| 7 | `it_looks` | "It looks" | 0.6132 | 0.2411 | 0.7589 | 0 | 0 |
| 8 | `it_sits` | "It sits" | 0.6246 | 0.7595 | 0.2405 | 1 | 1 |
| 9 | `this_object_is` | "This object is" | 0.6434 | 0.1995 | 0.8005 | 0 | 0 |
| 10 | `the_object_is` | "The object is" | 0.6623 | 0.1790 | 0.8210 | 0 | 0 |
| 11 | `it_is` | "It is" | 0.7510 | 0.1328 | 0.8672 | 0 | 0 |

## Top-10 Tokens at λ=0 (unsteered, averaged over all captions)

### `image_shows` — "The image shows"
*current baseline — predicts 'a' first*

Top-10: `the` (0.127) | `shows` (0.093) | `show` (0.075) | `a` (0.072) | `,` (0.038) | `.` (0.033) | `that` (0.023) | `how` (0.023) | `shown` (0.020) | `showing` (0.020)

### `it_stands` — "It stands"
*stand → spatial: on/beside/near/in front of*

Top-10: `on` (0.125) | `,` (0.075) | `stand` (0.060) | `.` (0.058) | `in` (0.043) | `at` (0.041) | `it` (0.027) | `as` (0.022) | `its` (0.019) | `and` (0.019)

### `it_appears` — "It appears"
*appear → adj: bright/large/old OR spatial: near*

Top-10: `as` (0.226) | `,` (0.063) | `.` (0.054) | `to` (0.036) | `in` (0.035) | `that` (0.033) | `it` (0.025) | `very` (0.022) | `from` (0.020) | `on` (0.019)

### `the_item_is` — "The item is"
*another object reference*

Top-10: `is` (0.075) | `a` (0.058) | `not` (0.057) | `the` (0.056) | `in` (0.041) | `item` (0.039) | `.` (0.024) | `,` (0.022) | `made` (0.019) | `an` (0.017)

### `it_lies` — "It lies"
*lie → spatial: on/beside/along/beneath*

Top-10: `on` (0.172) | `in` (0.094) | `lying` (0.062) | `beside` (0.035) | `.` (0.026) | `,` (0.025) | `flat` (0.022) | `about` (0.021) | `at` (0.019) | `lay` (0.019)
  - **Spatial hits**: `beside`
  - **Descriptive hits**: `flat`

### `it_rests` — "It rests"
*rest → spatial prepositions: on/against/beside*

Top-10: `on` (0.243) | `resting` (0.062) | `its` (0.051) | `,` (0.047) | `in` (0.043) | `.` (0.033) | `beside` (0.026) | `at` (0.025) | `flat` (0.019) | `it` (0.018)
  - **Spatial hits**: `beside`
  - **Descriptive hits**: `flat`

### `it_looks` — "It looks"
*look → adj: bright/large OR preposition: like*

Top-10: `like` (0.261) | `as` (0.067) | `very` (0.057) | `,` (0.038) | `.` (0.022) | `about` (0.019) | `look` (0.016) | `a` (0.015) | `and` (0.013) | `its` (0.013)

### `it_sits` — "It sits"
*sit → spatial prepositions: on/beside/near*

Top-10: `on` (0.388) | `in` (0.056) | `at` (0.052) | `,` (0.031) | `beside` (0.030) | `sitting` (0.026) | `.` (0.021) | `sit` (0.014) | `flat` (0.010) | `with` (0.010)
  - **Spatial hits**: `beside`
  - **Descriptive hits**: `flat`

### `this_object_is` — "This object is"
*similar to above, different determiner*

Top-10: `a` (0.080) | `not` (0.077) | `the` (0.063) | `is` (0.050) | `object` (0.028) | `in` (0.026) | `very` (0.020) | `an` (0.018) | `this` (0.016) | `that` (0.015)

### `the_object_is` — "The object is"
*direct predicate: adj or preposition next*

Top-10: `is` (0.223) | `a` (0.106) | `the` (0.060) | `are` (0.031) | `.` (0.026) | `not` (0.026) | `was` (0.025) | `,` (0.019) | `in` (0.016) | `an` (0.014)

### `it_is` — "It is"
*minimal predicate: adj or preposition next*

Top-10: `a` (0.134) | `the` (0.062) | `.` (0.059) | `not` (0.047) | `it` (0.037) | `,` (0.032) | `its` (0.029) | `also` (0.022) | `all` (0.021) | `very` (0.018)

## RMS Bias Reduction: Top-3 Templates

Formula: `RMS = sqrt(mean((spatial_prob - descriptive_prob)²))` over all eval examples.
Constrained softmax — probabilities normalised over spatial+descriptive tokens only.

| Template | Prefix | baseline_rms | best_rms | best_coeff | reduction% |
|---|---|---|---|---|---|
| `image_shows` | "The image shows" | 0.3070 | 0.2543 | -100 | 17.1% |
| `it_stands` | "It stands" | 0.3991 | 0.1858 | -50 | 53.4% |
| `it_appears` | "It appears" | 0.4558 | 0.3123 | -100 | 31.5% |

### Lambda sweep curves

#### `image_shows`
| coeff | rms | reduction% |
|---|---|---|
| -150 | 0.4186 | -36.4% |
| -100 | 0.2543 | 17.1% |
| -50 | 0.3357 | -9.4% |
| +0 | 0.3938 | -28.3% |
| +50 | 0.5459 | -77.8% |
| +100 | 0.8748 | -185.0% |
| +150 | 0.9443 | -207.6% |

#### `it_stands`
| coeff | rms | reduction% |
|---|---|---|
| -150 | 0.4050 | -1.5% |
| -100 | 0.2126 | 46.7% |
| -50 | 0.1858 | 53.4% |
| +0 | 0.4015 | -0.6% |
| +50 | 0.7868 | -97.2% |
| +100 | 0.8940 | -124.0% |
| +150 | 0.9396 | -135.4% |

#### `it_appears`
| coeff | rms | reduction% |
|---|---|---|
| -150 | 0.4723 | -3.6% |
| -100 | 0.3123 | 31.5% |
| -50 | 0.3367 | 26.1% |
| +0 | 0.4786 | -5.0% |
| +50 | 0.6378 | -39.9% |
| +100 | 0.8691 | -90.7% |
| +150 | 0.9394 | -106.1% |
