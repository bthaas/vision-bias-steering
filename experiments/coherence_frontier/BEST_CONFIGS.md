# Best Configurations — Coherent + Max RMS Reduction

Top configurations that maximize RMS reduction while maintaining coherent text.

Baseline RMS: 0.9670

## #1: Layer 11, λ=-50, 1 tokens

- **Experiment**: Exp3
- **Layer config**: single_11  layers=[11]
- **λ**: -50
- **Token limit**: 1
- **RMS**: 0.3167
- **Reduction**: 67.2%
- **Coherence**: coherent

### Generation examples (greedy, 20 tokens)

- _A lone hiker stands on top of a snow-dusted ridge looking out across a wide vall_
  → `Positioned no more than 10 feet away from me is a person standing at the edge of a dense`  [coherent]
- _A bright orange and yellow maple tree stands beside a small dark pond, its round_
  → `Positioned no more than 10 feet away from the speaker is a large, round object with a smooth`  [coherent]
- _A wide red double-decker bus is parked beside a row of tall narrow buildings, it_
  → `Positioned no more than 10 feet away from each other are two large, round tables with chairs that`  [coherent]
- _A tall iron gate stands at the far end of a narrow stone path, with a wooden ben_
  → `Positioned a total of 12 questions to answer:

  1. What is the name of the`  [coherent]
- _A round wooden table sits at the center of a small kitchen, with copper pots han_
  → `Positioned a few questions to clarify your request:

1. Are you referring to multiple chairs or is there only`  [coherent]

## #2: Layer 11, λ=-50, 5 tokens

- **Experiment**: Exp3
- **Layer config**: single_11  layers=[11]
- **λ**: -50
- **Token limit**: 5
- **RMS**: 0.3167
- **Reduction**: 67.2%
- **Coherence**: coherent

### Generation examples (greedy, 20 tokens)

- _A lone hiker stands on top of a snow-dusted ridge looking out across a wide vall_
  → `Positioned no other any other one thing that is not a chair.`  [coherent]
- _A bright orange and yellow maple tree stands beside a small dark pond, its round_
  → `Positioned no other two a certain number of times.`  [coherent]
- _A wide red double-decker bus is parked beside a row of tall narrow buildings, it_
  → `Positioned no other two a certain number of times.`  [coherent]
- _A tall iron gate stands at the far end of a narrow stone path, with a wooden ben_
  → `Positioned a total  one a. b. c. d. e. f. g. h.`  [coherent]
- _A round wooden table sits at the center of a small kitchen, with copper pots han_
  → `Positioned a total  one a. b. c. d. e. f. g. h.`  [coherent]

## #3: Layer 11, λ=-40, 1 tokens

- **Experiment**: Exp3
- **Layer config**: single_11  layers=[11]
- **λ**: -40
- **Token limit**: 1
- **RMS**: 0.3892
- **Reduction**: 59.8%
- **Coherence**: coherent

### Generation examples (greedy, 20 tokens)

- _A lone hiker stands on top of a snow-dusted ridge looking out across a wide vall_
  → `Positioned no more than a few feet away from me is a person standing on a rocky outcropping overlooking`  [coherent]
- _A bright orange and yellow maple tree stands beside a small dark pond, its round_
  → `Positioned no more than 10 feet away from the speaker is a large, round object with a smooth`  [coherent]
- _A wide red double-decker bus is parked beside a row of tall narrow buildings, it_
  → `Positioned no more than 10 feet away from each other, there are two large windows on either side`  [coherent]
- _A tall iron gate stands at the far end of a narrow stone path, with a wooden ben_
  → `Positioned no more than 10 feet away from the nearest wall is a large, ornate chandelier`  [coherent]
- _A round wooden table sits at the center of a small kitchen, with copper pots han_
  → `Positioned no more than 10 feet away from the speaker is a large, rectangular table made of dark`  [coherent]
