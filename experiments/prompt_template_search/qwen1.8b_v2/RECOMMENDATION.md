# Recommendation: Best Natural Template for Qwen-1.8B

## Winner: `B_positioned` — "Positioned"

### Why

- **Approach**: B — spatial-leaning: on/beside/above/near as first token
- **baseline_rms**: 0.9670
- **RMS reduction**: 94.2% at coeff=+100
- **Spatial tokens in top-10**: 4  (no spatial+desc both present)
- **Descriptive tokens in top-10**: 0

### Lambda sweep curve

| coeff | rms | reduction% |
|---|---|---|
| -150 | 0.0942 | 90.3% |
| -100 | 0.1217 | 87.4% |
| -50 | 0.3167 | 67.2% |
| +0 | 0.9625 | 0.5% |
| +50 | 0.9283 | 4.0% |
| +100 | 0.0563 | 94.2% |
| +150 | 0.2309 | 76.1% |

### Generation examples (λ=−100, λ=0, λ=+100)

**λ=−100:**
- _A lone hiker stands on top of a snow-dusted ridge looking out across a wide vall_
  → `Positioned a one a no a a no a no a no a no a no a no a no a`
- _A bright orange and yellow maple tree stands beside a small dark pond, its round_
  → `Positioned a one a no a one a no a no a no a no a no a no a no`
- _A wide red double-decker bus is parked beside a row of tall narrow buildings, it_
  → `Positioned a one a no a no a no a no a no a no a no a no a no`

**λ=0 (unsteered):**
- _A lone hiker stands on top of a snow-dusted ridge looking out across a wide vall_
  → `Positioned at the edge of a vast, snowy landscape, a solitary figure can be seen standing atop a rocky`
- _A bright orange and yellow maple tree stands beside a small dark pond, its round_
  → `Positioned in a serene natural setting, this image captures the beauty of autumn's colors. The maple tree is`
- _A wide red double-decker bus is parked beside a row of tall narrow buildings, it_
  → `Positioned in a busy urban area, this image captures the essence of a bustling city street. The bus itself`

**λ=+100:**
- _A lone hiker stands on top of a snow-dusted ridge looking out across a wide vall_
  → `Positioned ,,,,,,,,,,,,,,,,,,,,`
- _A bright orange and yellow maple tree stands beside a small dark pond, its round_
  → `Positioned ,,,,,,,,,,,,,,,,,,,,`
- _A wide red double-decker bus is parked beside a row of tall narrow buildings, it_
  → `Positioned ,,,,,,,,,,,,,,,,,,,,`

---

## Full Rankings

| Rank | Name | reduction% | rms@λ=0 | both_in_top10 |
|---|---|---|---|---|
| 1 | `A2_in_the` | 75.7% | 0.9958 | ✗ |
| 2 | `A_in_the` | 89.8% | 0.8885 | ✗ |
| 3 | `B_positioned` | 94.2% | 0.9670 | ✗ |
| 4 | `A2_main_subject` | 62.5% | 0.6094 | ✗ |
| 5 | `A_subject_looks` | 91.0% | 0.6377 | ✗ |
