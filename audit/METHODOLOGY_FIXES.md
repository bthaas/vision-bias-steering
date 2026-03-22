# Methodology Fixes

Implements corrections flagged in `DIVERGENCE_REPORT.md`.
All changed lines are marked with `# REALIGNED:` comments in the source.

---

## Fix C1 — WMD weights restored to raw bias scores

**File**: `bias_steering/steering/extract.py` (lines 53-57)

**Change**:
```python
# BEFORE (diverged):
pos_weights = torch.Tensor(pos_examples.bias.abs().tolist()) ** 2
neg_weights = torch.Tensor(neg_examples.bias.abs().tolist()) ** 2

# AFTER (aligned with reference):
pos_weights = torch.Tensor(pos_examples.bias.tolist())
neg_weights = torch.Tensor(neg_examples.bias.tolist())
```

**Why**: The reference repo uses raw signed bias scores as WMD weights. The previous
local version applied `.abs().square()`, which (a) discards the sign (breaking the
convex-combination property for negative weights) and (b) quadratically amplifies
outlier examples, shifting the steering vector away from the modal distribution.
This is not what the paper's WMD method describes. Any comparison to published
WMD results was invalid under the old formula.

---

## Fix C2 — Default validation reverts to coeff=0 (pure projection removal)

**File**: `bias_steering/config.py` (line 29)

**Change**:
```python
# BEFORE:
optimize_coeff: bool = True

# AFTER:
optimize_coeff: bool = False  # REALIGNED
```

**Why**: The reference `validate()` always uses `coeff=0`, which tests pure orthogonal
projection removal — the theoretically motivated operation for determining whether the
extracted vector captures a real semantic axis. Sweeping coefficients and picking the
best-RMS result (the old default) optimizes the metric on the validation set, producing
an inflated upper bound that conflates vector quality with coefficient selection.

The coeff sweep path (`optimize_coeff=True`) is still available and can be activated
explicitly via CLI `--optimize_coeff`, but is no longer the default.

---

## Fix M3 — Score mode default changed from "adaptive" to "prob_diff"

**Files**:
- `bias_steering/config.py` (line 27)
- `bias_steering/run.py` (line 1562)

**Changes**:
```python
# config.py — BEFORE:
score_mode: str = "adaptive"

# config.py — AFTER:
score_mode: str = "prob_diff"  # REALIGNED

# run.py — BEFORE:
score_mode=args.score_mode or "adaptive",

# run.py — AFTER:
score_mode=args.score_mode or "prob_diff",  # REALIGNED
```

**Why**: The reference exclusively uses `prob_diff = pos_probs - neg_probs` for both
WMD weight construction (via the `.bias` column on the training DataFrame) and
validation RMS computation. The local `"adaptive"` mode heuristically switched between
`prob_diff` and `logit_margin` using label accuracy and dynamic range heuristics,
creating a pipeline mismatch: the WMD vector was extracted to explain `prob_diff`
variance but validation layers were then ranked by `logit_margin` RMSE. This made
layer ranking non-comparable across runs depending on which mode the heuristic selected.

`"logit_margin"` and `"adaptive"` remain available as explicit CLI options
(`--score_mode logit_margin`) for diagnostic experiments.

---

## Unchanged (intentional deviations from reference retained)

| Item | Decision |
|---|---|
| `constrained_softmax` default `False` | Already matches reference (reference lacks this option). Default is safe. |
| `remove_overlapping_target_ids()` | Methodological improvement over reference — kept. |
| GPT-2 module detection (`model.py`) | Practical compatibility addition — kept. |
| `apply_chat_template` try/except (`model.py`) | GPT-2 compatibility — kept. |
| Lazy imports in `__init__.py` | No behavioral impact — kept. |
| `normalized_bias`, label metrics, template sweep in `validate.py` | Additive diagnostics — kept. |
| `run_handcrafted_eval` and `run_generation_log` in `run.py` | New features added in this session — kept, not affected by above fixes. |

---

## Verification of non-regression

- `bias_steering/eval/generation_logger.py` is not affected by any of the above changes.
  It imports only from `..steering.intervention` (unchanged).
- The `save_subdir` parameter added to `validate()` in the previous session is
  untouched and still defaults to `"validation"`.
- CLI flags `--log_generations`, `--eval_set handcrafted`, and all `--log_gen_*` args
  are unaffected.
