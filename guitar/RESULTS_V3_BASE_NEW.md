# V3 Base + Interaction Features — Results

**Option A execution (see `OPTION_A_IMPROVEMENT_PLAN.md`).** Goal was to close the
gap to Random Forest (0.331 acc) by (a) dropping the rhythm-aware features and
(b) adding five hand-crafted *interaction* descriptors. Both bets **did not
improve RubricNet**; the interaction features consistently *hurt* accuracy across
three independent model families. V3 remains the best interpretable model.

## Headline comparison (RubricNet, 3 seeds × 5 folds, test)

| Model | Accuracy | Balanced Acc | Acc ± 1 | MAE | MSE | Kendall τ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Random Forest V3 (black box) | **0.3310** | 0.2888 | N/A | 1.1216 | 2.5266 | N/A |
| **RubricNet V3 (kept as final)** | 0.3101 | **0.2911** | 0.7328 | **1.0693** | **2.1474** | **0.6340** |
| RubricNet V3 Base (no rhythm) | 0.3012 | 0.2821 | 0.7318 | 1.0834 | 2.1810 | 0.6327 |
| RubricNet V3 Base + Interaction | 0.2849 | 0.2695 | 0.7215 | 1.1304 | 2.3401 | 0.5972 |

Numbers regenerated from:
`guitar/rubricnet_results_v3.json`, `guitar/rubricnet_results_v3_base.json`,
`guitar/rubricnet_results_v3_base_new.json`.

## Baseline sanity check (test, 5-fold)

The same ordering holds for the non-neural baselines, so this is not a RubricNet
optimisation artifact:

| Feature set (n feats) | Ordinal Reg acc | Ordinal Reg bal | RF acc | RF bal |
|:---|:---:|:---:|:---:|:---:|
| V3 base (24, no rhythm) | 0.3254 | 0.3114 | 0.3254 | 0.2796 |
| V3 base + interaction (29) | 0.2821 | 0.2730 | 0.3101 | 0.2683 |
| V3 full (26, with rhythm) | 0.3087 | 0.2878 | 0.3310 | 0.2888 |

From `guitar/baseline_results_v3_base.json`,
`guitar/baseline_results_v3_base_new.json`, `guitar/baseline_results_v3.json`.

## Interpretation

- **Interaction features hurt (−2.5 acc pts on RubricNet, −0.037 τ).** Every
  metric degrades when the five descriptors are added, and it reproduces on the
  additive ordinal-regression baseline (0.325 → 0.282) and Random Forest
  (0.325 → 0.310). Root cause: each interaction descriptor is a product/ratio of
  descriptors already in the set (e.g. `open_string_efficiency` reuses
  `open_string_ratio` and `max_position_shift`). In an additive-over-monotone-
  scores model these collinear terms *double-count* existing signal rather than
  supplying new information, inflating variance without raising accuracy. This is
  despite all five having a healthy univariate correlation with difficulty
  (|Spearman ρ| = 0.21–0.46) — marginal signal did not survive being conditioned
  on the parents.
- **Dropping rhythm does not help.** V3 base (no rhythm) is slightly *worse* than
  V3 full (0.301 vs 0.310), within noise. The rhythm-aware window/velocity
  features are what give Random Forest its edge (0.331); removing them only
  narrows the RF advantage by weakening RF, not by strengthening RubricNet.
- **V3 stays the final interpretable model.** It keeps the best RubricNet
  accuracy (0.310), the best error metrics of any model in the study
  (MAE 1.069 < RF 1.122; MSE 2.147 < RF 2.527), the best ranking correlation
  (τ 0.634), and a balanced accuracy edge over RF (0.291 vs 0.289). The
  interpretable model trails the black box on raw exact-match accuracy by ~2
  points while beating it on every ranking/error metric — the defensible thesis
  claim.

See `guitar/ABLATION_INTERACTION_FEATURES.md` for the full negative-result
write-up (thesis appendix).
