# Ablation: Hand-Crafted Interaction Descriptors (Negative Result)

**Status:** Rejected. The interaction descriptors did not improve any model and
are not used in the final thesis pipeline. This document records the attempt as a
negative result, per the fallback in `OPTION_A_IMPROVEMENT_PLAN.md`.

## Motivation

RubricNet is an **additive** model: it scores each descriptor independently with a
monotone sub-network and sums the scores. By construction it cannot represent
*interactions* — e.g. "a barre is only hard when the left hand is also shifting a
lot." The hypothesis was that supplying such interactions as explicit, named
descriptors would (i) give the additive model access to multiplicative difficulty
structure and (ii) close the ~2-point exact-accuracy gap to Random Forest while
staying fully interpretable.

## What was implemented

Five descriptors in `guitar_features.calculate_interaction_descriptors_v3`, each a
composition of existing V2/V3 base descriptors:

| Descriptor | Definition | Intuition |
|:---|:---|:---|
| `barre_difficulty_tempo` | `barre_ratio · (1 + std_position_shift)` | barres are harder amid unstable positions |
| `stretch_under_time_pressure` | `p90_chord_stretch / (min_inter_onset_interval + 0.1)` | wide stretches with little time to shift |
| `position_shift_entropy` | `std_position_shift · string_entropy` | shifts + string variety = left-hand load |
| `open_string_efficiency` | `(1 − open_string_ratio) · (max_position_shift + 1)` | avoiding open strings while shifting |
| `arpeggio_stretch_coupling` | `arpeggio_density · avg_chord_stretch` (gated at density > 0.1) | wide-spanning arpeggios |

Extraction: `guitar/extract_features_v3_base.py` →
`features/guitar_descriptors_v3_base.csv` (716 rows, all resolved, 0 parse
failures). Feature-set wiring: `ALL_FEATURES_V3_BASE_NEW` in
`guitar/prepare_splits.py`; `--v3-base-new` flag in `baselines.py`,
`optuna_guitar_tuning.py`, and `train_guitar_rubricnet.py`.

## Univariate signal was fine

Spearman ρ against raw Difficulty (1–20), all 716 pieces:

| Descriptor | ρ |
|:---|:---:|
| `open_string_efficiency` | +0.4584 |
| `barre_difficulty_tempo` | +0.3592 |
| `stretch_under_time_pressure` | +0.3406 |
| `position_shift_entropy` | +0.3387 |
| `arpeggio_stretch_coupling` | +0.2091 |

All five clear the plan's |ρ| ≥ 0.15 acceptance bar. Signal existed in isolation.

## But every model got worse when they were added

RubricNet (3 seeds × 5 folds), test:

| Model | Acc | Bal | MAE | MSE | τ |
|:---|:---:|:---:|:---:|:---:|:---:|
| V3 base (24 feats) | 0.3012 | 0.2821 | 1.0834 | 2.1810 | 0.6327 |
| V3 base + interaction (29 feats) | **0.2849** | **0.2695** | **1.1304** | **2.3401** | **0.5972** |

The regression reproduces on both non-neural baselines (test, 5-fold), so it is
not specific to RubricNet or its hyperparameters:

| Model | V3 base acc | V3 base + interaction acc |
|:---|:---:|:---:|
| Ordinal regression | 0.3254 | 0.2821 |
| Random Forest | 0.3254 | 0.3101 |

Random Forest — which *can* model interactions natively — also drops, confirming
the added columns carry no new information for it and only add noise. The plan's
Phase 3 gate ("RF accuracy improves ≥ 0.5 pts") therefore fails.

## Why it failed: collinearity with the parent descriptors

Each interaction descriptor is a deterministic product/ratio of descriptors
already present in the feature set. It is therefore highly collinear with its
parents and contributes almost no variance orthogonal to them. Consequences:

- **Additive models (RubricNet, ordinal regression):** summing the monotone score
  of an interaction descriptor on top of the scores of its parents *double-counts*
  the same underlying signal. This shifts and widens the summed logit
  distribution without adding discriminative information — variance up, accuracy
  down. The additive models are hit hardest (ordinal regression −4.3 acc pts).
- **Random Forest:** trees already recover `a·b` structure via nested splits on
  `a` and `b`, so the explicit product is redundant; it merely enlarges the
  candidate-split space and slightly dilutes each tree.

In short: the interactions the additive model "can't see" are ones RF already
captures, and handing them to the additive model as collinear extra inputs is
strictly worse than leaving the base descriptors clean.

## Implication for guitar difficulty modelling

Multiplicative difficulty structure is real (RF's edge is partly interaction
capture), but **naively appending products of existing descriptors is not a viable
way to inject it into an additive model.** A productive future direction would be
interactions built from *decorrelated* primitives (e.g. residualising each
interaction against its parents, or replacing a parent pair with a single learned
interaction term) rather than adding collinear redundancy. That is left as future
work; it was out of scope for Option A.

## Decision

- Interaction descriptors **rejected**.
- Dropping the rhythm-aware features (V3 base) is also **not adopted** — it is
  marginally worse than V3 full and removes the features that most help the RF
  comparison hold up as a fair black-box baseline.
- **V3 remains the final interpretable model** (acc 0.310, bal 0.291, MAE 1.069,
  MSE 2.147, τ 0.634). See `guitar/RESULTS_V3_BASE_NEW.md`.

Code, feature CSV, and result JSONs are retained (nothing removed) so the ablation
is reproducible: run `python guitar/extract_features_v3_base.py`, then
`python guitar/baselines.py --v3-base-new` and
`python guitar/train_guitar_rubricnet.py --v3-base-new`.
