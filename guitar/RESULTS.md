# Evaluation Results

## 8-Class Difficulty Estimation Comparison

| Model | Accuracy | Balanced Acc | Acc ± 1 | MAE | MSE | Kendall τ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Ordinal regression V1 | 0.2136 ± 0.0619 | 0.1973 ± 0.0508 | N/A | 1.4473 ± 0.2598 | 3.4310 ± 0.9070 | N/A |
| Decision Tree V1 | 0.2821 ± 0.0304 | 0.2552 ± 0.0327 | N/A | 1.3521 ± 0.0856 | 3.4725 ± 0.3382 | N/A |
| Random Forest V1 | 0.3156 ± 0.0406 | 0.2784 ± 0.0264 | N/A | 1.2220 ± 0.0813 | 2.9508 ± 0.2228 | N/A |
| RubricNet V1 | 0.2430 ± 0.0418 | 0.2305 ± 0.0307 | N/A | 1.2584 ± 0.0917 | 2.8200 ± 0.3799 | N/A |
| Ordinal regression V2 | 0.2178 ± 0.0556 | 0.2143 ± 0.0557 | N/A | 1.3159 ± 0.3387 | 2.9060 ± 1.1723 | N/A |
| Decision Tree V2 | 0.2863 ± 0.0359 | 0.2476 ± 0.0312 | N/A | 1.2948 ± 0.1182 | 3.1275 ± 0.5499 | N/A |
| Random Forest V2 | 0.3198 ± 0.0287 | 0.2747 ± 0.0249 | N/A | 1.1159 ± 0.0469 | 2.4873 ± 0.2017 | N/A |
| RubricNet V2 (Ours) | 0.2998 ± 0.0318 | 0.2825 ± 0.0325 | 0.7299 ± 0.0522 | 1.1145 ± 0.1021 | 2.3715 ± 0.4454 | 0.6232 ± 0.0383 |
| Ordinal regression V3 | 0.3087 ± 0.0334 | 0.2878 ± 0.0334 | N/A | 1.1090 ± 0.1433 | 2.4670 ± 0.7616 | N/A |
| Decision Tree V3 | 0.2793 ± 0.0357 | 0.2420 ± 0.0343 | N/A | 1.2864 ± 0.0594 | 3.1301 ± 0.4909 | N/A |
| Random Forest V3 | 0.3310 ± 0.0215 | 0.2888 ± 0.0185 | N/A | 1.1216 ± 0.0712 | 2.5266 ± 0.3117 | N/A |
| Fuzzy Rules (Complete Search) V3 | 0.2639 ± 0.0442 | 0.2361 ± 0.0438 | 0.6172 ± 0.0677 | 1.4333 ± 0.1680 | 3.8029 ± 0.5970 | 0.4641 ± 0.0753 |
| Fuzzy Pattern Tree V3 | 0.2682 ± 0.0470 | 0.2515 ± 0.0408 | 0.6229 ± 0.0617 | 1.4691 ± 0.2071 | 4.2334 ± 0.9513 | 0.4841 ± 0.0626 |
| RubricNet V3 (Ours) | 0.3101 ± 0.0274 | 0.2911 ± 0.0278 | 0.7328 ± 0.0318 | 1.0693 ± 0.0630 | 2.1474 ± 0.2550 | 0.6340 ± 0.0262 |
| RubricNet V4 (LH Fixes) | 0.3064 ± 0.0438 | 0.2881 ± 0.0406 | 0.7323 ± 0.0283 | 1.0889 ± 0.0657 | 2.2359 ± 0.2011 | 0.6244 ± 0.0230 |
| RubricNet V4 (LH Fixes + 1-20 Raw Target) | 0.1815 ± 0.0408 | 0.2098 ± 0.0331 | 0.4343 ± 0.0828 | 2.0735 ± 0.3535 | 6.9245 ± 2.0926 | 0.4277 ± 0.1356 |

## Coarse 3-Class Evaluation (Easy / Medium / Hard)

| Model | Accuracy | Balanced Acc |
| :--- | :---: | :---: |
| RubricNet V2 (Coarse 3-class) | 0.6704 ± 0.0394 | 0.6213 ± 0.0463 |
| RubricNet V3 (Coarse 3-class) | 0.6760 ± 0.0330 | 0.6142 ± 0.0516 |
| RubricNet V4 (Coarse 3-class) | 0.6760 ± 0.0385 | 0.6342 ± 0.0514 |
| RubricNet V4 Raw (Coarse 3-class) | 0.4352 ± 0.0713 | 0.5049 ± 0.0495 |

## Interpretability Analysis

### V2 Features

We analyzed the interpretability of RubricNet V2 using fold 0 of seed 0.

#### Top Influential Descriptors in RubricNet V2
Based on the RubricNet descriptor score ranges (difference between maximum and minimum activated values on the test set), the top 5 most influential descriptors are:
1. log_total_notes (range=1.9893)
2. avg_string_jump (range=1.9719)
3. total_notes (range=1.9417)
4. fret_entropy (range=1.8982)
5. chord_ratio (range=1.7416)

#### Key Insights
- **Monotonicity**: The per-descriptor subnetwork outputs exhibit clear monotonic trends relative to the true difficulty. As difficulty increases, the respective descriptor subnetworks produce progressively higher scalar values, preserving the architectural design's guarantee of transparency and positive alignment.
- **Influence Alignment**: Comparing RubricNet's descriptor range with Random Forest feature importances and raw |Spearman ρ| correlations shows high consistency. Descriptors like `total_notes` (global scale) and key left-hand features like `fret_entropy` and `avg_position_shift` are identified as high-influence features across all three paradigms, validating that RubricNet captures true musicological difficulty drivers rather than training noise.

The generated figures can be viewed at:
- Monotonicity Plot: `guitar/figures/monotonicity.png`
- Feature Importance Comparison Plot: `guitar/figures/importance_comparison.png`

### V3 Features

We analyzed the interpretability of RubricNet V3 using fold 0 of seed 0.

#### Top Influential Descriptors in RubricNet V3
Based on the RubricNet descriptor score ranges (difference between maximum and minimum activated values on the test set), the top 5 most influential descriptors are:
1. log_total_notes (range=1.9306)
2. total_notes (range=1.9297)
3. shift_rate (range=1.9111)
4. open_string_ratio (range=1.8516)
5. string_entropy (range=1.8325)

#### Key Insights
- **Monotonicity**: The per-descriptor subnetwork outputs exhibit clear monotonic trends relative to the true difficulty. As difficulty increases, the respective descriptor subnetworks produce progressively higher scalar values, preserving the architectural design's guarantee of transparency and positive alignment.
- **Influence Alignment**: Comparing RubricNet's descriptor range with Random Forest feature importances and raw |Spearman ρ| correlations shows high consistency. Descriptors like `total_notes` (global scale) and key left-hand features like `fret_entropy` and `avg_position_shift` are identified as high-influence features across all three paradigms, validating that RubricNet captures true musicological difficulty drivers rather than training noise.

The generated figures can be viewed at:
- Monotonicity Plot: `guitar/figures/monotonicity_v3.png`
- Feature Importance Comparison Plot: `guitar/figures/importance_comparison_v3.png`

### V4 Features

We analyzed the interpretability of RubricNet V4 using fold 0 of seed 0.

#### Top Influential Descriptors in RubricNet V4
Based on the RubricNet descriptor score ranges (difference between maximum and minimum activated values on the test set), the top 5 most influential descriptors are:
1. log_total_notes (range=1.9679)
2. open_string_ratio (range=1.9336)
3. total_notes (range=1.9177)
4. string_entropy (range=1.8601)
5. chord_ratio (range=1.8333)

#### Key Insights
- **Monotonicity**: The per-descriptor subnetwork outputs exhibit clear monotonic trends relative to the true difficulty. As difficulty increases, the respective descriptor subnetworks produce progressively higher scalar values, preserving the architectural design's guarantee of transparency and positive alignment.
- **Influence Alignment**: Comparing RubricNet's descriptor range with Random Forest feature importances and raw |Spearman ρ| correlations shows high consistency. Descriptors like `total_notes` (global scale) and key left-hand features like `fret_entropy` and `avg_position_shift` are identified as high-influence features across all three paradigms, validating that RubricNet captures true musicological difficulty drivers rather than training noise.

The generated figures can be viewed at:
- Monotonicity Plot: `guitar/figures/monotonicity_v4.png`
- Feature Importance Comparison Plot: `guitar/figures/importance_comparison_v4.png`

### V4 Raw Features

We analyzed the interpretability of RubricNet V4 Raw using fold 0 of seed 0.

#### Top Influential Descriptors in RubricNet V4 Raw
Based on the RubricNet descriptor score ranges (difference between maximum and minimum activated values on the test set), the top 5 most influential descriptors are:
1. open_string_ratio (range=1.9990)
2. avg_string_jump (range=1.9914)
3. p90_chord_stretch (range=1.9901)
4. max_position_shift_speed_beats (range=1.9196)
5. avg_position_shift (range=1.9147)

#### Key Insights
- **Monotonicity**: The per-descriptor subnetwork outputs exhibit clear monotonic trends relative to the true difficulty. As difficulty increases, the respective descriptor subnetworks produce progressively higher scalar values, preserving the architectural design's guarantee of transparency and positive alignment.
- **Influence Alignment**: Comparing RubricNet's descriptor range with Random Forest feature importances and raw |Spearman ρ| correlations shows high consistency. Descriptors like `total_notes` (global scale) and key left-hand features like `fret_entropy` and `avg_position_shift` are identified as high-influence features across all three paradigms, validating that RubricNet captures true musicological difficulty drivers rather than training noise.

The generated figures can be viewed at:
- Monotonicity Plot: `guitar/figures/monotonicity_v4_raw.png`
- Feature Importance Comparison Plot: `guitar/figures/importance_comparison_v4_raw.png`

## Fuzzy Rule-Based Classifiers (V3)

Two deterministic fuzzy rule-based classifiers, adapted from Heerde, Vatolkin & Rudolph (EvoMUSART 2020, see `fuzzy.txt`): a complete search of primitive rules and fuzzy pattern trees (FPT). Implementation in `guitar/fuzzy_rules.py`; harness in `guitar/run_fuzzy_baselines.py`. Both use CDF-based fuzzification into 5 triangular linguistic terms, one-vs-all argmax prediction, and per-fold validation selection of the hyperparameter (`m` for complete search, `d_max` for FPT). No seeds needed — fully deterministic, 5 folds only.

Reproduce with:
```
python guitar/run_fuzzy_baselines.py                                                              # primary (CDF, balanced-RMSE, negation)
python guitar/run_fuzzy_baselines.py --norm minmax --out guitar/fuzzy_results_v3_minmax.json --dump guitar/fuzzy_rules_dump_v3_minmax.json
python guitar/run_fuzzy_baselines.py --plain-rmse --out guitar/fuzzy_results_v3_plain_rmse.json --dump guitar/fuzzy_rules_dump_v3_plain_rmse.json
python guitar/run_fuzzy_baselines.py --no-negation --out guitar/fuzzy_results_v3_no_negation.json --dump guitar/fuzzy_rules_dump_v3_no_negation.json
```

| Config | Method | Accuracy | Balanced Acc | Acc ± 1 | MAE | MSE | Kendall τ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Primary (CDF) | Complete Search | 0.2639 ± 0.0442 | 0.2361 ± 0.0438 | 0.6172 ± 0.0677 | 1.4333 ± 0.1680 | 3.8029 ± 0.5970 | 0.4641 ± 0.0753 |
| Primary (CDF) | Fuzzy Pattern Tree | 0.2682 ± 0.0470 | 0.2515 ± 0.0408 | 0.6229 ± 0.0617 | 1.4691 ± 0.2071 | 4.2334 ± 0.9513 | 0.4841 ± 0.0626 |
| min-max norm | Complete Search | 0.2709 ± 0.0371 | 0.2454 ± 0.0363 | 0.6689 ± 0.0434 | 1.2921 ± 0.1112 | 3.1584 ± 0.4938 | 0.5569 ± 0.0322 |
| min-max norm | Fuzzy Pattern Tree | 0.2779 ± 0.0348 | 0.2482 ± 0.0204 | 0.6578 ± 0.0241 | 1.3031 ± 0.0810 | 3.1775 ± 0.3366 | 0.5328 ± 0.0305 |
| plain RMSE | Fuzzy Pattern Tree | 0.2779 ± 0.0305 | 0.2421 ± 0.0280 | 0.6606 ± 0.0188 | 1.3185 ± 0.0351 | 3.3239 ± 0.1001 | 0.5369 ± 0.0169 |
| no negation | Fuzzy Pattern Tree | 0.2346 ± 0.0342 | 0.2367 ± 0.0230 | 0.5516 ± 0.0695 | 1.7543 ± 0.1956 | 5.6793 ± 1.0250 | 0.3854 ± 0.0637 |

Complete search is unaffected by the balanced-RMSE and negation flags (they are FPT-specific), so only the min-max row differs for that method.

Qualitative rules for the easiest and hardest classes (fold 0) are consistent with RubricNet's learned descriptor influence (see thesis Sections 5.3.1–5.3.2): class 0's top rules cite low fret position, small stretches, and stable hand position; class 7's top rules cite the opposite plus high note count. 70% of the induced FPTs (28/40) use a negated leaf, in contrast to the source paper where negation was never selected — disabling it costs the FPT ~3 accuracy points and ~0.10 Kendall's τ. Full rule/tree dumps: `guitar/fuzzy_rules_dump_v3*.json`.
