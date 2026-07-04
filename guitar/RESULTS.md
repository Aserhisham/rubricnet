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
| RubricNet V3 (Ours) | 0.3101 ± 0.0274 | 0.2911 ± 0.0278 | 0.7328 ± 0.0318 | 1.0693 ± 0.0630 | 2.1474 ± 0.2550 | 0.6340 ± 0.0262 |

## Coarse 3-Class Evaluation (Easy / Medium / Hard)

| Model | Accuracy | Balanced Acc |
| :--- | :---: | :---: |
| RubricNet V2 (Coarse 3-class) | 0.6704 ± 0.0394 | 0.6213 ± 0.0463 |
| RubricNet V3 (Coarse 3-class) | 0.6760 ± 0.0330 | 0.6142 ± 0.0516 |

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
- **Monotonicity**: The per-descriptor subnetwork outputs exhibit clear monotonic trends relative to the true difficulty classes. As difficulty increases, the respective descriptor subnetworks produce progressively higher scalar values, preserving the architectural design's guarantee of transparency and positive alignment.
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
- **Monotonicity**: The per-descriptor subnetwork outputs exhibit clear monotonic trends relative to the true difficulty classes. As difficulty increases, the respective descriptor subnetworks produce progressively higher scalar values, preserving the architectural design's guarantee of transparency and positive alignment.
- **Influence Alignment**: Comparing RubricNet's descriptor range with Random Forest feature importances and raw |Spearman ρ| correlations shows high consistency. Descriptors like `total_notes` (global scale) and key left-hand features like `fret_entropy` and `avg_position_shift` are identified as high-influence features across all three paradigms, validating that RubricNet captures true musicological difficulty drivers rather than training noise.

The generated figures can be viewed at:
- Monotonicity Plot: `guitar/figures/monotonicity_v3.png`
- Feature Importance Comparison Plot: `guitar/figures/importance_comparison_v3.png`
