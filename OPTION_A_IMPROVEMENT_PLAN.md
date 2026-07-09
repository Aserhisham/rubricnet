# Option A: Push RubricNet to 38–40% Accuracy — Execution Plan

**Goal:** Improve RubricNet V3 from 31.0% to 35%+ accuracy through better feature engineering and re-tuning, to reach a competitive position vs Random Forest (33.1%) and justify the thesis claim that an interpretable model can match black-box performance.

**Timeline:** 5–7 days of parallel work. Estimated effort: 40 hours.

**Success Criteria:**
- Achieve 34%+ accuracy on V3 base without rhythm features
- New hand-crafted interaction descriptors increase accuracy to 35%+
- Re-tuned hyperparams push model to 36–38%+
- All results committed with clear versioning

---

## Phase 1: Diagnose Rhythm Features (Days 1–2)

### 1a. Extract V3 Base Descriptors (No Rhythm)

**Objective:** Create a feature set with V3's core descriptors but *without* the rhythm-aware features (`max_avg_chord_stretch_window`, `p95_position_shift_window`, `max_note_density_window`, `avg_stretch_velocity_beats`, `avg_position_shift_speed_beats`, `polyphonic_arpeggio_intensity_beats`).

**Action:**
1. In `guitar/guitar_features.py`, examine `calculate_descriptors_v3()` and identify all rhythm-aware features (search for `window`, `velocity_beats`, `intensity_beats`).
2. Create a new function `calculate_descriptors_v3_base()` that:
   - Calls the same chord extraction as V3
   - Computes only the non-rhythm features (position height, fret entropy, stretch, shift_rate, string_entropy, etc.)
   - Returns the same 20–24 descriptors, excluding the 6 rhythm-aware ones
3. Create `guitar/extract_features_v3_base.py` (copy of `extract_features_v3.py`, use `calculate_descriptors_v3_base()`):
   - Re-extract all 716 pieces
   - Output: `features/guitar_descriptors_v3_base.csv`
   - Should have same 716 rows, ~20 feature columns instead of ~26

**Acceptance Criteria:**
- CSV exists with 716 rows, no NaN/inf in feature columns
- Feature count is ~20 (document which 6 were dropped)
- Spearman correlations printed and logged to `guitar/v3_base_audit.txt`

**Time:** 2 hours

---

### 1b. Run Baseline: V3 Base vs V3 Full Comparison

**Objective:** Measure whether rhythm-aware features help or hurt.

**Action:**
1. In `guitar/prepare_splits.py`, add:
   ```python
   ALL_FEATURES_V3_BASE = [list of base features from extracted CSV above]
   ```
2. Run baselines on both feature sets:
   ```bash
   python guitar/baselines.py --features guitar_all_v3_base --output guitar/baseline_results_v3_base.json
   python guitar/baselines.py --features guitar_all_v3 --output guitar/baseline_results_v3.json
   ```
3. Compare RF performance:
   - V3 base: expected ~31–32% accuracy (if rhythm was neutral/hurt)
   - V3 full: known 33.1% accuracy (current)
4. If V3 base > V3 full: rhythm features are hurting; proceed with base-only
5. If V3 base ≈ V3 full: rhythm features add minimal value; proceed with base + new features

**Output:** 
- `guitar/baseline_results_v3_base.json` 
- Text file with comparison table (RF base vs full)
- Decision: use base or full for next phase

**Acceptance Criteria:**
- Clear decision made: "Rhythm features [help / hurt / neutral]"
- If hurt: proceed with V3 base only
- If neutral: proceed with V3 full + new features

**Time:** 1–2 hours (mostly waiting for training)

---

## Phase 2: Hand-Crafted Interaction Features (Days 2–3)

### 2a. Design Interaction Descriptors

**Objective:** Create 5 new descriptors that capture multi-factor difficulty (not learnable via additive model alone, but interpretable via composition).

**Descriptors to add** (in `guitar/guitar_features.py`, new function `calculate_interaction_descriptors_v3()`):

1. **`barre_difficulty_tempo`** = `barre_ratio * (1.0 + std_position_shift)`
   - Intuition: barres are harder when positions are unstable
   - Value: 0–1 × 0–2 ≈ 0–2

2. **`stretch_under_time_pressure`** = `p90_chord_stretch / (min_inter_onset_interval_beats + 0.1)`
   - Intuition: large stretches are harder if little time to shift positions
   - Compute `min_inter_onset_interval_beats` from onsets in timed chord extraction
   - Value: 0–8 / (0.25 to 4 beats) ≈ 0–32

3. **`position_shift_entropy`** = `std_position_shift * string_entropy`
   - Intuition: frequent position shifts + string variety = high left-hand demand
   - Value: 0–6 × 0–3 ≈ 0–18

4. **`open_string_efficiency`** = `(1.0 - open_string_ratio) * (max_position_shift + 1.0)`
   - Intuition: pieces that avoid open strings and shift positions are demanding
   - Value: (0–1) × (0–12) ≈ 0–12

5. **`arpeggio_stretch_coupling`** = `arpeggio_density * avg_chord_stretch` (if arpeggio_density > 0.1 else 0)
   - Intuition: arpeggios spanning wide stretches are harder
   - Value: 0–1 × 0–8 ≈ 0–8

**Implementation:**
- Add to `guitar_features.py`:
  ```python
  def calculate_interaction_descriptors_v3(chords, onsets, fret_sequence):
      """Hand-crafted multi-factor descriptors for guitar difficulty.
      
      Args:
          chords: list of chord events (from get_timed_chords_from_xml)
          onsets: list of onset times in beats (parallel to chords)
          fret_sequence: flattened list of all frets in order
      
      Returns:
          dict of 5 interaction descriptors
      """
      # Implement each descriptor with edge-case guards (div by zero, empty chords, etc.)
      # Return dict: {'barre_difficulty_tempo': x, 'stretch_under_time_pressure': y, ...}
  ```
- Update `extract_features_v3_base.py` to call this and append 5 new columns
- Re-run extraction: `python guitar/extract_features_v3_base.py`
- Output: `features/guitar_descriptors_v3_base.csv` with 25 total columns

**Acceptance Criteria:**
- No NaN/inf in new columns (all edge cases handled)
- Spearman correlations for new descriptors printed; expect at least 3 of 5 to have |ρ| ≥ 0.20
- New CSV written successfully

**Time:** 4–6 hours (careful implementation + testing)

---

### 2b. Measure Feature Signal

**Objective:** Confirm new features add signal.

**Action:**
1. Compute Spearman correlation of all 25 features against raw Difficulty (1–20):
   ```bash
   python -c "
   import pandas as pd
   from scipy.stats import spearmanr
   df = pd.read_csv('features/guitar_descriptors_v3_base.csv')
   for col in [c for c in df.columns if c not in ['Difficulty', 'Title', 'Composer', ...]]:
       rho, p = spearmanr(df[col], df['Difficulty'])
       print(f'{col}: {rho:.3f}')
   " | sort -t: -k2 -rn > guitar/feature_correlations_v3_base_new.txt
   ```
2. Document which new features have |ρ| ≥ 0.15 (meaningful signal)

**Acceptance Criteria:**
- At least 3 of 5 new descriptors have |ρ| ≥ 0.15
- If fewer than 3: review descriptor design; iterate before proceeding

**Time:** 30 minutes

---

## Phase 3: Re-run Baselines (Day 3)

### 3a. Baseline with New Features

**Objective:** Measure whether Random Forest improves with new features.

**Action:**
1. In `guitar/prepare_splits.py`, add:
   ```python
   ALL_FEATURES_V3_BASE_NEW = [all 25 feature names]
   ```
2. Run:
   ```bash
   python guitar/baselines.py --features guitar_all_v3_base_new --output guitar/baseline_results_v3_base_new.json
   ```
3. Compare RF accuracy:
   - V3 base (20 features): ~31–32%
   - V3 base + new (25 features): expected 32–34%
   - If RF improves by 1–2 points, new features have signal

**Acceptance Criteria:**
- RF accuracy improves by at least 0.5 points; if not, reconsider feature design

**Time:** 1–2 hours

---

## Phase 4: Optuna Re-tuning (Days 4–5)

### 4a. Set Up Feature Set in Optuna

**Objective:** Hyperparameter search for the new feature set.

**Action:**
1. In `guitar/optuna_guitar_tuning.py`, add:
   ```python
   FEATURE_SETS = {
       ...
       "guitar_all_v3_base_new": ALL_FEATURES_V3_BASE_NEW
   }
   ```
   and update `load_data()` to handle the v3_base_new CSV path.

2. Run Optuna sweep (use frozen v3 hyperparams as priors):
   ```bash
   python guitar/optuna_guitar_tuning.py \
     --features guitar_all_v3_base_new \
     --n-trials 40 \
     --study-name guitar_v3_base_new
   ```
   - LR: log-uniform [5e-3, 1e-1]
   - Batch size: [16, 64]
   - Dropout: [0.1, 0.5]
   - Weight decay: log-uniform [1e-4, 1e-1]
   - Decay LR: [0.3, 0.9]

3. Select best config via robust method:
   - Take top 4 trials by val balanced accuracy
   - Re-evaluate each with 3 torch seeds (0, 1, 2) across 5 folds = 15 runs
   - Average val balanced accuracy over 15 runs
   - Pick highest average; save to `guitar/best_hyperparams_guitar_all_v3_base_new.json`

**Acceptance Criteria:**
- Best config's mean val balanced accuracy ≥ 0.30 (v3 baseline was 0.2911)
- If not: report top of feature correlation table and Optuna plateau; stop for review

**Time:** 8–12 hours (mostly Optuna wall-clock; can run overnight)

---

### 4b. Final Training & Results (Day 5–6)

**Objective:** Train RubricNet with best config across 3 seeds.

**Action:**
1. Update `guitar/train_guitar_rubricnet.py`:
   - Add `--v3-base-new` flag (loads v3_base_new CSV + feature set + hyperparams JSON)
   - Run 5-fold × 3 seeds (seeds 0, 1, 2)
   - Compute: accuracy (exact), balanced accuracy, MAE, MSE, Kendall τ
   - Output: `guitar/rubricnet_results_v3_base_new.json` with per-fold, per-seed breakdowns

2. Run:
   ```bash
   python guitar/train_guitar_rubricnet.py --v3-base-new --is-experimental False
   ```

3. Compare to baselines:
   - Random Forest V3: 33.1% accuracy
   - RubricNet V3 base new: expected 34–36%

**Acceptance Criteria:**
- RubricNet ≥ 34% accuracy (within 1 point of RF)
- Balanced accuracy ≥ 0.295 (ahead of RF's 0.289)
- MAE ≤ 1.09 (better than RF)
- Kendall τ ≥ 0.62 (competitive)

**Output:**
- `guitar/rubricnet_results_v3_base_new.json`
- Updated monotonicity + importance plots in `guitar/figures/monotonicity_v3_base_new.png` etc.

**Time:** 4–6 hours (training wall-clock)

---

## Phase 5: Commit & Report (Day 6–7)

### 5a. Document Results

**Action:**
1. Create comparison table in `guitar/RESULTS_V3_BASE_NEW.md`:
   ```markdown
   | Model | Accuracy | Balanced Acc | MAE | MSE | Kendall τ |
   |:---|:---:|:---:|:---:|:---:|:---:|
   | Random Forest V3 | 0.331 | 0.289 | 1.122 | 2.527 | 0.624 |
   | RubricNet V3 (Original) | 0.310 | 0.291 | 1.069 | 2.147 | 0.634 |
   | RubricNet V3 Base (No Rhythm) | ? | ? | ? | ? | ? |
   | **RubricNet V3 Base + New (Final)** | **?** | **?** | **?** | **?** | **?** |
   ```

2. Write 3–4 sentences on:
   - Why rhythm features were dropped (or kept)
   - How interaction descriptors improve interpretability
   - Whether this closes the gap to original RubricNet piano baseline

### 5b. Commit

**Action:**
```bash
git add guitar/extract_features_v3_base.py
git add guitar/rubricnet_results_v3_base_new.json
git add guitar/best_hyperparams_guitar_all_v3_base_new.json
git add guitar/RESULTS_V3_BASE_NEW.md
git add guitar/figures/monotonicity_v3_base_new.png
git add guitar/figures/importance_comparison_v3_base_new.png
git commit -m "feat: V3 base + interaction features, optuna retune (target 35%+ accuracy)"
```

### 5c. Report Back

**Provide to Aser:**
1. Final accuracy numbers (full table)
2. Did you hit ≥34% accuracy? Yes/No
3. Which interaction descriptors were most influential?
4. One sentence on whether to use this as final thesis result or fall back to Option B

---

## Fallback: If Phase 3/4 Show Negative Results

If new features don't help (RF plateaus, or new features have |ρ| < 0.10):

1. **Don't panic.** Fall back to Option B:
   - Use V3 original (31% accuracy)
   - Reframe thesis as: "Interpretable model achieves comparable ranking correlation (τ=0.634) and error metrics (MAE=1.069) with full transparency, while RF achieves marginally better raw accuracy."
   - This is still defensible; just weaker headline.

2. **Document the attempt:**
   - Create `guitar/ABLATION_INTERACTION_FEATURES.md`: explain what was tried, why it didn't work, implications for guitar difficulty modeling
   - This becomes "valuable negative result" in thesis appendix

---

## Key Constraints (Do Not Violate)

- Never modify `guitar/guitar_splits.json` or bin edges
- RubricNet architecture frozen (hidden_size=1, additive, ordinal loss)
- All new feature sets use same 5-fold splits
- Keep all old result files; write new ones with version suffixes
- Every commit must be reproducible from scratch via scripts

---

## Success Definition

**This plan succeeds if:**
- Accuracy reaches 34–38% (vs current 31% and RF 33.1%)
- **OR** you have a well-documented reason why it maxed out at 31%, making Option B the cleaner choice

**Effort budget:** 40–50 hours over 5–7 days (parallel work where possible)
