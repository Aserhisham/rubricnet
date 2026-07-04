# Guitar RubricNet Improvement Plan

Goal: close the gap between Guitar RubricNet (currently 24.3% acc / 23.1% balanced acc / 1.26 MAE, 5-fold mean) and the Random Forest baseline (31.6% / 27.8% / 1.22), and produce thesis-ready metrics and interpretability analysis. Target: RubricNet balanced accuracy within 2 points of RF (ideally ≥ 27%), with fold-4 instability fixed.

## Context you must not rediscover (already established)

- Dataset: 716 pieces, 8 difficulty classes (equal-frequency bins of levels 1–20), class sizes 37–136. Chance = 12.5%, majority-class = 19.0%, always-predict-median MAE = 1.80.
- Data file: `features/guitar_descriptors.csv` (716 rows, 12 feature columns + metadata incl. `xml_path`/`token_path`/`pdf_path`, `source` = pdf:655 / dada_gp:49 / gaps:12).
- Splits: `guitar/guitar_splits.json`, fixed 5-fold stratified. **NEVER regenerate or modify this file.** All experiments use these exact splits.
- Feature–label Spearman correlations (computed on the full CSV, `Difficulty` raw 1–20):
  - Strong: `total_notes` +0.68, `max_chord_stretch` +0.44, `barre_ratio` +0.34, `max_string_jump` +0.32, `avg_chord_stretch` +0.32, `avg_position_shift` +0.26
  - Dead (|ρ| ≤ 0.14): `fret_change_rate` −0.05, `arpeggio_density` −0.10, `avg_string_jump` −0.07, `avg_polyphony` +0.05, `special_technique_ratio` +0.06 (nonzero for only 29/716 pieces — XML source only), `tempo_bpm` −0.14 (14 missing values, currently fillna(0))
- Optuna history (`guitar/guitar_rubricnet_guitar_all.db`, 20 trials): search converged to lr ≈ 0.03–0.1, small batch (16–43), high dropout (0.35–0.5), plateau at val balanced acc ≈ 0.31. The bottleneck is the features, not the tuning.
- RubricNet architecture constraint: each descriptor is scored *independently* and scores are summed (additive model, no feature interactions). Interaction effects must be provided as explicit hand-crafted features. `hidden_size=1, num_layers=1` are by design — do not tune them.
- Class weighting is already built into RubricNet's ordinal loss (`RubricnetSklearn.calculate_weights`) — do not add it again.
- Fold 4 (split index 4) collapses to 17.5% acc with the current lr=0.036 config; other folds get 23–27%.

## Ground rules

1. Never modify `guitar/guitar_splits.json` or the bin edges in `guitar/prepare_splits.py`.
2. Never modify `rubricnet/rubricnet.py` except where a phase explicitly says so.
3. StandardScaler is always fit on train only (existing code does this — keep it).
4. Use the project venv: `/home/aser/programming/thesis/rubricnet/.venv/bin/python`. Run everything from the repo root.
5. Keep existing result JSONs; write new results to new filenames (versioned) so before/after comparison is possible.
6. Commit at the end of each phase with a short message (`git add` only the files that phase touched).
7. If a phase's acceptance check fails, stop and report rather than proceeding.

---

## Phase 0 — Reproduce current numbers (sanity check)

1. Run `.venv/bin/python guitar/baselines.py` and `.venv/bin/python guitar/train_guitar_rubricnet.py`.
2. Confirm the printed 5-fold means match the context above within ±1.5 points (RubricNet has seed variance; exact reproduction not required).

**Acceptance:** both scripts run to completion; RF accuracy mean ≈ 0.31, RubricNet accuracy mean ≈ 0.22–0.27.

---

## Phase 1 — Descriptor v2: fix dead features, add new ones

### 1a. Refactor extraction to expose chord sequences

In `guitar/guitar_features.py`, refactor so chord extraction is separated from descriptor computation:

- Split `parse_guitar_xml(filepath)` into `get_chords_from_xml(filepath) -> (chords, tempo_bpm, technique_ratio)` plus the existing wrapper that calls `calculate_descriptors_from_chords`.
- Same for `extract_from_tokens` → `get_chords_from_tokens(filepath) -> chords` and `parse_guitar_pdf` → `get_chords_from_pdf(filepath) -> (chords, tempo_bpm)`.
- The existing public functions must keep their current signatures and outputs (wrappers around the new functions), so nothing else breaks.

### 1b. Implement `calculate_descriptors_v2(chords)`

New function in `guitar/guitar_features.py`. It returns everything `calculate_descriptors_from_chords` returns **plus** the following. All new features must be computable from fret/string chord lists alone (no rhythm/duration data exists for the 655 PDF pieces). Chords are lists of `{'fret': int, 'string': int}`; a "chord event" is one element of `chords`.

Length/quantity (replace raw-count confound with tamed versions):
- `log_total_notes` = `np.log1p(total_notes)` — keep `total_notes` too.

Position height (currently missing entirely; higher positions are harder):
- `avg_fret` — mean fret over all notes.
- `p90_fret` — 90th percentile of fret over all notes.
- `high_position_ratio` — fraction of notes with fret ≥ 7.
- `open_string_ratio` — fraction of notes with fret == 0 (open strings are easier; expect negative correlation).

Stretch/shape distribution (max_* worked; add distribution shape):
- `p90_chord_stretch` — 90th percentile of per-chord stretch (over multi-note chords with ≥1 fretted note, same population as `avg_chord_stretch`).
- `chord_ratio` — fraction of chord events with ≥ 2 notes.
- `avg_string_span` — mean over multi-note chords of (max string − min string).
- `unique_shape_rate` — number of distinct (sorted tuple of (string, fret)) chord shapes among multi-note chords, divided by number of multi-note chords; 0 if none. Measures shape vocabulary vs repetition.

Movement (replace the dead `fret_change_rate` normalization):
- `shift_rate` — fraction of consecutive event pairs where the mean fret changes by > 2 (a position shift), i.e. `np.mean(np.abs(np.diff(avg_frets)) > 2)`.
- `max_position_shift` — max of `np.abs(np.diff(avg_frets))`, 0 if < 2 events.
- `std_position_shift` — std of the same diffs, 0 if < 2 events.

Variety/entropy:
- `fret_entropy` — Shannon entropy (base 2) of the distribution of fret values over all notes.
- `string_entropy` — same over string values.
- `repetition_ratio` — fraction of consecutive event pairs whose (string, fret) sets are identical (repeated figure = easier; expect negative correlation).

Guard all of these against edge cases: empty chords, single event, no multi-note chords → return 0 for the affected feature, never NaN/inf. Add a unit-style check at the bottom of the extraction script (1c) that asserts no NaN/inf in the output DataFrame.

### 1c. Extraction script

Create `guitar/extract_features_v2.py`:

1. Read `features/guitar_descriptors.csv` (this is the source of truth for which 716 pieces are in the dataset and where their files are).
2. For each row, dispatch on `source`: `gaps` → `get_chords_from_xml(xml_path)`, `dada_gp` → `get_chords_from_tokens(token_path)`, `pdf` → `get_chords_from_pdf(pdf_path)`. If a path column is NaN, try `file_path`. If parsing fails for a piece, keep the piece and log it; fill its new features from the v1 row where names overlap and 0 otherwise (do not drop pieces — the splits reference all 716 piece ids).
3. Compute `calculate_descriptors_v2(chords)`, carry over `tempo_bpm` where the source provides it.
4. `tempo_bpm`: impute the 14 missing values with the train-independent global median (this is a fixed constant, acceptable; note it in the thesis) instead of 0.
5. Merge with all metadata columns from the v1 CSV (`Title`, `Composer`, `Difficulty`, `source`, path columns, etc.). Write `features/guitar_descriptors_v2.csv` with the same 716 rows.
6. Print a Spearman correlation table (feature vs `Difficulty`) for all v1+v2 features, sorted by |ρ|.

Expect PDF re-parsing of 655 files to take a while; that's fine. Print progress every 50 pieces.

### 1d. Prune

Based on the printed Spearman table: drop `special_technique_ratio` unconditionally (29/716 nonzero — it's a data-availability artifact, not a descriptor). Drop any *new* feature with |ρ| < 0.05. Keep all other v1 features even if weak (needed for the ablation story), but record the table in `guitar/feature_audit_v2.md` (a short markdown table of ρ values plus one line on what was dropped and why).

### 1e. Wire in the v2 feature set

In `guitar/prepare_splits.py`:
- Add `FEATURE_GROUPS_V2` (lh / rh / global groups, assigning each surviving new feature to the appropriate group: position/stretch/shift/fret-entropy → lh; string-jump/string-entropy/arpeggio → rh; totals/tempo/polyphony/chord_ratio/repetition → global) and `ALL_FEATURES_V2`.
- Do **not** touch `main()`, the bin edges, or `NUM_CLASSES`.

In `guitar/baselines.py`: give `load_data` parameters `csv_path` and `columns` (already there) — just verify it works when called with the v2 CSV and `ALL_FEATURES_V2`.

**Acceptance:** `features/guitar_descriptors_v2.csv` exists with 716 rows, no NaN/inf in feature columns; at least 3 new features have |ρ| ≥ 0.25; `guitar/feature_audit_v2.md` written.

---

## Phase 2 — Re-run baselines on v2 features

1. Add a `--v2` flag (or a small separate `main`) to `guitar/baselines.py` that loads the v2 CSV + `ALL_FEATURES_V2` and writes `guitar/baseline_results_v2.json`. Input dim for the ordinal baseline must use `len(ALL_FEATURES_V2)`, and use a distinct `alias_experiment` (e.g. `guitar_baseline_ordinal_v2`) so checkpoints don't collide.
2. Run it. Record the RF numbers — this is the new bar RubricNet has to meet.

**Acceptance:** `guitar/baseline_results_v2.json` written; RF balanced accuracy on v2 ≥ v1 RF (if v2 features *hurt* RF, the new descriptors are broken — stop and report).

---


## Phase 4 — Retune RubricNet on v2 features

1. In `guitar/optuna_guitar_tuning.py`, add `"guitar_all_v2": ALL_FEATURES_V2` to `FEATURE_SETS` and make `load_data` receive the v2 csv path when a `_v2` feature set is chosen.
2. Narrow the search using the existing history: `lr` log-uniform in [5e-3, 1e-1], `batch_size` int in [16, 64], `dropout` [0.1, 0.5], `weight_decay` log-uniform [1e-4, 1e-1], `decay_lr` [0.3, 0.9]. Keep patience=20, hidden_size=1, num_layers=1.
3. Run 60 trials: `.venv/bin/python guitar/optuna_guitar_tuning.py --features guitar_all_v2 --n-trials 60`. This trains 5 folds per trial — it is the longest step of the plan; let it run.
4. Robust selection instead of trusting one noisy run: take the Pareto-best trials (`study.best_trials`), keep the top 4 by val balanced accuracy, and re-evaluate each with 3 different torch seeds (write a small script `guitar/select_best_config.py`: for each candidate config, run the 5-fold training 3× with `torch.manual_seed(seed)` for seeds 0/1/2, average val balanced acc over the 15 runs). Pick the config with the best mean. Save it to `guitar/best_hyperparams_guitar_all_v2.json` in the same format as the v1 file, adding a `"seed_mean_val_bacc"` field.

**Acceptance:** best config's mean val balanced accuracy ≥ 0.33 (v1 plateau was 0.31 — v2 features should clear it). If not, report the top of the Spearman table and the Optuna plateau value and stop for review.

---

## Phase 5 — Final training and results

Note: there is no Phase 3. An earlier draft had a Phase 3 that relaxed accuracy to an "off-by-one level counts as correct" metric — explicitly rejected; accuracy must always mean exact predicted level/class, never ±1. `acc±1` may still be *reported* as an extra informational column in the results table below, but it is never the optimization target and never substitutes for exact accuracy in any acceptance check.

1. Update `guitar/train_guitar_rubricnet.py`: load the v2 CSV / `ALL_FEATURES_V2` / v2 hyperparams file; run the 5-fold evaluation **3 times with seeds 0/1/2**; report each metric as mean ± std over the 15 fold-runs (also keep the per-fold lists). Metrics: accuracy (exact), balanced accuracy (exact), acc±1 (informational only), MAE, MSE, Kendall τ. Write `guitar/rubricnet_results_v2.json` with structure `{"hyperparams": ..., "seeds": [0,1,2], "metrics": {metric: {"per_fold_per_seed": [[...]], "mean": x, "std": y}}}`.
2. Also write a single comparison table to `guitar/RESULTS.md`: rows = ordinal regression / decision tree / RF / RubricNet-v1 (from the old JSON) / RubricNet-v2; columns = accuracy, balanced accuracy, acc±1, MAE, MSE, Kendall τ (mean ± std where available). Mark the best value per column in bold.

**Acceptance:** RubricNet-v2 balanced accuracy within 2 points of RF-v2, and no single fold below 0.20 accuracy at the chosen config. If the gap is larger, still write everything, and add a short "gap analysis" section to `RESULTS.md` (per-fold comparison, confusion matrix of the worst fold via `sklearn.metrics.confusion_matrix`).

---

## Phase 6 — Interpretability deliverable (RubricNet's selling point)

1. Inspect `rubricnet/rubricnet.py` to find how per-descriptor scores are exposed (the model computes one scalar score per descriptor before aggregation — find the module/attribute; the original repo has a way to read them, likely via the forward pass of the per-descriptor subnetworks).
2. Write `guitar/interpret_rubricnet.py`: load the trained v2 checkpoint of one representative fold (fold 0, seed 0), run the test set through the model, and export per-descriptor scores to `guitar/descriptor_scores_fold0.csv` (rows = pieces, columns = descriptors + true label + predicted label).
3. Produce two matplotlib figures into `guitar/figures/` (simple, publication-grade, no seaborn): (a) mean descriptor score per true difficulty class (line per descriptor — shows monotonicity, the paper's key interpretability plot); (b) RF feature importances vs |Spearman ρ| vs RubricNet descriptor-score range, one grouped bar chart per descriptor.
4. Add 3–5 sentences to `guitar/RESULTS.md` interpreting which descriptors drive predictions and whether they behave monotonically with difficulty.

**Acceptance:** CSV + 2 figures exist; descriptor-score-vs-class plot shows a clearly monotone trend for at least the top-3 descriptors.

---

## Phase 7 (optional, only if Phases 1–6 succeeded) — Coarse 3-class evaluation

Map the 8 classes to 3 (0–2 → easy, 3–5 → medium, 6–7 → hard) **at evaluation time only** (no retraining): take the Phase-5 predictions, map both y_true and y_pred, recompute accuracy/balanced accuracy. Add one row per model to `RESULTS.md`. This gives the thesis a headline number comparable to prior 3-level guitar difficulty work.

## Explicitly out of scope

- Do not use the `Reading` or `Max Position` metadata columns as input features (they are human annotations correlated with the label — leakage risk; at most mention them as future work).
- Do not change the 8-class binning, the splits, or the RubricNet architecture.
- Do not delete v1 result files or v1 feature columns from the codebase.
