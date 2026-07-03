# Roadmap to Completion

Written 2026-07-03. Estimates are **focused, Claude-assisted work hours** — actual
time spent solving the problem together, not calendar time. Your own review/thinking
and advisor feedback cycles sit on top of these numbers and aren't compressible by
Claude.

## Where things stand today

You are substantially ahead of the proposal's own timeline (which allotted May–July
for exactly this):

- **Dataset**: 712 classical guitar pieces matched across GuitarBurst/GAPS/DADA-GP/PDF
  sources, each with a difficulty label (1–20) **and** full symbolic feature
  extraction already computed (`features/guitar_descriptors.csv`).
- **Feature extraction** (`guitar/guitar_features.py`, documented in
  `docs/guitar_descriptors.md`): 13 descriptors across the three dimensions the
  proposal calls for — LH (barre_ratio, avg/max_chord_stretch, avg_position_shift,
  fret_change_rate), RH (arpeggio_density, avg/max_string_jump,
  special_technique_ratio), Global (avg_polyphony, total_notes, tempo_bpm).
- **MIDI rhythm alignment**: 512/651 pieces aligned cleanly, 136 skipped
  (low coverage), 3 unmatched — accepted as-is (pitch content and attack timing,
  which is what feature extraction needs, are correct even where full rhythmic
  notation isn't recovered).
- **Model infrastructure inherited from the piano RubricNet codebase**
  (`rubricnet/rubricnet.py`, `optuna_bayesian_optimization.py`,
  `interpretability.py`) is architecture-agnostic — it takes a column list and a
  JSON split file, so most of it can be **repointed at guitar data rather than
  rebuilt from scratch**. This is the single biggest time-saver left on the table.

One concrete bug found while reviewing: `guitar/train_guitar_rubricnet.py`
references `max_position_shift` / `total_position_shift`, columns that don't
exist in `guitar_descriptors.csv` (only `avg_position_shift` does) — it will crash
as-is and needs a rewrite, not a patch.

Difficulty label distribution (712 pieces) also matters for what's next:

```
1-6:   274 pieces   7-12:  346 pieces   13-18: 88 pieces   19-20: 5 pieces (17: 7, 18: 9, 19: 3, 20: 2)
```

The top of the scale is sparse — levels 17–20 combined are 21 pieces. Equal-width
bins (as in the current draft script) will leave near-empty top classes; this
needs a deliberate binning decision, not the current fixed-width scheme.

## Phases

### Phase 1 — Data/eval scaffolding (~2–4 hrs)
- Fix the broken feature list in the training script; lock the final per-group
  descriptor set (LH/RH/Global).
- Decide the 1–20 → N-class binning, informed by the distribution above (e.g.
  merge levels 15+ into one top class).
- Generate a fixed, stratified k-fold split file (`guitar_splits.json`) mirroring
  the existing `rubricnet/cipi_splits.json`, so baselines/RubricNet/ablations all
  train and evaluate on identical folds.
- **Decision needed from you**: final number of ordinal classes and bin edges.

### Phase 2 — Baselines (~2–3 hrs)
- Ordinal regression — `LogisticRegressionOrdinal` already exists in
  `rubricnet/rubricnet.py`, reuse directly.
- Random Forest / decision tree baseline with feature importances.
- Report accuracy / MAE / MSE across folds for both.

### Phase 3 — Guitar RubricNet + hyperparameter tuning (~3–5 hrs)
- Rewrite `train_guitar_rubricnet.py` around the corrected features and new
  splits.
- Adapt `optuna_bayesian_optimization.py`: point data loading at the guitar CSV,
  add a `FEATURES="guitar_all"` branch, rerun the existing multi-objective
  (accuracy + MSE) search — the search loop itself needs no changes.

### Phase 4 — Ablation study (~2–3 hrs)
- Reuse the same `FEATURES` switch pattern to define `lh_only`, `rh_only`,
  `global_only`, `lh_rh`, `all` column sets.
- Train the 5 variants across folds — this directly produces the comparison
  table required by the proposal's Evaluation Plan §5.1.

### Phase 5 — Interpretability analysis (~4–6 hrs)
- Adapt `rubricnet/interpretability.py` (built for CIPI/piano) to guitar:
  descriptor-contribution plots, class-boundary plots, per-piece local
  explainability tables.
- Group-level feature-selection sweep (§5.2): with only ~13 descriptors a full
  power-set search is computationally cheap (small NN, 712 rows) — this is
  scripting effort, not an algorithmic challenge.
- Qualitative pedagogical check on a handful of pieces at different levels —
  inherently a judgment call, budget your own time here, not Claude's.

### Phase 6 — Optional: fuzzy-logic classifier (~2–4 hrs)
Explicitly "if time permits" in the proposal. Cut first if the schedule is tight.

### Phase 7 — Writing (Methods, Evaluation, Results)
Once phases 2–5 produce figures/tables, Claude can draft these sections quickly
(~1 day of writing-with-Claude for a first pass). The real bottleneck is advisor
review cycles, which aren't compressible — plan calendar time for that
separately from effort time.

## Total effort estimate

Phases 1–5 (the core "does it work, can I explain it" path): **~15–20 hours of
focused work**, realistically **3–5 working days** back-to-back — none of this
needs long unattended training runs (small model, 712 rows, trains on CPU in
minutes per fold).

If you work through this continuously, the technical work could be done within
**about a week**, leaving the rest of the proposal's July–September window as
slack for writing, advisor revisions, and the optional fuzzy-logic stretch goal
— well ahead of the original May–September plan.

## Risks

- **Class imbalance at the top of the scale** (5 pieces at levels 19–20) — needs
  the rebinning decision in Phase 1 before anything downstream is trustworthy.
- **136 low-coverage MIDI alignments** — already usable for features, no action
  needed unless finer rhythmic descriptors are wanted later.
- **Group-level feature-selection results could show 1–2 descriptors dominate**
  — would weaken the "multi-dimensional combination helps" claim (Research
  Question 3). Worth surfacing in Phase 5 early rather than discovering it while
  writing Results.
