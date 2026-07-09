# Project History — Interpretable Difficulty Estimation for Classical Guitar

This document reconstructs the full story of this repository: where it came from, what
you (Aser Abdelhakeem) and Claude did to it, in what order, and why. It's built from
`git log`, the thesis proposal PDF, the docs already in the repo, and session memory.
Read it top to bottom once, then use it as a map back into the code.

---

## 1. What this project actually is

Your bachelor's thesis, **"Interpretable Difficulty Estimation for Classical Guitar
from Symbolic Notation"** (proposal dated May 2026), asks: can an interpretable,
"rubric-like" model predict how hard a classical guitar piece is to play, using only
features extracted from the score (MusicXML/tab), the same way a teacher would judge
difficulty from barre chords, position shifts, chord stretches, etc.?

This is a direct adaptation of an existing MIR paper to a new instrument:

- **RubricNet** (Ramoneda et al., ISMIR 2024, *"Towards Explainable and Interpretable
  Musical Difficulty Estimation: A Parameter-Efficient Approach"*) — a white-box neural
  model for **piano** difficulty estimation, built on the CIPI dataset. Each hand-crafted
  descriptor gets its own tiny subnetwork (a linear layer + nonlinearity) whose scalar
  output is summed with the others into an ordinal difficulty score. Because each
  descriptor is scored independently before summation, you can read off exactly how much
  each one contributed to a given piece's predicted difficulty — that's the "rubric."
- This repo is a **fork of PRamoneda/rubricnet** (`origin` = `Aserhisham/rubricnet`,
  `upstream` = `PRamoneda/rubricnet`). The original piano code
  (`rubricnet/rubricnet.py`, the CIPI feature extractor under `extractor/`) is largely
  **untouched infrastructure** — architecture-agnostic enough that the thesis work is
  mostly about *repointing* it at a new dataset and a new descriptor set, not rewriting it.

The thesis's four research questions (from the proposal):
1. Which guitar-specific symbolic descriptors best capture technical difficulty, as
   reflected in a graded repertoire resource (GuitarBurst)?
2. Can a rubric-based model predict piece-level difficulty with useful accuracy?
3. Does combining LH + RH + Global dimensions beat any single dimension alone?
4. Do the model's descriptor contributions align with pedagogical intuition?

---

## 2. Timeline

### Era 0 — Inherited piano codebase (2024-07 to 2025-12, pre-thesis)

This is the original `PRamoneda/rubricnet` repository as published alongside the ISMIR
paper, plus your early fork housekeeping:
- `rubricnet/rubricnet.py`: `Rubricnet` (the additive descriptor-subnetwork model),
  `LogisticRegressionOrdinal`, `OrdinalLoss`, `RubricnetSklearn` (an sklearn-style
  wrapper used everywhere downstream).
- `extractor/`: the CIPI piano feature extractor (raw_data extractors, fingering models,
  bayesian difficulty calculators) — piano-specific, not reused for guitar except as a
  reference for how descriptor extraction pipelines are structured.
- 2024-12: "Add all changes to push to rubricnet repository", README/license cleanup.
- 2025-12-02: MIT License added.

Nothing guitar-related exists yet. This era exists purely so the thesis has a proven
interpretable-model architecture to adapt rather than invent from scratch.

### Era 1 — Dataset acquisition (2026-05-13)

Two commits kick off the actual thesis work:
- **"feat: setup guitarburst scraper, GAPS dataset integration, and matching logic for
  thesis"**: adds a Mutopia downloader, pulls in `gaps_v1_metadata.csv` (7MB — the GAPS
  dataset, Guitar Aligned Performance Scores), and a large `guitarburst_full.json`
  (21k+ lines) — the scraped GuitarBurst graded-repertoire database that supplies your
  **difficulty labels** (1–20 scale) for essentially the whole project. `PROJECT_STATUS.md`
  appears as a running log.
- **"GAPS extracted"**: syncs GAPS labels against GuitarBurst (`sync_gaps_labels.py`),
  produces `found_pieces.csv` — the first cut at pieces with both a score source and a
  difficulty label.

GuitarBurst is the *label* source throughout the project (the proposal explicitly names
it as the preferred difficulty-label resource); GAPS, DADA-GP, and a large body of
Mutopia/other PDF scores are the *score* sources that need to be matched against it.

### Era 2 — Matching pipeline & dataset assembly (2026-06-23, "dataset created")

The single largest commit in the repo's history (64 files, +36,442/−455). This is where
the "curate a dataset of classical guitar pieces with symbolic scores + difficulty
labels" problem gets solved in earnest:
- `scripts/scraping/`: `extract_guitarburst.py`, `download_mutopia_midi_ly.py`.
- `scripts/matching/`: one script per source to reconcile against GuitarBurst —
  `match_gaps.py`, `match_dada.py`, `match_mutopia.py`, `match_pdf.py`,
  `match_ly_pieces.py`, `match_high_difficulty.py`, plus `loose_scan.py` for fuzzy
  matches and `sync_datasets.py`/`sync_stats.py` to reconcile everything into one table.
- `scripts/analysis/`: `compare_pieces.py`, `piece_stats.py`, `piece_visual_analysis.py`,
  `generate_targets.py`, `update_json_and_stats.py` — sanity-checking the matches (plots
  of found/unfound difficulty distributions land in `data/plots/`).
- `scripts/utilities/`: the beginnings of format conversion/copying infrastructure
  (`copy_from_found_pieces.py`, `copy_to_verified.py`, `export_to_excel.py`,
  `final_clean_csv.py`).
- First version of `guitar/guitar_features.py` (305 lines) and
  `guitar/train_guitar_rubricnet.py` (77 lines) — early, not-yet-working descriptor
  extraction and a training script stub.
- `data/checked_pieces.md` — a **manual spot-check log**: for a handful of pieces per
  difficulty level (1 through ~18), you personally verified the assigned GuitarBurst
  level against your own judgment ("correct" / "incorrect" per piece). This is the
  human sanity check behind trusting GuitarBurst's labels at all.
- Notebooks (`notebooks/*.ipynb`) for exploring the dataset and the unmatched pieces.

By the end of this commit, pieces come from three tiers: **PDF** (Mutopia and other
public-domain scores, largest by far), **DADA-GP** (a GuitarPro tab dataset, tokenized
format), and **GAPS** (small, but has genuinely aligned performance-score data).

### Era 3 — Unifying formats and fixing rhythm (2026-07-03, "all transferred to xml")

At this point the dataset exists but is fragmented across three incompatible symbolic
formats (raw PDF layout parsing, DADA-GP tokens, GAPS MusicXML), each requiring separate
feature-extraction code paths. This commit (+14,675/−3,958) does the consolidation:
- `scripts/utilities/convert_pdf_to_musicxml.py` (411 lines) and
  `convert_gp_to_musicxml.py` (226 lines): convert the PDF-vector-graphics tab scores and
  the DADA-GP GuitarPro tokens into MusicXML, so everything can eventually live in one
  format.
- `scripts/utilities/align_midi_rhythm.py` (577 lines) — the big one. PDF-extracted
  scores only carry correct **pitch** content (via OCR/vector parsing); they don't carry
  real rhythm (every note comes out as a quarter note by default). This script takes a
  companion MIDI file for each piece and a Needleman-Wunsch sequence alignment between
  the PDF's note sequence and the MIDI's note sequence, then **transplants the MIDI's
  timing onto the PDF's (correct) pitches**, writing rhythm-corrected MusicXML to
  `verified_pieces/pdf/xml_rhythm/`.
  - Key fixes made here: time signature is read from the MIDI meta-event rather than
    trusted from the PDF parser (which defaults to 4/4 and misses vector-graphic time
    signatures); note duration is computed from inter-onset interval (time to the next
    onset) rather than raw sounding duration, because guitar bass strings ring out much
    longer than they're "held" musically, which was inflating chord durations.
  - Result at the time: 512 pieces aligned cleanly, 136 skipped for low alignment
    coverage, 3 with no MIDI match at all. Accepted limitation: output is single-voice
    only (a classical guitar piece really has a treble melody + a bass pedal voice, but
    splitting them into separate MusicXML voice streams was judged ~2–4h of work for low
    payoff, since pitch + attack-position correctness — the things feature extraction
    actually needs — were already achieved).
- `docs/roadmap.md` and `docs/guitar_descriptors.md` written — the first real planning
  document, laying out Phases 1–7 for the rest of the thesis (data scaffolding →
  baselines → RubricNet + tuning → ablation → interpretability → optional fuzzy-logic
  classifier → writing). It also flags a concrete bug found during review:
  `train_guitar_rubricnet.py` referenced feature columns
  (`max_position_shift`/`total_position_shift`) that didn't exist in the CSV — a rewrite
  needed, not a patch.
- `guitar/guitar_features.py` grows substantially (+425/−… lines) to support all three
  source formats.

### Era 4 — Splits, baselines, and the first real run (2026-07-03)

Two commits same day:
- **"baseline done"**: `guitar/prepare_splits.py` produces `guitar/guitar_splits.json` —
  a **fixed, stratified 5-fold split** (following the roadmap's Phase 1), later declared
  a hard invariant: never regenerate or modify it, so every subsequent experiment (v1/v2/v3
  features, every baseline, every RubricNet variant) trains/evaluates on identical folds.
  `guitar/baselines.py` implements ordinal regression (reusing
  `rubricnet.rubricnet.LogisticRegressionOrdinal`), a decision tree, and a Random Forest.
  Also: `scripts/analysis/find_duplicate_pieces.py` +
  `drop_cross_batch_duplicates.py` — cleaning cross-source duplicates out of the merged
  dataset (this is where the piece count settles near 716, after dropping 7 confirmed
  duplicates and 1 mismatched-content PDF).
- **"first run, 30% accuracy only"**: the first actual RubricNet training run against
  guitar data, plus `guitar/optuna_guitar_tuning.py` for hyperparameter search. Despite
  the commit message, the "30%" refers to Random Forest's accuracy — RubricNet itself
  underperformed at 24.3% accuracy / 23.1% balanced accuracy, well behind RF's 31.6% /
  27.8%. This is the moment the project's central engineering problem became concrete:
  **RubricNet was losing to a plain Random Forest**, and `guitar/IMPROVEMENT_PLAN.md`
  gets written to diagnose why and fix it (see §4 below).

### Era 5 — Feature engineering v2: closing the gap with RF (2026-07-04, "2nd run done")

Guided by `IMPROVEMENT_PLAN.md`'s diagnosis (features, not tuning, were the bottleneck —
an Optuna sweep had already converged and plateaued at ~31% balanced accuracy regardless
of hyperparameters), this commit:
- Audited every v1 feature's Spearman correlation with the raw 1–20 difficulty label.
  Found several **dead** features (|ρ| ≤ 0.14): `fret_change_rate`, `arpeggio_density`,
  `avg_string_jump`, `avg_polyphony`, `special_technique_ratio` (nonzero for only 29/716
  pieces — an XML-only artifact, not a real descriptor), `tempo_bpm` (also weak, and 14
  pieces had it missing entirely).
- Added a new v2 descriptor set (`calculate_descriptors_v2` in `guitar_features.py`):
  log-scaled note counts, fret-height features (`avg_fret`, `p90_fret`,
  `high_position_ratio`, `open_string_ratio`), stretch/shape distribution features
  (`p90_chord_stretch`, `chord_ratio`, `avg_string_span`, `unique_shape_rate`), better
  movement features (`shift_rate`, `max_position_shift`, `std_position_shift` — replacing
  the dead `fret_change_rate`), and entropy/variety features (`fret_entropy`,
  `string_entropy`, `repetition_ratio`).
  - `special_technique_ratio` was dropped outright; documented in
    `guitar/feature_audit_v2.md`.
- `guitar/extract_features_v2.py` re-extracts all 716 pieces →
  `features/guitar_descriptors_v2.csv`.
- Re-tuned RubricNet (`best_hyperparams_guitar_all_v2.json`), retrained across 3 seeds ×
  5 folds, and wrote `guitar/interpret_rubricnet.py` — pulling the per-descriptor scalar
  scores out of the trained model to produce the actual "rubric" interpretability plots
  (`monotonicity.png`, `importance_comparison.png`) that are the thesis's whole selling
  point.
- **Result: RubricNet V2 reaches 0.300 accuracy / 0.283 balanced accuracy / 1.11 MAE /
  τ=0.62, and for the first time *beats* Random Forest V2 on balanced accuracy, MAE, MSE,
  and Kendall's τ** (RF still nominally ahead on raw accuracy). This is the thesis's
  headline result: an interpretable additive model matching/beating a black-box model.

### Era 6 — Unifying rhythm and going V3 (2026-07-04, "3rd run")

Up to this point a belief had settled in (visible in earlier session memory) that only a
small minority of pieces (49/716) had any real timing information, so rhythm-aware
features seemed out of reach for most of the dataset. On 2026-07-04 you consolidated
**every** source (PDF-with-rhythm, DADA-GP, GAPS) into one directory,
`verified_pieces/all_xmls/` (723 files) — the new single source of truth. This
overturned the earlier belief: **623/716 (87%) of pieces actually have real note
durations** (varied `<type>`/`<duration>`, `<divisions>`, `<backup>` + multi-voice
structure); only 93/716 (90 PDF + 3 DADA-GP) remain dummy uniform-quarter placeholders.
Each fretted `<note>` carries fret+string+duration together, making a proper
rhythm-aware parser feasible for the large majority of the dataset.

This unlocked a fast pivot:
- `get_timed_chords_from_xml` (in `guitar_features.py`) — a rhythm-aware MusicXML parser
  handling `divisions`/`chord`/`backup`/`forward`/`voice` to compute onset times in
  **beats** (deliberately not seconds — `tempo_bpm` had already tested as a dead feature,
  ρ=−0.14, so absolute time was abandoned in favor of tempo-independent beat position).
- `guitar/extract_features_v3.py` re-extracts base descriptors for all 716 pieces from
  the unified `all_xmls` source (not the old PDF parser), producing
  `features/guitar_descriptors_v3.csv` with a `has_rhythm` flag (639/716 True).
- **Unexpected finding**: switching the *base* (non-rhythm) descriptors from the old PDF
  parser to the unified XML source alone produced a large jump in per-feature signal —
  e.g. `fret_entropy`'s |ρ| went from weak to 0.66, `high_position_ratio` to 0.55,
  `avg_fret` to 0.54. **The old PDF parser had been silently corrupting fret numbers**;
  this had been quietly capping V1/V2 performance the whole time.
- `calculate_descriptors_v3` then layers on rhythm/context-aware features expressed in
  beats: windowed bottleneck features (`max_avg_chord_stretch_window`,
  `p95_position_shift_window`, `max_note_density_window` — because a piece's difficulty
  is usually set by its hardest passage, not its average, per
  `guitar/FUTURE_WORK_PLAN.md` §3) and context-aware physical proxies
  (`avg_stretch_velocity_beats`, `avg_position_shift_speed_beats`,
  `polyphonic_arpeggio_intensity_beats` — combining a physical demand with how much time
  is available to execute it, per `FUTURE_WORK_PLAN.md` §4).
- Gate scripts (`guitar/gate_v3.py`, `gate_rubricnet_v3.py`) confirmed no regression vs
  v2 before committing to the switch.
- **Result: RubricNet V3 reaches 0.310 accuracy / 0.291 balanced accuracy / 1.07 MAE /
  2.15 MSE / τ=0.63**, now ahead of Random Forest V3 on balanced accuracy, MAE, MSE, and
  Kendall's τ, with RF only nominally ahead on raw accuracy (0.331 vs 0.310). Also added:
  a coarse 3-class (Easy/Medium/Hard) evaluation view (RubricNet V3: 67.6% accuracy).

### Era 7 — Post-headline experimentation: pruning, label smoothing, multi-task, retune (2026-07-06)

With the thesis headline already earned (V3 beats RF on 4/6 metrics), the next question
was whether raw accuracy (still ~30%, and the one metric RF nominally still wins on) could
be pushed further **without relaxing the frozen constraint**: RubricNet stays a strictly
additive, single-hidden-layer-per-descriptor model. Four candidate levers were considered;
two were tried and rejected on evidence, one was tried and kept, one is mid-flight.

**Reviewed and rejected up front** (`geminiRect.txt`, an external brainstorm doc): GA2M
pairwise-interaction subnetworks were rejected outright — they add
$\sum g_{ij}(x_i,x_j)$ terms, which directly breaks the "no hidden interactions" invariant
that the whole interpretability story depends on. "Joint fingering optimization" was found
to not even apply to this codebase: grepping for `fingering` shows it only exists under
`extractor/` (the inherited *piano* CIPI pipeline) — guitar tab/MusicXML already commits to
a specific string+fret per note, so there's no solver step to make "joint" with anything.
Generative rhythm-feature reconstruction (replacing train-fold-median imputation for the
~11% no-rhythm pieces with a learned imputer) was judged legitimate future work but heavy
engineering for a bounded payoff — parked, not attempted.

**Experiment 1 — prune near-zero-signal legacy features (rejected).** `feature_audit_v2.md`
had already flagged `tempo_bpm`, `arpeggio_density`, `fret_change_rate`, `avg_string_jump`,
`chord_ratio`, `avg_polyphony` as weak (`|Spearman rho| <= 0.16` against raw Difficulty) back
in the v2 audit, but marked them "Kept" rather than dropped, and they were still present in
`ALL_FEATURES_V3`. Added `ALL_FEATURES_V3_PRUNED` (26 of 32 v3 columns) to
`guitar/prepare_splits.py`, wired a `--v3-pruned` flag through `baselines.py`,
`train_guitar_rubricnet.py`, and `optuna_guitar_tuning.py`. Result: **negative across the
board** — RubricNet acc 0.302 vs 0.310 baseline, bacc 0.285 vs 0.291, MAE/MSE/tau all
slightly worse, variance roughly doubled, and one fold collapsed to 0.196 acc. Random
Forest/ordinal-regression baselines were flat-to-mixed too, no clean win anywhere. Likely
confound: V3's existing Optuna-tuned regularization (`weight_decay`, `dropout`) was reused
as-is for 6-fewer-feature inputs rather than retuned for them. Code path kept (opt-in,
unused by default); idea abandoned. Numbers live in
`guitar/{baseline,rubricnet}_results_v3_pruned.json`, not merged into `RESULTS.md`.

**Bug found + fixed along the way:** `train_guitar_rubricnet.py`'s `write_results_markdown()`
unconditionally overwrote `guitar/RESULTS.md`'s manually-written Interpretability Analysis
section with a placeholder on *every* run, including throwaway experimental ones. It fired
once during the pruning experiment and clobbered the V2/V3 interpretability write-up;
recovered via `git checkout` since `RESULTS.md` is committed. Fixed by adding an
`is_experimental` guard so only the canonical `--v2`/`--v3` runs regenerate `RESULTS.md` now.

**Experiment 2 — Ordinal Label Smoothing (kept).** Implemented in `rubricnet/rubricnet.py`.
`OrdinalLoss`'s hard cumulative target (a step from 1 to 0 at the true class boundary, per
the Cao et al. rank encoding this model uses — not one-hot, so the smoothing had to be
adapted from `FUTURE_WORK_PLAN.md` §1's one-hot sketch) is now optionally softened into a
sigmoid ramp via a `label_smoothing_temp` parameter; `0` (default) reproduces the exact old
hard-step behavior (verified: vectorized hard-target path is bit-identical to the old
per-sample loop). Threaded through `LogisticRegressionOrdinal` and `RubricnetSklearn` as an
opt-in kwarg (`getattr(..., default 0.0)`, so every existing piano/V1/V2 caller is
unaffected). Exposed via `--label-smoothing-temp` on `train_guitar_rubricnet.py`. Swept
`T in {0.15, 0.3, 0.6}` against V3's frozen tuned hyperparams: **T=0.15 gave a small,
directionally-consistent win** (acc 0.312 vs 0.310, bacc 0.294 vs 0.291, no fold collapse);
T=0.3 and T=0.6 were flat-to-worse, matching the theory that over-softening erodes the
signal the ordinal loss needs. Effect size is within ~1 std error — a real direction, not a
decisive one on its own.

**Experiment 3 — multi-task coarse-3-class auxiliary head (rejected).** Also implemented in
`rubricnet/rubricnet.py`: `Rubricnet` gained an optional second `nn.Linear(1,
num_coarse_classes)` reading off the *same* `aggregated_score` scalar (no new hidden layers,
no cross-descriptor interaction — still additive), trained jointly via
`fine_loss + coarse_loss_weight * coarse_loss` where the coarse target is the existing
`map_8_to_3` grouping. Fully opt-in (`num_coarse_classes=None` by default), verified
backward-compatible via forward-shape and end-to-end fit/predict/load_model smoke tests.
Exposed via `--coarse-loss-weight` on `train_guitar_rubricnet.py`. Swept weight in
`{0.15, 0.3, 0.6}`: **all three underperformed the V3 baseline on every metric** (acc
0.293-0.307 vs 0.310), no fold collapses, just consistently worse. Diagnosis: unlike a
typical multi-task net with a wide shared hidden layer, this model's entire "shared trunk"
is a single scalar by design — the auxiliary head has no independent capacity to exploit and
can only compete with the fine-grained head for how that one number gets shaped, and the two
coarse-class boundaries don't align with the fine-grained loss's own eight thresholds.
Abandoned; code path kept (opt-in, unused by default) for the record.

**Net result of this round:** label smoothing (T=0.15) is the one keeper; feature pruning
and the multi-task head were both tried in good faith and rejected on evidence.

**Phase E (launched 2026-07-06, results pending):** an Optuna retune of V3 with
`label_smoothing_temp` folded in as a joint search dimension (range 0.0-0.4, centered on the
empirically-good T=0.15) alongside the existing `batch_size`/`weight_decay`/`dropout`/
`decay_lr`/`lr`. Uses a new `guitar_all_v3_smooth` feature-set key in
`optuna_guitar_tuning.py` (same 32 V3 columns as `guitar_all_v3`) so it writes to its own
study DB and `guitar/best_hyperparams_guitar_all_v3_smooth.json`, deliberately not
overwriting the frozen `best_hyperparams_guitar_all_v3.json` that the committed
`RESULTS.md` V3 numbers depend on.

### Era 8 — RubricNet V4 and 20-Level Ordinal Training (2026-07-06 to 2026-07-07)

This era expanded RubricNet to V4, introducing new Left-Hand position and barre detection features, and tackled direct 20-level ordinal difficulty training:
- **Stabilizing 20-class target space**: Training directly on raw 1-20 labels initially suffered from severe gradient collapse and loss explosion.
  - *Monotonic Initialization*: In the 20-class target space, random initialization of the final projection layer thresholds caused gradient signals to cancel out. We implemented monotonic initialization (weights set to 1.0, biases set to decreasing steps `-0.5 * i`) to provide a strong inductive bias and restore gradient flow.
  - *Capped vs. Unweighted Loss*: Inverse-frequency class weighting (even when capped) was found to destabilize learning because extremely sparse classes (only 3-5 samples) introduced high-variance, noisy updates. Disabling class weights for `num_classes > 8` and training with an unweighted loss achieved a stable Kendall $\tau$ correlation of 0.5569.
- **Solving the Checkpoint Versioning Gotcha**: Discovered a critical bug where PyTorch Lightning auto-incremented checkpoint files (saving as `split_0-v1.ckpt` instead of overwriting) while the evaluation script loaded the stale, collapsed first run. Added systematic directory cleanup at the start of each seed and tuning trial.
- **Optuna Tuning & Validation**:
  - Updated `optuna_guitar_tuning.py` to support V4 and raw 1-20 target mappings.
  - Executed a 15-trial Optuna sweep specifically for the raw 20-class target space, producing `guitar/best_hyperparams_guitar_all_v4_raw.json` with optimized regularization and label smoothing parameters.
  - Retrained the final 20-level model across 3 seeds × 5 folds. The tuned model improved validation accuracy from **0.168** to **0.3028** and mapped test accuracy to **0.1815** (Coarse 3-class accuracy of **0.4352**), resolving the model collapse.
- **Interpretability Analysis**: Updated `interpret_rubricnet.py` to support V4 and V4 Raw. Generated publication-grade monotonicity and feature importance charts for all four models (V2, V3, V4, V4 Raw) in `guitar/figures/` and populated the results in `RESULTS.md`.

---

## 3. Current state (as of 2026-07-07)

`guitar/RESULTS.md` has four full model generations side by side (V1/V2/V3/V4 × ordinal
regression / decision tree / Random Forest / RubricNet), plus a coarse 3-class table and
an interpretability section for V2, V3, V4, and V4 Raw. The dataset stands at **716 pieces**
(source breakdown: 655 PDF-derived, 49 DADA-GP, 12 GAPS), 8 difficulty classes from
equal-frequency binning of the 1–20 GuitarBurst scale (edges `[1-3, 4-5, 6-7, 8, 9-10,
11-12, 13-15, 16-20]`, class sizes 37–136), with 639/716 (89%) carrying real rhythm data.

**Thesis-relevant headline claim is now earned**: the interpretable, additive RubricNet
model matches or beats an opaque Random Forest on 4 of 6 metrics (balanced accuracy, MAE,
MSE, Kendall's τ), while remaining fully interpretable via per-descriptor scores.

### Hard invariants established along the way (do not violate without deliberate review)
- Never modify `guitar/guitar_splits.json` or its bin edges.
- RubricNet architecture is frozen at `hidden_size=1, num_layers=1` — this is not an
  undertuned hyperparameter, it's the point: one scalar score per descriptor, additive,
  no hidden interactions, or the interpretability story collapses.
- `Reading` and `Max Position` columns in the raw metadata are human annotations
  correlated with the label — excluded as leakage risk, at most mentioned as future work.
- Class weighting already lives inside RubricNet's ordinal loss — never double it up in
  the data loader.
- StandardScaler always fit on train fold only.
- Never delete older (v1/v2) result files or feature columns — needed for the ablation
  narrative and before/after comparisons.

### What's next (per `guitar/FUTURE_WORK_PLAN.md` / most recent session memory)
- ~~Phase D: Optuna retune specifically for the V3 feature set~~ — **done** (this is the
  "3rd run" commit; `guitar/best_hyperparams_guitar_all_v3.json`, trial 51, is what
  `RESULTS.md`'s V3 row already reflects). A correction to earlier session memory, which
  had this listed as still pending.
- Phase E retune (Era 7, in progress): same idea, but with `label_smoothing_temp` folded in
  as a joint search dimension — see Era 7 above for status.
- Rhythm-dependent features already get train-fold-median imputation (confirmed in
  `train_guitar_rubricnet.py`/`baselines.py`, not just a plan); the still-open half of this
  idea is ablating on the rhythm-complete subset only, to see how much of V3's gain is
  attributable to rhythm vs. the fixed base features — not yet done.
- The context-free-vs-context-aware A/B model comparison proposed in
  `FUTURE_WORK_PLAN.md` §4 (train one RubricNet on context-free descriptors, one on the
  context-aware/velocity versions, compare monotonicity and balanced accuracy).
- Phase 7 (optional): finalize the coarse 3-class table as a citable headline number
  comparable to prior guitar-difficulty work.
- Eventually: Phase 6/7 of the original roadmap — interpretability write-up polish and
  drafting the thesis Methods/Evaluation chapters from the now-complete results.

---

## 4. Why RubricNet was initially losing to Random Forest (the core diagnosis)

This is worth calling out on its own because it's the intellectual crux of the middle of
the project. After the first real run, RubricNet trailed RF by ~7 points of raw accuracy.
The `IMPROVEMENT_PLAN.md` diagnosis, later validated by the V3 rewrite, was:

1. **RubricNet is an additive model by architectural constraint** — no descriptor can
   interact with another before the final sum. If the underlying features don't carry
   enough independent signal, no amount of tuning fixes it (confirmed empirically: a
   20-trial Optuna sweep converged to a stable balanced-accuracy plateau regardless of
   learning rate, dropout, batch size).
2. So the fix had to be in the **features themselves**, not the model or the tuning:
   - Half the v1 descriptors were statistically dead (|ρ| < 0.15) — pure noise the
     additive model still had to "score" and sum in.
   - The PDF-derived scores (89% of the dataset) had been extracted through a parser
     that was silently mangling fret numbers — the single biggest hidden problem,
     invisible until the unified `all_xmls` rewrite surfaced it via a Spearman-correlation
     jump on the exact same descriptor definitions.
3. Fixing both (killing dead features + fixing the corrupted parse path + adding
   rhythm-aware and context-aware descriptors once rhythm data was confirmed available
   for 87% of pieces) is what took RubricNet from behind-RF to ahead-of-RF on most
   metrics, without touching the model architecture at all — which is itself a piece of
   evidence for the thesis's actual point: the descriptors, not the model, are the
   difficulty story.

---

## 5. Repo map (where things live today)

```
docs/
  PROJECT_HISTORY.md        this file
  roadmap.md                 phase-by-phase plan written 2026-07-03
  guitar_descriptors.md      v1 descriptor definitions/formulas

data/
  checked_pieces.md          manual spot-checks of GuitarBurst labels vs. your judgment
  midi_alignment_report.csv  align_midi_rhythm.py output: ok/skipped/no-match per piece
  verified_pieces_full.json  consolidated piece metadata

scripts/
  scraping/    pulling GuitarBurst + Mutopia source data
  matching/    reconciling GAPS/DADA-GP/PDF/Mutopia pieces against GuitarBurst labels
  analysis/    duplicate detection, piece stats, distribution plots
  utilities/   format conversion (PDF/GP -> MusicXML), MIDI rhythm alignment, consolidation

guitar/
  guitar_features.py         all descriptor extraction (v1/v2/v3, XML/tokens/PDF parsers)
  prepare_splits.py          the frozen 5-fold split + FEATURE_GROUPS definitions
  baselines.py                ordinal regression / decision tree / random forest
  train_guitar_rubricnet.py  final RubricNet training/eval, seeds x folds
  optuna_guitar_tuning.py    hyperparameter search
  interpret_rubricnet.py     pulls per-descriptor scores for the rubric plots
  IMPROVEMENT_PLAN.md         the v1->v2 diagnosis and phase plan
  FUTURE_WORK_PLAN.md         v3 ideas: label smoothing, barre inference, windowed/context features
  RESULTS.md                  the full V1/V2/V3 comparison table (the thesis results)
  figures/                    monotonicity + feature-importance-comparison plots

rubricnet/
  rubricnet.py                inherited piano architecture: Rubricnet, OrdinalLoss,
                               LogisticRegressionOrdinal, RubricnetSklearn wrapper. As of
                               Era 7: OrdinalLoss takes an opt-in label_smoothing_temp
                               (0 = old hard-step behavior, exact); Rubricnet/
                               LogisticRegressionOrdinal/RubricnetSklearn take an opt-in
                               num_coarse_classes + coarse_loss_weight for the (rejected,
                               but still-available) multi-task coarse-3-class head. Both
                               default off, so every pre-Era-7 caller is unaffected.

extractor/                    inherited piano (CIPI) feature extraction, not used for guitar
                               (this is also where the *piano* fingering solver lives --
                               guitar has no equivalent, tabs already commit to string+fret)

features/guitar_descriptors_v4.csv   current feature table with Left-Hand/barre fixes, 716 rows.

Era 7 & 8 outputs:
  - guitar/best_hyperparams_guitar_all_v4_raw.json (Optuna tuned params for 20-level ordinal training)
  - guitar/rubricnet_results_v4_raw.json (15-run detailed metrics for raw target model)
  - guitar/rubricnet_results_v4.json (15-run detailed metrics for V4 8-class model)
  - guitar/descriptor_scores_fold0_v4.csv & descriptor_scores_fold0_v4_raw.csv (extracted descriptor ranges on the test set)
  - guitar/figures/monotonicity_v4.png & monotonicity_v4_raw.png (V4 monotonicity curves)
  - guitar/figures/importance_comparison_v4.png & importance_comparison_v4_raw.png (V4 feature comparison plots)
```
