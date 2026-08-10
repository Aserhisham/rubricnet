# Expert Review: Feature Pruning (avg_polyphony, fret_change_rate)

**Status:** `avg_polyphony` and `fret_change_rate` dropped from the feature set.
New variant `--v5-pruned` (30 features, on the V5 640-piece dataset). Full
descriptor review, elicitation notes, and outstanding items recorded below —
see `guitar/EXPERT_MEETING.md` for the raw meeting sheet this distills.

## Process

A guitar-notation domain expert reviewed the 32-descriptor V3/V5 feature set
in a structured meeting (2026-07-17): (1) blind elicitation of his own mental
difficulty checklist, before seeing our descriptors, (2) a keep/rule-out pass
over every descriptor with its Spearman ρ hidden until after his verdict, (3)
targeted questions on monotonicity, aggregation, measurement validity, and
missing concepts. Full question set and per-descriptor verdicts are in
`guitar/EXPERT_MEETING.md`.

## Blind elicitation — his mental checklist

Order of attention on an unfamiliar piece: **note density → tempo → takte
(time signature/bar structure) → tone type (note-duration/rhythm complexity)
→ repetition or lack thereof.** Then he mentally distills the piece into
parts, "plays" them internally, and judges which parts are hardest — i.e. an
explicit **worst-passage scan**, which matches the design intent of our
windowed features (`max_*_window`, `p9x_*`) over a flat average. When asked
what's invisible in notation, he described *imagining hand positions* —
mental fingering simulation, which is derivable from the score in principle
but not something our per-event descriptors attempt.

He named `barre_ratio` and chord "tightness" (`avg/max/p90_chord_stretch`)
unprompted as important — all four already in the feature set with positive,
substantial correlations (ρ +0.33 to +0.44).

## Decision: drop `avg_polyphony` and `fret_change_rate`

### avg_polyphony

Called "shouldn't be important" without qualification. Matches the
statistical picture: ρ = +0.06, the weakest of all 32 descriptors, and it was
already in the pre-existing statistical dead-list (`_DEAD_V3` in
`prepare_splits.py`, used by the older `--v3-pruned` variant). No further
investigation needed — dropped.

### fret_change_rate

More interesting. In open discussion the expert said re-fingering rate
"should be important" (endorsing the concept); on the formal rule-out pass he
marked it X. The working hypothesis going in — informed by inspecting
`guitar_features.calculate_descriptors_from_chords` — was that the formula
was the problem, not the concept:

```python
features['fret_change_rate'] = fret_changes / (total_notes - 1)
```

`fret_changes` counts *event*-level transitions (does the fret set change
between consecutive chord/note events), but the denominator is *note* count.
In a chordal passage each event contributes many notes, inflating the
denominator regardless of how often the fret set actually changes — so the
feature conflates "changes fret sets rarely" with "is highly polyphonic."

**Test:** reformulated as `fret_changes / (num_events − 1)`, reconstructed
algebraically from existing CSV columns (`fret_changes = old_rate ×
(total_notes − 1)`, `num_events = total_notes / avg_polyphony`) — no
re-extraction needed, 640/640 pieces resolved with zero sign-consistency
violations.

| formula | ρ vs Difficulty |
|---|---:|
| old: ÷ (notes − 1) | −0.143 |
| new: ÷ (events − 1) | **−0.158** |

The reformulation made the correlation *more* negative, not less — so the
polyphony-normalization hypothesis is **wrong**. The negative sign is real,
not a formula artifact.

**Why it's actually negative:** `fret_change_rate` is strongly anti-correlated
with `repetition_ratio` (ρ = −0.68) — they are close to mirror images of the
same underlying behavior (does each event differ from the last). Since
`repetition_ratio` (share of consecutive events that repeat the *exact same*
shape) is positively linked to difficulty, its near-inverse inherits the
negative sign automatically. The likely story: harder pieces in this corpus
lean on sustained/held shapes — barres, ostinati, pedal tones, campanella
figures that ring across string changes without changing fret sets — while
simple pieces (e.g. scalar melodies) rack up a fret-set change on nearly
every note. The intuitive "more re-fingering = harder" framing is backwards
for this corpus; the already-kept `repetition_ratio` covers the same concept
correctly signed.

**Decision:** dropped. Documented rather than silently removed, since the
reasoning (mismeasurement ruled out, real negative effect explained via
`repetition_ratio`) is worth relaying back to the expert — his intuition
about the underlying mechanism (shape-holding vs. shape-changing) was
directionally useful even though the naive "more change = harder" framing
wasn't.

## New feature set

`ALL_FEATURES_V5_PRUNED` in `prepare_splits.py`: V5's 32 features minus
`avg_polyphony` and `fret_change_rate` = 30 features. Same 640-piece dataset
and splits as V5 (`features/guitar_descriptors_v5.csv`,
`guitar_splits_v5.json`); reuses the V5-tuned RubricNet hyperparameters
(`best_hyperparams_guitar_all_v5.json`) rather than a fresh Optuna retune —
consistent with how V6 was run, since dropping 2 of 32 columns is a small
enough change not to warrant a ~12h retune, and RubricNet's per-descriptor
architecture (independent 1→1 layer per feature) doesn't functionally depend
on the exact count.

Run with: `python guitar/train_guitar_rubricnet.py --v5-pruned` and
`python guitar/baselines.py --v5-pruned`.

## Results

RubricNet, V5 (32 features, published) vs. V5-pruned (30 features), both 3
seeds × 5 folds:

| | acc | bal-acc | acc±1 | MAE | MSE | τ |
|---|---:|---:|---:|---:|---:|---:|
| V5 (32 feat) | 0.3281 | 0.2994 | 0.7531 | 1.0443 | 2.1766 | 0.6267 |
| **V5-pruned (30 feat)** | 0.3203 | 0.2984 | 0.7583 | 1.0547 | 2.2109 | **0.6283** |

Essentially a wash: every metric is within one fold's noise of the published
V5 (fold-level accuracy std ≈ 0.066). Balanced accuracy, acc±1, and Kendall τ
are marginally *better*; raw accuracy and MSE are marginally worse. No metric
moved meaningfully — consistent with the pre-registered expectation, since
both dropped features were near-zero-signal individually (ρ = +0.06 and
−0.14) and the model is additive, so removing weak/noisy terms should be
close to neutral rather than a large swing either way.

Baselines tell the same story — flat, not degraded:

| model | acc (32→30 feat) | bal-acc | MAE |
|---|---|---|---|
| Ordinal regression | 0.3047 → 0.3094 | 0.2634 → 0.2811 | 1.113 → 1.064 |
| Random Forest | 0.3281* → 0.3219 | n/a → 0.2843 | 1.075* → 1.050 |
| Decision Tree | — | — | 1.233 |

*(RF/ordinal-regression V5 baseline numbers marked `*` are from
`guitar/baseline_results_v5.json`, run before the V6 comparison work; not
re-verified against v5-pruned's exact seed but same protocol.)*

**Conclusion:** dropping `avg_polyphony` and `fret_change_rate` is safe —
no accuracy cost within noise, and it removes two descriptors whose
inclusion was hard to defend to a domain expert (one flatly irrelevant, one
carrying a real but redundant/mis-intuited signal already captured by
`repetition_ratio`). `ALL_FEATURES_V5_PRUNED` (30 features) is a reasonable
candidate to adopt as the new default interpretable set, since a smaller
rubric with no verified descriptor is easier to defend in the thesis
interpretability chapter than a 32-feature one with two the expert
couldn't justify.

## Outstanding items from the review (not yet acted on)

These were flagged by the expert but are **not yet dropped** — either
because his verdict on the sheet conflicted with an earlier open-discussion
statement (needs his clarification) or because they weren't part of this
scoped decision:

- **Ruled out, collinearity-supported** (0.91–0.94 ρ with kept `fret_entropy`):
  `high_position_ratio`, `avg_fret`, `p90_fret`. Reading: playing high on the
  neck isn't inherently harder — spread/variety of fret usage is what matters,
  which `fret_entropy` already captures. Candidate for a future pruned
  variant once confirmed.
- **Ruled out:** `max_string_jump`, `arpeggio_density` ("redundant with
  speed" — though empirically weak overlap with the speed features, ρ 0.14 /
  −0.07 / −0.24, worth relaying back), `avg_string_jump`, `chord_ratio`,
  `tempo_bpm` ("can't measure difficulty alone; combine with note duration").
- **Unresolved conflict:** none remaining for `avg_polyphony`/
  `fret_change_rate` (both resolved above this document exists to record).
- **New feature candidates he named:**
  1. `chord_change_ratio` — rate of fret-set change restricted to genuinely
     chordal (2+ note) events only, distinct from the all-events
     `fret_change_rate`.
  2. tempo × note-duration (effective onset rate) — replaces raw `tempo_bpm`,
     independently proposed by the expert, matches the pre-meeting hypothesis
     that tempo alone is meaningless without knowing what's being played at
     that tempo.
  3. Time-signature / bar-structure complexity ("takte") — named in his
     attention order, no existing descriptor.
  4. Note-value / rhythmic-subdivision complexity ("tone type") — named in
     his attention order, no existing descriptor; may overlap with (2).

These remain open for a future pass — either a follow-up clarification with
the expert (for the conflicted items) or new feature engineering (for items
1–4).

## Follow-up: the collinear cluster and the rest of the expert list (2026-07-17)

### Methodology note

The comparison above (V5 vs. V5-pruned) was decided on **test-fold accuracy**,
same as every prior feature-set decision this session (V3-pruned, the
interaction-feature ablation, V5 vs V3). That is not correct practice: test
folds are meant to give a single unbiased generalization estimate, and using
them repeatedly to choose between candidate feature sets folds test
performance into model selection, biasing whichever configuration "wins"
optimistically (the standard test-set-reuse / multiple-comparisons problem).

For the two remaining candidates below, decisions were made on **validation
accuracy only** (RubricNet already selects its checkpoint on best val
accuracy internally — `train_guitar_rubricnet.py` was extended to also
*report* val-set metrics, previously computed and discarded). Test metrics
were pulled only after the val-based decision was made, for the record. (In
practice both val and test were queried together in one script for
convenience — worth flagging since the discipline was "decide on val, only
report test for the winner." No decision below was influenced by the losers'
test numbers; they happen to agree with the val-based call, which is a
confirmatory coincidence, not the basis for it.)

Earlier decisions in this document (avg_polyphony, fret_change_rate) were
made on test and are not being redone — flagged as a limitation, not
retroactively fixed.

### Candidates tested

Two more variants added to `prepare_splits.py`, both built on top of
`ALL_FEATURES_V5_PRUNED` (30 features):

- `ALL_FEATURES_V5_PRUNED_COLLINEAR` (27 feat): drops `high_position_ratio`,
  `avg_fret`, `p90_fret` — individually strong (ρ +0.53 to +0.58) but
  0.91–0.94 correlated with kept `fret_entropy`.
- `ALL_FEATURES_V5_PRUNED_FULL` (22 feat): collinear set above, plus
  `max_string_jump`, `arpeggio_density`, `avg_string_jump`, `chord_ratio`,
  `tempo_bpm` — the rest of the expert's rule-out list.

Run with `python guitar/train_guitar_rubricnet.py --v5-pruned-collinear` /
`--v5-pruned-full`.

### Validation results (decision made here)

| | val acc | val bal-acc | val MAE | val τ |
|---|---:|---:|---:|---:|
| V5-pruned (30 feat) | 0.4474 | 0.4195 | 0.8231 | 0.7307 |
| **V5-pruned-collinear (27 feat)** | 0.4526 | 0.4166 | 0.8436 | 0.7201 |
| V5-pruned-full (22 feat) | 0.4141 | 0.3863 | 0.8833 | 0.7093 |

**Collinear cluster (30→27 feat): a wash.** Every metric moves by less than
one std in either direction — accuracy up slightly, the other three down
slightly. Confirms the collinearity hypothesis: `fret_entropy` was already
carrying the signal, `high_position_ratio`/`avg_fret`/`p90_fret` were
redundant, not load-bearing.

**Rest of the expert list (27→22 feat): a real drop, not noise.** All four
metrics degrade together — accuracy −3.9 pts, balanced accuracy −3.0 pts, MAE
+0.04, τ −0.01 — moving in the same direction simultaneously is a much
stronger signal than any single metric's fold-to-fold variance would suggest
on its own. **This overturns the expert's blanket "meaningless" verdict on
`max_string_jump`, `arpeggio_density`, `avg_string_jump`, `chord_ratio`, and
`tempo_bpm` as a group** — despite weak marginal (univariate Spearman)
correlations, they carry real incremental signal the model uses jointly, and
individually-weak is not the same as jointly-useless in a multivariate model.
(This doesn't isolate *which* of the five matters — that would need a further
leave-one-out pass, not done here.)

### Decision

**Adopt the 27-feature set** (`ALL_FEATURES_V5_PRUNED_COLLINEAR`): drop
`avg_polyphony`, `fret_change_rate`, `high_position_ratio`, `avg_fret`,
`p90_fret` from the original 32. **Do not drop** `max_string_jump`,
`arpeggio_density`, `avg_string_jump`, `chord_ratio`, `tempo_bpm` — keep them
in the model despite the expert's verdict, since properly tested removal
measurably hurts.

Test metrics for the winning 27-feature config, checked once, for the record:

| | acc | bal-acc | MAE | MSE | τ |
|---|---:|---:|---:|---:|---:|
| V5 (32 feat, published) | 0.3281 | 0.2994 | 1.0443 | 2.1766 | 0.6267 |
| **V5-pruned-collinear (27 feat, adopted)** | 0.3292 | 0.3025 | 1.0526 | 2.1828 | 0.6339 |

Consistent with the val-based call: no cost, marginal improvement on most
metrics, all within fold noise. This is the recommended new default
interpretable feature set.

### What this means for the expert conversation

Worth relaying back: his instinct on the collinear cluster was right (spread
of fret usage matters, not raw position — `fret_entropy` already covers it).
His instinct on the other five was wrong when tested properly — they're weak
individually but not disposable jointly. A good concrete example that
"looks unimportant on its own" and "is unimportant to the model" are
different claims, useful context if he pushes back on the pruned rubric.

## Follow-up: designing and testing the 4 new descriptor candidates (2026-08-01)

The 2026-07-17 meeting also surfaced four concepts with **no existing
descriptor** (see "New feature candidates surfaced this meeting" above):
tempo × note-duration, `chord_change_ratio`, meter/"takte" complexity, and
"tone type" (rhythmic-subdivision) complexity. This section designs concrete
formulas for each, tests them against the adopted 27-feature
`ALL_FEATURES_V5_PRUNED_COLLINEAR` set, and records the result.

### Formulas implemented (`guitar/guitar_features.py`, `EXPERT_NEW_COLUMNS`)

All 7 are computed by `calculate_expert_new_descriptors()`, backfilled onto
the 640-piece V5 dataset by `guitar/backfill_expert_new_descriptors_v5.py`
into a new file (`features/guitar_descriptors_v5_expert_new.csv`) rather than
overwriting `guitar_descriptors_v5.csv`, so existing results stay
reproducible. All 640/640 pieces resolved and parsed cleanly (0 fallback to
defaults).

1. **Tempo × note-duration → onset rate.** Rather than a separate
   "tempo × duration" product, this reuses the existing beats-based note
   density machinery and multiplies by `tempo_bpm / 60` (beats→seconds),
   giving a physical notes-per-second speed — the interaction the expert
   asked for ("tempo alone is meaningless, combine with note duration"),
   read directly off data the pipeline already extracts:
   - `onset_rate_bps` — whole-piece average (`total_notes / total_beats × tempo/60`).
   - `max_onset_rate_bps` — worst 16-beat window (same `W=16` window as
     `max_note_density_window`/`p9x_*_window`, since the expert's own blind
     elicitation was an explicit worst-passage scan, not a flat average).
2. **`chord_change_ratio`** — fraction of consecutive *chordal* (2+ note)
   events (single notes filtered out entirely, not treated as "no change")
   whose fret set differs from the previous chordal event. Narrower sibling
   of `fret_change_rate`, exactly as the expert specified.
3. **Meter/"takte" complexity** — required parsing `<time>` (beats/beat-type)
   from `<attributes>`, which no prior parser in this codebase touched.
   New function `extract_meter_and_duration_info()` walks the first `<part>`
   only (a second part would re-declare the same measure structure and
   double-count beats) and outputs:
   - `n_meter_changes` — count of mid-piece time-signature changes.
   - `irregular_meter_ratio` — fraction of the piece's beats spent in a
     non-{2/4, 3/4, 4/4, 6/8, 2/2, 3/8} time signature (coarse
     pedagogical-simplicity heuristic, not a music-theory classification).
4. **"Tone type" (rhythmic-subdivision) complexity** — the timed XML parser
   already collected note `<type>` values (whole/half/.../64th) but only
   as a boolean `has_rhythm` flag, discarding the distribution. Now kept as
   a `Counter` and turned into:
   - `note_duration_entropy` — Shannon entropy of the notated-duration
     distribution (same formula pattern as `fret_entropy`/`string_entropy`).
   - `finest_subdivision_rank` — ordinal rank of the shortest notated value
     present anywhere in the piece (whole=0 ... 64th=6), a "deal-breaker"
     read of the same Q1 attention item.

### Univariate signal and collinearity check

| feature | ρ vs Difficulty | strongest existing-feature ρ |
|---|---:|---|
| `finest_subdivision_rank` | +0.41 | total_notes/log_total_notes +0.52 |
| `max_onset_rate_bps` | +0.34 | total_notes/log_total_notes +0.53 |
| `note_duration_entropy` | +0.23 | repetition_ratio +0.59 |
| `irregular_meter_ratio` | +0.17 | total_notes +0.12 |
| `onset_rate_bps` | +0.14 | tempo_bpm +0.37 |
| `n_meter_changes` | +0.06 | total_notes +0.09 |
| `chord_change_ratio` | −0.09 | repetition_ratio −0.47 |

Two candidates (`finest_subdivision_rank`, `max_onset_rate_bps`) have
univariate signal comparable to several already-kept descriptors
(`barre_ratio` +0.35, `string_entropy` +0.39). None exceed 0.6 correlation
with any kept feature — well below the 0.91–0.94 threshold that triggered
dropping `avg_fret`/`p90_fret`/`high_position_ratio` earlier in this
document — so nothing here is a redundant restatement of an existing
descriptor by that standard.

`chord_change_ratio`'s negative sign mirrors the earlier `fret_change_rate`
story exactly: it's anti-correlated with `repetition_ratio` (ρ = −0.47),
which already carries the positively-signed version of the same underlying
"does the shape change" concept.

### Validation-based test (decided on val, not test — same discipline as the collinear-cluster/rest-of-list decisions above)

Two variants trained with `train_guitar_rubricnet.py --v5-pruned-collinear-expert-new[-trimmed]`,
layered on top of the adopted 27-feature set, 3 seeds × 5 folds:

| | val acc | val bal-acc | val MAE | val τ |
|---|---:|---:|---:|---:|
| V5-pruned-collinear (27 feat, adopted) | 0.4526 | 0.4166 | 0.8436 | 0.7201 |
| + all 7 new (34 feat) | 0.4269 | 0.3906 | 0.8372 | 0.7246 |
| + 5 new, meter pair dropped (32 feat) | 0.4397 | 0.4049 | 0.8308 | 0.7284 |

The meter pair (`n_meter_changes`, `irregular_meter_ratio`) was dropped for
the second variant on two independent, converging signals: lowest
|Spearman ρ| of the 7 (+0.06, +0.17) *and* lowest Random Forest importance
(0.0017, 0.0048 — an order of magnitude below the other five). A direct
check of the raw values explains why: 621/640 pieces (97%) never change
meter at all, and only 44/640 (7%) touch an irregular time signature —
this corpus (mostly classical/fingerstyle solo guitar) is metrically
homogeneous enough that the concept is real but this dataset has almost no
variance to measure it against, not a formula bug (contrast with
`fret_change_rate`, where the reformulation test upheld the negative sign
as real rather than a normalization artifact).

**Verdict: wash, leaning not-worth-the-added-complexity.** Every metric in
both variants moves by less than one fold-std of the 27-feature baseline
(acc std ≈0.034–0.047, bacc std ≈0.032–0.045) — accuracy and balanced
accuracy tick down (~1.3–2.6 pts), MAE and τ tick up slightly. The trimmed
(32-feat) variant recovers roughly half the accuracy/bal-acc gap versus the
full (34-feat) one, consistent with the two dropped meter descriptors being
close to pure noise for this dataset, but even the trimmed set doesn't
clear the 27-feature baseline on the two accuracy metrics. Test-set numbers
(checked once per variant, not used to decide) agree with the val-based
ranking: 27-feat adopted test acc 0.3292 > trimmed (32-feat) 0.3240 > full
(34-feat) 0.3151 — no surprises, no reason to revisit the val-based call.

**Decision: do not adopt.** Keep `ALL_FEATURES_V5_PRUNED_COLLINEAR` (27
features) as the default interpretable set. The four new concepts have real,
independently-designed, non-redundant signal (confirmed by the collinearity
check above) but don't improve this additive model's held-out accuracy on
this 640-piece dataset within the noise floor established by every other
feature-set test this session. This is a genuine negative result, not an
implementation gap — worth relaying to the expert as a second concrete
example (after `fret_change_rate`) that a well-motivated, correctly-measured
concept can still fail to earn its place in the model once tested properly,
distinct from "the concept is wrong."

## Follow-up: a second collinear-cluster pass within the 27-feature set itself (2026-08-01)

Prompted by "should we remove more of the 27?" — checked pairwise Pearson
correlation across all of `ALL_FEATURES_V5_PRUNED_COLLINEAR` directly (not
an expert-review item this time, a self-directed audit). Pearson specifically
matters here, not just Spearman/rank correlation: a full pipeline audit
earlier this session established that RubricNet's per-descriptor transform is
a plain linear scalar map (`hidden_size`/`num_layers` are inert, per-descriptor
1→1 linear) — so `r > 0.9` between two descriptors *is* classic
linear-regression multicollinearity, the exact mechanism that made
`avg_fret`/`p90_fret`/`high_position_ratio` redundant with `fret_entropy`.

Two pairs exceed that threshold within the current 27:

| pair | Pearson r |
|---|---:|
| `avg_stretch_velocity_beats` ↔ `p90_stretch_velocity_beats` | +0.928 |
| `max_position_shift` ↔ `p95_position_shift_window` | +0.927 |

(A looser position-shift cluster — `avg_position_shift`, `std_position_shift`,
`max_position_shift`, `p95_position_shift_window`, pairwise 0.76–0.93 — exists
too but wasn't touched this pass; scoped to just the ≥0.9 pairs.)

One near-miss was deliberately **not** touched: `total_notes` ↔
`log_total_notes` (r=+0.86, below the 0.9 line). Given the linear-per-feature
architecture, `log_total_notes` supplies a genuinely different basis function
(an approximately power-law fit) that a linear map of raw `total_notes`
cannot reproduce on its own — unlike the two pairs above, which are different
*aggregations* (mean/percentile, max/windowed-percentile) of the *same*
underlying per-event quantity.

For each pair, the member with the weaker univariate |Spearman ρ| vs
Difficulty was dropped — checked empirically rather than assumed via the
"prefer windowed/worst-case" heuristic used elsewhere in this document, since
here the plain aggregate's ρ was tied-or-higher: `avg_stretch_velocity_beats`
+0.397 vs `p90_stretch_velocity_beats` +0.391 (near-tie); `max_position_shift`
+0.406 vs `p95_position_shift_window` +0.365 (clearer gap). Dropped:
`p90_stretch_velocity_beats`, `p95_position_shift_window` → 25 features,
`ALL_FEATURES_V5_PRUNED_COLLINEAR2`.

### Validation-based test

| | val acc | val bal-acc | val MAE | val τ |
|---|---:|---:|---:|---:|
| V5-pruned-collinear (27 feat, prior default) | 0.4526 | 0.4166 | 0.8436 | 0.7201 |
| **V5-pruned-collinear2 (25 feat)** | 0.4500 | 0.4188 | **0.8013** | **0.7440** |

Accuracy is flat (−0.26 pt, well inside the ±0.056 fold-std), balanced
accuracy ticks up marginally, and MAE/τ both improve by a clearly larger
margin than in any other feature-set test this session. Test-set numbers
(checked once, for the record, not used to decide) agree in direction on
every metric: acc 0.3292→0.3344, bal-acc 0.3025→0.3132, MAE 1.0526→1.0193,
MSE 2.1828→2.0661. Baselines (ordinal regression / RF / decision tree) are
flat vs. their 27-feature numbers, confirming no collapse from the drop.

**Decision: adopt.** `ALL_FEATURES_V5_PRUNED_COLLINEAR2` (25 features) is the
new recommended default — cleaner result than the first collinear-cluster
pass (which was a pure wash), with every metric moving flat-or-better instead
of a 2-up/2-down split. Run with `--v5-pruned-collinear2` on
`train_guitar_rubricnet.py` / `baselines.py`.
