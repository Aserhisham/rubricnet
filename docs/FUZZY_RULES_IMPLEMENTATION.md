# Fuzzy Rule-Based Classifiers: Implementation Notes

This document explains what was built, why, and how it fits into the rest of
the guitar difficulty pipeline. It supplements `guitar/RESULTS.md` (numbers)
and `thesis/chapters/4_method.tex` / `thesis/chapters/5_evaluation.tex`
(the polished write-up) with the implementation reasoning that doesn't belong
in the thesis itself.

## 1. Motivation

`fuzzy.txt` (Heerde, Vatolkin & Rudolph, *"Comparing Fuzzy Rule Based
Approaches for Music Genre Classification"*, EvoMUSART 2020) describes three
fuzzy rule-based classifiers as an interpretable alternative to black-box
models for music classification. The thesis already argues that RubricNet is
preferable to a Random Forest because it's interpretable; adding fuzzy rules
gives it a second, independently-sourced interpretable baseline to compare
against — one that reads rules in natural language rather than summing
learned per-descriptor scores. The goal was to implement two of the paper's
three methods (the evolutionary rule-base approach was excluded — it's
nondeterministic, slowest, and produced the least readable rules in the
source paper) and integrate the results into the thesis as peer rows in the
main comparison table.

## 2. What was implemented

### 2.1 Complete Search of Primitive Rules

A primitive rule has the form *"if descriptor X is linguistic-term T then
class C"*. For every (descriptor, term, class) triple, a relevance score is
computed on the training fold:

```
R(C, X, T) = P(X is T | C) · (1 − P(X is T))
```

The first factor rewards terms that are typical of class C; the second
penalizes terms that are common across the whole corpus (so a term nearly
everyone satisfies scores low even if it's common within C too). Per class,
rules are ranked by relevance and the top-`m` rules' truth values are
averaged; a piece is predicted as whichever class's top-`m` rules are, on
average, most true (one-vs-all argmax). `m` is a single hyperparameter,
swept from 1 to the full rule pool size (32 descriptors × 5 terms = 160) and
selected per fold on the validation split.

### 2.2 Fuzzy Pattern Trees (FPT)

One tree per class. Leaves are fuzzy statements (optionally negated); inner
nodes are fuzzy operators — `and` (min), `or` (max), `avg` (mean of two
children). Induction is deterministic and greedy: start from the single best
statement (lowest RMSE against the one-vs-all 0/1 target), then repeatedly
replace one leaf with an operator node whose children are the old leaf and a
new candidate statement, always picking whichever (leaf, operator,
statement) combination most reduces error. Induction stops when the depth
limit `d_max` is reached, no legal extension remains, or the best available
extension would make things worse by more than a factor `(1 + γ)`. `d_max`
is swept over `{1, ..., 5}` and selected per fold on validation, exactly
like `m` for the complete search; `γ = 0.05` is fixed (the paper never
states a value).

### 2.3 Shared fuzzification

Both methods need raw descriptor values converted into truth degrees for
five linguistic terms (very low, low, moderate, high, very high). This
happens in two steps:

1. **Normalize to [0, 1].** Two methods are implemented:
   - `cdf` (default): each feature is mapped through its own empirical CDF
     fit on the training fold — a value becomes the fraction of training
     pieces at or below it. Chosen as the default because several V3
     descriptors are heavy-tailed (`total_notes`, window maxima); plain
     min-max would compress most pieces into the lowest term.
   - `minmax`: literal `(x - min) / (max - min)`, closer to the paper's
     description. Kept as a `--norm minmax` flag for comparison.
   Values outside the training range (in val/test) are clipped to [0, 1].
2. **Triangular membership.** Five evenly spaced triangles centered at
   0, 0.25, 0.5, 0.75, 1, each with half-width 0.25, so adjacent terms
   always sum to exactly 1 at any point.

## 3. Design decisions that deviate from the paper (and why)

These were flagged explicitly during planning and validated against results
after the fact:

| Decision | Choice | Why |
|---|---|---|
| Ordinal handling | None — faithful one-vs-all argmax | Keeps the comparison to RubricNet honest: the gap in the results *is* the value of RubricNet's ordinal-aware design, not a confound from a different decision rule. |
| Hyperparameter selection | Per-fold validation split, balanced accuracy | Matches how every other model in this pipeline (RubricNet, ordinal regression) uses the val split, rather than the paper's train-error selection. |
| FPT loss under class imbalance | Class-balanced RMSE by default | Every one-vs-all target is a 5–19% minority (class sizes 37–136 of 716); unweighted RMSE is minimized by predicting near-zero everywhere. Balancing means a constant predictor's best achievable output is 0.5, not "always negative." An ablation (`--plain-rmse`) confirmed this barely changes results here — the imbalance turned out to be mild enough not to matter — but it's kept as the safer default. |
| Negated leaves | Included, `--no-negation` to disable | The paper found negation was *never* selected by their induction algorithm. On this dataset the opposite happened: 70% of induced trees (28/40) use at least one negated leaf, and disabling negation costs the FPT about 3 accuracy points and 0.10 Kendall's τ. This ended up being one of the more interesting findings and is written up in the thesis. |

## 4. Files

### New code
- **`guitar/fuzzy_rules.py`** — pure numpy, no CLI, no torch import (kept
  separate from `guitar/baselines.py` deliberately, since that module pulls
  in `rubricnet.rubricnet` and torch, which these methods don't need).
  - `triangular_memberships`, `Fuzzifier`
  - `CompleteSearchClassifier` (`fit`, `scores`, `predict`,
    `rules_for_class`, static `sweep_m` for the cheap m-sweep)
  - `FPTNode`, `FuzzyPatternTreeClassifier` (`fit`, `scores`, `predict`,
    `tree_expression`, `tree_dict`, `is_constant`)
- **`guitar/run_fuzzy_baselines.py`** — argparse harness, modeled on the
  fold loop in `guitar/thesis_extra_results.py`: reads
  `guitar/guitar_splits.json`, V3 features from
  `features/guitar_descriptors_v3.csv`, does train-fold median imputation,
  fits/selects/refits per fold, computes the same six metrics as
  everywhere else in the pipeline (accuracy, balanced accuracy, acc±1, MAE,
  MSE, Kendall τ — copied from `guitar/train_guitar_rubricnet.py`'s
  `compute_metrics`), and writes results + a human-readable rules dump.
  Flags: `--norm {cdf,minmax}`, `--gamma`, `--plain-rmse`, `--no-negation`,
  `--out`, `--dump`.
- **`tests/test_fuzzy_rules.py`** — 10 unit tests: partition-of-unity of
  memberships, fuzzifier clipping/monotonicity, complete search on a
  synthetic separable dataset, FPT `d_max=1` matching a brute-force best
  single statement, the balanced-RMSE-of-a-constant-predictor identity
  (0.5), and determinism across repeated fits.

### Outputs (generated by running the harness)
- `guitar/fuzzy_results_v3.json` — primary run (CDF norm, balanced RMSE,
  negation enabled): per-fold metrics + mean/std summary + config block.
- `guitar/fuzzy_results_v3_minmax.json`, `_plain_rmse.json`,
  `_no_negation.json` — the three ablations, each writing to its own file
  so they don't clobber the primary result.
- `guitar/fuzzy_rules_dump_v3*.json` (one per run) — per fold, per class:
  top-10 complete-search rules as text with relevance scores, and the FPT's
  logical expression + tree dict + whether negation was used. This is what
  the thesis's qualitative rules table and FPT expression figure were
  pulled from directly.

### Thesis edits
- `thesis/references.bib` — 6 new entries: `zadeh1965fuzzy`,
  `heerde2020comparing`, `vatolkin2015interpretable`, `senge2011topdown`,
  `senge2015fast`, `huang2008patterntrees`.
- `thesis/chapters/3_preliminaries.tex` — extended §Baseline Models
  ("Three" → "Five" model families) with a paragraph introducing fuzzy
  sets/statements/rules.
- `thesis/chapters/4_method.tex` — new §Fuzzy Rule-Based Comparison Methods
  (`sec:fuzzy_method`) covering fuzzification, the complete search
  relevance formula, and FPT induction/stopping/balancing, mirroring the
  level of detail used for the rest of the method chapter.
- `thesis/chapters/5_evaluation.tex`:
  - Two new rows in the main comparison table (`tab:8class_results`, V3
    block), between Random Forest V3 and RubricNet V3.
  - A fourth "observation" paragraph explaining *why* the fuzzy methods
    trail RubricNet (nominal vs. ordinal treatment, not weak rules).
  - A new qualitative subsection (`sec:fuzzy_qualitative`) with a table of
    the top-5 rules for the easiest and hardest classes, the fold-0 FPT
    expression for class 7, and the negation-usage finding.
  - A new auxiliary-experiments subsection (`sec:fuzzy_ablation`)
    reporting the three ablations (normalization, balanced vs. plain RMSE,
    negation on/off) with numbers and interpretation.
- `guitar/RESULTS.md` — two new rows in the 8-class table, plus a new
  "Fuzzy Rule-Based Classifiers (V3)" section with the full ablation table
  and reproduction commands.

## 5. Results

Primary run (CDF normalization, class-balanced RMSE, negation enabled),
5-fold mean ± std:

| Method | Accuracy | Balanced Acc. | Acc ±1 | MAE | MSE | Kendall τ |
|---|---|---|---|---|---|---|
| Complete Search | 0.264 ± 0.044 | 0.236 ± 0.044 | 0.617 ± 0.068 | 1.433 ± 0.168 | 3.803 ± 0.597 | 0.464 ± 0.075 |
| Fuzzy Pattern Tree | 0.268 ± 0.047 | 0.252 ± 0.041 | 0.623 ± 0.062 | 1.469 ± 0.207 | 4.233 ± 0.951 | 0.484 ± 0.063 |

For context, this lands between Ordinal Regression V1 (0.214 accuracy) and
Random Forest V3 (0.331 accuracy) — a sensible place for a simple,
deterministic, fully-interpretable method with no ordinal awareness. The
strongest correctness signal came from reading the actual rules: fold 0's
top rules for class 0 (easiest) all cite *low* fret position, stretch, and
position-shift; class 7's (hardest) cite the *high* counterparts plus high
note count. That's an independent (non-neural) confirmation of exactly the
descriptor story RubricNet's own influence/monotonicity analysis tells.

## 6. Verification performed

- All 10 unit tests pass (`.venv/bin/python -m pytest tests/test_fuzzy_rules.py`).
- Full 5-fold run completes in ~10 seconds.
- Re-running the harness twice produces byte-identical JSON output
  (confirmed via `diff`) — both methods are genuinely deterministic, no
  hidden randomness.
- Sanity-checked accuracy/balanced-accuracy/τ ranges against the existing
  baselines in `guitar/baseline_results_v3.json` and
  `guitar/rubricnet_results_v3.json`.
- Read the actual induced rules (not just metrics) to confirm they encode
  real difficulty signal rather than degenerating to noise.
- Ran three ablations (`--norm minmax`, `--plain-rmse`, `--no-negation`) to
  check the design decisions in §3 actually mattered where claimed (they
  did, for negation; they didn't much, for balanced-vs-plain RMSE, which is
  itself a useful finding — the class imbalance here is milder than
  anticipated).
- Rebuilt the full thesis PDF with `latexmk -pdf`: compiles cleanly to 64
  pages, no undefined references or citations (checked via
  `grep -i undefined main.log` and confirming all 6 new bib keys appear in
  `main.bbl`).

## 7. How to reproduce

```bash
# primary result
.venv/bin/python guitar/run_fuzzy_baselines.py

# ablations
.venv/bin/python guitar/run_fuzzy_baselines.py --norm minmax \
    --out guitar/fuzzy_results_v3_minmax.json --dump guitar/fuzzy_rules_dump_v3_minmax.json
.venv/bin/python guitar/run_fuzzy_baselines.py --plain-rmse \
    --out guitar/fuzzy_results_v3_plain_rmse.json --dump guitar/fuzzy_rules_dump_v3_plain_rmse.json
.venv/bin/python guitar/run_fuzzy_baselines.py --no-negation \
    --out guitar/fuzzy_results_v3_no_negation.json --dump guitar/fuzzy_rules_dump_v3_no_negation.json

# tests
.venv/bin/python -m pytest tests/test_fuzzy_rules.py -v

# thesis rebuild
cd thesis && latexmk -pdf -interaction=nonstopmode main.tex
```

## 8. What's not done / possible follow-ups

- The evolutionary rule-base approach from the paper (§3.4) was
  deliberately excluded per the original scoping decision — nondeterministic,
  needs multi-seed repetition, and produced the least interpretable rules
  in the source paper.
- Selected `m` for complete search varies quite a bit across folds (11 to
  150, out of a pool of 160) — noticeably less stable than the paper's own
  m=30 finding. This is likely a consequence of the small per-fold
  validation split (~10% of 716 pieces) making balanced-accuracy selection
  noisy; worth a sentence if this becomes a point of scrutiny in defense,
  but wasn't treated as a bug since predictions on the actual test set
  remained sensible across folds.
- No visual (TikZ) rendering of the FPT trees was added — the thesis uses a
  typeset logical expression instead, which was judged sufficient at the
  observed tree sizes (depth ≤ 5) and avoids a chunk of LaTeX/TikZ
  plumbing for a single-tree illustration.
