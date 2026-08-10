"""Do generic symbolic features add anything to the hand-crafted guitar descriptors?

The experiment
--------------
Three feature sets, one protocol, identical folds:

    A. jSymbolic only  -- generic instrument-agnostic symbolic features (music21's
                          jSymbolic reimplementations), extracted by
                          `guitar/extract_jsymbolic_features.py`.
    B. Guitar only     -- the thesis's 25-descriptor V5-pruned-collinear2 set.
    C. Both            -- A + B concatenated.

This answers a question the thesis cannot currently answer: *was the hand-crafted
guitar-specific descriptor work necessary, or would an off-the-shelf symbolic feature
library have done the job?* jSymbolic features cannot encode fret or string choice --
they are computed from pitch, duration and voice structure, and a pitch does not say
which of up to six fretboard positions produced it. So the comparison directly measures
what the guitar-specific representation buys.

Interpretation guide, fixed before running to avoid post-hoc storytelling:
  * B >> A   -> guitar-specific descriptors reach information generic features cannot.
                Justifies the descriptor engineering by measurement, not assertion.
  * C ~= B   -> generic features add nothing on top. Strengthens the final set.
  * C >> B   -> generic features carry signal the thesis is missing; the honest response
                is to adopt the useful ones, not to bury the result.
  * A >= B   -> an awkward result that would need reporting plainly.

Model choice
------------
Random Forest, matching the existing per-dimension ablation
(`AIM-thesis/chapters/05-evaluation.tex`, Table `tab:dimension_ablation`), which uses the
forest specifically "to isolate the feature groups from RubricNet's training variance".
Using the same model keeps the two ablations directly comparable. Running this on
RubricNet instead would confound feature-set quality with the additive architecture's
sensitivity to input count: with ~150 generic features against 25 descriptors, a model
that must score and sum every input is penalised for dimensionality alone.

Provenance screening
--------------------
Some jSymbolic features encode *how a file was produced* rather than what the music is:
instrumentation features read the MusicXML instrument declaration, which is present for
DadaGP-derived scores and absent for PDF-derived ones. Since source correlates with
difficulty in this corpus, such features leak provenance into the model. They are dropped
by name prefix before any model sees them; `--keep-instrumentation` disables the screen so
the size of that effect can be measured rather than assumed.

Usage
-----
    python -m guitar.jsymbolic_comparison
    python -m guitar.jsymbolic_comparison --n-iter 40 --keep-instrumentation
"""

import argparse
import json
import os
import sys
from statistics import mean, stdev

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from guitar.metric_study import METRIC_LABEL, METRIC_ORDER, all_metrics
from guitar.prepare_splits import ALL_FEATURES_V5_PRUNED_COLLINEAR2, make_piece_id
from guitar.tuned_baselines import INNER_FOLDS, SEARCH_SPACES, SEED

DESCRIPTOR_CSV = "features/guitar_descriptors_v5.csv"
JSYMBOLIC_CSV = "features/guitar_jsymbolic_v5.csv"
SPLITS_PATH = "guitar/guitar_splits_v5.json"
OUT_JSON = "guitar/jsymbolic_comparison_results.json"
REPORT_PATH = "guitar/JSYMBOLIC_STUDY.md"

# jSymbolic families that describe the *file*, not the *music*. Instrumentation features
# read the MusicXML instrument declaration; on this corpus that declaration fingerprints
# the source dataset rather than any property of the piece.
PROVENANCE_PREFIXES = (
    "js_AcousticGuitar", "js_ElectricGuitar", "js_ElectricInstrument", "js_Violin",
    "js_Saxophone", "js_Brass", "js_Woodwinds", "js_Orchestral", "js_String",
    "js_Percussion", "js_UnpitchedInstruments", "js_PitchedInstruments",
    "js_NumberOfPitchedInstruments", "js_NumberOfUnpitchedInstruments",
)


def load_frames(keep_instrumentation=False):
    desc = pd.read_csv(DESCRIPTOR_CSV)
    desc["piece_id"] = desc.apply(make_piece_id, axis=1)
    desc = desc.set_index("piece_id")

    if not os.path.exists(JSYMBOLIC_CSV):
        raise SystemExit(
            f"{JSYMBOLIC_CSV} not found. Run:\n"
            f"  python -m guitar.extract_jsymbolic_features --workers 10 --exclude-slow"
        )
    js = pd.read_csv(JSYMBOLIC_CSV, index_col=0)

    dropped = []
    if not keep_instrumentation:
        dropped = [c for c in js.columns if c.startswith(PROVENANCE_PREFIXES)]
        js = js.drop(columns=dropped)

    with open(SPLITS_PATH) as f:
        splits = json.load(f)

    common = desc.index.intersection(js.index)
    return desc.loc[common], js.loc[common], splits, dropped


def fold_xy(frame, splits, split_idx, subset):
    labels = splits[str(split_idx)][subset]
    ids = [i for i in labels if i in frame.index]
    return frame.loc[ids], pd.Series([labels[i] for i in ids], index=ids)


def evaluate_feature_set(frame, splits, name, n_iter):
    """Tuned Random Forest over one feature set, under the standard 5-fold protocol."""
    estimator, space = SEARCH_SPACES["random_forest"]
    fold_metrics, chosen = [], []

    for split_idx in range(5):
        X_tr, y_tr = fold_xy(frame, splits, split_idx, "train")
        X_va, y_va = fold_xy(frame, splits, split_idx, "val")
        X_te, y_te = fold_xy(frame, splits, split_idx, "test")

        X_pool = pd.concat([X_tr, X_va])
        y_pool = pd.concat([y_tr, y_va])
        medians = X_pool.median().fillna(0.0)
        X_pool = X_pool.fillna(medians).replace([np.inf, -np.inf], 0.0)
        X_te = X_te.fillna(medians).replace([np.inf, -np.inf], 0.0)

        search = RandomizedSearchCV(
            estimator=estimator, param_distributions=space, n_iter=n_iter,
            scoring="balanced_accuracy",
            cv=StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED),
            random_state=SEED + split_idx, n_jobs=-1, refit=True,
        )
        search.fit(X_pool, y_pool)
        y_pred = search.best_estimator_.predict(X_te)
        m = all_metrics(y_te.to_numpy(), y_pred)
        fold_metrics.append(m)
        chosen.append({k: (int(v) if isinstance(v, np.integer) else v)
                       for k, v in search.best_params_.items()})
        print(f"  {name} split {split_idx}: acc={m['accuracy']:.4f} "
              f"macroMAE={m['macro_mae']:.4f} tau_b={m['kendall_tau_b']:.4f}", flush=True)

    out = {k: [m[k] for m in fold_metrics] for k in METRIC_ORDER}
    out["chosen_params_per_fold"] = chosen
    out["n_features"] = int(frame.shape[1])

    # Which family the forest actually leans on, for set C -- the question of whether the
    # generic features displace the hand-crafted ones or merely sit alongside them.
    if hasattr(search.best_estimator_, "feature_importances_"):
        imp = sorted(zip(frame.columns, search.best_estimator_.feature_importances_),
                     key=lambda t: -t[1])
        out["top_features_last_fold"] = [(c, float(v)) for c, v in imp[:20]]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-iter", type=int, default=40)
    parser.add_argument("--keep-instrumentation", action="store_true")
    args = parser.parse_args()

    desc, js, splits, dropped = load_frames(args.keep_instrumentation)
    guitar_cols = [c for c in ALL_FEATURES_V5_PRUNED_COLLINEAR2 if c in desc.columns]

    sets = {
        "A_jsymbolic_only": js,
        "B_guitar_only": desc[guitar_cols],
        "C_both": pd.concat([desc[guitar_cols], js], axis=1),
    }

    print(f"Pieces in both tables : {len(desc)}")
    print(f"jSymbolic features    : {js.shape[1]} ({len(dropped)} provenance features dropped)")
    print(f"Guitar descriptors    : {len(guitar_cols)}")
    print(f"Search budget         : {args.n_iter} configs/fold, inner {INNER_FOLDS}-fold CV\n", flush=True)

    results = {"dropped_provenance_features": dropped, "n_pieces": int(len(desc))}
    for name, frame in sets.items():
        print(f"--- {name} ({frame.shape[1]} features) ---", flush=True)
        results[name] = evaluate_feature_set(frame, splits, name, args.n_iter)

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    # ---------------- report ----------------
    L = []
    L.append("# Do Generic Symbolic Features Add Anything?\n")
    L.append("Generated by `guitar/jsymbolic_comparison.py`. Tuned Random Forest, 5 folds, ")
    L.append("identical splits, hyperparameters selected on inner-CV balanced accuracy.\n")
    L.append("\nThe question: the thesis hand-crafted every descriptor. Would an off-the-shelf ")
    L.append("symbolic feature library have done the job? jSymbolic features are computed from ")
    L.append("pitch, duration and voice structure and cannot encode fret or string choice, so ")
    L.append("this measures what the guitar-specific representation actually buys.\n")
    # Extraction failures are not uniformly distributed over difficulty, which changes how
    # set B's numbers may be quoted. Compute the bias rather than leave it implicit.
    all_labels = {}
    for k in ("train", "val", "test"):
        all_labels.update(splits["0"][k])
    missing = [p for p in all_labels if p not in desc.index]
    if missing:
        mean_missing = sum(all_labels[p] for p in missing) / len(missing)
        mean_all = sum(all_labels.values()) / len(all_labels)
        L.append(f"\n> **Coverage caveat.** music21 failed to parse {len(missing)} of "
                 f"{len(all_labels)} pieces, and the failures are *not* uniform over "
                 f"difficulty: their mean class is {mean_missing:.2f} against the corpus "
                 f"mean of {mean_all:.2f}. All three feature sets below are evaluated on "
                 f"the same {len(desc)} surviving pieces, so the A/B/C comparison is "
                 f"internally valid -- but set B's numbers here are **not** comparable to "
                 f"the thesis's 640-piece headline and must not be quoted as such.\n")

    L.append(f"\n- Pieces: {len(desc)}")
    L.append(f"- jSymbolic features after screening: {js.shape[1]}")
    L.append(f"- Provenance features dropped: {len(dropped)}")
    if dropped:
        shown = "`, `".join(dropped[:12])
        L.append(f"  (`{shown}`{' ...' if len(dropped) > 12 else ''})")
    L.append(f"- Guitar descriptors: {len(guitar_cols)}\n")

    L.append("\n## Results\n")
    L.append("| Metric | A: jSymbolic only | B: Guitar only | C: Both |")
    L.append("|---|---:|---:|---:|")
    for m in METRIC_ORDER:
        cells = [f"{mean(results[s][m]):.4f} ± {stdev(results[s][m]):.3f}" for s in sets]
        L.append(f"| {METRIC_LABEL[m]} | {cells[0]} | {cells[1]} | {cells[2]} |")
    L.append(f"| **Feature count** | {results['A_jsymbolic_only']['n_features']} | "
             f"{results['B_guitar_only']['n_features']} | {results['C_both']['n_features']} |")

    # Where the combined model's importance mass sits.
    top_c = results["C_both"].get("top_features_last_fold", [])
    if top_c:
        n_guitar = sum(1 for c, _ in top_c if not c.startswith("js_"))
        L.append("\n## What the combined model leans on\n")
        L.append(f"Of the 20 most important features in set C (last fold), **{n_guitar} are ")
        L.append(f"hand-crafted guitar descriptors** and {20 - n_guitar} are generic jSymbolic ")
        L.append("features. Full ranking:\n")
        L.append("| Rank | Feature | Importance | Family |")
        L.append("|---:|---|---:|---|")
        for i, (c, v) in enumerate(top_c, 1):
            fam = "jSymbolic" if c.startswith("js_") else "**guitar**"
            L.append(f"| {i} | `{c}` | {v:.4f} | {fam} |")

    report = "\n".join(L)
    with open(REPORT_PATH, "w") as f:
        f.write(report)
    print(f"\nWrote {OUT_JSON} and {REPORT_PATH}\n")
    print(report)


if __name__ == "__main__":
    main()
