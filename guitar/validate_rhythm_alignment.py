"""Validate the MIDI rhythm-transplant pipeline against ground-truth durations.

Why
---
The alignment pipeline (scripts/utilities/align_midi_rhythm.py) recovers note durations for
the PDF-derived majority of the corpus by matching pitch sets against a MIDI performance and
copying that performance's durations onto the score. The only gate applied is alignment
*coverage* -- the fraction of score chords that found a MIDI partner, thresholded at 0.9.

Coverage tests whether the two pitch sequences correspond. It does not test whether the
transplanted durations are correct. Since the dataset is a claimed contribution of this
thesis, "why should the recovered rhythm be believed?" needs an answer grounded in ground
truth, and coverage cannot supply one.

WHAT WAS WRONG WITH THE PREVIOUS VERSION OF THIS SCRIPT
-------------------------------------------------------
It selected its "native rhythm" population as `gp_path.notna() | file_path.notna()`, which
yields 159 rows -- but 106 of those are `source == 'pdf'`, admitted because their `file_path`
holds a stale GAPS mapping. It then paired scores to MIDIs by normalised `xml_path` basename.
Only ClassClef-derived filenames follow the `"<Title> by <Composer>"` convention the MIDI
collection uses, so all 49 DadaGP and all 4 GAPS candidates were skipped as `no_midi` and all
106 pdf rows matched. Ground truth was read from `verified_pieces/all_xmls/`, which for a pdf
piece *is the alignment pipeline's own output*.

Net effect: 83/83 "validated" pieces were pdf, and the pipeline was compared against itself.
The reported 99.4% median exact agreement measured nothing. Superseded output is kept at
guitar/rhythm_validation_SUPERSEDED_circular.json for provenance; do not cite it.

Design
------
Population is restricted to `source in {dada_gp, gaps}` -- 61 pieces whose durations come
from their source format and which the alignment pipeline was never run on. Their entries in
`verified_pieces/all_xmls/` are byte-identical copies of the native conversions, so reading
ground truth there is legitimate for these sources and only these.

MIDI pairing cannot use the pipeline's exact-filename lookup, since native-source filenames
do not follow the ClassClef convention. Scores are paired to MIDIs by composer surname plus
title similarity, with any structured work identifier (opus, ordinal number, BWV, roman
ordinal) present on both sides required to agree exactly; identifiers are read from the
GuitarBurst title *and* the score filename, so a piece whose two metadata sources disagree is
rejected rather than guessed at. This is deliberately conservative: it accepts 14 of 61.

What is therefore under test is the *duration transplant given a correct MIDI*, not the
file-matching step of the pipeline. That is the right scope -- the transplant is what the
rhythm-aware descriptors depend on -- but it must be stated, and the n is small.

For each accepted pair this script:

  1. reads the true chord sequence and onsets from the score (ground truth durations);
  2. runs the *actual* alignment function against the matched MIDI;
  3. derives durations exactly as the pipeline does -- the matched MIDI event's duration,
     falling back to a quarter note when a chord is unmatched;
  4. compares recovered durations against the true ones.

The chord sequence is read with the descriptor pipeline's own parser rather than
align_midi_rhythm.extract_skeleton, because the latter requires the tab staff that only the
PDF-derived skeletons carry; the alignment function under test is used unmodified.

Reported
--------
  * exact-match rate: fraction of chords whose recovered duration equals the true duration
  * within-a-factor-of-2 rate: tolerant version, catching halving/doubling errors
  * median absolute log2 ratio: scale-free error, 0 = perfect
  * total-length ratio: whether the piece ends up the right overall duration
  * how much of the agreement is carried by the unmatched-chord quarter-note fallback
"""
import difflib
import glob
import json
import os
import re
import sys
import unicodedata

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "scripts/utilities")

from align_midi_rhythm import align, extract_midi_events  # noqa: E402

from guitar.guitar_features import get_timed_chords_from_xml  # noqa: E402
from guitar.prepare_splits import make_piece_id  # noqa: E402

MIDI_DIR = "midi-20260630T194726Z-3-001/midi"
CSV = "features/guitar_descriptors_v5.csv"
ALL_XMLS = "verified_pieces/all_xmls"
OUT = "guitar/rhythm_validation.json"
COVERAGE_THRESHOLD = 0.9

# Sources whose all_xmls entry is the untouched native conversion rather than pipeline
# output. Anything else would make the comparison circular.
NATIVE_SOURCES = ("dada_gp", "gaps")

# Title-similarity floors. Agreement on a structured identifier (Op./No./BWV) is strong
# evidence on its own, so it buys tolerance on wording; without one, the titles themselves
# have to carry the match.
SIM_WITH_ID = 0.62
SIM_WITHOUT_ID = 0.85

ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8, "ix": 9,
         "x": 10, "xi": 11, "xii": 12, "xiii": 13, "xiv": 14, "xv": 15, "xx": 20,
         "xxii": 22, "xxiv": 24}


def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", strip_accents(s).lower()).strip()


def work_ids(title):
    """Structured work identifiers: opus, ordinal number, BWV, roman ordinal."""
    t = norm(title)
    out = set()
    for m in re.finditer(r"\b(?:op|opus)\s*(\d+)", t):
        out.add(("op", int(m.group(1))))
    for m in re.finditer(r"\b(?:no|n|num|nr)\s*(\d+)", t):
        out.add(("no", int(m.group(1))))
    for m in re.finditer(r"\bbwv\s*(\d+)", t):
        out.add(("bwv", int(m.group(1))))
    for tok in t.split():
        if tok in ROMAN:
            out.add(("rom", ROMAN[tok]))
    return out


def _kinds(idset):
    return {k for k, _ in idset}


def _vals(idset, kind):
    return {v for k, v in idset if k == kind}


def build_midi_index():
    """MIDI stems follow '<Title> by <Composer>'; split and normalise both halves."""
    index = {}
    for path in glob.glob(os.path.join(MIDI_DIR, "*.mid*")):
        stem = os.path.splitext(os.path.basename(path))[0]
        title, composer = stem.rsplit(" by ", 1) if " by " in stem else (stem, "")
        index[path] = (norm(title), norm(composer), work_ids(title))
    return index


def match_midi(row, midi_index):
    """Best MIDI for a score, or None. Conservative by design -- see module docstring."""
    score_title = norm(row["Title"])
    surnames = {t for t in norm(row["Composer"]).split() if len(t) > 2}
    # Identifiers from both metadata sources; a disagreement between them propagates into
    # the comparison below and rejects the piece rather than resolving it arbitrarily.
    score_ids = work_ids(row["Title"]) | work_ids(os.path.basename(str(row["xml_path"])))

    best = None
    for path, (midi_title, midi_composer, midi_ids) in midi_index.items():
        if surnames and not (surnames & set(midi_composer.split())):
            continue
        shared = _kinds(score_ids) & _kinds(midi_ids)
        if any(_vals(score_ids, k) != _vals(midi_ids, k) for k in shared):
            continue
        if score_ids and midi_ids and not shared:
            continue  # both identified, but under different systems -- not comparable
        sim = difflib.SequenceMatcher(None, score_title, midi_title).ratio()
        if sim < (SIM_WITH_ID if shared else SIM_WITHOUT_ID):
            continue
        key = (len(shared), sim)
        if best is None or key > best[0]:
            best = (key, path, sim, len(shared))
    return best


def main():
    midi_index = build_midi_index()
    df = pd.read_csv(CSV)
    df["piece_id"] = df.apply(make_piece_id, axis=1)
    native = df[df["source"].isin(NATIVE_SOURCES)]
    print(f"Native-rhythm pieces (source in {NATIVE_SOURCES}): {len(native)}")

    rows, pairings = [], []
    skipped = {"no_midi_match": 0, "no_rhythm": 0, "parse_fail": 0,
               "low_coverage": 0, "len_mismatch": 0}

    for _, r in native.iterrows():
        best = match_midi(r, midi_index)
        if best is None:
            skipped["no_midi_match"] += 1
            continue
        _, midi_path, sim, n_shared_ids = best

        xml = os.path.join(ALL_XMLS, os.path.basename(str(r["xml_path"])))
        if not os.path.exists(xml):
            skipped["parse_fail"] += 1
            continue
        chords, onsets, _, _, has_rhythm = get_timed_chords_from_xml(xml)
        if chords is None or not has_rhythm or len(chords) < 8:
            skipped["no_rhythm"] += 1
            continue
        try:
            events, _ = extract_midi_events(midi_path)
        except Exception:
            skipped["parse_fail"] += 1
            continue
        if not events:
            skipped["parse_fail"] += 1
            continue

        # Ground truth: inter-onset intervals. The final chord has no successor, so it is
        # dropped from both sequences rather than guessed.
        true_dur = np.diff(np.asarray(onsets, dtype=float))
        if len(true_dur) < 8:
            skipped["len_mismatch"] += 1
            continue

        matches, coverage = align(chords, events)
        pairings.append({
            "piece": r["piece_id"], "source": r["source"],
            "midi": os.path.basename(midi_path),
            "title_similarity": round(float(sim), 3),
            "shared_identifiers": n_shared_ids,
            "coverage": round(float(coverage), 3),
            "passed_gate": bool(coverage >= COVERAGE_THRESHOLD),
        })
        if coverage < COVERAGE_THRESHOLD:
            skipped["low_coverage"] += 1
            continue

        rec = np.array([float(events[matches[i]][1]) if i in matches else 1.0
                        for i in range(len(true_dur))])
        matched_mask = np.array([i in matches for i in range(len(true_dur))])

        ok = (true_dur > 0) & (rec > 0)
        if ok.sum() < 8:
            skipped["len_mismatch"] += 1
            continue
        t, p = true_dur[ok], rec[ok]
        ratio = np.log2(p / t)

        rows.append({
            "piece": r["piece_id"],
            "source": r["source"],
            "midi": os.path.basename(midi_path),
            "n": int(ok.sum()),
            "coverage": float(coverage),
            "matched_frac": float(matched_mask[ok].mean()),
            "exact": float(np.mean(np.isclose(p, t, rtol=1e-6))),
            "within_2x": float(np.mean(np.abs(ratio) <= 1.0)),
            "median_abs_log2": float(np.median(np.abs(ratio))),
            "total_len_ratio": float(p.sum() / t.sum()),
        })
        print(f"  {r['piece_id'][:46]:48s} cov={coverage:.2f} exact={rows[-1]['exact']:.2f} "
              f"med|log2|={rows[-1]['median_abs_log2']:.2f}", flush=True)

    print("\nCandidate pairings (all matched pieces, gate-passing or not):")
    for p in sorted(pairings, key=lambda x: -x["coverage"]):
        flag = "keep" if p["passed_gate"] else "GATED"
        print(f"  [{flag}] cov={p['coverage']:.2f} sim={p['title_similarity']:.2f} "
              f"ids={p['shared_identifiers']}  {p['piece'][:40]:42s} <- {p['midi']}")

    if not rows:
        print("\nNo pieces validated.", flush=True)
        json.dump({"summary": {"n_pieces": 0, "skipped": skipped}, "pairings": pairings,
                   "per_piece": []}, open(OUT, "w"), indent=2)
        return

    d = pd.DataFrame(rows)
    summary = {
        "population": f"source in {list(NATIVE_SOURCES)}",
        "n_candidates": len(native),
        "n_midi_matched": len(pairings),
        "n_pieces": len(d),
        "skipped": skipped,
        "coverage_threshold": COVERAGE_THRESHOLD,
        "exact_match_mean": float(d["exact"].mean()),
        "exact_match_median": float(d["exact"].median()),
        "within_2x_mean": float(d["within_2x"].mean()),
        "median_abs_log2_median": float(d["median_abs_log2"].median()),
        "total_len_ratio_median": float(d["total_len_ratio"].median()),
        "matched_frac_mean": float(d["matched_frac"].mean()),
        "pieces_exact_over_50pct": int((d["exact"] > 0.5).sum()),
        "pieces_exact_over_80pct": int((d["exact"] > 0.8).sum()),
    }
    json.dump({"summary": summary, "pairings": pairings, "per_piece": rows},
              open(OUT, "w"), indent=2)

    print("\n" + "=" * 70)
    print(f"Native candidates             : {summary['n_candidates']}")
    print(f"MIDI-matched                  : {summary['n_midi_matched']}")
    print(f"Pieces validated              : {summary['n_pieces']}")
    print(f"Skipped                       : {skipped}")
    print(f"Exact duration match  (mean)  : {summary['exact_match_mean']:.3f}")
    print(f"Exact duration match  (median): {summary['exact_match_median']:.3f}")
    print(f"Within factor 2       (mean)  : {summary['within_2x_mean']:.3f}")
    print(f"Median |log2(rec/true)|       : {summary['median_abs_log2_median']:.3f}")
    print(f"Total-length ratio    (median): {summary['total_len_ratio_median']:.3f}")
    print(f"Chords matched (mean frac)    : {summary['matched_frac_mean']:.3f}")
    print(f"Pieces >50% exact             : {summary['pieces_exact_over_50pct']}/{summary['n_pieces']}")
    print(f"Pieces >80% exact             : {summary['pieces_exact_over_80pct']}/{summary['n_pieces']}")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
