"""Extract generic symbolic-music features (jSymbolic set, via music21) for the V5 corpus.

Why
---
Every descriptor in this thesis was designed by hand (see `guitar/DESCRIPTOR_PROVENANCE.md`).
That invites an obvious question at a defense: *why not just use an off-the-shelf symbolic
feature library?* jSymbolic (McKay & Fujinaga) is the standard answer to that question in
symbolic MIR, and music21 ships Python reimplementations of its extractors, which run
directly on the MusicXML corpus without a Java round-trip or MIDI conversion.

The point is not to hope the generic features win. It is that jSymbolic features are
**instrument-agnostic** -- pitch, melody, rhythm, texture, dynamics. None of them can see a
fret or a string, because MIDI-derived representations do not encode which of the six
possible positions a pitch was played at. So extracting them sets up the experiment that
actually answers the question:

    A. generic jSymbolic features alone      -> how far does off-the-shelf get you?
    B. the 25 hand-crafted guitar descriptors -> the thesis's set
    C. A + B                                  -> does generic add anything on top?

If A trails B, the hand-crafted guitar-specific descriptor work is justified by measurement
rather than by assertion. If C beats B, the thesis gains descriptors it did not have.
Either outcome is a publishable result; the current thesis can claim neither.

Cost
----
Feature extraction is expensive: roughly 3 minutes per piece for the full extractor set on
a mid-length score, dominated by a handful of autocorrelation-based rhythm extractors. This
script therefore
  * runs pieces in parallel across processes (`--workers`),
  * caches each piece's result to `features/jsymbolic_cache/` so runs are resumable,
  * supports an `--exclude` denylist for the pathological extractors,
  * and enforces a per-piece timeout so one degenerate score cannot stall the run.

Usage
-----
    python -m guitar.extract_jsymbolic_features --list-extractors
    python -m guitar.extract_jsymbolic_features --workers 14 --exclude-slow
    python -m guitar.extract_jsymbolic_features --workers 14 --limit 20   # trial run
"""

import argparse
import json
import os
import signal
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

warnings.filterwarnings("ignore")
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from guitar.prepare_splits import make_piece_id

ALL_XMLS = "verified_pieces/all_xmls"
CSV_PATH = "features/guitar_descriptors_v5.csv"
CACHE_DIR = "features/jsymbolic_cache"
OUT_CSV = "features/guitar_jsymbolic_v5.csv"

# Extractors that raise on this corpus regardless of the piece: they need instrument
# metadata, unpitched percussion, or MIDI controller events that solo-guitar MusicXML
# derived from score engraving simply does not carry.
ALWAYS_BROKEN = {
    "PitchedInstrumentsPresentFeature",
    "UnpitchedInstrumentsPresentFeature",
    "NotePrevalenceOfPitchedInstrumentsFeature",
    "DominantSpreadFeature",
    "StrongTonalCentresFeature",
    "GlissandoPrevalenceFeature",
    "AverageRangeOfGlissandosFeature",
    "VibratoPrevalenceFeature",
    "HarmonicityOfTwoStrongestRhythmicPulsesFeature",
    "StrengthRatioOfTwoStrongestRhythmicPulsesFeature",
    "NumberOfStrongPulsesFeature",
    "NumberOfModeratePulsesFeature",
    "RhythmicLoosenessFeature",
    "PolyrhythmsFeature",
    "RhythmicVariabilityFeature",
}

# Profiling result (one mid-length score, all 96 working extractors, 184s total): cost is
# spread almost evenly at ~1.9s per extractor, because each one re-walks the parsed score
# independently rather than sharing intermediate representations. The slowest twelve account
# for only 20% of the total, so excluding extractors buys very little -- parallelism across
# pieces is the only lever that matters. The list below is kept for the handful whose cost
# is high *and* whose relevance to difficulty is doubtful (independent-voice counting on
# single-staff guitar scores, where voice separation is unreliable anyway).
SLOW_EXTRACTORS = {
    "VariabilityOfNumberOfIndependentVoicesFeature",
    "AverageNumberOfIndependentVoicesFeature",
    "MaximumNumberOfIndependentVoicesFeature",
    "AverageTimeBetweenAttacksForEachVoiceFeature",
    "ImportanceOfLoudestVoiceFeature",
    "VoiceEqualityDynamicsFeature",
}


def resolve_xml(row):
    """Map a features-CSV row to its file in all_xmls (same logic as extract_features_v3)."""
    for col in ("xml_path", "file_path"):
        p = row.get(col)
        if isinstance(p, str) and p:
            cand = os.path.join(ALL_XMLS, os.path.basename(p))
            if os.path.exists(cand):
                return cand
    return None


# music21's *native* extractors are not part of jSymbolic. They matter here because they
# cover a dimension absent from both the jSymbolic subset and the thesis's 25 descriptors:
# vertical harmony (triad/seventh prevalence, set-class simultaneities), tonal clarity, and
# notated-duration variety. The 25 guitar descriptors measure chords only physically
# (chord_ratio counts 2+ notes; chord_stretch is fret span) and never harmonically.
# All 20 run cleanly on this corpus at ~1.1s each.
NATIVE_EXCLUDE = {
    "LanguageFeature",   # text/lyrics language -- meaningless for untexted guitar scores
}


def get_extractor_classes(exclude=frozenset(), include_native=False):
    from music21 import features

    out = []
    for group in sorted(features.jSymbolic.extractorsById):
        for cls in features.jSymbolic.extractorsById[group]:
            if cls is None:
                continue
            if cls.__name__ in ALWAYS_BROKEN or cls.__name__ in exclude:
                continue
            out.append(cls)

    if include_native:
        for cls in features.native.featureExtractors:
            if cls.__name__ in NATIVE_EXCLUDE or cls.__name__ in exclude:
                continue
            out.append(cls)
    return out


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def extract_one(task):
    """Worker: parse one score and run every extractor. Returns (piece_id, dict|None, secs)."""
    piece_id, path, exclude, per_piece_timeout, include_native, cache_dir = task
    cache_path = os.path.join(cache_dir, _cache_name(piece_id))
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return piece_id, json.load(f), 0.0

    from music21 import converter

    t0 = time.time()
    values = {}
    try:
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(per_piece_timeout)
        score = converter.parse(path)
        for cls in get_extractor_classes(exclude, include_native):
            name = cls.__name__.replace("Feature", "")
            try:
                vector = cls(score).extract().vector
            except Exception:
                continue
            if len(vector) == 1:
                values[f"js_{name}"] = float(vector[0])
            else:
                for i, v in enumerate(vector):
                    values[f"js_{name}_{i}"] = float(v)
        signal.alarm(0)
    except _Timeout:
        signal.alarm(0)
        return piece_id, None, time.time() - t0
    except Exception:
        signal.alarm(0)
        return piece_id, None, time.time() - t0

    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(values, f)
    return piece_id, values, time.time() - t0


def _cache_name(piece_id):
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in piece_id)
    return f"{safe[:180]}.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    parser.add_argument("--limit", type=int, default=None, help="process only the first N pieces (trial run)")
    parser.add_argument("--timeout", type=int, default=900, help="per-piece timeout in seconds")
    parser.add_argument("--exclude", default="", help="comma-separated extractor class names to skip")
    parser.add_argument("--exclude-slow", action="store_true", help="also skip the profiled slow extractors")
    parser.add_argument("--list-extractors", action="store_true")
    parser.add_argument(
        "--include-native", action="store_true",
        help="also run music21's 20 native extractors (harmony/simultaneity, tonal "
             "certainty, notated-duration variety) -- a dimension the jSymbolic subset "
             "and the thesis's 25 descriptors both miss entirely",
    )
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    exclude = {e.strip() for e in args.exclude.split(",") if e.strip()}
    if args.exclude_slow:
        exclude |= SLOW_EXTRACTORS

    # Separate cache and output when native extractors are on, so the completed
    # jSymbolic-only run stays valid and reproducible.
    cache_dir = args.cache_dir or (CACHE_DIR + "_native" if args.include_native else CACHE_DIR)
    out_path = args.out or (
        "features/guitar_jsymbolic_native_v5.csv" if args.include_native else OUT_CSV
    )

    if args.list_extractors:
        active = get_extractor_classes(exclude, args.include_native)
        for cls in active:
            print(cls.__name__)
        print(f"\n{len(active)} extractors active "
              f"({len(ALWAYS_BROKEN)} always-broken + {len(exclude)} excluded skipped)")
        return

    df = pd.read_csv(CSV_PATH)
    df["piece_id"] = df.apply(make_piece_id, axis=1)

    tasks = []
    unresolved = 0
    for _, row in df.iterrows():
        path = resolve_xml(row)
        if path is None:
            unresolved += 1
            continue
        tasks.append((row["piece_id"], path, frozenset(exclude), args.timeout,
                      args.include_native, cache_dir))
    if args.limit:
        tasks = tasks[: args.limit]

    n_active = len(get_extractor_classes(exclude, args.include_native))
    print(f"Pieces      : {len(tasks)} resolved, {unresolved} unresolved")
    print(f"Extractors  : {n_active} active (native included: {args.include_native})")
    print(f"Workers     : {args.workers}, per-piece timeout {args.timeout}s")
    print(f"Cache       : {cache_dir}/\n")

    os.makedirs(cache_dir, exist_ok=True)
    rows = {}
    failed = []
    done = 0
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(extract_one, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            piece_id, values, secs = fut.result()
            done += 1
            if values is None:
                failed.append(piece_id)
                status = "FAIL"
            else:
                rows[piece_id] = values
                status = f"{len(values)} feats"
            elapsed = time.time() - t_start
            rate = done / elapsed if elapsed else 0
            eta = (len(tasks) - done) / rate if rate else 0
            print(f"[{done}/{len(tasks)}] {status:12s} {secs:6.1f}s  ETA {eta/60:5.1f}m  {piece_id[:55]}")

    if not rows:
        print("\nNo pieces extracted.")
        return

    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "piece_id"
    out = out.sort_index(axis=1)

    # Drop columns that are constant across the corpus: they carry no information for any
    # model and would only inflate the descriptor count in the comparison against the 25.
    nunique = out.nunique(dropna=False)
    constant = nunique[nunique <= 1].index.tolist()
    out = out.drop(columns=constant)

    out.to_csv(args.out)
    print(f"\nExtracted {len(out)} pieces x {out.shape[1]} features -> {args.out}")
    print(f"Dropped {len(constant)} constant columns")
    if failed:
        print(f"Failed on {len(failed)} pieces: {failed[:10]}")


if __name__ == "__main__":
    main()
