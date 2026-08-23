"""Decoy audit of the PDF -> MIDI file matching step of the rhythm pipeline.

The rhythm pipeline (scripts/utilities/align_midi_rhythm.py) pairs a PDF-derived score
to a MIDI performance by exact filename, then gates the pair on Needleman-Wunsch
alignment coverage. Section 5 of the thesis validates the *duration transplant* given a
correct MIDI, on the natively-timed pieces; it cannot validate the *file matching*,
because that step had to be replaced to reach that population at all.

This script tests the matching step directly and without ground truth, by asking whether
the named MIDI is discriminably the right partner rather than merely an acceptable one.
For each sampled piece the PDF skeleton is aligned against its own name-matched MIDI and
against N decoy MIDIs drawn at random from the same ~7,200-file ClassClef pool. If the
filename convention identifies the correct performance, the true MIDI should rank first
by coverage and by a wide margin; if coverage merely reflects "some guitar piece", decoys
would score comparably and the gate would be admitting arbitrary partners.

Outputs guitar/midi_matching_audit.json.
"""
import argparse
import json
import os
import random
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.utilities.align_midi_rhythm import (
    MIDI_DIR, SKELETON_XML_DIR, align, extract_midi_events, extract_skeleton,
)

OUTPUT_XML_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "verified_pieces", "pdf", "xml_rhythm",
)
OUT_PATH = "guitar/midi_matching_audit.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pieces", type=int, default=25)
    ap.add_argument("--n-decoys", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # The audit population is the pieces the pipeline actually kept: those with a
    # written xml_rhythm output, i.e. a name-matched MIDI that also cleared the gate.
    aligned = sorted(os.path.splitext(f)[0] for f in os.listdir(OUTPUT_XML_DIR)
               if f.endswith((".xml", ".musicxml")))
    midi_pool = sorted(f for f in os.listdir(MIDI_DIR) if f.lower().endswith((".mid", ".midi")))
    print(f"{len(aligned)} aligned pieces, {len(midi_pool)} MIDI files in the pool")

    sample = rng.sample(aligned, min(args.n_pieces, len(aligned)))
    rows = []

    for stem in sample:
        skel_path = next((p for p in (os.path.join(SKELETON_XML_DIR, f"{stem}{e}")
                          for e in (".musicxml", ".xml")) if os.path.exists(p)), "")
        true_midi = next((f for f in midi_pool if os.path.splitext(f)[0] == stem), None)
        if not skel_path or true_midi is None:
            print(f"  skip {stem}: no skeleton or no name-matched MIDI")
            continue
        try:
            skeleton = extract_skeleton(skel_path)
            true_events, _ = extract_midi_events(os.path.join(MIDI_DIR, true_midi))
            _, true_cov = align(skeleton["chords"], true_events)
        except Exception as exc:
            print(f"  skip {stem}: {exc}")
            continue

        decoy_names = rng.sample([f for f in midi_pool if f != true_midi], args.n_decoys)
        decoy_covs = []
        for d in decoy_names:
            try:
                ev, _ = extract_midi_events(os.path.join(MIDI_DIR, d))
                _, cov = align(skeleton["chords"], ev)
                decoy_covs.append((d, cov))
            except Exception:
                continue
        if not decoy_covs:
            continue

        best_decoy, best_decoy_cov = max(decoy_covs, key=lambda t: t[1])
        rank = 1 + sum(1 for _, c in decoy_covs if c > true_cov)
        rows.append({
            "piece": stem, "true_midi": true_midi, "true_coverage": true_cov,
            "best_decoy": best_decoy, "best_decoy_coverage": best_decoy_cov,
            "mean_decoy_coverage": sum(c for _, c in decoy_covs) / len(decoy_covs),
            "n_decoys": len(decoy_covs), "rank_of_true": rank,
            "decoys_above_gate": sum(1 for _, c in decoy_covs if c >= 0.9),
        })
        print(f"  {stem[:52]:52s} true={true_cov:.3f} best_decoy={best_decoy_cov:.3f} rank={rank}")

    n = len(rows)
    summary = {
        "n_pieces_audited": n,
        "n_decoys_per_piece": args.n_decoys,
        "seed": args.seed,
        "true_rank_1": sum(1 for r in rows if r["rank_of_true"] == 1),
        "mean_true_coverage": sum(r["true_coverage"] for r in rows) / n if n else None,
        "mean_best_decoy_coverage": sum(r["best_decoy_coverage"] for r in rows) / n if n else None,
        "max_decoy_coverage": max((r["best_decoy_coverage"] for r in rows), default=None),
        "decoys_reaching_gate": sum(r["decoys_above_gate"] for r in rows),
        "total_decoy_alignments": sum(r["n_decoys"] for r in rows),
    }
    print("\n" + json.dumps(summary, indent=2))
    with open(OUT_PATH, "w") as f:
        json.dump({"summary": summary, "pieces": rows}, f, indent=2)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
