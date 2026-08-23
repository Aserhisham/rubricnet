"""Audit the corpus for the same composition appearing under two source formats.

Why
---
Section "Matching and Deduplication" removes seven duplicates found by
`scripts/analysis/find_duplicate_pieces.py`. Those seven are cross-*batch*
duplicates: both members are ClassClef PDFs, scraped in two differently-named
batches. A duplicate that spans two *sources* -- the same composition present
once as a PDF and once as a DadaGP or GAPS transcription -- is titled differently
on each side ("Fugue, BWV 1000" against "Fuge in A Minor, BWV 1000") and so
survives title-similarity deduplication.

Such a pair matters for two reasons: if one member lands in a training fold while
the other lands in the matching test fold, the model has seen the composition it
is being tested on; and where the two members carry different GuitarBurst grades,
the label source disagrees with itself.

Method
------
A candidate pair requires an exact match on composer surname and on at least one
structured work identifier (BWV / opus+number / Kirkpatrick-Longo number) parsed
from both titles, with one member PDF-derived and the other native. Candidates are
then filtered to those whose identifiers pin the same *movement*, and every
surviving pair was inspected by hand. The script prints, for each confirmed pair,
the two grades, the two note counts, and the fold roles across the five frozen
splits.

Usage
-----
    python -m guitar.cross_source_duplicate_audit
"""

import json
import re

import pandas as pd

CSV_PATH = "features/guitar_descriptors_v5.csv"
SPLITS_PATH = "guitar/guitar_splits_v5.json"
OUT_PATH = "guitar/cross_source_duplicates.json"

# Pairs surviving the identifier match and confirmed by hand as the same movement.
# (native title, pdf title, composer)
CONFIRMED = [
    ("Etude, Op. 6 no. 11", "Allegretto moderato, Op. 6 no. 11", "Fernando Sor"),
    ("Fugue, BWV 1000", "Fuge in A Minor, BWV 1000", "J.S. Bach"),
    ("Lesson, Op. 60 no. 5",
     "Einleitende Etuden (Introductory Studies); Op. 60 no. 5", "Fernando Sor"),
    ("Study, Op. 139 no. 3", "Lesson, Op. 139 no. 3", "Mauro Giuliani"),
    ("Studio in Si Minore, Op. 35 no. 22", "Allegretto, Op. 35 no. 22", "Fernando Sor"),
    ("Study, Op. 60 no. 8", "Etude, Op. 60 no. 8", "Matteo Carcassi"),
]


def work_ids(title):
    t = str(title).lower()
    out = set()
    for m in re.finditer(r"bwv\s*(\d+)", t):
        out.add("bwv" + m.group(1))
    for m in re.finditer(r"op\.?\s*(\d+)\s*(?:no\.?\s*(\d+))?", t):
        out.add("op" + m.group(1) + ("n" + m.group(2) if m.group(2) else ""))
    for m in re.finditer(r"\bl\.?\s*(\d{3})", t):
        out.add("l" + m.group(1))
    return out


def surname(composer):
    return str(composer).lower().replace(".", " ").split()[-1]


def candidates(df):
    df = df.assign(_ids=df["Title"].map(work_ids), _ln=df["Composer"].map(surname))
    pdf = df[df.source == "pdf"]
    native = df[df.source != "pdf"]
    out = []
    for _, r in native.iterrows():
        if not r._ids:
            continue
        for _, p in pdf[pdf._ln == r._ln].iterrows():
            if r._ids & p._ids:
                out.append((r.Title, p.Title, r.Composer))
    return out


def fold_role(splits, key, fold):
    for role in ("train", "val", "test"):
        if key in splits[str(fold)][role]:
            return role
    return "absent"


def main():
    df = pd.read_csv(CSV_PATH)
    splits = json.load(open(SPLITS_PATH))
    cand = candidates(df)
    print(f"identifier-matched candidate pairs : {len(cand)}")
    print(f"confirmed the same movement by hand: {len(CONFIRMED)}\n")

    rows, leak_instances = [], 0
    for native_title, pdf_title, composer in CONFIRMED:
        a = df[(df.Title == native_title) & (df.Composer == composer)].iloc[0]
        b = df[(df.Title == pdf_title) & (df.Composer == composer)].iloc[0]
        ka = f"{native_title}||{composer}"
        kb = f"{pdf_title}||{composer}"
        roles, leaks = [], 0
        for fold in range(5):
            ra, rb = fold_role(splits, ka, fold), fold_role(splits, kb, fold)
            roles.append((ra, rb))
            if "absent" not in (ra, rb) and (ra == "test") != (rb == "test"):
                leaks += 1
        leak_instances += leaks
        rows.append({
            "composer": composer,
            "native_title": native_title, "native_source": a.source,
            "native_grade": int(a.Difficulty), "native_notes": int(a.total_notes),
            "pdf_title": pdf_title,
            "pdf_grade": int(b.Difficulty), "pdf_notes": int(b.total_notes),
            "grade_disagreement": int(abs(a.Difficulty - b.Difficulty)),
            "note_count_ratio": round(max(a.total_notes, b.total_notes)
                                      / max(1.0, min(a.total_notes, b.total_notes)), 3),
            "folds_with_one_member_in_test": leaks,
            "fold_roles": roles,
        })
        print(f"{composer:18s} | {native_title[:38]:40s} g={a.Difficulty:2.0f} n={a.total_notes:5.0f}"
              f" | {pdf_title[:38]:40s} g={b.Difficulty:2.0f} n={b.total_notes:5.0f}"
              f" | leaky folds: {leaks}")

    grade_diffs = [r["grade_disagreement"] for r in rows]
    summary = {
        "confirmed_pairs": len(rows),
        "pairs_with_grade_disagreement": sum(1 for d in grade_diffs if d),
        "max_grade_disagreement": max(grade_diffs),
        "leaked_test_instances_total": leak_instances,
        "leaked_test_instances_per_fold": leak_instances / 5.0,
        "test_fold_size": 128,
        "leaked_share_of_test_fold": round(leak_instances / 5.0 / 128, 4),
    }
    print("\n" + json.dumps(summary, indent=2))
    json.dump({"summary": summary, "pairs": rows}, open(OUT_PATH, "w"), indent=2)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
