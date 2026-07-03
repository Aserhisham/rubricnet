"""
Drop 7 confirmed duplicate pieces found by find_duplicate_pieces.py: each pair
is the same composition scraped as two different PDF files/arrangements, one
from the main "Title by Composer.pdf" batch and one from a second, informally
-named batch (e.g. bach-bwv-996-courante.pdf) merged in separately. Unlike
Asturias, these are genuinely different files (different note counts/tempo),
not the same file counted twice -- but keeping both still double-counts one
composition in the dataset.

Resolution rule applied per pair: keep whichever entry has the more complete/
precise title and transcription (tempo present, more plausible note count for
the movement), drop the other. Where difficulty disagreed (BWV 1002 Double),
kept the more precisely-titled entry's label rather than averaging or guessing.
"""
import pandas as pd

MAIN_XLSX = "data/verified pieces.xlsx"

# (title to drop) -> reason
DROP_TITLES = {
    "Suite, BWV 996: iii. Courante": "dup of 'Suite I: iii. Courante; BWV 996' (fuller title, has tempo)",
    "Suite, BWV 996: vi. Gigue": "dup of 'Suite I: vi. Gigue; BWV 996' (fuller title, more notes)",
    "Double, BWV 1002": "dup of 'Partita I: BWV 1002: viii. Double (Tempo di Borea)' (more precise title/movement attribution)",
    "A mi madre": "dup of 'A mi madre (Sonatina)' (more complete title)",
    "Waltz, Op. 8 no. 2": "dup of 'Waltz (from Six Divertimentos); Op. 8 no. 2' (more complete title)",
    "Study, Op. 60 no. 16": "dup of 'Study No. 16' (fuller transcription, filename has key signature)",
    "Sonata L. 483": "dup of 'Sonata in A Major, L 483/ K 322' (cross-references both catalog numbers)",
}


def main():
    df = pd.read_excel(MAIN_XLSX)
    before = len(df)
    mask = df["Title"].isin(DROP_TITLES)
    found = set(df.loc[mask, "Title"])
    missing = set(DROP_TITLES) - found
    assert not missing, f"expected titles not found in xlsx: {missing}"
    assert mask.sum() == len(DROP_TITLES), f"expected {len(DROP_TITLES)} rows to drop, matched {mask.sum()}"

    df = df[~mask].copy()
    df.to_excel(MAIN_XLSX, index=False)
    print(f"Main xlsx: {before} -> {len(df)} rows (-{len(DROP_TITLES)} cross-batch duplicates)")
    for title, reason in DROP_TITLES.items():
        print(f"  dropped '{title}': {reason}")


if __name__ == "__main__":
    main()
