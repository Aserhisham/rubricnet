"""
Copy each piece's XML (the one actually read by feature extraction, per its
xml_path in features/guitar_descriptors.csv) into a single flat directory,
verified_pieces/all_xmls/, for convenience.

Deliberately uses the plain skeleton xml for pdf-sourced pieces (not the MIDI
rhythm-corrected xml_rhythm version) -- that's what guitar_features.py actually
reads, and no descriptor in the current set needs real note durations besides
tempo_bpm, which is already read from the PDF text at conversion time.
"""
import os
import shutil

import pandas as pd

CSV_PATH = "features/guitar_descriptors.csv"
OUT_DIR = "verified_pieces/all_xmls"


def main():
    df = pd.read_csv(CSV_PATH)
    os.makedirs(OUT_DIR, exist_ok=True)

    basenames = df["xml_path"].apply(os.path.basename)
    assert basenames.is_unique, "basename collisions would overwrite files when flattened"

    n_copied = 0
    for xml_path in df["xml_path"]:
        assert os.path.exists(xml_path), f"missing file: {xml_path}"
        dest = os.path.join(OUT_DIR, os.path.basename(xml_path))
        shutil.copy2(xml_path, dest)
        n_copied += 1

    print(f"Copied {n_copied} / {len(df)} xml files into {OUT_DIR}")


if __name__ == "__main__":
    main()
