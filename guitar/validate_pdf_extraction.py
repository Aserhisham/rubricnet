"""Validate the ClassClef PDF extractor against independent transcriptions.

Why
---
655 of the 716 pieces enter the corpus through `scripts/utilities/
convert_pdf_to_musicxml.py`, which reads vector-graphic tablature with pdfplumber.
Everything downstream -- every descriptor, and the pitch sequence the rhythm
alignment consumes -- rests on that step, yet the rhythm validation and the decoy
audit both test links *below* it and take its output as given. The V2 descriptor
audit moreover found the extractor had been corrupting fret values, so its
correctness is demonstrably not self-evident.

The corpus contains its own control. Six compositions appear twice, once as a
ClassClef PDF and once as a DadaGP or GAPS transcription produced independently by
a different transcriber in a format that carries string and fret natively
(`guitar/cross_source_duplicate_audit.py`). Aligning the extractor's output against
the native transcription of the same composition therefore measures whether the
extractor reads the notes that are on the page -- against a reference it had no
part in producing.

What this does and does not test
--------------------------------
The two members of a pair are different *transcriptions*, not different renderings
of one file: they may take repeats differently, add or omit an ornament, or realise
a voice an octave apart. Disagreement therefore bounds extractor error from above
and does not isolate it. Pitch-set agreement on the aligned portion is the
informative quantity; the length ratio is reported separately because it is
dominated by repeat structure rather than by extraction.

Usage
-----
    python -m guitar.validate_pdf_extraction
"""

import json
import sys
import xml.etree.ElementTree as ET

import pandas as pd

sys.path.insert(0, "scripts/utilities")
from align_midi_rhythm import OPEN_STRINGS, align  # noqa: E402

CSV_PATH = "features/guitar_descriptors_v5.csv"
PAIRS_PATH = "guitar/cross_source_duplicates.json"
OUT_PATH = "guitar/pdf_extraction_validation.json"


def read_tab_chords(xml_path):
    """Return [[{'string':s,'fret':f}, ...], ...], one entry per chord event.

    Handles both layouts present in the corpus: a single part whose tab notes sit
    on staff 2 (ClassClef PDFs, DadaGP), and a two-part score whose tab part
    carries no staff element (GAPS).
    """
    root = ET.parse(xml_path).getroot()
    best = []
    for part in root.findall("part"):
        staves = {n.findtext("staff") for n in part.iter("note")}
        for staff in sorted(staves, key=lambda s: (s is None, s)):
            chords, cur = [], None
            for measure in part.findall("measure"):
                for note in measure.findall("note"):
                    if note.findtext("staff") != staff:
                        continue
                    if note.find("rest") is not None or note.find("grace") is not None:
                        continue
                    tech = note.find("notations/technical")
                    if tech is None:
                        continue
                    s_el, f_el = tech.find("string"), tech.find("fret")
                    if s_el is None or f_el is None:
                        continue
                    nd = {"string": int(s_el.text), "fret": int(f_el.text)}
                    if note.find("chord") is not None and cur is not None:
                        cur.append(nd)
                    else:
                        cur = [nd]
                        chords.append(cur)
            if len(chords) > len(best):
                best = chords
    return best


def pitch_sets(chords):
    return [frozenset(OPEN_STRINGS.get(n["string"], 40) + n["fret"] for n in c) for c in chords]


def compare(pdf_xml, native_xml):
    pdf_chords = read_tab_chords(pdf_xml)
    nat_chords = read_tab_chords(native_xml)
    if not pdf_chords or not nat_chords:
        return None
    nat_events = [(ps,) for ps in pitch_sets(nat_chords)]
    matches, coverage = align(pdf_chords, nat_events)

    pdf_sets, nat_sets = pitch_sets(pdf_chords), pitch_sets(nat_chords)
    exact = sum(1 for i, j in matches.items() if pdf_sets[i] == nat_sets[j])
    # String/fret agreement on the chords whose pitch content matches exactly:
    # a pitch can be produced at several positions, so this is the stricter test.
    sf_total = sf_exact = 0
    for i, j in matches.items():
        if pdf_sets[i] != nat_sets[j]:
            continue
        a = sorted((n["string"], n["fret"]) for n in pdf_chords[i])
        b = sorted((n["string"], n["fret"]) for n in nat_chords[j])
        sf_total += 1
        sf_exact += int(a == b)
    return {
        "pdf_chords": len(pdf_chords),
        "native_chords": len(nat_chords),
        "length_ratio": round(max(len(pdf_chords), len(nat_chords))
                              / min(len(pdf_chords), len(nat_chords)), 3),
        "coverage": round(coverage, 4),
        "matched_chords": len(matches),
        "pitch_set_exact_of_matched": round(exact / len(matches), 4) if matches else 0.0,
        "string_fret_exact_of_pitch_matched": round(sf_exact / sf_total, 4) if sf_total else None,
    }


def main():
    df = pd.read_csv(CSV_PATH)
    pairs = json.load(open(PAIRS_PATH))["pairs"]
    rows = []
    for p in pairs:
        nat = df[(df.Title == p["native_title"]) & (df.Composer == p["composer"])].iloc[0]
        pdf = df[(df.Title == p["pdf_title"]) & (df.Composer == p["composer"])].iloc[0]
        res = compare(pdf.xml_path, nat.xml_path)
        if res is None:
            print(f"skipped {p['composer']} / {p['pdf_title']} (no tab notes readable)")
            continue
        res.update(composer=p["composer"], pdf_title=p["pdf_title"],
                   native_title=p["native_title"], native_source=p["native_source"])
        rows.append(res)
        print(f"{p['composer']:18s} {p['pdf_title'][:34]:36s} vs {p['native_source']:8s}"
              f" cov={res['coverage']:.3f} pitch-exact={res['pitch_set_exact_of_matched']:.3f}"
              f" str/fret={res['string_fret_exact_of_pitch_matched']}"
              f" len-ratio={res['length_ratio']}")

    cov = [r["coverage"] for r in rows]
    pex = [r["pitch_set_exact_of_matched"] for r in rows]
    sfx = [r["string_fret_exact_of_pitch_matched"] for r in rows
           if r["string_fret_exact_of_pitch_matched"] is not None]
    summary = {
        "pairs": len(rows),
        "mean_coverage": round(sum(cov) / len(cov), 4),
        "median_coverage": round(sorted(cov)[len(cov) // 2], 4),
        "mean_pitch_exact_of_matched": round(sum(pex) / len(pex), 4),
        "min_pitch_exact_of_matched": round(min(pex), 4),
        "mean_string_fret_exact": round(sum(sfx) / len(sfx), 4) if sfx else None,
        "total_matched_chords": sum(r["matched_chords"] for r in rows),
    }
    print("\n" + json.dumps(summary, indent=2))
    json.dump({"summary": summary, "pairs": rows}, open(OUT_PATH, "w"), indent=2)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
