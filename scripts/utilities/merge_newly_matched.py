"""
One-off merge of data/newly_matched_pieces.xlsx (second matching sweep) into
data/verified pieces.xlsx, the source xlsx read by extract_all_features.py.

Resolution decisions (see thesis roadmap discussion):
- 12 rows are genuinely new pieces -> appended, with xml_path pointed at the
  skeleton MusicXML already sitting on disk (verified_pieces/pdf/xml/).
- Asturias (Leyenda): same PDF as an existing row, but the existing row's
  difficulty (8, titled "...(Theme)") looks like a catalog mislabel -- the
  MusicXML has 240 measures / 1179 notes, far too long to be a short theme
  excerpt, and Albeniz's Asturias is a substantial virtuosic piece. The new
  row's difficulty (15) is more credible for that content, so the existing
  row's Title/Difficulty are corrected in place rather than adding a duplicate.
- Grand Overture Op.61, La Maja de Goya, BWV 1006a Bourree, Valses Poeticos
  No.6: same PDF as an existing row with a near-identical or unresolvable
  difficulty conflict -> dropped, existing row kept as-is.
- El Decameron Negro (II/III) and Grand Solo Op.14 (3 editions): different
  GuitarBurst catalog difficulty per movement/edition, but all sharing one
  PDF that isn't segmented by movement -> left out entirely, documented as a
  known dataset limitation rather than a false training signal.
"""
import os

import pandas as pd

MAIN_XLSX = "data/verified pieces.xlsx"
NEW_XLSX = "data/newly_matched_pieces.xlsx"

CLEAN_NEW_TITLES = [
    "24 Caprices, Op. 1: 1",
    "24 Caprices, Op. 1: 13",
    "24 Caprices, Op. 1: 16",
    "24 Caprices, Op. 1: 2",
    "24 Caprices, Op. 1: 20",
    "24 Caprices, Op. 1: 24",
    "Bajando de la meseta",
    "Choro da Saudade",
    "Entre olivares",
    "Goldberg Variations, BWV 988: Variatio 1",
    "Jesu, Joy of Man's Desiring (from Cantata No. 147)",
    "Suite IV, BWV 1006a: iii. Gavotte en rondeau",
]

ASTURIAS_OLD_TITLE = "Asturias (Leyenda) (Theme)"
ASTURIAS_NEW_TITLE = "Leyenda (Asturias) from Suite Espanola"
ASTURIAS_NEW_DIFFICULTY = 15


def main():
    main_df = pd.read_excel(MAIN_XLSX)
    new_df = pd.read_excel(NEW_XLSX)

    # 1. Correct the Asturias difficulty/title in place.
    mask = main_df["Title"] == ASTURIAS_OLD_TITLE
    assert mask.sum() == 1, f"expected exactly 1 Asturias row, found {mask.sum()}"
    main_df.loc[mask, "Title"] = ASTURIAS_NEW_TITLE
    main_df.loc[mask, "Difficulty"] = ASTURIAS_NEW_DIFFICULTY

    # 2. Append the 12 clean new rows, pointing xml_path at the skeleton
    # MusicXML already produced on disk for each PDF.
    to_add = new_df[new_df["Title"].isin(CLEAN_NEW_TITLES)].copy()
    assert len(to_add) == len(CLEAN_NEW_TITLES), (
        f"expected {len(CLEAN_NEW_TITLES)} clean rows, matched {len(to_add)}"
    )

    def xml_path_for(pdf_path):
        basename = os.path.splitext(os.path.basename(pdf_path))[0]
        xml_path = os.path.join("verified_pieces", "pdf", "xml", basename + ".musicxml")
        assert os.path.exists(xml_path), f"missing skeleton xml: {xml_path}"
        return xml_path

    to_add["xml_path"] = to_add["pdf_path"].apply(xml_path_for)
    to_add["validated"] = 1

    merged = pd.concat([main_df, to_add], ignore_index=True)
    merged.to_excel(MAIN_XLSX, index=False)
    print(f"Main xlsx: {len(main_df)} -> {len(merged)} rows (+{len(to_add)} new, 1 corrected)")


if __name__ == "__main__":
    main()
