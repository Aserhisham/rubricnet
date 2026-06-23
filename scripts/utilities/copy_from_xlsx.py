#!/usr/bin/env python3
"""
Copy files from 'data/to_copy.xlsx' where flag (col 0) == 1 into 'verified_pieces/'.

Routing:
  pdf     -> verified_pieces/pdf/      (path from col 9)
  dada_gp -> verified_pieces/dada/     (col 10 tokens + col 11 gp file)
  gaps    -> verified_pieces/gaps/     (path from col 13)

Conflict cases go to 'data/review_duplicates.xlsx' instead of being copied:
  - Same title+composer appearing more than once with flag=1
  - Any row with a secondary path in col 14

Existing destination files are skipped (not overwritten).
"""

import os
import shutil
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
XLSX_PATH = os.path.join(BASE_DIR, "data", "to_copy.xlsx")
TARGET_DIR = os.path.join(BASE_DIR, "verified_pieces")
REVIEW_XLSX = os.path.join(BASE_DIR, "data", "review_duplicates.xlsx")

COL_NAMES = {
    0: "copy_flag",
    1: "title",
    2: "composer",
    3: "difficulty_num",
    4: "difficulty_num2",
    5: "difficulty_grade",
    6: "era",
    7: "found_status",
    8: "source",
    9: "pdf_path",
    10: "tokens_path",
    11: "gp_path",
    12: "gaps_id",
    13: "gaps_xml_path_1",
    14: "gaps_xml_path_2",
}


def is_filled(val) -> bool:
    return isinstance(val, str) and val.strip() and val.strip().lower() != "nan"


def safe_copy(src_rel: str, dest_subdir: str) -> tuple[str, str]:
    src = os.path.join(BASE_DIR, src_rel.strip())
    if not os.path.exists(src):
        return "missing", src
    dest_dir = os.path.join(TARGET_DIR, dest_subdir)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, os.path.basename(src))
    if os.path.exists(dest):
        return "skipped", dest
    shutil.copy2(src, dest)
    return "copied", dest


def main():
    df = pd.read_excel(XLSX_PATH, header=None)

    flag1 = df[df[0] == 1].copy()
    print(f"Rows with flag=1: {len(flag1)}")

    # Conflict mask 1: same title+composer appearing more than once
    dupe_sizes = flag1.groupby([1, 2])[0].transform("size")
    dupe_mask = dupe_sizes > 1

    # Conflict mask 2: secondary path in col 14
    col14_mask = flag1[14].apply(lambda v: is_filled(str(v)))

    conflict_mask = dupe_mask | col14_mask
    conflict_rows = flag1[conflict_mask].copy()
    conflict_rows["conflict_reason"] = ""
    both = dupe_mask & col14_mask
    conflict_rows.loc[both, "conflict_reason"] = "Duplicate piece AND secondary path"
    conflict_rows.loc[dupe_mask & ~col14_mask, "conflict_reason"] = (
        "Piece appears multiple times with flag=1"
    )
    conflict_rows.loc[~dupe_mask & col14_mask, "conflict_reason"] = (
        "Has secondary file path in col 14"
    )

    to_copy = flag1[~conflict_mask].copy()
    print(f"  → {len(to_copy)} rows to copy, {len(conflict_rows)} rows to review")

    results: dict[str, list[str]] = {"copied": [], "skipped": [], "missing": []}

    for _, row in to_copy.iterrows():
        source = str(row[8]).strip()

        if source == "pdf":
            path = str(row[9])
            if is_filled(path):
                status, dest = safe_copy(path, "pdf")
                results[status].append(dest)

        elif source == "dada_gp":
            for col in [10, 11]:
                path = str(row[col])
                if is_filled(path):
                    status, dest = safe_copy(path, "dada")
                    results[status].append(dest)

        elif source == "gaps":
            path = str(row[13])
            if is_filled(path):
                status, dest = safe_copy(path, "gaps")
                results[status].append(dest)

        else:
            print(f"  [WARN] Unknown source '{source}' for: {row[1]}")

    # Write review excel with readable column names
    review = conflict_rows.rename(columns=COL_NAMES)
    review["conflict_reason"] = conflict_rows["conflict_reason"].values
    review.to_excel(REVIEW_XLSX, index=False)

    print(f"\nResults:")
    print(f"  Copied:  {len(results['copied'])} files")
    print(f"  Skipped: {len(results['skipped'])} files (already exist)")
    print(f"  Missing: {len(results['missing'])} files (source not found)")
    print(f"  Review excel: {len(conflict_rows)} rows → {REVIEW_XLSX}")

    if results["missing"]:
        print("\nMissing source files:")
        for m in results["missing"]:
            print(f"  {m}")


if __name__ == "__main__":
    main()
