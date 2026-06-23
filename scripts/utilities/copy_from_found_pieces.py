#!/usr/bin/env python3
"""
Copy files from 'features/found_pieces.xlsx' where validated == 1 into 'verified_pieces/'.

Routing:
  pdf     -> verified_pieces/pdf/      (pdf_path)
  dada_gp -> verified_pieces/dada/     (token_path + gp_path)
  gaps    -> verified_pieces/gaps/     (xml_path)

Conflict cases go to 'data/review_duplicates.xlsx' instead of being copied:
  - Same Title+Composer appearing more than once with validated=1
  - Any row with a secondary path in 'file_path'

Existing destination files are skipped (not overwritten).
"""

import os
import shutil
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
XLSX_PATH = os.path.join(BASE_DIR, "features", "found_pieces.xlsx")
TARGET_DIR = os.path.join(BASE_DIR, "verified_pieces")
REVIEW_XLSX = os.path.join(BASE_DIR, "data", "review_duplicates.xlsx")


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
    df = pd.read_excel(XLSX_PATH)

    validated = df[df["validated"] == 1].copy()
    print(f"Rows with validated=1: {len(validated)}")

    # Conflict mask 1: same Title+Composer appearing more than once
    dupe_sizes = validated.groupby(["Title", "Composer"])["validated"].transform("size")
    dupe_mask = dupe_sizes > 1

    # Conflict mask 2: secondary path in file_path
    file_path_mask = validated["file_path"].apply(lambda v: is_filled(str(v)))

    conflict_mask = dupe_mask | file_path_mask
    conflict_rows = validated[conflict_mask].copy()
    conflict_rows["conflict_reason"] = ""
    both = dupe_mask & file_path_mask
    conflict_rows.loc[both, "conflict_reason"] = "Duplicate piece AND secondary path"
    conflict_rows.loc[dupe_mask & ~file_path_mask, "conflict_reason"] = (
        "Piece appears multiple times with validated=1"
    )
    conflict_rows.loc[~dupe_mask & file_path_mask, "conflict_reason"] = (
        "Has secondary file path in 'file_path'"
    )

    to_copy = validated[~conflict_mask].copy()
    print(f"  -> {len(to_copy)} rows to copy, {len(conflict_rows)} rows to review")

    results: dict[str, list[str]] = {"copied": [], "skipped": [], "missing": []}

    for _, row in to_copy.iterrows():
        source = str(row["source"]).strip()

        if source == "pdf":
            path = str(row["pdf_path"])
            if is_filled(path):
                status, dest = safe_copy(path, "pdf")
                results[status].append(dest)

        elif source == "dada_gp":
            for col in ["token_path", "gp_path"]:
                path = str(row[col])
                if is_filled(path):
                    status, dest = safe_copy(path, "dada")
                    results[status].append(dest)

        elif source == "gaps":
            path = str(row["xml_path"])
            if is_filled(path):
                status, dest = safe_copy(path, "gaps")
                results[status].append(dest)

        else:
            print(f"  [WARN] Unknown source '{source}' for: {row['Title']}")

    conflict_rows.to_excel(REVIEW_XLSX, index=False)

    print(f"\nResults:")
    print(f"  Copied:  {len(results['copied'])} files")
    print(f"  Skipped: {len(results['skipped'])} files (already exist)")
    print(f"  Missing: {len(results['missing'])} files (source not found)")
    print(f"  Review excel: {len(conflict_rows)} rows -> {REVIEW_XLSX}")

    if results["missing"]:
        print("\nMissing source files:")
        for m in results["missing"]:
            print(f"  {m}")


if __name__ == "__main__":
    main()
