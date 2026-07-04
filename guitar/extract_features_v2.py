import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

# Set up package paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from guitar.guitar_features import (
    get_chords_from_xml,
    get_chords_from_tokens,
    get_chords_from_pdf,
    calculate_descriptors_v2
)

V1_COLUMNS = [
    'barre_ratio', 'avg_chord_stretch', 'max_chord_stretch',
    'avg_position_shift', 'fret_change_rate', 'arpeggio_density',
    'avg_string_jump', 'max_string_jump', 'special_technique_ratio',
    'avg_polyphony', 'total_notes'
]

NEW_COLUMNS = [
    'log_total_notes', 'avg_fret', 'p90_fret', 'high_position_ratio',
    'open_string_ratio', 'p90_chord_stretch', 'chord_ratio',
    'avg_string_span', 'unique_shape_rate', 'shift_rate',
    'max_position_shift', 'std_position_shift', 'fret_entropy',
    'string_entropy', 'repetition_ratio'
]

def parse_row_chords(row):
    source = row['source']
    chords = None
    tempo_bpm = None
    technique_ratio = None
    
    # 1. Gaps (XML)
    if source == 'gaps':
        for path_col in ['xml_path', 'file_path']:
            path = row.get(path_col)
            if pd.notna(path):
                if os.path.exists(path):
                    chords, tempo_bpm, technique_ratio = get_chords_from_xml(path)
                    if chords is not None:
                        break
        if chords is None:
            xml_name = os.path.basename(row.get('xml_path', ''))
            alt_path = os.path.join('verified_pieces/all_xmls', xml_name)
            if os.path.exists(alt_path):
                chords, tempo_bpm, technique_ratio = get_chords_from_xml(alt_path)
                
    # 2. Dada GP (Tokens)
    elif source == 'dada_gp':
        for path_col in ['token_path', 'file_path']:
            path = row.get(path_col)
            if pd.notna(path):
                actual_tp = path
                if not os.path.exists(actual_tp):
                    actual_tp = os.path.join('verified_pieces/dada', os.path.basename(path))
                if os.path.exists(actual_tp):
                    chords = get_chords_from_tokens(actual_tp)
                    if chords is not None:
                        break
        # Fallback to XML if tokens failed
        if chords is None:
            for path_col in ['xml_path', 'file_path']:
                path = row.get(path_col)
                if pd.notna(path):
                    actual_xp = path
                    if not os.path.exists(actual_xp):
                        actual_xp = os.path.join('verified_pieces/all_xmls', os.path.basename(path))
                    if os.path.exists(actual_xp):
                        chords, tempo_bpm, technique_ratio = get_chords_from_xml(actual_xp)
                        if chords is not None:
                            break
                            
    # 3. PDF
    elif source == 'pdf':
        for path_col in ['pdf_path', 'file_path']:
            path = row.get(path_col)
            if pd.notna(path):
                actual_pdf = path
                if not os.path.exists(actual_pdf):
                    actual_pdf = os.path.join('verified_pieces/pdf', os.path.basename(path))
                if os.path.exists(actual_pdf):
                    chords, tempo_bpm = get_chords_from_pdf(actual_pdf)
                    if chords is not None:
                        break
                        
    return chords, tempo_bpm, technique_ratio

def process_row(args):
    idx, row = args
    try:
        chords, tempo_bpm, technique_ratio = parse_row_chords(row)
    except Exception as e:
        print(f"Exception parsing row {idx}: {e}")
        chords, tempo_bpm, technique_ratio = None, None, None
        
    res = {}
    if chords is not None:
        res = calculate_descriptors_v2(chords)
        if tempo_bpm is not None:
            res['tempo_bpm'] = tempo_bpm
        else:
            res['tempo_bpm'] = row.get('tempo_bpm', np.nan)
        if technique_ratio is not None:
            res['special_technique_ratio'] = technique_ratio
        else:
            res['special_technique_ratio'] = row.get('special_technique_ratio', 0.0)
    else:
        # Fallback to V1 features
        for col in V1_COLUMNS:
            res[col] = row.get(col, 0.0)
        res['tempo_bpm'] = row.get('tempo_bpm', np.nan)
        for col in NEW_COLUMNS:
            res[col] = 0.0
            
    res['idx'] = idx
    return res

def main():
    csv_path = 'features/guitar_descriptors.csv'
    df = pd.read_csv(csv_path)
    
    print(f"Starting feature extraction V2 for {len(df)} pieces...")
    
    args_list = list(df.iterrows())
    
    results = [None] * len(df)
    completed = 0
    
    with ProcessPoolExecutor() as executor:
        for res in executor.map(process_row, args_list):
            results[res['idx']] = res
            completed += 1
            if completed % 50 == 0:
                print(f"Processed {completed}/{len(df)} pieces...")
                
    # Create the new DataFrame
    new_rows = []
    for idx, res in enumerate(results):
        row = df.iloc[idx].copy()
        for col in V1_COLUMNS + NEW_COLUMNS + ['tempo_bpm']:
            row[col] = res[col]
        new_rows.append(row)
        
    df_v2 = pd.DataFrame(new_rows)
    
    # Impute tempo_bpm missing values
    global_tempo_median = df_v2['tempo_bpm'].dropna().median()
    missing_tempo_count = df_v2['tempo_bpm'].isna().sum()
    df_v2['tempo_bpm'] = df_v2['tempo_bpm'].fillna(global_tempo_median)
    print(f"Imputed {missing_tempo_count} missing tempo values with median: {global_tempo_median}")
    
    # Assert no NaN or inf in feature columns
    all_feature_cols = V1_COLUMNS + NEW_COLUMNS + ['tempo_bpm']
    for col in all_feature_cols:
        assert not df_v2[col].isna().any(), f"NaN found in column {col}"
        assert np.isfinite(df_v2[col]).all(), f"Inf found in column {col}"
        
    # Write to v2 CSV
    out_csv = 'features/guitar_descriptors_v2.csv'
    df_v2.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} successfully.")
    
    # Print Spearman correlation table
    correlations = []
    y = df_v2['Difficulty'].values
    for col in all_feature_cols:
        x = df_v2[col].values
        coef, _ = spearmanr(x, y)
        correlations.append((col, coef))
        
    correlations_sorted = sorted(correlations, key=lambda t: -abs(t[1]))
    
    print("\nSpearman correlations vs Difficulty (raw 1-20):")
    print(f"{'Feature':30s} | {'Correlation (rho)':18s}")
    print("-" * 53)
    for col, coef in correlations_sorted:
        print(f"{col:30s} | {coef:+.4f}")
        
    # Drop features
    dropped_features = []
    surviving_features = []
    
    # Unconditionally drop special_technique_ratio
    dropped_features.append(('special_technique_ratio', 'Data availability artifact (nonzero in 29/716 only)'))
    
    for col, coef in correlations:
        if col == 'special_technique_ratio':
            continue
        # Drop new features if |rho| < 0.05
        if col in NEW_COLUMNS and abs(coef) < 0.05:
            dropped_features.append((col, f"Weak correlation (|rho| = {abs(coef):.4f} < 0.05)"))
        else:
            surviving_features.append((col, coef))
            
    print(f"\nDropped features:")
    for col, reason in dropped_features:
        print(f"  - {col}: {reason}")
        
    # Write feature audit markdown
    audit_path = 'guitar/feature_audit_v2.md'
    with open(audit_path, 'w') as f:
        f.write("# Feature Audit V2\n\n")
        f.write("This audit document evaluates the correlation of both v1 and new v2 features against the raw `Difficulty` target (values 1–20) on the full 716-piece dataset.\n\n")
        f.write("## Spearman Correlation Table\n\n")
        f.write("| Feature | Spearman $\\rho$ | Status |\n")
        f.write("| --- | --- | --- |\n")
        for col, coef in correlations_sorted:
            # Find status
            if col == 'special_technique_ratio':
                status = "Dropped (artifact)"
            elif col in NEW_COLUMNS and abs(coef) < 0.05:
                status = f"Dropped ($|\\rho| < 0.05$)"
            else:
                status = "**Kept**"
            f.write(f"| `{col}` | {coef:+.4f} | {status} |\n")
            
        f.write("\n## Decisions & Rationale\n\n")
        f.write("- `special_technique_ratio` was unconditionally dropped because it is nonzero for only 29/716 pieces, making it a data-availability artifact rather than a meaningful descriptor.\n")
        for col, reason in dropped_features:
            if col != 'special_technique_ratio':
                f.write(f"- `{col}` was dropped due to weak correlation ($|\\rho| < 0.05$).\n")
                
    print(f"\nWrote feature audit to {audit_path}")

if __name__ == '__main__':
    main()
