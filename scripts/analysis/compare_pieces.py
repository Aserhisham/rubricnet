import os
import pandas as pd

def main():
    # Paths relative to the script's directory
    dir_path = os.path.dirname(os.path.abspath(__file__))
    csv_abs = os.path.join(dir_path, 'found_pieces.csv')
    excel_in_abs = os.path.join(dir_path, 'found_pieces.xlsx')
    excel_out_abs = os.path.join(dir_path, 'found_pieces_not_in_excel.xlsx')
    
    print(f"Reading CSV from: {csv_abs}")
    df_csv = pd.read_csv(csv_abs)
    print(f"Loaded {len(df_csv)} rows.")
    
    print(f"Reading Excel from: {excel_in_abs}")
    df_xlsx = pd.read_excel(excel_in_abs)
    print(f"Loaded {len(df_xlsx)} rows.")
    
    # Normalize Title and Composer for robust matching
    def normalize_key(row):
        return (str(row['Title']).strip().lower(), str(row['Composer']).strip().lower())
    
    xlsx_keys = set(df_xlsx.apply(normalize_key, axis=1))
    print(f"Total unique (Title, Composer) keys in Excel: {len(xlsx_keys)}")
    
    df_csv['normalized_key'] = df_csv.apply(normalize_key, axis=1)
    
    # Filter rows that are not in the Excel file
    df_diff = df_csv[~df_csv['normalized_key'].isin(xlsx_keys)].copy()
    df_diff = df_diff.drop(columns=['normalized_key'])
    
    # Prepend the 'validated' column filled with '?' to match the original Excel structure
    df_diff.insert(0, 'validated', '?')
    
    print(f"Exporting {len(df_diff)} rows to: {excel_out_abs}")
    df_diff.to_excel(excel_out_abs, index=False)
    print("Export completed successfully!")

if __name__ == "__main__":
    main()
