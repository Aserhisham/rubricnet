import os
import json
import pandas as pd
from guitar_features import parse_guitar_xml, extract_from_tokens, parse_guitar_pdf

XLSX_PATH = 'data/verified pieces.xlsx'

def resolve_verified_path(rel_path, category):
    if not isinstance(rel_path, str) or not rel_path.strip() or str(rel_path).lower() == 'nan':
        return None
    rel_path_stripped = rel_path.strip()
    if os.path.exists(rel_path_stripped):
        return rel_path_stripped
    basename = os.path.basename(rel_path_stripped)
    verified_path = os.path.join("verified_pieces", category, basename)
    if os.path.exists(verified_path):
        return verified_path
    if category == 'dada' and basename.endswith('.musicxml'):
        verified_path_xml = os.path.join("verified_pieces", "dada", "xml", basename)
        if os.path.exists(verified_path_xml):
            return verified_path_xml
    if category == 'pdf' and basename.endswith('.musicxml'):
        verified_path_xml = os.path.join("verified_pieces", "pdf", "xml", basename)
        if os.path.exists(verified_path_xml):
            return verified_path_xml
    return None


def main():
    if not os.path.exists(XLSX_PATH):
        print(f"Error: {XLSX_PATH} not found.")
        return
        
    print(f"Loading verified pieces from {XLSX_PATH}...")
    df = pd.read_excel(XLSX_PATH)
    
    # Filter for validated pieces
    validated_df = df[df['validated'] == 1].copy()
    data = validated_df.to_dict('records')
    
    results = []
    print(f"Starting feature extraction for {len(data)} validated pieces...")
    
    for item in data:
        feat = None
        source = item.get('source')
        title = item.get('Title', 'Unknown')
        
        if source == 'gaps':
            xml_path = resolve_verified_path(item.get('xml_path'), 'gaps')
            if xml_path:
                feat = parse_guitar_xml(xml_path)
            else:
                print(f"Warning: XML file for gaps piece '{title}' not found.")
                
        elif source == 'dada_gp':
            xml_path = resolve_verified_path(item.get('xml_path'), 'dada')
            if xml_path:
                feat = parse_guitar_xml(xml_path)
            if not feat:
                # Fallback to tokens
                token_path = resolve_verified_path(item.get('token_path'), 'dada')
                if token_path:
                    feat = extract_from_tokens(token_path)
                else:
                    print(f"Warning: Neither XML nor tokens file for dada_gp piece '{title}' found.")
                    
        elif source == 'pdf':
            xml_path = resolve_verified_path(item.get('xml_path'), 'pdf')
            if xml_path:
                feat = parse_guitar_xml(xml_path)
            else:
                # Fallback to direct PDF parsing
                pdf_path = resolve_verified_path(item.get('pdf_path'), 'pdf')
                if pdf_path:
                    # Verify ClassClef
                    is_classclef = False
                    if "classclef" in pdf_path.lower() or os.path.basename(pdf_path).lower().startswith("classclef"):
                        is_classclef = True
                    else:
                        import subprocess
                        try:
                            out_sample = subprocess.check_output(['pdftotext', '-l', '1', pdf_path, '-'])
                            if b'classclef' in out_sample.lower():
                                is_classclef = True
                        except:
                            pass
                    if is_classclef:
                        feat = parse_guitar_pdf(pdf_path)
                else:
                    print(f"Warning: Neither XML nor PDF file for pdf piece '{title}' found.")
                
        if feat:
            # Merge features with metadata
            row = item.copy()
            row.update(feat)
            results.append(row)
            
    if results:
        df_out = pd.DataFrame(results)
        df_out.to_csv('features/guitar_descriptors.csv', index=False)
        print(f"Successfully extracted features for {len(results)} / {len(data)} pieces.")
    else:
        print("No features extracted.")

if __name__ == "__main__":
    main()

