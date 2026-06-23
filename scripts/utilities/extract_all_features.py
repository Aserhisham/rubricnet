import os
import json
import pandas as pd
from guitar_features import parse_guitar_xml, extract_from_tokens, parse_guitar_pdf

def main():
    with open('features/guitarburst_full.json', 'r') as f:
        data = json.json_load(f)
    
    results = []
    print(f"Starting extraction for {len([i for i in data if i.get('status') == 'found'])} pieces...")
    
    for item in data:
        if item.get('status') != 'found':
            continue
            
        feat = None
        if item.get('source') == 'gaps' and 'xml_path' in item:
            feat = parse_guitar_xml(item['xml_path'])
        elif item.get('source') == 'dada_gp' and 'token_path' in item:
            feat = extract_from_tokens(item['token_path'])
        elif item.get('source') == 'pdf' and 'pdf_path' in item:
            path = item['pdf_path']
            # Verify ClassClef
            is_classclef = False
            if "classclef" in path.lower() or os.path.basename(path).lower().startswith("classclef"):
                is_classclef = True
            else:
                import subprocess
                try:
                    out_sample = subprocess.check_output(['pdftotext', '-l', '1', path, '-'])
                    if b'classclef' in out_sample.lower():
                        is_classclef = True
                except:
                    pass
            if is_classclef:
                feat = parse_guitar_pdf(path)
            
        if feat:
            # Merge features with metadata
            row = item.copy()
            row.update(feat)
            results.append(row)
            
    if results:
        df = pd.DataFrame(results)
        # Drop non-numeric or extra columns for the training set
        df.to_csv('features/guitar_descriptors.csv', index=False)
        print(f"Successfully extracted features for {len(results)} pieces.")
    else:
        print("No features extracted.")

if __name__ == "__main__":
    from guitar_features import parse_guitar_xml, extract_from_tokens, parse_guitar_pdf
    import json
    # Fix the json_load typo
    json.json_load = json.load 
    main()
