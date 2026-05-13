import json
import os
import pandas as pd
import re

def clean_filename(name):
    """Matches the logic used in rename_gaps.py"""
    if not isinstance(name, str):
        return "Unknown"
    name = re.sub(r'[\\/*?:"<>|]', "-", name)
    name = name.encode('ascii', 'ignore').decode('ascii')
    return name.strip()

def sync_found_pieces():
    json_path = 'features/guitarburst_full.json'
    gaps_xml_dir = 'gaps_v1/musicxml/'
    readable_xml_dir = 'gaps_v1/readable_musicxml/'
    output_csv = 'features/found_pieces.csv'
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    available_hashes = {f.replace('.xml', '') for f in os.listdir(gaps_xml_dir) if f.endswith('.xml')}
    
    # Map scorehashes to readable filenames using the SAME logic as rename_gaps.py
    hash_to_readable = {}
    if os.path.exists(readable_xml_dir):
        try:
            metadata = pd.read_csv('gaps_v1/gaps_v1_metadata.csv', encoding='latin1')
            for _, row in metadata.iterrows():
                h = str(row['scorehash'])
                title = clean_filename(str(row['title']))
                composer = clean_filename(str(row.get('composer_name_normalized', 'Unknown')))
                
                # Check for standard naming
                readable_name = f"{composer} - {title}.xml"
                
                # Check for the version with (scorehash) in case of duplicates
                if not os.path.exists(os.path.join(readable_xml_dir, readable_name)):
                    alt_name = f"{composer} - {title} ({h}).xml"
                    if os.path.exists(os.path.join(readable_xml_dir, alt_name)):
                        readable_name = alt_name
                
                hash_to_readable[h] = readable_name
        except Exception as e:
            print(f"Warning: Could not load metadata for readable mapping: {e}")

    print(f"Found {len(available_hashes)} XML files in GAPS directory.")

    found_entries = []
    updated_count = 0
    already_found = 0
    
    for entry in data:
        scorehash = entry.get('scorehash') or entry.get('file_path', '').split('/')[-1].replace('.xml', '')
        
        if scorehash in available_hashes:
            if entry.get('status') != 'found':
                entry['status'] = 'found'
                updated_count += 1
            else:
                already_found += 1
            
            entry['scorehash'] = scorehash
            entry['xml_path'] = os.path.join(gaps_xml_dir, scorehash + '.xml')
            
            # Remove redundant file_path if it exists
            if 'file_path' in entry:
                del entry['file_path']
            
            # Find readable path
            readable_name = hash_to_readable.get(scorehash)
            if readable_name and os.path.exists(os.path.join(readable_xml_dir, readable_name)):
                entry['readable_path'] = os.path.join(readable_xml_dir, readable_name)
            else:
                entry['readable_path'] = "not_available"

            found_entries.append(entry)
        else:
            if entry.get('status') == 'found':
                 entry['status'] = 'not_found'

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"Updated JSON: {updated_count} new pieces found, {already_found} were already marked.")

    if found_entries:
        df = pd.DataFrame(found_entries)
        # Select and reorder columns for the CSV
        columns_to_save = ['Composer', 'Title', 'Difficulty', 'scorehash', 'xml_path', 'readable_path']
        existing_cols = [c for c in columns_to_save if c in df.columns]
        df[existing_cols].to_csv(output_csv, index=False)
        print(f"Exported {len(found_entries)} found pieces to {output_csv}")
    else:
        print("No pieces found to export to CSV.")

if __name__ == "__main__":
    sync_found_pieces()
