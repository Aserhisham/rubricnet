import os
import pandas as pd
import re
from rapidfuzz import process, fuzz

# Paths
MIDI_DIR = "symbolic_data/mutopia_midi"
TARGET_LIST = "features/to_find_list.csv"
FOUND_CSV = "features/found_pieces.csv"

def extract_numbers(text):
    """Extracts BWV or Opus numbers from text."""
    # BWV numbers (e.g., BWV 1006, BWV1006)
    bwv = re.findall(r'BWV\s*(\d+[a-z]?)', text, re.IGNORECASE)
    # Opus numbers (e.g., Op. 1, Opus 35)
    opus = re.findall(r'Op(?:[us\.]*)?\s*(\d+)', text, re.IGNORECASE)
    # Number (e.g., No. 1, nr1)
    no = re.findall(r'(?:No|nr|num)\.?\s*(\d+)', text, re.IGNORECASE)
    
    return {
        'bwv': bwv[0].lower() if bwv else None,
        'opus': opus[0] if opus else None,
        'no': no[0] if no else None
    }

def match_files():
    if not os.path.exists(TARGET_LIST):
        print("Target list not found.")
        return

    targets = pd.read_csv(TARGET_LIST)
    midi_files = os.listdir(MIDI_DIR)
    
    matches = []
    
    print(f"Attempting to match {len(midi_files)} files against {len(targets)} targets...")
    
    for filename in midi_files:
        if not filename.endswith('.mid'):
            continue
            
        file_nums = extract_numbers(filename)
        best_match = None
        best_score = 0
        
        # Filter targets by composer if possible (rough check)
        # Mutopia filenames often start with composer name or initials
        # but let's just search all targets for now to be safe
        
        for idx, row in targets.iterrows():
            target_title = f"{row['Composer']} {row['Title']}"
            target_nums = extract_numbers(target_title)
            
            score = 0
            
            # Numeric matching (highest priority)
            if file_nums['bwv'] and target_nums['bwv']:
                if file_nums['bwv'] == target_nums['bwv']:
                    score = 100
            elif file_nums['opus'] and target_nums['opus']:
                if file_nums['opus'] == target_nums['opus']:
                    score = 80
                    if file_nums['no'] and target_nums['no'] and file_nums['no'] == target_nums['no']:
                        score = 100
            
            # Fuzzy title matching if score is not perfect
            if score < 90:
                fuzzy_score = fuzz.token_set_ratio(filename.replace('_', ' '), target_title)
                if fuzzy_score > score:
                    score = fuzzy_score
            
            if score > best_score:
                best_score = score
                best_match = row
                
        if best_match is not None and best_score > 70:
            matches.append({
                'Composer': best_match['Composer'],
                'Title': best_match['Title'],
                'Filename': filename,
                'Score': best_score
            })
    
    matches_df = pd.DataFrame(matches)
    if not matches_df.empty:
        # Drop duplicates in case multiple files match the same target
        matches_df = matches_df.sort_values('Score', ascending=False).drop_duplicates(['Composer', 'Title'])
        print(f"Successfully matched {len(matches_df)} pieces.")
        matches_df.to_csv("features/mutopia_matches.csv", index=False)
    else:
        print("No matches found.")

if __name__ == "__main__":
    match_files()
