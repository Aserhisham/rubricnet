import os
import json
import re
from rapidfuzz import process, fuzz

# Paths
LY_DIR = "symbolic_data/mutopia_ly"
JSON_PATH = "features/guitarburst_full.json"

def roman_to_int(s):
    rom_val = {'i': 1, 'v': 5, 'x': 10, 'l': 50, 'c': 100, 'd': 500, 'm': 1000}
    int_val = 0
    s = s.lower()
    for i in range(len(s)):
        if i > 0 and rom_val[s[i]] > rom_val[s[i-1]]:
            int_val += rom_val[s[i]] - 2 * rom_val[s[i-1]]
        else:
            int_val += rom_val[s[i]]
    return int_val

def extract_numbers(text):
    """Extracts BWV or Opus/Caprice numbers from text."""
    # BWV numbers: match BWV followed by digits
    bwv_match = re.search(r'BWV\s*(\d+)([a-g])?(?![a-z])', text, re.IGNORECASE)
    bwv = (bwv_match.group(1) + (bwv_match.group(2) or "")) if bwv_match else None
    
    # Opus numbers
    opus = re.findall(r'Op(?:[us\.]*)?\s*(\d+)', text, re.IGNORECASE)
    # Number (e.g., No. 1, nr1, #1)
    no = re.findall(r'(?:No|nr|num|#)\.?\s*(\d+)', text, re.IGNORECASE)
    # Caprice number
    caprice = re.findall(r'Caprice\s*(\d+)', text, re.IGNORECASE)
    
    # Movement number (Roman or Digits)
    # Match things like "v. Sarabande" or "4. Gigue" or "iv. Presto"
    mvmt_match = re.search(r'(?:^|[\.\:\s])([ivxl]+|[1-9])\.\s+', text, re.IGNORECASE)
    mvmt = None
    if mvmt_match:
        m = mvmt_match.group(1).lower()
        if all(c in 'ivxl' for c in m):
            mvmt = str(roman_to_int(m))
        else:
            mvmt = m

    return {
        'bwv': bwv.lower() if bwv else None,
        'opus': opus[0] if opus else None,
        'no': no[0] if no else None,
        'caprice': caprice[0] if caprice else None,
        'mvmt': mvmt
    }

def match_ly():
    if not os.path.exists(JSON_PATH):
        print("JSON data not found.")
        return

    with open(JSON_PATH, 'r') as f:
        data = json.load(f)
        
    ly_files = os.listdir(LY_DIR)
    ly_files = [f for f in ly_files if f.endswith('.ly')]
    
    matches_count = 0
    
    for entry in data:
        if entry.get('status') == 'found':
            continue
            
        title = entry['Title']
        composer = entry['Composer']
        target_nums = extract_numbers(f"{composer} {title}")
        
        best_match = None
        best_score = 0
        
        # Mapping for specific titles
        if "Bist du bei mir" in title:
            target_nums['custom'] = 'bistdubeimir'

        for ly_file in ly_files:
            file_nums = extract_numbers(ly_file.replace('-', ' ').replace('_', ' '))
            score = 0
            
            # Custom hardcoded matches
            if 'custom' in target_nums and target_nums['custom'] in ly_file.lower().replace('-', '').replace('_', ''):
                score = 100

            # Strict Numeric Matching - VERY HIGH priority
            elif target_nums['bwv'] and file_nums['bwv']:
                if target_nums['bwv'] == file_nums['bwv']:
                    score = 90
                    # Check movement
                    # ly files often have _1.ly or -1.ly
                    f_mvmt_match = re.search(r'[_-](\d+)\.ly$', ly_file)
                    f_mvmt = f_mvmt_match.group(1) if f_mvmt_match else None
                    
                    if target_nums['mvmt'] and f_mvmt and target_nums['mvmt'] == f_mvmt:
                        score = 100
                    elif target_nums['mvmt'] and target_nums['mvmt'] in clean_name(ly_file):
                        score = 98
            
            elif target_nums['caprice'] and file_nums['caprice']:
                if target_nums['caprice'] == file_nums['caprice']:
                    score = 95
            elif target_nums['opus'] and file_nums['opus']:
                if target_nums['opus'] == file_nums['opus']:
                    composer_match = composer.split()[-1].lower() in ly_file.lower()
                    if composer_match:
                        score = 85
                        if target_nums['no'] and file_nums['no'] and target_nums['no'] == file_nums['no']:
                            score = 100
            
            # Fuzzy fallback
            if score < 85:
                clean_ly = ly_file.replace('_', ' ').replace('-', ' ').replace('.ly', '')
                f_score = fuzz.token_set_ratio(clean_ly, f"{composer} {title}")
                if f_score > score:
                    score = f_score
            
            if score > best_score:
                best_score = score
                best_match = ly_file
        
        if best_match and best_score > 85:
            entry['status'] = 'found'
            entry['source'] = 'mutopia_ly'
            entry['ly_path'] = os.path.abspath(os.path.join(LY_DIR, best_match))
            matches_count += 1
            print(f"Matched: {composer} - {title} -> {best_match} ({best_score:.1f})")

    if matches_count > 0:
        with open(JSON_PATH, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Updated JSON with {matches_count} new LilyPond matches.")
    else:
        print("No new matches found.")

def clean_name(s):
    return s.lower().replace('_', ' ').replace('-', ' ').replace('.ly', '')

if __name__ == "__main__":
    match_ly()