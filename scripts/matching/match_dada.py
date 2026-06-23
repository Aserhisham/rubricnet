import json
import os
import re
import pandas as pd
from difflib import SequenceMatcher

def normalize(s):
    if not isinstance(s, str): return ""
    return re.sub(r'[^a-z0-9]', '', s.lower())

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def get_words(s):
    if not s: return set()
    s = s.lower()
    s = re.sub(r'[^a-z0-9 ]', ' ', s)
    return set(s.split())

def match_dada_gp():
    print("Loading datasets...")
    json_path = 'features/guitarburst_full.json'
    dada_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "datasets", "DadaGP-v1.1"))
    meta_path = os.path.join(dada_root, '_DadaGP_all_metadata.json')
    
    if not os.path.exists(dada_root):
        print(f"Error: {dada_root} not found.")
        return
    if not os.path.exists(meta_path):
        print(f"Error: Metadata JSON {meta_path} not found.")
        return

    with open(json_path, 'r') as f:
        gb_data = json.load(f)

    with open(meta_path, 'r') as f:
        dada_meta = json.load(f)

    # 1. Pre-index GuitarBurst by composer name words for fast lookup
    from collections import defaultdict
    comp_to_items = defaultdict(list)
    title_to_items = defaultdict(list)
    num_to_items = defaultdict(list)

    for item in gb_data:
        comp_norm = item.get('Composer', '').lower().replace('mandore', 'mangore')
        comp_words = set(re.sub(r'[^a-z0-9 ]', ' ', comp_norm).split())
        title_norm = normalize(item.get('Title', ''))
        
        item['_comp_words'] = comp_words
        item['_title_norm'] = title_norm
        
        for w in comp_words:
            comp_to_items[w].append(item)
        if title_norm:
            title_to_items[title_norm].append(item)
        
        # Numbers for Op., BWV, etc.
        for n in re.findall(r'\d+', title_norm):
            num_to_items[n].append(item)

    found_count = 0
    important_composers = {'bach', 'sor', 'giuliani', 'tarrega', 'carcassi', 'aguado', 'barrios', 'sanz', 'coste', 'paganini'}

    print(f"Processing {len(dada_meta)} Dada-GP entries...")
    
    for rel_path, info in dada_meta.items():
        # rel_path is e.g. "B/Bach, Johann Sebastian/Bach...tokens.txt"
        artist_token = info.get('artist_token', '').replace('artist:', '').replace('_', ' ')
        artist_words = set(re.sub(r'[^a-z0-9 ]', ' ', artist_token.lower()).split())
        
        filename = os.path.basename(rel_path)
        song_full = filename.split('.tokens.txt')[0]
        song_title = song_full
        artist_part = ""
        if ' - ' in song_full:
            parts = song_full.split(' - ')
            artist_part = parts[0].lower()
            song_title = ' - '.join(parts[1:])
        
        song_title = re.sub(r'\.gp[345x]$', '', song_title)
        song_norm = normalize(song_title)
        song_words = set(re.sub(r'[^a-z0-9 ]', ' ', song_title.lower()).split())
        all_art_words = artist_words | set(artist_part.split()) | set(re.sub(r'[^a-z0-9 ]', ' ', os.path.dirname(rel_path).lower()).split())

        # Candidates
        candidates = []
        for w in all_art_words:
            candidates.extend(comp_to_items[w])
        
        # Broad Search: if no candidates, check unique titles
        if not candidates and len(song_norm) > 12:
            candidates.extend(title_to_items.get(song_norm, []))

        unique_candidates = {id(i): i for i in candidates}.values()
        
        for item in unique_candidates:
            if item.get('status') == 'found' and item.get('source') == 'dada_gp':
                continue
            
            # Match Composer
            intersect = all_art_words.intersection(item['_comp_words'])
            is_imp = any(c in all_art_words and c in item['_comp_words'] for c in important_composers)
            
            art_match = (len(intersect) >= 2) or (len(intersect) >= 1 and is_imp)
            
            if art_match:
                # Match Title
                match = False
                it_norm = item['_title_norm']
                item_title_words = get_words(item.get('Title',''))
                
                if song_norm == it_norm:
                    match = True
                else:
                    # Parse numbers properly
                    file_nums = sorted(re.findall(r'\d+', song_norm))
                    item_nums = sorted(re.findall(r'\d+', it_norm))
                    
                    # 1. Number subset match (Opus AND Number must exist if present)
                    gt_set = set(file_nums)
                    it_set = set(item_nums)
                    if gt_set and it_set and (gt_set.issubset(it_set) or it_set.issubset(gt_set)):
                        common = song_words.intersection(item_title_words)
                        if len(common) >= 1 or len(item_title_words) == 0:
                            match = True
                    
                    # 2. Movement/Part matching (Substrings) - only if numbers don't conflict
                    elif len(song_norm) > 6 and (song_norm in it_norm or it_norm in song_norm):
                        if file_nums == item_nums:
                            match = True
                    
                    # 3. Fuzzy matching for long titles
                    elif len(item_title_words) > 3 and is_imp:
                        common = song_words.intersection(item_title_words)
                        if len(common) / len(item_title_words) >= 0.7:
                            match = True
                
                if match:
                    item['status'] = 'found'
                    item['source'] = 'dada_gp'
                    item['token_path'] = os.path.join(dada_root, rel_path)
                    
                    # Try to find GP file
                    gp_base = filename.split('.tokens.txt')[0]
                    gp_path = os.path.join(dada_root, os.path.dirname(rel_path), gp_base)
                    if os.path.exists(gp_path):
                        item['gp_path'] = gp_path
                    
                    found_count += 1
                    break

    # Finalize
    for item in gb_data:
        for k in ['_comp_words', '_title_norm']:
            if k in item: del item[k]

    with open(json_path, 'w') as f:
        json.dump(gb_data, f, indent=4)
    
    print(f"Successfully matched/updated pieces in total.")

if __name__ == "__main__":
    match_dada_gp()
