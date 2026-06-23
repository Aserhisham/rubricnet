import json
import pandas as pd
import re
from collections import defaultdict

def normalize(text):
    if not text: return ""
    text = text.lower()
    text = re.sub(r'[^\w\s\d]', ' ', text)
    return " ".join(text.split())

def loose_match():
    print("Loading data...")
    with open('features/guitarburst_full.json') as f: gb_data = json.load(f)
    with open('datasets/DadaGP-v1.1/_DadaGP_all_metadata.json') as f: dada_meta = json.load(f)
    gaps_df = pd.read_csv('datasets/gaps_v1/gaps_v1_metadata.csv')
    
    targets = [i for i in gb_data if i.get('status', 'not_found') == 'not_found']
    vetted = {normalize(i.get('Title','')) + normalize(i.get('Composer','')) for i in gb_data if i.get('status') == 'found'}
    
    candidates = []
    
    # 1. Index Dada-GP by important classical guitar composer last names
    important = {'bach', 'sor', 'giuliani', 'carcassi', 'aguado', 'tarrega', 'albeniz', 'barrios', 'villa', 'lobos', 'brower', 'ponce'}
    dada_index = defaultdict(list)
    for path in dada_meta.keys():
        p_norm = normalize(path)
        p_words = set(p_norm.split())
        # Check if any important composer is in path
        inter = p_words.intersection(important)
        if inter:
            for c in inter:
                dada_index[c].append({'path': path, 'p_norm': p_norm, 'p_words': p_words})

    print(f"Scanning {len(targets)} pieces...")
    for item in targets:
        title, composer = item.get('Title', ''), item.get('Composer', '')
        t_norm, c_norm = normalize(title), normalize(composer)
        if not t_norm or not c_norm: continue
        
        t_words = set(t_norm.split())
        t_nums = set(re.findall(r'\d+', t_norm))
        c_words = set(c_norm.split())
        c_search = c_words.intersection(important)
        
        # Search DadaGP
        potential = []
        if c_search:
            for c in c_search: potential.extend(dada_index.get(c, []))
        else:
            # Fallback for other composers
            c_last = c_norm.split()[-1] if c_norm.split() else ''
            if len(c_last) > 3:
                for path, p_info in dada_index.items(): # This is slow but subset
                    pass # skip for now to keep speed
        
        seen = set()
        for d in potential:
            if d['path'] in seen: continue
            seen.add(d['path'])
            
            match_reason = None
            p_nums = set(re.findall(r'\d+', d['p_norm']))
            common_words = t_words.intersection(d['p_words'])
            
            # Match Logic
            if t_nums and p_nums and t_nums.intersection(p_nums):
                if common_words or len(t_words) == 0:
                    match_reason = f"Num:{t_nums.intersection(p_nums)}"
            elif len(common_words) >= max(1, len(t_words)*0.4):
                match_reason = f"Words:{len(common_words)}/{len(t_words)}"
            
            if match_reason:
                candidates.append({
                    'Target': f"{composer} - {title}",
                    'Candidate': d['path'],
                    'Source': 'DadaGP',
                    'Reason': match_reason,
                    'Path': d['path']
                })
        
        # Search GAPS
        g_matches = gaps_df[gaps_df['composer'].fillna('').str.lower().str.contains(c_norm.split()[-1] if c_norm.split() else 'xyz', na=False)]
        for _, g in g_matches.iterrows():
            g_title = str(g.get('title', ''))
            g_norm = normalize(g_title)
            g_words = set(g_norm.split())
            g_nums = set(re.findall(r'\d+', g_norm))
            
            match_reason = None
            common = t_words.intersection(g_words)
            if t_nums and g_nums and t_nums.intersection(g_nums):
                if common or len(t_words) == 0:
                    match_reason = "Num"
            elif len(common) >= max(1, len(t_words)*0.4):
                match_reason = "Words"
            
            if match_reason:
                candidates.append({
                    'Target': f"{composer} - {title}",
                    'Candidate': g_title,
                    'Source': 'GAPS',
                    'Reason': match_reason,
                    'Path': g.get('scorehash', '')
                })

    if not candidates:
        print("Done. No candidates.")
        return
        
    df = pd.DataFrame(candidates).drop_duplicates(subset=['Target', 'Candidate'])
    df.to_csv('features/loose_matches_to_vet.csv', index=False)
    print(f"Done. Found {len(df)} candidates in features/loose_matches_to_vet.csv")

if __name__ == "__main__":
    loose_match()
