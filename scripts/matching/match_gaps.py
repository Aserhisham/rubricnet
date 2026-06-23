import json
import pandas as pd
import re
import os

def normalize(s):
    if not isinstance(s, str): return ""
    return re.sub(r'[^a-z0-9]', '', s.lower())

def clean_gaps_composer(c):
    c = str(c)
    if 'Transcription:' in c: return normalize(c.split('Transcription:')[1])
    if 'Arr.' in c: return normalize(c.split('Arr.')[1])
    if 'Arr:' in c: return normalize(c.split('Arr:')[1])
    return normalize(c)

def match_datasets():
    print("Loading data...")
    json_path = 'features/guitarburst_full.json'
    gaps_meta_path = 'datasets/gaps_v1/gaps_v1_metadata.csv'
    
    with open(json_path, 'r') as f:
        gb_data = json.load(f)

    gaps_metadata = pd.read_csv(gaps_meta_path, encoding='latin1')

    # Pre-index GB
    for item in gb_data:
        item['_norm_title'] = normalize(item.get('Title', ''))
        item['_words'] = set(re.sub(r'[^a-z0-9 ]', ' ', item.get('Title','').lower()).split())
        item['_comp'] = item.get('Composer', '').lower()

    found_count = 0
    important_comps = ['bach', 'sor', 'giuliani', 'tarrega', 'barrios', 'sanz', 'aguado', 'carulli', 'carcassi']

    for _, row in gaps_metadata.iterrows():
        gt = row.get('title', '')
        if not isinstance(gt, str): continue
        
        gt_norm = normalize(gt)
        gt_words = set(re.sub(r'[^a-z0-9 ]', ' ', gt.lower()).split())
        gt_nums = set(re.findall(r'\d+', gt_norm))
        scorehash = str(row['scorehash'])

        for item in gb_data:
            if item.get('status') == 'found': continue
            
            it_norm = item['_norm_title']
            it_nums = set(re.findall(r'\d+', it_norm))
            it_comp = item['_comp']
            
            match = False
            # 1. Exact
            if gt_norm == it_norm:
                match = True
            # 2. Number subset match (BWV/Op) + composer check
            elif gt_nums and it_nums and (gt_nums.issubset(it_nums) or it_nums.issubset(gt_nums)):
                # Check if it's a common guitar piece composer
                is_guitar_comp = any(c in it_comp for c in important_comps)
                common = gt_words.intersection(item['_words'])
                if (is_guitar_comp and len(common) >= 1) or 'bwv' in gt_norm:
                    match = True
            
            # 3. Fuzzy match for longer titles
            elif len(item['_words']) > 3:
                common = gt_words.intersection(item['_words'])
                if len(common) / len(item['_words']) >= 0.7:
                    match = True
            
            if match:
                item['status'] = 'found'
                item['source'] = 'gaps'
                item['scorehash'] = scorehash
                item['xml_path'] = f"datasets/gaps_v1/musicxml/{scorehash}.musicxml"
                if not os.path.exists(item['xml_path']):
                    item['xml_path'] = f"datasets/gaps_v1/musicxml/{scorehash}.xml"
                
                found_count += 1
                break

    for item in gb_data:
        for k in ['_norm_title', '_words', '_comp']:
            if k in item: del item[k]

    with open(json_path, 'w') as f:
        json.dump(gb_data, f, indent=4)
    
    print(f"Aggressive GAPS scan matched {found_count} pieces.")

if __name__ == "__main__":
    match_datasets()
