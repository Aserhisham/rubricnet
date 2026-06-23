import json
import os
import re
from difflib import SequenceMatcher

def normalize(s):
    if not isinstance(s, str): return ""
    # Replace ordinal numbers
    s = s.lower()
    s = re.sub(r'1st', '1', s)
    s = re.sub(r'2nd', '2', s)
    s = re.sub(r'3rd', '3', s)
    s = re.sub(r'([0-9]+)th', r'\1', s)
    s = s.replace('opus', 'op').replace('number', 'no')
    return re.sub(r'[^a-z0-9]', '', s)

def get_words(s):
    if not isinstance(s, str): return set()
    return set(re.findall(r'[a-z0-9]+', s.lower()))

def parse_filename(filename, gb_composers_words):
    # Remove extension
    name, ext = os.path.splitext(filename)
    if ext.lower() not in ['.pdf', '.zip']:
        return None, None
    
    # Normalize spaces
    name = name.replace('_', ' ').strip()
    
    title_part = name
    composer_part = ""
    
    if ' by ' in name:
        parts = name.split(' by ')
        title_part = parts[0].strip()
        composer_part = parts[1].strip()
    elif ' - ' in name:
        parts = name.split(' - ')
        part0 = parts[0].strip()
        part1 = parts[1].strip()
        # Check which part matches composer words
        words0 = set(re.findall(r'[a-z0-9]+', part0.lower()))
        words1 = set(re.findall(r'[a-z0-9]+', part1.lower()))
        
        matches0 = words0.intersection(gb_composers_words)
        matches1 = words1.intersection(gb_composers_words)
        
        if matches0 and not matches1:
            composer_part = part0
            title_part = part1
        elif matches1 and not matches0:
            composer_part = part1
            title_part = part0
        else:
            # Default to part0 as composer
            composer_part = part0
            title_part = part1
    else:
        # Try to extract composer from words
        words = set(re.findall(r'[a-z0-9]+', name.lower()))
        matched_comps = words.intersection(gb_composers_words)
        # Filter out small words or numbers
        matched_comps = {w for w in matched_comps if len(w) > 2 and not w.isdigit()}
        if matched_comps:
            composer_part = list(matched_comps)[0]
            title_part = name
        else:
            if 'bwv' in name.lower():
                composer_part = "bach"
            else:
                title_part = name
                composer_part = ""
            
    return title_part, composer_part

def match_pdfs():
    print("Loading data...")
    json_path = 'features/guitarburst_full.json'
    pdf_dir = 'pdf-20260524T103936Z-3-001/pdf'
    
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return
    if not os.path.exists(pdf_dir):
        print(f"Error: PDF directory {pdf_dir} not found.")
        return

    with open(json_path, 'r') as f:
        gb_data = json.load(f)

    # Pre-index GB composers and items
    from collections import defaultdict
    comp_to_items = defaultdict(list)
    title_to_items = defaultdict(list)
    num_to_items = defaultdict(list)
    gb_composers_words = set()

    for item in gb_data:
        comp = item.get('Composer', '').replace('Mandore', 'Mangore')
        comp_words = get_words(comp)
        gb_composers_words.update(comp_words)
        
        title_norm = normalize(item.get('Title', ''))
        item['_comp_words'] = comp_words
        item['_title_norm'] = title_norm
        item['_title_words'] = get_words(item.get('Title', ''))
        
        for w in comp_words:
            comp_to_items[w].append(item)
        if title_norm:
            title_to_items[title_norm].append(item)
            
        for n in re.findall(r'\d+', title_norm):
            num_to_items[n].append(item)

    # List all files in pdf directory
    files = sorted(os.listdir(pdf_dir))
    found_count = 0
    important_composers = {'bach', 'sor', 'giuliani', 'tarrega', 'carcassi', 'aguado', 'barrios', 'sanz', 'coste', 'paganini'}

    print(f"Processing {len(files)} files in {pdf_dir}...")
    for filename in files:
        title_part, composer_part = parse_filename(filename, gb_composers_words)
        if not title_part:
            continue
        
        # Gather candidate items
        candidates = []
        c_words = get_words(composer_part)
        for w in c_words:
            candidates.extend(comp_to_items[w])
            
        # If no composer matched, check title index if filename has numbers or is long
        title_norm = normalize(title_part)
        if not candidates:
            if title_norm in title_to_items:
                candidates.extend(title_to_items[title_norm])
            nums = re.findall(r'\d+', title_norm)
            for n in nums:
                candidates.extend(num_to_items[n])
                
        unique_candidates = {id(i): i for i in candidates}.values()
        
        for item in unique_candidates:
            if item.get('status') == 'found':
                continue
                
            # Check composer match
            it_comp_words = item['_comp_words']
            intersect = c_words.intersection(it_comp_words)
            is_imp = any(c in c_words and c in it_comp_words for c in important_composers)
            
            comp_match = False
            if not composer_part:
                comp_match = len(title_to_items.get(title_norm, [])) == 1
            else:
                comp_match = (len(intersect) >= min(len(c_words), len(it_comp_words), 2)) or \
                             (len(intersect) >= 1 and (len(c_words) == 1 or len(it_comp_words) == 1 or is_imp))
                             
            if comp_match:
                it_norm = item['_title_norm']
                it_words = item['_title_words']
                song_words = get_words(title_part)
                
                match = False
                if title_norm == it_norm:
                    match = True
                else:
                    file_nums = re.findall(r'\d+', title_norm)
                    item_nums = re.findall(r'\d+', it_norm)
                    
                    if file_nums and item_nums and any(n in item_nums for n in file_nums):
                        common = song_words.intersection(it_words)
                        if len(common) >= 1 or len(it_words) == 0:
                            match = True
                    else:
                        common = song_words.intersection(it_words)
                        if len(it_words) > 3 and len(common) / len(it_words) >= 0.7:
                            match = True
                            
                if match:
                    item['status'] = 'found'
                    item['source'] = 'pdf'
                    item['pdf_path'] = os.path.join(pdf_dir, filename)
                    found_count += 1
                    print(f"Matched: {item.get('Composer')} - {item.get('Title')} -> {filename}")
                    break

    # Clean temporary keys
    for item in gb_data:
        for k in ['_comp_words', '_title_norm', '_title_words']:
            if k in item: del item[k]

    with open(json_path, 'w') as f:
        json.dump(gb_data, f, indent=4)
        
    print(f"Successfully matched/updated {found_count} pieces in total.")

if __name__ == "__main__":
    match_pdfs()
