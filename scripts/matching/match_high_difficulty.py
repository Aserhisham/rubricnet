import json
import os
import re

def normalize(s):
    if not isinstance(s, str): return ""
    s = s.lower()
    s = re.sub(r'1st', '1', s)
    s = re.sub(r'2nd', '2', s)
    s = re.sub(r'3rd', '3', s)
    s = re.sub(r'([0-9]+)th', r'\1', s)
    s = s.replace('opus', 'op').replace('number', 'no')
    return re.sub(r'[^a-z0-9]', ' ', s).strip()

def get_words(s):
    return set(normalize(s).split())

def extract_numbers(s):
    return set(re.findall(r'\d+', s.lower()))

def match_high_difficulty():
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
        
    pdf_files = sorted(os.listdir(pdf_dir))
    
    # We only match pieces that are NOT_FOUND and have Difficulty >= 12
    targets = [d for d in gb_data if d.get('status') == 'not_found' and d.get('Difficulty', 0) >= 12]
    print(f"Scanning {len(targets)} unmatched high-difficulty pieces...")
    
    matched_count = 0
    
    for item in targets:
        comp = item.get('Composer', '')
        title = item.get('Title', '')
        
        comp_words = get_words(comp)
        title_words = get_words(title)
        
        best_candidate = None
        
        for filename in pdf_files:
            f_norm = filename.lower()
            file_words = get_words(filename)
            
            # Composer check (strict last name or major part matching)
            comp_match = False
            for cw in comp_words:
                if len(cw) > 3 and cw in file_words:
                    comp_match = True
                    break
            if not comp_match:
                if 'paganini' in comp_words and 'paganini' in f_norm:
                    comp_match = True
                elif 'barrios' in comp_words and 'barrios' in f_norm:
                    comp_match = True
                    
            if not comp_match:
                continue
                
            clean_title_words = {w for w in title_words if w not in comp_words and w != 'by'}
            clean_file_words = {w for w in file_words if w not in comp_words and w != 'by'}
            
            # Specific high-confidence match rules
            is_match = False
            
            # 1. Paganini Caprices matching (Caprice number must match exactly)
            if 'paganini' in comp_words and 'caprice' in title.lower():
                # Extract caprice number from target
                target_num_match = re.search(r'caprices?.*?(?:op.*?1.*?)?:\s*(\d+)', title.lower())
                if target_num_match:
                    target_num = target_num_match.group(1)
                    # Extract caprice number from candidate filename
                    cand_num_match = re.search(r'(\d+)(?:st|nd|rd|th)?\s+caprice', f_norm)
                    if not cand_num_match:
                        cand_num_match = re.search(r'(?:caprice|capriccio|capricce)\s*(?:no)?\.?\s*(\d+)', f_norm)
                    if cand_num_match:
                        cand_num = cand_num_match.group(1)
                        if target_num == cand_num:
                            is_match = True

            # 2. Bach Goldberg Variations matching
            elif 'bach' in comp_words and 'goldberg' in title.lower():
                target_num_match = re.search(r'variatio\s*(\d+)', title.lower())
                if target_num_match:
                    target_num = target_num_match.group(1)
                    cand_num_match = re.search(r'goldberg\s*variation\s*(\d+)', f_norm)
                    if cand_num_match:
                        cand_num = cand_num_match.group(1)
                        if target_num == cand_num:
                            is_match = True

            # 3. Bach Chaconne / Ciaccona matching
            elif 'bach' in comp_words and 'chaconne' in title.lower():
                if 'ciaccona' in f_norm or 'chaconne' in f_norm:
                    is_match = True
                    
            # 4. Granados La Maja de Goya
            elif 'granados' in comp_words and 'maja' in title.lower():
                if 'maja' in f_norm and 'goya' in f_norm:
                    is_match = True
                    
            # 5. Valses Poeticos numbers matching
            elif 'poeticos' in title.lower() and 'vals' in title.lower():
                if 'valse' in f_norm or 'valses' in f_norm or 'poeticos' in f_norm:
                    target_num_match = re.search(r'(?:valse|no\.?)\s*(\d+)', title.lower())
                    if target_num_match:
                        target_num = target_num_match.group(1)
                        cand_num_match = re.search(r'(?:valse|no\.?)\s*(\d+)', f_norm)
                        if cand_num_match:
                            cand_num = cand_num_match.group(1)
                            if target_num == cand_num:
                                is_match = True
                    
            # 6. Sor Grand Solo / Sonata matching
            elif 'sor' in comp_words and 'grand' in title.lower() and 'solo' in title.lower():
                if 'grand' in f_norm or 'gran' in f_norm:
                    if 'solo' in f_norm and '14' in f_norm:
                        is_match = True
                        
            # 7. General strict fallback
            else:
                # Key mismatch check
                if ('sharp' in clean_title_words) != ('sharp' in clean_file_words):
                    continue
                if ('flat' in clean_title_words) != ('flat' in clean_file_words):
                    continue
                if ('minor' in clean_title_words) != ('minor' in clean_file_words):
                    continue
                if ('major' in clean_title_words) != ('major' in clean_file_words):
                    continue
                    
                # Movement mismatch check
                mvt_words = {'prelude', 'preludio', 'fugue', 'fuga', 'gigue', 'giga', 'sarabande', 'courante', 'allemande', 'minuet', 'gavotte', 'bourree', 'chaconne', 'ciaccona', 'adagio', 'allegro', 'andante'}
                target_mvts = {w for w in clean_title_words if w in mvt_words}
                file_mvts = {w for w in clean_file_words if w in mvt_words}
                if target_mvts and file_mvts:
                    norm_target_mvts = {w.replace('fuga', 'fugue').replace('giga', 'gigue').replace('ciaccona', 'chaconne').replace('preludio', 'prelude') for w in target_mvts}
                    norm_file_mvts = {w.replace('fuga', 'fugue').replace('giga', 'gigue').replace('ciaccona', 'chaconne').replace('preludio', 'prelude') for w in file_mvts}
                    if not norm_target_mvts.intersection(norm_file_mvts):
                        continue
                
                # Check digits/numbers in names (like catalog numbers BWV 1005 vs 1035)
                title_numbers = extract_numbers(title)
                file_numbers = extract_numbers(filename)
                if title_numbers and file_numbers:
                    cat_title_nums = {n for n in title_numbers if len(n) > 2}
                    cat_file_nums = {n for n in file_numbers if len(n) > 2}
                    if cat_title_nums and cat_file_nums:
                        if not cat_title_nums.intersection(cat_file_nums):
                            continue
                    elif not title_numbers.intersection(file_numbers):
                        continue
                
                exclude_words = {
                    'no', 'op', 'opus', 'by', 'pdf', 'number', 'vol', 'volume', 'part', 'zip', 'bwv', 'k', 'rv', 'hob',
                    'in', 'of', 'and', 'the', 'a', 'to', 'for', 'with', 'on', 'at', 'from',
                    'de', 'la', 'el', 'le', 'un', 'une', 'les', 'des', 'pour', 'del', 'da', 'di', 'du', 'en', 'et', 'y',
                    'i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x',
                    'suite', 'sonata', 'sonate', 'variations', 'variation', 'theme', 'valse', 'vals', 'dance', 'danza',
                    'prelude', 'preludio', 'fuga', 'fugue', 'chaconne', 'ciaccona', 'study', 'etude', 'espanola', 'espana',
                    'c', 'd', 'e', 'f', 'g', 'a', 'b', 'major', 'minor', 'sharp', 'flat', 'm',
                    'allegro', 'adagio', 'andante', 'largo', 'lento', 'presto', 'moderato', 'vivace', 'grave', 'assai', 'risoluto',
                    'be', 'thou', 'me', 'with', 'castles', 'spain', 'cinco', 'piezas', 'para', 'guitarra', 'guitar', 
                    'right', 'honorable', 'honourable', 'earl', 'galliard', 'al', 'del', 'los', 'las', 'por', 'campos', 'espanol'
                }
                meaningful_title_words = {w for w in clean_title_words if w not in exclude_words and not w.isdigit()}
                meaningful_file_words = {w for w in clean_file_words if w not in exclude_words and not w.isdigit()}
                meaningful_intersect = meaningful_title_words.intersection(meaningful_file_words)
                
                if len(meaningful_intersect) >= 2:
                    is_match = True
                        
            if is_match:
                best_candidate = filename
                break
                
        if best_candidate:
            item['status'] = 'found'
            item['source'] = 'pdf'
            item['pdf_path'] = os.path.join(pdf_dir, best_candidate)
            matched_count += 1
            print(f"Matched Target: {comp} - {title} -> {best_candidate}")
            
    if matched_count > 0:
        with open(json_path, 'w') as f:
            json.dump(gb_data, f, indent=4)
        print(f"Successfully matched and saved {matched_count} high-difficulty pieces.")
    else:
        print("No new high-difficulty pieces matched.")

if __name__ == '__main__':
    match_high_difficulty()
