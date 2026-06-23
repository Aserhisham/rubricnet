import json
import os
import re
import pandas as pd
from collections import defaultdict

def clean_filename_for_path(name):
    """Remove characters that are illegal in filenames."""
    if not isinstance(name, str):
        return "Unknown"
    name = re.sub(r'[\\/*?:"<>|]', "-", name)
    name = name.encode('ascii', 'ignore').decode('ascii')
    return name.strip()

def clean_composer(c):
    if not isinstance(c, str): return ""
    c = c.lower().strip()
    return c.replace('mandore', 'mangore')

STOP_WORDS = {
    'j', 's', 'f', 'm', 'l', 'd', 'de', 'di', 'von', 'unknown', 'anonymous', 'traditional', 'arr', 'arranged',
    'johann', 'sebastian', 'domenico', 'fernando', 'matteo', 'mateo', 'mauro', 'agustin', 'augustin', 'gaspar', 
    'francisco', 'isaac', 'heitor', 'robert', 'john', 'silvius', 'leopold', 'ludwig', 'antonio', 'napoleon', 
    'giulio', 'luigi', 'dionisio', 'dioniso', 'emilio', 'albert', 'alexandre'
}

def get_composer_words(c):
    if not isinstance(c, str): return set()
    c = clean_composer(c)
    words = set(re.findall(r'[a-z]+', c))
    return words - STOP_WORDS

def composers_match(gb, candidate):
    if gb['composer'] == candidate['composer']:
        return True
    if not gb['composer_words'] or not candidate['composer_words']:
        return False
    return len(gb['composer_words'].intersection(candidate['composer_words'])) >= 1

def extract_key(title):
    if not isinstance(title, str):
        return None
    title_lower = title.lower()
    key_match = re.search(r'\bin\s+([a-g])\s*(sharp|flat|#|s)?\s*(major|minor|dur|moll|m)?\b', title_lower)
    if key_match:
        note = key_match.group(1).upper()
        acc = key_match.group(2) or ''
        mode = key_match.group(3) or ''
    else:
        key_match = re.search(r'\b([a-g])(m|major|minor)\b', title_lower)
        if key_match:
            note = key_match.group(1).upper()
            acc = ''
            mode = key_match.group(2) or ''
        else:
            return None
            
    acc = acc.replace('s', '#').replace('sharp', '#').replace('flat', 'b')
    if acc not in ['#', 'b']:
        acc = ''
    mode_str = mode.lower()
    acc_str = acc.lower()
    if 'moll' in mode_str or 'minor' in mode_str or 'm' in mode_str or 'm' in acc_str:
        mode = 'minor'
    else:
        mode = 'major'
    return f"{note}{acc} {mode}"

def roman_to_int(s):
    s = s.upper().strip()
    if not re.match(r'^(?:[IVX]+)$', s):
        return None
    roman_map = {
        'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10,
        'XI': 11, 'XII': 12, 'XIII': 13, 'XIV': 14, 'XV': 15, 'XVI': 16, 'XVII': 17, 'XVIII': 18, 'XIX': 19, 'XX': 20,
        'XXI': 21, 'XXII': 22, 'XXIII': 23, 'XXIV': 24, 'XXV': 25, 'XXVI': 26, 'XXVII': 27, 'XXVIII': 28, 'XXIX': 29, 'XXX': 30
    }
    return roman_map.get(s, None)

def extract_catalog_info(title):
    if not isinstance(title, str):
        return {'bwv': None, 'opus': None, 'number': None, 'study_no': None, 'scarlatti_no': None, 'genre': None, 'key': None, 'norm_title': ''}
    
    title_lower = title.lower().replace('-', ' ')
    
    # BWV (Bach)
    bwv_match = re.search(r'bwv\s*(\d+)', title_lower)
    bwv = int(bwv_match.group(1)) if bwv_match else None
    
    # Scarlatti K or L
    scarlatti_matches = re.findall(r'\b[kl]\.?\s*(\d+)\b', title_lower)
    if not scarlatti_matches:
        scarlatti_matches = re.findall(r'\b[kl](\d+)\b', title_lower)
    scarlatti_no = set(int(x) for x in scarlatti_matches) if scarlatti_matches else None
    
    # Opus and Number
    opus = None
    number = None
    
    op_match = re.search(r'(?:op|opus)\.?\s*(\d+)', title_lower)
    if op_match:
        opus = int(op_match.group(1))
        no_match = re.search(r'(?:no|number|#|n\.?|a)\.?\s*([ivx\d]+)\b', title_lower)
        if no_match:
            val = no_match.group(1)
            number = int(val) if val.isdigit() else roman_to_int(val)
        else:
            sub_title = title_lower[op_match.end():]
            no_after = re.search(r'^(?:\s*[:,.-]\s*|\s+no\.?\s+|\s+number\s+|\s+#\s*|\s+a\s*|\s+)([ivx\d]+)\b', sub_title)
            if no_after:
                val = no_after.group(1)
                number = int(val) if val.isdigit() else roman_to_int(val)
                
    # Study/Etude number and genre identification
    study_no = None
    genre = None
    
    genre_keywords = r'(?:study|studies|estudio|estudios|etude|etudes|prelude|preludes|preludio|caprice|caprices|capriccio|ghiribizzo|sonata|sonatas|partita|partitas|suite|suites|fantasia|fantasias|fantasie|fancy|rondo|rondeau|minuet|menuet|gavotte|allemande|allemanda|courante|corrente|sarabande|sarabanda|gigue|giga|bourree|bouree|siciliana|toccata|fugue|fuga|lesson|lessons|leccion|lecciones|exercise|exercises|ejercicio|ejercicios|pavan|pavane|pavanas|pavana|galliard|galliards|passacaglia)'
    
    study_match = re.search(rf'\b({genre_keywords})\b(?:\s*(?:no|number|#|n\.?|part)?\.?\s*|[:\s]+)([ivx\d]+)\b', title_lower)
    if study_match:
        genre_word = study_match.group(1)
        val = study_match.group(2)
        if genre_word not in ['suite', 'suites', 'sonata', 'sonatas', 'partita', 'partitas']:
            study_no = int(val) if val.isdigit() else roman_to_int(val)
        
        # Normalize genre to singular English representation
        if genre_word in ['study', 'studies', 'estudio', 'estudios', 'etude', 'etudes', 'lesson', 'lessons', 'leccion', 'lecciones', 'exercise', 'exercises', 'ejercicio', 'ejercicios']:
            genre = 'study'
        elif genre_word in ['prelude', 'preludes', 'preludio']:
            genre = 'prelude'
        elif genre_word in ['caprice', 'caprices', 'capriccio', 'ghiribizzo']:
            genre = 'caprice'
        elif genre_word in ['sonata', 'sonatas']:
            genre = 'sonata'
        elif genre_word in ['partita', 'partitas']:
            genre = 'partita'
        elif genre_word in ['suite', 'suites']:
            genre = 'suite'
        elif genre_word in ['fantasia', 'fantasias', 'fantasie', 'fancy']:
            genre = 'fantasia'
        elif genre_word in ['rondo', 'rondeau']:
            genre = 'rondo'
        elif genre_word in ['minuet', 'menuet']:
            genre = 'minuet'
        elif genre_word in ['gavotte']:
            genre = 'gavotte'
        elif genre_word in ['allemande', 'allemanda']:
            genre = 'allemande'
        elif genre_word in ['courante', 'corrente']:
            genre = 'courante'
        elif genre_word in ['sarabande', 'sarabanda']:
            genre = 'sarabande'
        elif genre_word in ['gigue', 'giga']:
            genre = 'gigue'
        elif genre_word in ['bourree', 'bouree']:
            genre = 'bourree'
        elif genre_word in ['siciliana']:
            genre = 'siciliana'
        elif genre_word in ['toccata']:
            genre = 'toccata'
        elif genre_word in ['fugue', 'fuga']:
            genre = 'fugue'
        elif genre_word in ['pavan', 'pavane', 'pavanas', 'pavana']:
            genre = 'pavan'
        elif genre_word in ['galliard', 'galliards']:
            genre = 'galliard'
        elif genre_word in ['passacaglia']:
            genre = 'passacaglia'

    if not genre:
        # Scan for the first keyword that matches
        for word in title_lower.split():
            clean_w = re.sub(r'[^a-z]', '', word)
            if clean_w in ['study', 'studies', 'estudio', 'estudios', 'etude', 'etudes', 'lesson', 'lessons', 'leccion', 'lecciones', 'exercise', 'exercises', 'ejercicio', 'ejercicios']:
                genre = 'study'
                break
            elif clean_w in ['prelude', 'preludes', 'preludio']:
                genre = 'prelude'
                break
            elif clean_w in ['caprice', 'caprices', 'capriccio', 'ghiribizzo']:
                genre = 'caprice'
                break
            elif clean_w in ['sonata', 'sonatas']:
                genre = 'sonata'
                break
            elif clean_w in ['partita', 'partitas']:
                genre = 'partita'
                break
            elif clean_w in ['suite', 'suites']:
                genre = 'suite'
                break
            elif clean_w in ['fantasia', 'fantasias', 'fantasie', 'fancy']:
                genre = 'fantasia'
                break
            elif clean_w in ['rondo', 'rondeau']:
                genre = 'rondo'
                break
            elif clean_w in ['minuet', 'menuet']:
                genre = 'minuet'
                break
            elif clean_w in ['gavotte']:
                genre = 'gavotte'
                break
            elif clean_w in ['allemande', 'allemanda']:
                genre = 'allemande'
                break
            elif clean_w in ['courante', 'corrente']:
                genre = 'courante'
                break
            elif clean_w in ['sarabande', 'sarabanda']:
                genre = 'sarabande'
                break
            elif clean_w in ['gigue', 'giga']:
                genre = 'gigue'
                break
            elif clean_w in ['bourree', 'bouree']:
                genre = 'bourree'
                break
            elif clean_w in ['siciliana']:
                genre = 'siciliana'
                break
            elif clean_w in ['toccata']:
                genre = 'toccata'
                break
            elif clean_w in ['fugue', 'fuga']:
                genre = 'fugue'
                break
            elif clean_w in ['pavan', 'pavane', 'pavanas', 'pavana']:
                genre = 'pavan'
                break
            elif clean_w in ['galliard', 'galliards']:
                genre = 'galliard'
                break
            elif clean_w in ['passacaglia']:
                genre = 'passacaglia'
                break
            
    # Fallback study/number extraction:
    # Scan for any standalone number or Roman numeral that has not been consumed by BWV, Opus, or Scarlatti
    if study_no is None:
        title_temp = re.sub(r'\b\d+\s+(?:pieces|piezas|studies|estudios|etudes|preludes|lessons|lecciones|exercises|ejercicios|caprices|sonatas|valses|waltzes|minuets|menuets|gavottes|allemandes|courantes|sarabandes|gigues|bourrees)\b', '', title_lower)
        title_temp = re.sub(r'\b(?:suite|sonata|partita|concerto|book|volume|vol|bk|op|opus)\s*(?:no|number|#|n\.?)?\.?\s*[ivx\d]+\b', '', title_temp)
        all_nums = re.findall(r'\b(?:no|number|#|n\.?|part)?\.?\s*([ivx\d]+)\b', title_temp)
        for num_str in reversed(all_nums):
            val = int(num_str) if num_str.isdigit() else roman_to_int(num_str)
            if val is not None:
                if val in [bwv, opus, number] or (scarlatti_no is not None and val in scarlatti_no):
                    continue
                if val > 100:
                    continue
                study_no = val
                break

    norm_title = title_lower
    # Replace archaic/spelling variants to align normalized titles
    norm_title = norm_title.replace('forlorne', 'forlorn')
    norm_title = norm_title.replace('fantasdega', 'fantasia')
    norm_title = norm_title.replace('fantasye', 'fantasia')
    norm_title = norm_title.replace('fuga', 'fugue')
    norm_title = norm_title.replace('menuet', 'minuet')
    norm_title = norm_title.replace('estudio', 'study')
    norm_title = norm_title.replace('etude', 'study')
    
    # Strip Opus / BWV / Scarlatti numbers
    norm_title = re.sub(r'(?:op|opus)\.?\s*\d+(?:\s*(?:no|number|#|n\.?|a)\.?\s*\d+)?', '', norm_title)
    norm_title = re.sub(r'bwv\s*\d+', '', norm_title)
    norm_title = re.sub(r'\b[kl]\.?\s*\d+\b', '', norm_title)
    
    # Strip number indicators followed by Roman numerals or digits
    norm_title = re.sub(r'\b(?:no|number|#|n\.?|part)\.?\s*[ivx\d]+\b', '', norm_title)
    # Strip standalone Roman numerals and digits representing parts/movements
    norm_title = re.sub(r'\b[ivx]+\b', '', norm_title)
    norm_title = re.sub(r'\b\d+\b', '', norm_title)
    
    # Strip collection size patterns from normalized title as well
    norm_title = re.sub(r'\b\d+\s+(?:pieces|piezas|studies|estudios|etudes|preludes|lessons|lecciones|exercises|ejercicios|caprices|sonatas|valses|waltzes|minuets|menuets|gavottes|allemandes|courantes|sarabandes|gigues|bourrees)\b', '', norm_title)
    
    norm_title = re.sub(r'[^a-z0-9\s]', '', norm_title)
    filler = {
        'by', 'of', 'in', 'for', 'the', 'a', 'an', 'major', 'minor', 'flat', 'sharp', 'key', 'dur', 'moll',
        'em', 'am', 'dm', 'gm', 'cm', 'fm', 'bm', 'fsharp', 'g', 'c', 'd', 'a', 'e', 'f', 'b'
    }
    words = [w for w in norm_title.split() if w not in filler]
    norm_title = " ".join(words)
    
    key = extract_key(title)
    
    return {
        'bwv': bwv,
        'opus': opus,
        'number': number,
        'study_no': study_no,
        'scarlatti_no': scarlatti_no,
        'genre': genre,
        'key': key,
        'norm_title': norm_title
    }

def try_match(gb, candidate, strict_only=False):
    # Composers must match
    if not composers_match(gb, candidate):
        return False
        
    gb_cat = gb['cat']
    c_cat = candidate['cat']
    
    GENERIC_FORMS = {
        'study', 'estudio', 'etude', 'estudios', 'etudes', 'sencillos', 'sencillo', 'prelude', 'preludes', 'preludio', 
        'fantasia', 'fantasie', 'fancy', 'rondo', 'rondeau', 'minuet', 'menuet', 'gavotte', 'allemande', 'allemanda',
        'courante', 'corrente', 'sarabande', 'sarabanda', 'gigue', 'giga', 'bourree', 'bouree', 'siciliana', 'sonata', 
        'sonatina', 'divertimento', 'chaconne', 'passacaglia', 'pavane', 'pavan', 'galliard', 'piece', 'pieces', 'piezas', 
        'stuck', 'stuckchen', 'variation', 'variations', 'fugue', 'fuga', 'lesson', 'lessons', 'leccion', 
        'lecciones', 'exercise', 'exercises', 'ejercicio', 'ejercicios', 'caprice', 'caprices', 'capriccio', 'ghiribizzo',
        'toccata', 'double', 'adagio', 'presto', 'grave', 'andante', 'allegro', 'largo', 'maestoso', 'moderato',
        'borey', 'ciaccona', 'chacona'
    }
    
    def is_generic(t):
        words = t.split()
        if not words: return True
        return all(w in GENERIC_FORMS for w in words)
        
    MOVEMENT_KEYWORDS = {
        'adagio', 'fuga', 'fugue', 'siciliana', 'presto', 'grave', 'andante', 'allegro', 'largo', 'maestoso', 'moderato', 
        'chaconne', 'allemande', 'courante', 'sarabande', 'gigue', 'bourree', 'prelude', 'loure', 'gavotte', 'minuet', 'menuet',
        'allemanda', 'corrente', 'sarabanda', 'giga', 'bouree', 'borey', 'ciaccona', 'chacona', 'preludio', 'double',
        'passacaglia', 'passacaille'
    }
    
    # Helper to check movement conflict using original/full titles
    def has_movement_conflict(t1, t2):
        t1_clean = re.sub(r'[^a-z\s]', ' ', t1.lower())
        t2_clean = re.sub(r'[^a-z\s]', ' ', t2.lower())
        w1 = set(t1_clean.split())
        w2 = set(t2_clean.split())
        mv1 = w1.intersection(MOVEMENT_KEYWORDS)
        mv2 = w2.intersection(MOVEMENT_KEYWORDS)
        if mv1 and mv2 and mv1 != mv2:
            return True
        return False
        
    # Check for general catalog conflicts (where both specify a value)
    if gb_cat['bwv'] is not None and c_cat['bwv'] is not None and gb_cat['bwv'] != c_cat['bwv']:
        return False
    if gb_cat['scarlatti_no'] is not None and c_cat['scarlatti_no'] is not None:
        if not (gb_cat['scarlatti_no'] & c_cat['scarlatti_no']):
            return False
    if gb_cat['opus'] is not None and c_cat['opus'] is not None and gb_cat['opus'] != c_cat['opus']:
        return False
    if gb_cat['number'] is not None and c_cat['number'] is not None and gb_cat['number'] != c_cat['number']:
        return False
    if gb_cat['study_no'] is not None and c_cat['study_no'] is not None and gb_cat['study_no'] != c_cat['study_no']:
        return False
    if gb_cat['genre'] is not None and c_cat['genre'] is not None:
        g1, g2 = gb_cat['genre'], c_cat['genre']
        collection_genres = {'suite', 'sonata', 'partita', 'concerto'}
        if g1 != g2 and g1 not in collection_genres and g2 not in collection_genres:
            return False
        
    def is_valid_study_opus(composer, opus):
        if opus is None:
            return True
        c_lower = composer.lower()
        if 'carcassi' in c_lower:
            return opus in [59, 60]
        if 'sor' in c_lower:
            return opus in [6, 29, 31, 35, 44, 60]
        if 'giuliani' in c_lower:
            return opus in [48, 51, 100, 111, 139]
        if 'aguado' in c_lower:
            return opus in [6]
        return True

    # Check strict generic conflicts (where one having a number and the other not having it is a conflict)
    if is_generic(gb_cat['norm_title']) or is_generic(c_cat['norm_title']):
        if gb_cat['bwv'] is not None or c_cat['bwv'] is not None:
            if gb_cat['bwv'] != c_cat['bwv']:
                return False
        elif gb_cat['scarlatti_no'] is not None or c_cat['scarlatti_no'] is not None:
            if gb_cat['scarlatti_no'] is None or c_cat['scarlatti_no'] is None or not (gb_cat['scarlatti_no'] & c_cat['scarlatti_no']):
                return False
        else:
            # Opus conflict logic
            if gb_cat['opus'] is not None and c_cat['opus'] is not None:
                if gb_cat['opus'] != c_cat['opus']:
                    return False
            elif gb_cat['opus'] is not None or c_cat['opus'] is not None:
                opus_val = gb_cat['opus'] if gb_cat['opus'] is not None else c_cat['opus']
                is_study = (gb_cat['genre'] == 'study' or c_cat['genre'] == 'study')
                composer = gb.get('Composer', gb.get('ComposerName', ''))
                if is_study:
                    if not is_valid_study_opus(composer, opus_val):
                        return False
                else:
                    return False

            # Unified study_no/number conflict logic
            gb_num = gb_cat['study_no'] if gb_cat['study_no'] is not None else gb_cat['number']
            c_num = c_cat['study_no'] if c_cat['study_no'] is not None else c_cat['number']
            if (gb_num is not None or c_num is not None) and gb_num != c_num:
                return False
            
    # 1. BWV Match (Bach)
    if gb_cat['bwv'] is not None and c_cat['bwv'] is not None:
        if gb_cat['bwv'] == c_cat['bwv']:
            return not has_movement_conflict(gb.get('Title', '').lower(), candidate.get('title', '').lower())
        return False
        
    # 2. Scarlatti Match
    if gb_cat['scarlatti_no'] is not None and c_cat['scarlatti_no'] is not None:
        return bool(gb_cat['scarlatti_no'] & c_cat['scarlatti_no'])
        
    # 3. Opus Match
    if gb_cat['opus'] is not None and c_cat['opus'] is not None:
        if gb_cat['number'] is not None and c_cat['number'] is not None:
            if gb_cat['opus'] == c_cat['opus'] and gb_cat['number'] == c_cat['number']:
                return not has_movement_conflict(gb.get('Title', '').lower(), candidate.get('title', '').lower())
            return False
        elif gb_cat['number'] is None and c_cat['number'] is None:
            words_gb = set(gb_cat['norm_title'].split())
            words_c = set(c_cat['norm_title'].split())
            if not words_gb or not words_c:
                return gb_cat['opus'] == c_cat['opus'] and not has_movement_conflict(gb.get('Title', '').lower(), candidate.get('title', '').lower())
            if gb_cat['opus'] == c_cat['opus'] and len(words_gb.intersection(words_c)) >= 1:
                return not has_movement_conflict(gb.get('Title', '').lower(), candidate.get('title', '').lower())
            return False
            
    # 4. Study Number Match
    gb_num = gb_cat['study_no'] if gb_cat['study_no'] is not None else gb_cat['number']
    c_num = c_cat['study_no'] if c_cat['study_no'] is not None else c_cat['number']
    if gb_num is not None and c_num is not None:
        words_gb = set(gb_cat['norm_title'].split())
        words_c = set(c_cat['norm_title'].split())
        common = words_gb.intersection(words_c)
        same_genre = gb_cat['genre'] is not None and gb_cat['genre'] == c_cat['genre']
        is_study = (gb_cat['genre'] == 'study' or c_cat['genre'] == 'study')
        if gb_num == c_num and (len(common) >= 1 or len(words_gb) == 0 or len(words_c) == 0 or same_genre or is_study):
            return not has_movement_conflict(gb.get('Title', '').lower(), candidate.get('title', '').lower())
        return False
            
    # 5. Exact Title Match
    if gb_cat['norm_title'] and gb_cat['norm_title'] == c_cat['norm_title']:
        return not has_movement_conflict(gb.get('Title', '').lower(), candidate.get('title', '').lower())
        
    if strict_only:
        return False
        
    # 6. Fuzzy Match (Pass 2)
    # Check for movement conflicts
    if has_movement_conflict(gb.get('Title', '').lower(), candidate.get('title', '').lower()):
        return False
        
    gb_title = gb_cat['norm_title']
    c_title = c_cat['norm_title']
        
    if is_generic(gb_title) or is_generic(c_title):
        return False
        
    if len(gb_title) > 4 and len(c_title) > 4:
        if gb_title in c_title or c_title in gb_title:
            return True
        words_gb = set(gb_title.split())
        words_c = set(c_title.split())
        if words_gb and words_c:
            common = words_gb.intersection(words_c)
            min_len = min(len(words_gb), len(words_c))
            if min_len > 0 and len(common) / min_len >= 0.7:
                return True
                
    return False

def parse_filename(filename):
    name, ext = os.path.splitext(filename)
    if ext.lower() not in ['.pdf', '.zip']:
        return None
        
    name = name.replace('_', ' ').replace('-', ' ').strip()
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
        comp0 = clean_composer(part0)
        comp1 = clean_composer(part1)
        
        known_keys = {'bach', 'sor', 'giuliani', 'tarrega', 'carcassi', 'carulli', 'aguado', 'coste', 'barrios', 'sanz', 'paganini', 'dowland', 'villa-lobos', 'brouwer', 'albeniz', 'lauro', 'weiss', 'mertz', 'scarlatti', 'satie', 'mudarra'}
        if comp0 in known_keys and comp1 not in known_keys:
            composer_part = part0
            title_part = part1
        elif comp1 in known_keys and comp0 not in known_keys:
            composer_part = part1
            title_part = part0
        else:
            composer_part = part0
            title_part = part1
    else:
        # Split on spaces, dashes, underscores to identify known composer keys
        words = re.split(r'[-_\s]', name.lower())
        found_comp = None
        known_keys = {'bach', 'sor', 'giuliani', 'tarrega', 'carcassi', 'carulli', 'aguado', 'coste', 'barrios', 'sanz', 'paganini', 'dowland', 'villa-lobos', 'brouwer', 'albeniz', 'lauro', 'weiss', 'mertz', 'scarlatti', 'satie', 'mudarra'}
        for w in words:
            if w in known_keys:
                found_comp = w
                break
        if found_comp:
            composer_part = found_comp
            title_part = re.sub(rf'\b{found_comp}\b', '', name, flags=re.IGNORECASE).strip()
        else:
            title_part = name
            composer_part = "unknown"
            
    composer = clean_composer(composer_part)
    composer_words = get_composer_words(composer_part)
    cat = extract_catalog_info(title_part)
    
    return {
        'composer': composer,
        'composer_words': composer_words,
        'cat': cat,
        'pdf_path': os.path.join('pdf-20260524T103936Z-3-001/pdf', filename),
        'title': title_part,
        'filename': filename
    }

def sync():
    json_path = 'features/guitarburst_full.json'
    gaps_meta_path = 'datasets/gaps_v1/gaps_v1_metadata.csv'
    dada_root = 'datasets/DadaGP-v1.1'
    
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return
        
    with open(json_path, 'r') as f:
        gb_data = json.load(f)
        
    print(f"Loaded {len(gb_data)} GuitarBurst pieces.")
    
    readable_xml_dir = 'datasets/gaps_v1/readable_musicxml/'
    hash_to_readable = {}
    if os.path.exists(readable_xml_dir):
        try:
            metadata = pd.read_csv(gaps_meta_path, encoding='latin1')
            for _, row in metadata.iterrows():
                h = str(row['scorehash'])
                title_clean = clean_filename_for_path(str(row['title']))
                composer_clean = clean_filename_for_path(str(row.get('composer_name_normalized', 'Unknown')))
                readable_name = f"{composer_clean} - {title_clean}.xml"
                if not os.path.exists(os.path.join(readable_xml_dir, readable_name)):
                    alt_name = f"{composer_clean} - {title_clean} ({h}).xml"
                    if os.path.exists(os.path.join(readable_xml_dir, alt_name)):
                        readable_name = alt_name
                hash_to_readable[h] = readable_name
        except Exception as e:
            print(f"Warning: Could not load metadata for readable mapping: {e}")
            
    # 1. Reset statuses & pre-parse GuitarBurst pieces
    for item in gb_data:
        item['status'] = 'not_found'
        # Clean temporary keys
        for key in ['source', 'xml_path', 'readable_path', 'token_path', 'gp_path', 'scorehash', 'pdf_path']:
            if key in item: del item[key]
            
        item['composer'] = clean_composer(item.get('Composer', ''))
        item['composer_words'] = get_composer_words(item.get('Composer', ''))
        item['cat'] = extract_catalog_info(item.get('Title', ''))

    # 2. Gather Candidates from GAPS
    gaps_candidates = []
    print("Loading GAPS metadata...")
    if os.path.exists(gaps_meta_path):
        gaps_meta = pd.read_csv(gaps_meta_path, encoding='latin1')
        for _, row in gaps_meta.iterrows():
            title = row.get('title', '')
            composer_str = ""
            for col in ['composer', 'ComposerName', 'composers']:
                val = row.get(col, '')
                if isinstance(val, str) and val:
                    composer_str = val
                    break
            
            composer = clean_composer(composer_str)
            composer_words = get_composer_words(composer_str)
            cat = extract_catalog_info(title)
            sh = str(row.get('scorehash', ''))
            gaps_candidates.append({
                'composer': composer,
                'composer_words': composer_words,
                'cat': cat,
                'scorehash': sh,
                'title': title,
                'used': False
            })
    print(f"Loaded {len(gaps_candidates)} GAPS candidates.")

    # 3. Gather Candidates from Dada-GP
    dada_candidates = []
    dada_meta_path = os.path.join(dada_root, "_DadaGP_all_metadata.json")
    print("Loading Dada-GP metadata...")
    if os.path.exists(dada_meta_path):
        with open(dada_meta_path, 'r') as f:
            dada_meta = json.load(f)
            
        for rel_token_path, info in dada_meta.items():
            token_path = os.path.join(dada_root, rel_token_path)
            file = os.path.basename(token_path)
            root = os.path.dirname(token_path)
            
            artist_token = info.get('artist_token', '').replace('artist:', '').replace('_', ' ')
            song_full = file.split('.tokens.txt')[0]
            artist_part = ""
            song_part = song_full
            
            if ' - ' in song_full:
                parts = song_full.split(' - ')
                artist_part = parts[0]
                song_part = ' - '.join(parts[1:])
            
            song_part = re.sub(r'\.gp[345x]$', '', song_part)
            comp_str = artist_part if artist_part else artist_token
            if not comp_str:
                comp_str = os.path.basename(root)
                
            composer = clean_composer(comp_str)
            composer_words = get_composer_words(comp_str)
            cat = extract_catalog_info(song_part)
            
            gp_base = file.replace('.tokens.txt', '')
            gp_path = os.path.join(root, gp_base)
            
            dada_candidates.append({
                'composer': composer,
                'composer_words': composer_words,
                'cat': cat,
                'token_path': token_path,
                'gp_path': gp_path if os.path.exists(gp_path) else None,
                'title': song_part,
                'used': False
            })
    print(f"Loaded {len(dada_candidates)} Dada-GP candidates.")

    # 4. Gather Candidates from PDF Folder
    pdf_candidates = []
    pdf_dir = 'pdf-20260524T103936Z-3-001/pdf'
    print("Loading PDF files...")
    if os.path.exists(pdf_dir):
        for filename in sorted(os.listdir(pdf_dir)):
            parsed = parse_filename(filename)
            if parsed:
                parsed['used'] = False
                pdf_candidates.append(parsed)
    print(f"Loaded {len(pdf_candidates)} PDF candidates.")

    # --- Pass 1: Strict Alignment ---
    print("\nStarting Pass 1 (Strict Catalog Match)...")
    for gb in gb_data:
        # Match GAPS first
        for cand in gaps_candidates:
            if cand['used']: continue
            if try_match(gb, cand, strict_only=True):
                gb['status'] = 'found'
                gb['source'] = 'gaps'
                gb['scorehash'] = cand['scorehash']
                
                # Try to use readable name
                readable_name = hash_to_readable.get(cand['scorehash'])
                p = None
                if readable_name:
                    temp_p = os.path.join(readable_xml_dir, readable_name)
                    if os.path.exists(temp_p):
                        p = temp_p
                if not p:
                    for ext in ['.musicxml', '.xml']:
                        temp_p = f"datasets/gaps_v1/musicxml/{cand['scorehash']}{ext}"
                        if os.path.exists(temp_p):
                            p = temp_p
                            break
                if p:
                    gb['xml_path'] = p
                cand['used'] = True
                break
                
        if gb['status'] == 'found': continue
        
        # Match Dada-GP next
        for cand in dada_candidates:
            if cand['used']: continue
            if try_match(gb, cand, strict_only=True):
                gb['status'] = 'found'
                gb['source'] = 'dada_gp'
                gb['token_path'] = cand['token_path']
                if cand['gp_path']:
                    gb['gp_path'] = cand['gp_path']
                cand['used'] = True
                break
                
        if gb['status'] == 'found': continue
        
        # Match PDF last
        for cand in pdf_candidates:
            if cand['used']: continue
            if try_match(gb, cand, strict_only=True):
                gb['status'] = 'found'
                gb['source'] = 'pdf'
                gb['pdf_path'] = cand['pdf_path']
                cand['used'] = True
                break

    # --- Pass 2: Fuzzy Alignment ---
    print("\nStarting Pass 2 (Fuzzy Title Match)...")
    for gb in gb_data:
        if gb['status'] == 'found': continue
        
        # Match GAPS first
        for cand in gaps_candidates:
            if cand['used']: continue
            if try_match(gb, cand, strict_only=False):
                gb['status'] = 'found'
                gb['source'] = 'gaps'
                gb['scorehash'] = cand['scorehash']
                
                # Try to use readable name
                readable_name = hash_to_readable.get(cand['scorehash'])
                p = None
                if readable_name:
                    temp_p = os.path.join(readable_xml_dir, readable_name)
                    if os.path.exists(temp_p):
                        p = temp_p
                if not p:
                    for ext in ['.musicxml', '.xml']:
                        temp_p = f"datasets/gaps_v1/musicxml/{cand['scorehash']}{ext}"
                        if os.path.exists(temp_p):
                            p = temp_p
                            break
                if p:
                    gb['xml_path'] = p
                cand['used'] = True
                break
                
        if gb['status'] == 'found': continue
        
        # Match Dada-GP next
        for cand in dada_candidates:
            if cand['used']: continue
            if try_match(gb, cand, strict_only=False):
                gb['status'] = 'found'
                gb['source'] = 'dada_gp'
                gb['token_path'] = cand['token_path']
                if cand['gp_path']:
                    gb['gp_path'] = cand['gp_path']
                cand['used'] = True
                break
                
        if gb['status'] == 'found': continue
        
        # Match PDF last
        for cand in pdf_candidates:
            if cand['used']: continue
            if try_match(gb, cand, strict_only=False):
                gb['status'] = 'found'
                gb['source'] = 'pdf'
                gb['pdf_path'] = cand['pdf_path']
                cand['used'] = True
                break

    # 5. Clean up temporary matching structures and save results
    found = []
    for item in gb_data:
        if 'composer' in item: del item['composer']
        if 'composer_words' in item: del item['composer_words']
        if 'cat' in item: del item['cat']
        
        if item.get('status') == 'found':
            found.append(item)
            
    print(f"\nMatching Results Summary:")
    print(f"Total Found Pieces: {len(found)}")
    print(f"  GAPS: {len([i for i in found if i.get('source') == 'gaps'])}")
    print(f"  Dada-GP: {len([i for i in found if i.get('source') == 'dada_gp'])}")
    print(f"  PDF: {len([i for i in found if i.get('source') == 'pdf'])}")

    with open(json_path, 'w') as f:
        json.dump(gb_data, f, indent=4)
        
    pd.DataFrame(found).to_csv('features/found_pieces.csv', index=False)
    print("Successfully updated guitarburst_full.json and saved features/found_pieces.csv.")

if __name__ == "__main__":
    sync()
