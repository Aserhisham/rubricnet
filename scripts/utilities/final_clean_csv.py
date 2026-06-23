import csv
import re
import os
import html

def extract_numbers(text):
    if not text:
        return {'opus': None, 'no': None}
    
    # Try to find Opus number
    opus_match = re.search(r'(?:Op\.|Opus|Opera)\s*(\d+)', text, re.IGNORECASE)
    # Try to find Number/Piece number
    no_match = re.search(r'(?:No\.|No|N\.|N|n\.|n|#)\s*(\d+)', text, re.IGNORECASE)
    
    if not no_match:
        # Match "12 variazioni" -> 12
        no_match = re.search(r'^(\d+)\s+', text)

    return {
        'opus': opus_match.group(1) if opus_match else None,
        'no': no_match.group(1) if no_match else None
    }

def get_keywords(text):
    if not text:
        return set()
    # Remove generic musical terms
    generic = {
        'allegretto', 'allegro', 'andante', 'andantino', 'adagio', 'largo', 'moderato', 'presto', 'vivace', 
        'study', 'estudio', 'etude', 'opus', 'op', 'no', 'piece', 'pieza', 'in', 'major', 'minor', 'sharp', 'flat',
        'guitar', 'solo', 'by', 'of', 'from', 'the', 'and', 'with', 'variations', 'variazioni', 'variaciones',
        'c', 'd', 'e', 'f', 'g', 'a', 'b', 'major', 'minor', 'sharp', 'flat', 'am', 'dm', 'em', 'bm'
    }
    
    text = text.lower()
    text = html.unescape(text)
    text = re.sub(r'[^\w\s]', ' ', text)
    words = text.split()
    return {w for w in words if w not in generic and not w.isdigit() and len(w) > 2}

def is_valid_match(row):
    status = row.get('status', '').lower()
    if status == 'not_found':
        return True # We keep non-matches for now, unless instructed otherwise.
        # Wait, the user said "remove any line where the matching is wrong".
        # If it's not_found, there is no match to be wrong.
        # Let's keep them so the user knows they are missing.
    
    title = row.get('Title', '')
    composer = row.get('Composer', '')
    
    # Paths to check
    paths = [row.get('pdf_path'), row.get('token_path'), row.get('gp_path'), row.get('xml_path'), row.get('file_path')]
    paths = [p for p in paths if p and p.lower() != 'none']
    
    if not paths:
        return False # Should have been not_found

    title_nums = extract_numbers(title)
    title_keywords = get_keywords(title)
    
    # Clean composer words
    comp_cleaned = re.sub(r'[^\w\s]', ' ', composer.lower())
    comp_words = {w for w in comp_cleaned.split() if len(w) > 2}

    for p in paths:
        filename = os.path.basename(p)
        path_nums = extract_numbers(filename)
        path_keywords = get_keywords(filename)
        path_lower = filename.lower()
        
        # Check composer match
        composer_match = any(word in path_lower for word in comp_words)
        
        # Check Opus/No match
        opus_conflict = (title_nums['opus'] and path_nums['opus'] and title_nums['opus'] != path_nums['opus'])
        no_conflict = (title_nums['no'] and path_nums['no'] and title_nums['no'] != path_nums['no'])
        
        if opus_conflict or no_conflict:
            continue # Hard mismatch
            
        # If numbers match perfectly
        num_perfect = (title_nums['opus'] and title_nums['opus'] == path_nums['opus']) or \
                      (title_nums['no'] and title_nums['no'] == path_nums['no'])
        
        # Keyword overlap
        overlap = title_keywords.intersection(path_keywords)
        
        if overlap:
            return True
        
        if composer_match and num_perfect:
            return True
            
        # Special case for Bach BWV
        bwv_title = re.search(r'BWV\s*(\d+)', title, re.IGNORECASE)
        bwv_path = re.search(r'BWV\s*(\d+)', filename, re.IGNORECASE)
        if bwv_title and bwv_path:
            if bwv_title.group(1) == bwv_path.group(1):
                return True
            else:
                continue

    return False

def clean_composer(name):
    name = name.strip()
    # Cleanup dictionary
    mapping = {
        'Agustin Barrios Mandore': 'Agustin Barrios Mangore',
        'Augustin Barrios Mangore': 'Agustin Barrios Mangore',
        'Alonso de Mudarra': 'Alonso Mudarra',
        'Anonyme': 'Anonymous',
        'Carl Philipp Emanuel Bach': 'C.P.E. Bach',
        'Dimitri Kabalevsky': 'Dmitri Kabalevsky',
        'Emanuel Andriaesen': 'Emanuel Adriaenssen',
        'Enriquez de Valderrabano': 'Enrique de Valderrabano',
        'Federico Moreno Torroba': 'Federico Moreno-Torroba',
        'F. Moreno-Torroba': 'Federico Moreno-Torroba',
        'Ferdinand Carulli': 'Ferdinando Carulli',
        'Ferdinando Sor': 'Fernando Sor',
        'Franz Josephn Haydn': 'Joseph Haydn',
        'Franz Joseph Haydn': 'Joseph Haydn',
        'George Friedrich Handel': 'George Frideric Handel',
        'Gerard Mountreuil': 'Gerard Montreuil',
        'J. S. Bach': 'J.S. Bach',
        'Jamie Mirtenbaum Zenamon': 'Jaime Mirtenbaum Zenamon',
        'Jean Baptiste Besard': 'Jean-Baptiste Besard',
        'Johann Antonin Logy': 'Johann Anton Logy',
        'Johann Anton Losy von Losinthal': 'Johann Anton Logy',
        'Johan Kaspar Mertz': 'Johann Kaspar Mertz',
        'Johann Kaspar Mertzr Mertz': 'Johann Kaspar Mertz',
        'Lodvico Roncalli': 'Ludovico Roncalli',
        'Lodovico Roncalli': 'Ludovico Roncalli',
        'Ludvico Roncalli': 'Ludovico Roncalli',
        'Luys Milasn': 'Luis Milan',
        'Manuel Ponce': 'Manuel M. Ponce',
        'Manuel M. Ponce': 'Manuel M. Ponce',
        'Mario Castelnuovo-TedescoCastelnuovo-Tedesco': 'Mario Castelnuovo-Tedesco',
        'Mateo Carcassi': 'Matteo Carcassi',
        'Melchoir Neusidler': 'Melchior Neusidler',
        'Norbert Craft': 'Norbert Kraft',
        'Phillip Rosseter': 'Philip Rosseter',
        'Reginald Smith-Brindle': 'Reginald Smith Brindle',
        'Robert De Visee': 'Robert de Visee',
        'Silvius Leopold Weiss': 'Silvius Leopold Weiss',
        'Sylvius Leopold Weiss': 'Silvius Leopold Weiss',
        'Sylvus Leopold Weiss': 'Silvius Leopold Weiss',
    }
    return mapping.get(name, name)

def clean():
    input_file = 'features/found_pieces.csv'
    output_temp = 'features/found_pieces_cleaned.csv'
    
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    
    # First pass: clean names and filter mismatches
    processed_rows = []
    removed_count = 0
    fixed_composer_count = 0
    
    for row in rows:
        # Fix composer
        orig_composer = row['Composer']
        # Special check for "Mandore" in title too
        if 'Mandore' in orig_composer or 'Mandore' in row['Title']:
            row['Composer'] = 'Agustin Barrios Mangore'
            fixed_composer_count += 1
        else:
            new_composer = clean_composer(orig_composer)
            if new_composer != orig_composer:
                row['Composer'] = new_composer
                fixed_composer_count += 1
            
        # Unescape title
        row['Title'] = html.unescape(row['Title'])
        
        if is_valid_match(row):
            processed_rows.append(row)
        else:
            removed_count += 1
            
    # Second pass: Deduplicate by (Title, Composer)
    unique_pieces = {}
    for row in processed_rows:
        key = (row['Title'], row['Composer'])
        if key not in unique_pieces:
            unique_pieces[key] = row
        else:
            # Merge paths
            existing = unique_pieces[key]
            for field in ['pdf_path', 'token_path', 'gp_path', 'xml_path', 'file_path']:
                if not existing.get(field) or existing.get(field).lower() == 'none':
                    existing[field] = row.get(field)
            # If status was not_found but now is found
            if existing['status'] == 'not_found' and row['status'] == 'found':
                existing['status'] = 'found'
                existing['source'] = row['source']
    
    cleaned_rows = list(unique_pieces.values())
        
    with open(output_temp, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cleaned_rows)
        
    print(f"Total rows processed: {len(rows)}")
    print(f"Fixed composer names: {fixed_composer_count}")
    print(f"Removed mismatched rows: {removed_count}")
    print(f"Deduplicated to: {len(cleaned_rows)} unique pieces")

if __name__ == '__main__':
    clean()
