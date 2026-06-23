import music21
import os
import json
import numpy as np
import pandas as pd
import re
import xml.etree.ElementTree as ET

def parse_guitar_xml(filepath):
    try:
        tree = ET.parse(filepath)
    except Exception as e:
        print(f"Error parsing XML {filepath}: {e}")
        return None
    root = tree.getroot()
    
    events = []
    # MusicXML notes
    for note in root.findall('.//note'):
        if note.find('rest') is not None: continue
        
        fret = note.find('.//fret')
        string = note.find('.//string')
        chord = note.find('chord') is not None
        
        if fret is not None and string is not None:
            try:
                events.append({
                    'fret': int(fret.text),
                    'string': int(string.text),
                    'chord': chord
                })
            except: pass
            
    if not events: return None
    
    chords = []
    current_chord = []
    for e in events:
        if not e['chord'] and current_chord:
            chords.append(current_chord)
            current_chord = []
        current_chord.append(e)
    if current_chord: chords.append(current_chord)
    
    features = {}
    barre_count, total_stretch, chord_count = 0, 0, 0
    avg_frets = []
    
    for c in chords:
        fs = [n['fret'] for n in c]
        avg_frets.append(np.mean(fs))
        if len(c) > 1:
            fixed_frets = [f for f in fs if f > 0]
            if fixed_frets:
                f_counts = {}
                for f in fixed_frets: f_counts[f] = f_counts.get(f, 0) + 1
                if any(cnt >= 3 for cnt in f_counts.values()): barre_count += 1
                total_stretch += max(fs) - min(fs)
                chord_count += 1
                
    features['barre_ratio'] = barre_count / len(chords) if chords else 0
    features['avg_chord_stretch'] = total_stretch / chord_count if chord_count else 0
    
    shifts = np.abs(np.diff(avg_frets))
    features['avg_position_shift'] = np.mean(shifts) if len(shifts) > 0 else 0
    features['max_position_shift'] = np.max(shifts) if len(shifts) > 0 else 0
    features['total_position_shift'] = np.sum(shifts) if len(shifts) > 0 else 0
    
    string_jumps = []
    for i in range(1, len(chords)):
        p_strs = [n['string'] for n in chords[i-1]]
        c_strs = [n['string'] for n in chords[i]]
        min_dist = min([abs(s1 - s2) for s1 in p_strs for s2 in c_strs])
        string_jumps.append(min_dist)
    features['avg_string_jump'] = np.mean(string_jumps) if string_jumps else 0
    return features

def extract_from_tokens(filepath):
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading tokens {filepath}: {e}")
        return None
        
    events = []
    current_chord = []
    for line in lines:
        line = line.strip()
        if 'note:s' in line:
            try:
                parts = line.split(':')
                s_part = [p for p in parts if p.startswith('s')][0]
                f_part = [p for p in parts if p.startswith('f')][0]
                current_chord.append({'string': int(s_part[1:]), 'fret': int(f_part[1:])})
            except: pass
        elif 'wait:' in line or 'new_measure' in line:
            if current_chord:
                events.append(current_chord)
                current_chord = []
    if current_chord: events.append(current_chord)
    if not events: return None
    
    features = {}
    barre_count, total_stretch, multi_note_chords = 0, 0, 0
    avg_frets = []
    for c in events:
        fs = [n['fret'] for n in c]
        avg_frets.append(np.mean(fs))
        if len(c) > 1:
            multi_note_chords += 1
            f_counts = {}
            for f in [n['fret'] for n in c if n['fret'] > 0]:
                f_counts[f] = f_counts.get(f, 0) + 1
            if any(cnt >= 3 for cnt in f_counts.values()): barre_count += 1
            total_stretch += max(fs) - min(fs)
            
    features['barre_ratio'] = barre_count / len(events)
    features['avg_chord_stretch'] = total_stretch / multi_note_chords if multi_note_chords else 0
    shifts = np.abs(np.diff(avg_frets))
    features['avg_position_shift'] = np.mean(shifts) if len(shifts) > 0 else 0
    features['max_position_shift'] = np.max(shifts) if len(shifts) > 0 else 0
    features['total_position_shift'] = np.sum(shifts) if len(shifts) > 0 else 0
    
    string_jumps = []
    for i in range(1, len(events)):
        p_strs = [n['string'] for n in events[i-1]]
        c_strs = [n['string'] for n in events[i]]
        min_dist = min([abs(s1 - s2) for s1 in p_strs for s2 in c_strs])
        string_jumps.append(min_dist)
    features['avg_string_jump'] = np.mean(string_jumps) if string_jumps else 0
    return features

def calculate_descriptors_from_chords(chords):
    features = {}
    barre_count, total_stretch, chord_count = 0, 0, 0
    avg_frets = []
    
    for c in chords:
        fs = [n['fret'] for n in c]
        avg_frets.append(np.mean(fs))
        if len(c) > 1:
            fixed_frets = [f for f in fs if f > 0]
            if fixed_frets:
                f_counts = {}
                for f in fixed_frets: f_counts[f] = f_counts.get(f, 0) + 1
                if any(cnt >= 3 for cnt in f_counts.values()): barre_count += 1
                total_stretch += max(fs) - min(fs)
                chord_count += 1
                
    features['barre_ratio'] = barre_count / len(chords) if chords else 0
    features['avg_chord_stretch'] = total_stretch / chord_count if chord_count else 0
    
    shifts = np.abs(np.diff(avg_frets))
    features['avg_position_shift'] = np.mean(shifts) if len(shifts) > 0 else 0
    features['max_position_shift'] = np.max(shifts) if len(shifts) > 0 else 0
    features['total_position_shift'] = np.sum(shifts) if len(shifts) > 0 else 0
    
    string_jumps = []
    for i in range(1, len(chords)):
        p_strs = [n['string'] for n in chords[i-1]]
        c_strs = [n['string'] for n in chords[i]]
        min_dist = min([abs(s1 - s2) for s1 in p_strs for s2 in c_strs])
        string_jumps.append(min_dist)
    features['avg_string_jump'] = np.mean(string_jumps) if string_jumps else 0
    return features

def parse_guitar_pdf(filepath):
    import subprocess
    cmd = ['pdftotext', '-bbox', filepath, '-']
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    
    if proc.returncode != 0:
        print(f"Error running pdftotext for {filepath}: {err.decode()}")
        return None
        
    try:
        out_str = out.decode('utf-8', errors='ignore')
        out_str = re.sub(r'\sxmlns="[^"]+"', '', out_str)
        root = ET.fromstring(out_str)
    except Exception as e:
        print(f"Error parsing XML for {filepath}: {e}")
        return None
        
    fret_re = re.compile(r'^\(?(\d+)(?:[/\-]\d+)?\)?$')
    
    all_chords = []
    
    for page in root.findall('.//page'):
        words = []
        for word in page.findall('.//word'):
            text = word.text.strip() if word.text else ""
            match = fret_re.match(text)
            if match:
                fret_val = int(match.group(1))
                x_min = float(word.attrib['xMin'])
                y_min = float(word.attrib['yMin'])
                x_max = float(word.attrib['xMax'])
                y_max = float(word.attrib['yMax'])
                words.append({
                    'fret': fret_val,
                    'x': (x_min + x_max) / 2.0,
                    'y': (y_min + y_max) / 2.0,
                })
        
        if not words:
            continue
            
        words.sort(key=lambda w: w['y'])
        
        clusters = []
        current_cluster = [words[0]]
        for w in words[1:]:
            if w['y'] - current_cluster[-1]['y'] > 20:
                clusters.append(current_cluster)
                current_cluster = [w]
            else:
                current_cluster.append(w)
        clusters.append(current_cluster)
        
        for cluster in clusters:
            ys = [w['y'] for w in cluster]
            y_span = max(ys) - min(ys)
            # Filter out tuning lines, headers, or footers
            if y_span < 15 or len(cluster) < 5:
                continue
                
            y_min = min(ys)
            system_notes = []
            for w in cluster:
                spacing = 6.75
                string_num = 1 + int(round((w['y'] - y_min) / spacing))
                if 1 <= string_num <= 6:
                    system_notes.append({
                        'fret': w['fret'],
                        'string': string_num,
                        'x': w['x']
                    })
                    
            system_notes.sort(key=lambda n: n['x'])
            
            chords = []
            if system_notes:
                current_chord = [system_notes[0]]
                for n in system_notes[1:]:
                    if n['x'] - current_chord[-1]['x'] < 3.0:
                        current_chord.append(n)
                    else:
                        chords.append(current_chord)
                        current_chord = [n]
                chords.append(current_chord)
                
            all_chords.extend(chords)
            
    if not all_chords:
        return None
        
    return calculate_descriptors_from_chords(all_chords)

def main():
    json_path = 'features/guitarburst_full.json'
    with open(json_path, 'r') as f:
        data = json.load(f)
    print(f"Extracting features for {len([i for i in data if i.get('status')=='found'])} pieces...")
    results = []
    for item in data:
        if item.get('status') != 'found': continue
        res = None
        if item.get('source') == 'gaps':
            path = item.get('xml_path')
            if path and os.path.exists(path):
                res = parse_guitar_xml(path)
        elif item.get('source') == 'dada_gp':
            path = item.get('token_path')
            if path and os.path.exists(path):
                res = extract_from_tokens(path)
        elif item.get('source') == 'pdf':
            path = item.get('pdf_path')
            if path and os.path.exists(path):
                # Verify it is ClassClef vector PDF
                is_classclef = False
                if "classclef" in path.lower() or os.path.basename(path).lower().startswith("classclef"):
                    is_classclef = True
                else:
                    import subprocess
                    try:
                        out_sample = subprocess.check_output(['pdftotext', '-l', '1', path, '-'])
                        if b'classclef' in out_sample.lower():
                            is_classclef = True
                    except:
                        pass
                if is_classclef:
                    res = parse_guitar_pdf(path)
                    
        if res:
            res.update({'Title': item['Title'], 'Composer': item['Composer'], 'Difficulty': item['Difficulty'], 'source': item['source']})
            results.append(res)
    
    pd.DataFrame(results).to_csv('features/guitar_technical_features.csv', index=False)
    print(f"Done. Extracted {len(results)} pieces.")

if __name__ == "__main__":
    main()
