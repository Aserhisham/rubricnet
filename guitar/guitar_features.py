import os
import json
import numpy as np
import pandas as pd
import re
import xml.etree.ElementTree as ET
from collections import Counter

FRET_RE = re.compile(r'^(\d{1,2})$')

def get_chords_from_xml(filepath):
    try:
        tree = ET.parse(filepath)
    except Exception as e:
        print(f"Error parsing XML {filepath}: {e}")
        return None, None, None
    root = tree.getroot()

    # Tempo: first <sound tempo="..."> element
    tempo_bpm = None
    for sound in root.findall('.//sound'):
        t = sound.get('tempo')
        if t:
            try:
                tempo_bpm = int(float(t))
                break
            except: pass

    events = []
    technique_count = 0
    total_note_count = 0

    for note in root.findall('.//note'):
        if note.find('rest') is not None: continue
        fret   = note.find('.//fret')
        string = note.find('.//string')
        chord  = note.find('chord') is not None
        if fret is not None and string is not None:
            try:
                events.append({
                    'fret':   int(fret.text),
                    'string': int(string.text),
                    'chord':  chord,
                })
                total_note_count += 1
                # Count special techniques from notations
                notations = note.find('notations')
                if notations is not None:
                    technical = notations.find('technical')
                    if technical is not None:
                        for tag in ('hammer-on', 'pull-off', 'slide'):
                            if technical.find(tag) is not None:
                                technique_count += 1
                                break
                    if notations.find('slur') is not None:
                        technique_count += 1
                    ornaments = notations.find('ornaments')
                    if ornaments is not None and ornaments.find('tremolo') is not None:
                        technique_count += 1
            except: pass

    if not events: return None, tempo_bpm, None

    chords, cur = [], []
    for e in events:
        if not e['chord'] and cur:
            chords.append(cur); cur = []
        cur.append(e)
    if cur: chords.append(cur)

    technique_ratio = technique_count / total_note_count if total_note_count > 0 else 0
    return chords, tempo_bpm, technique_ratio

def parse_guitar_xml(filepath):
    chords, tempo_bpm, technique_ratio = get_chords_from_xml(filepath)
    if chords is None:
        return None
    features = calculate_descriptors_from_chords(chords)
    if tempo_bpm:
        features['tempo_bpm'] = tempo_bpm
    if technique_ratio is not None:
        features['special_technique_ratio'] = technique_ratio
    return features

def get_chords_from_tokens(filepath):
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading tokens {filepath}: {e}")
        return None

    chords, cur = [], []
    for line in lines:
        line = line.strip()
        if 'note:s' in line:
            try:
                parts  = line.split(':')
                s_part = next(p for p in parts if p.startswith('s'))
                f_part = next(p for p in parts if p.startswith('f'))
                cur.append({'string': int(s_part[1:]), 'fret': int(f_part[1:])})
            except: pass
        elif ('wait:' in line or 'new_measure' in line) and cur:
            chords.append(cur); cur = []
    if cur: chords.append(cur)
    if not chords: return None
    return chords

def extract_from_tokens(filepath):
    chords = get_chords_from_tokens(filepath)
    if chords is None:
        return None
    return calculate_descriptors_from_chords(chords)

def calculate_descriptors_v2(chords):
    features = calculate_descriptors_from_chords(chords)
    if not chords:
        new_feats = {
            'log_total_notes': 0.0,
            'avg_fret': 0.0,
            'p90_fret': 0.0,
            'high_position_ratio': 0.0,
            'open_string_ratio': 0.0,
            'p90_chord_stretch': 0.0,
            'chord_ratio': 0.0,
            'avg_string_span': 0.0,
            'unique_shape_rate': 0.0,
            'shift_rate': 0.0,
            'max_position_shift': 0.0,
            'std_position_shift': 0.0,
            'fret_entropy': 0.0,
            'string_entropy': 0.0,
            'repetition_ratio': 0.0
        }
        features.update(new_feats)
        return features

    all_notes = [n for c in chords for n in c]
    total_notes = len(all_notes)
    all_frets = [n['fret'] for n in all_notes]
    all_strings = [n['string'] for n in all_notes]

    features['log_total_notes'] = float(np.log1p(total_notes))

    if all_frets:
        features['avg_fret'] = float(np.mean(all_frets))
        features['p90_fret'] = float(np.percentile(all_frets, 90))
        features['high_position_ratio'] = float(sum(1 for f in all_frets if f >= 7) / len(all_frets))
        features['open_string_ratio'] = float(sum(1 for f in all_frets if f == 0) / len(all_frets))
    else:
        features['avg_fret'] = 0.0
        features['p90_fret'] = 0.0
        features['high_position_ratio'] = 0.0
        features['open_string_ratio'] = 0.0

    stretches = []
    for c in chords:
        fs = [n['fret'] for n in c]
        fixed_frets = [f for f in fs if f > 0]
        if len(c) > 1 and fixed_frets:
            stretches.append(max(fixed_frets) - min(fixed_frets))
    
    if stretches:
        features['p90_chord_stretch'] = float(np.percentile(stretches, 90))
    else:
        features['p90_chord_stretch'] = 0.0

    features['chord_ratio'] = float(sum(1 for c in chords if len(c) >= 2) / len(chords))

    spans = [max(n['string'] for n in c) - min(n['string'] for n in c) for c in chords if len(c) >= 2]
    features['avg_string_span'] = float(np.mean(spans)) if spans else 0.0

    shapes = [tuple(sorted((n['string'], n['fret']) for n in c)) for c in chords if len(c) >= 2]
    features['unique_shape_rate'] = float(len(set(shapes)) / len(shapes)) if shapes else 0.0

    avg_frets = [float(np.mean([n['fret'] for n in c])) for c in chords]
    if len(chords) >= 2:
        diffs = np.abs(np.diff(avg_frets))
        features['shift_rate'] = float(np.mean(diffs > 2))
        features['max_position_shift'] = float(np.max(diffs))
        features['std_position_shift'] = float(np.std(diffs))
    else:
        features['shift_rate'] = 0.0
        features['max_position_shift'] = 0.0
        features['std_position_shift'] = 0.0

    if all_frets:
        counts = Counter(all_frets)
        probs = [cnt / len(all_frets) for cnt in counts.values()]
        features['fret_entropy'] = float(-sum(p * np.log2(p) for p in probs))
    else:
        features['fret_entropy'] = 0.0

    if all_strings:
        counts = Counter(all_strings)
        probs = [cnt / len(all_strings) for cnt in counts.values()]
        features['string_entropy'] = float(-sum(p * np.log2(p) for p in probs))
    else:
        features['string_entropy'] = 0.0

    if len(chords) >= 2:
        chord_sets = [frozenset((n['string'], n['fret']) for n in c) for c in chords]
        repeated = sum(1 for i in range(1, len(chords)) if chord_sets[i] == chord_sets[i - 1])
        features['repetition_ratio'] = float(repeated / (len(chords) - 1))
    else:
        features['repetition_ratio'] = 0.0

    return features

def calculate_descriptors_from_chords(chords):
    """
    Compute all 12 guitar difficulty descriptors from a list of chord events.
    Each chord is a list of {'fret': int, 'string': int} dicts.
    tempo_bpm and special_technique_ratio are NOT set here — callers add them
    after the fact when the source provides that data.
    """
    if not chords:
        return {}

    features = {}
    total_notes   = sum(len(c) for c in chords)
    barre_count   = 0
    total_stretch = 0
    max_stretch   = 0
    chord_count   = 0  # multi-note chords with at least one fretted note
    avg_frets     = []

    # ── LH: barre, stretch, position ────────────────────────────────────────
    for c in chords:
        fs          = [n['fret'] for n in c]
        fixed_frets = [f for f in fs if f > 0]
        avg_frets.append(float(np.mean(fs)))

        if len(c) > 1 and fixed_frets:
            f_counts = {}
            for f in fixed_frets:
                f_counts[f] = f_counts.get(f, 0) + 1
            if any(cnt >= 3 for cnt in f_counts.values()):
                barre_count += 1
            stretch        = max(fixed_frets) - min(fixed_frets)
            total_stretch += stretch
            max_stretch    = max(max_stretch, stretch)
            chord_count   += 1

    features['barre_ratio']      = barre_count / len(chords)
    features['avg_chord_stretch'] = total_stretch / chord_count if chord_count else 0
    features['max_chord_stretch'] = max_stretch

    shifts = np.abs(np.diff(avg_frets))
    features['avg_position_shift'] = float(np.mean(shifts)) if len(shifts) else 0

    # fret_change_rate: fraction of consecutive event-pairs where the fret set changes,
    # normalised by total notes (keeps it length-independent across polyphony levels)
    fret_sets     = [frozenset(n['fret'] for n in c) for c in chords]
    fret_changes  = sum(1 for i in range(1, len(chords)) if fret_sets[i] != fret_sets[i - 1])
    features['fret_change_rate'] = fret_changes / (total_notes - 1) if total_notes > 1 else 0

    # ── RH: string jumps, arpeggio density ──────────────────────────────────
    string_jumps        = []
    single_transitions  = 0
    string_changes      = 0

    for i in range(1, len(chords)):
        p_strs   = [n['string'] for n in chords[i - 1]]
        c_strs   = [n['string'] for n in chords[i]]
        min_dist = min(abs(s1 - s2) for s1 in p_strs for s2 in c_strs)
        string_jumps.append(min_dist)

        if len(chords[i - 1]) == 1 and len(chords[i]) == 1:
            single_transitions += 1
            if chords[i - 1][0]['string'] != chords[i][0]['string']:
                string_changes += 1

    features['arpeggio_density'] = string_changes / single_transitions if single_transitions else 0
    features['avg_string_jump']  = float(np.mean(string_jumps)) if string_jumps else 0
    features['max_string_jump']  = int(max(string_jumps)) if string_jumps else 0
    # special_technique_ratio: default 0; parse_guitar_xml overrides from XML notations
    features['special_technique_ratio'] = 0

    # ── Global ───────────────────────────────────────────────────────────────
    features['avg_polyphony'] = total_notes / len(chords)
    features['total_notes']   = total_notes
    # tempo_bpm: not in chord data; callers (parse_guitar_xml / parse_guitar_pdf) add it

    return features

def _find_tab_staves(page):
    h_lines = [
        l for l in page.lines
        if abs(l.get('height', 0)) < 1.5 and (l['x1'] - l['x0']) > 50
    ]
    ys = sorted(set(round(l['y0'], 1) for l in h_lines))
    if not ys:
        return []
    groups, cur = [], [ys[0]]
    for y in ys[1:]:
        if y - cur[-1] < 20:
            cur.append(y)
        else:
            groups.append(cur); cur = [y]
    groups.append(cur)
    tab_staves = []
    for g in groups:
        if len(g) != 6: continue
        spacings = [g[i+1] - g[i] for i in range(5)]
        if all(5 <= s <= 10 for s in spacings):
            tab_staves.append(g)
    return tab_staves


def get_chords_from_pdf(filepath):
    try:
        import pdfplumber
    except ImportError:
        print("pdfplumber not installed — run: pip install pdfplumber")
        return None, None

    all_chords = []
    tempo_bpm  = None
    header_checked = False

    try:
        with pdfplumber.open(filepath) as doc:
            for page in doc.pages:
                ph = page.height
                tab_staves = _find_tab_staves(page)
                if not tab_staves:
                    continue

                words = page.extract_words(x_tolerance=1, y_tolerance=3)

                if not header_checked:
                    header_checked = True
                    top_staff_y   = max(s[-1] for s in tab_staves)
                    header_cutoff = ph - top_staff_y - 5
                    header_words  = [w for w in words if w['top'] < header_cutoff]
                    for i, w in enumerate(header_words):
                        if w['text'] == '=' and i + 1 < len(header_words):
                            nw = header_words[i + 1]
                            if re.match(r'^\d{2,3}$', nw['text']):
                                bpm = int(nw['text'])
                                if 30 <= bpm <= 300:
                                    tempo_bpm = bpm
                                    break

                for staff_ys in sorted(tab_staves, key=lambda s: -s[-1]):
                    y_lo = staff_ys[0] - 8
                    y_hi = staff_ys[5] + 8
                    staff_notes = []
                    for w in words:
                        m = FRET_RE.match(w['text'])
                        if not m: continue
                        fret_val = int(m.group(1))
                        if fret_val > 24: continue
                        wy = ph - (w['top'] + w['bottom']) / 2  # digit centre → pdfplumber y
                        if not (y_lo <= wy <= y_hi): continue
                        dists  = [abs(wy - sy) for sy in staff_ys]
                        min_d  = min(dists)
                        if min_d > 8: continue
                        idx        = dists.index(min_d)
                        string_num = 6 - idx
                        staff_notes.append({
                            'fret':   fret_val,
                            'string': string_num,
                            'x':      (float(w['x0']) + float(w['x1'])) / 2,
                        })
                    if not staff_notes: continue
                    staff_notes.sort(key=lambda n: n['x'])
                    chords, cur_chord = [], [staff_notes[0]]
                    for n in staff_notes[1:]:
                        if n['x'] - cur_chord[-1]['x'] < 6:
                            cur_chord.append(n)
                        else:
                            chords.append(cur_chord); cur_chord = [n]
                    chords.append(cur_chord)
                    all_chords.extend(chords)
    except Exception as e:
        print(f"Error parsing PDF {filepath}: {e}")
        return None, None

    if not all_chords:
        return None, None

    return all_chords, tempo_bpm


def parse_guitar_pdf(filepath):
    chords, tempo_bpm = get_chords_from_pdf(filepath)
    if chords is None:
        return None
    features = calculate_descriptors_from_chords(chords)
    if tempo_bpm:
        features['tempo_bpm'] = tempo_bpm
    return features

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
