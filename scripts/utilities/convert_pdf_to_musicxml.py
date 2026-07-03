#!/usr/bin/env python3
import os
import re
import pandas as pd
import xml.etree.ElementTree as ET
from xml.dom import minidom
import zipfile
import tempfile

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
XLSX_PATH = os.path.join(BASE_DIR, "data", "verified pieces.xlsx")

FRET_RE = re.compile(r'^(\d{1,2})$')

def midi_to_pitch(midi_num):
    names  = ['C','C','D','D','E','F','F','G','G','A','A','B']
    alters = [ 0,  1,  0,  1,  0,  0,  1,  0,  1,  0,  1,  0]
    note_idx = midi_num % 12
    octave   = (midi_num // 12) - 1
    return names[note_idx], alters[note_idx], octave

def sort_pdf_files(pdf_paths):
    def key_func(path):
        name = os.path.basename(path)
        m = re.search(r'(\d+)\.pdf$', name, re.IGNORECASE)
        if m: return (int(m.group(1)), name)
        m2 = re.search(r'(\d+)\s', name)
        if m2: return (int(m2.group(1)), name)
        return (0, name)
    return sorted(pdf_paths, key=key_func)

def _find_tab_staves(page):
    """
    Return list of tab staves detected from graphical lines on the page.
    Each staff is a list of 6 y-values in pdfplumber coords (y=0 at bottom),
    sorted ascending: [string-6-y, ..., string-1-y].
    """
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
        if len(g) != 6:
            continue
        spacings = [g[i+1] - g[i] for i in range(5)]
        if all(5 <= s <= 10 for s in spacings):
            tab_staves.append(g)  # g[0]=string6(low E), g[5]=string1(high E)
    return tab_staves

def parse_pdf_to_chords(filepath):
    """
    Parse a ClassClef vector PDF using pdfplumber.

    Returns (chords, tempo_bpm, time_sig):
      - chords:    list of events; each event is a list of {'fret', 'string', 'x'} dicts
      - tempo_bpm: int or None
      - time_sig:  (beats, beat_type) tuple, default (4, 4)

    String assignment uses detected graphical staff lines, so it is correct even
    when the topmost note in a system is not on string 1.
    """
    try:
        import pdfplumber
    except ImportError:
        print("pdfplumber not installed — run: pip install pdfplumber")
        return None, None, (4, 4)

    all_chords = []
    tempo_bpm  = None
    time_sig   = (4, 4)
    header_checked = False

    try:
        with pdfplumber.open(filepath) as doc:
            for page in doc.pages:
                ph = page.height
                tab_staves = _find_tab_staves(page)
                if not tab_staves:
                    continue

                words = page.extract_words(x_tolerance=1, y_tolerance=3)

                # ── Header extraction (first page only) ─────────────────────
                if not header_checked:
                    header_checked = True
                    # Topmost tab staff: largest pdfplumber y = closest to page top
                    top_staff_y = max(s[-1] for s in tab_staves)
                    # Convert to screen-top distance for word comparison
                    header_cutoff = ph - top_staff_y - 5
                    header_words  = [w for w in words if w['top'] < header_cutoff]

                    # Tempo: look for "= NNN" pair in header
                    for i, w in enumerate(header_words):
                        if w['text'] == '=' and i + 1 < len(header_words):
                            nw = header_words[i + 1]
                            if re.match(r'^\d{2,3}$', nw['text']):
                                bpm = int(nw['text'])
                                if 30 <= bpm <= 300:
                                    tempo_bpm = bpm
                                    break

                    # Time signature: stacked digit pair near left margin, in header zone only
                    # Valid beats: 2,3,4,6,9,12  Valid beat-types: 2,4,8,16
                    VALID_BEATS     = {2, 3, 4, 6, 9, 12}
                    VALID_BEATTYPE  = {2, 4, 8, 16}
                    ts_words = [
                        w for w in header_words
                        if float(w['x0']) < 75
                        and FRET_RE.match(w['text'])
                        and int(w['text']) in VALID_BEATS | VALID_BEATTYPE
                    ]
                    if len(ts_words) >= 2:
                        ts_words.sort(key=lambda w: w['top'])
                        t, b = ts_words[0], ts_words[1]
                        bt, bb = int(t['text']), int(b['text'])
                        if (abs(float(t['x0']) - float(b['x0'])) < 15
                                and bt in VALID_BEATS and bb in VALID_BEATTYPE):
                            time_sig = (bt, bb)

                # ── Note extraction per tab staff ────────────────────────────
                # Process staves top-to-bottom (reading order = decreasing pdfplumber y)
                for staff_ys in sorted(tab_staves, key=lambda s: -s[-1]):
                    y_lo = staff_ys[0] - 8
                    y_hi = staff_ys[5] + 8

                    staff_notes = []
                    for w in words:
                        m = FRET_RE.match(w['text'])
                        if not m:
                            continue
                        fret_val = int(m.group(1))
                        if fret_val > 24:
                            continue

                        wy = ph - (w['top'] + w['bottom']) / 2  # digit centre → pdfplumber y
                        if not (y_lo <= wy <= y_hi):
                            continue

                        dists   = [abs(wy - sy) for sy in staff_ys]
                        min_d   = min(dists)
                        if min_d > 8:
                            continue
                        idx        = dists.index(min_d)
                        string_num = 6 - idx  # idx 0 = string 6, idx 5 = string 1

                        staff_notes.append({
                            'fret':   fret_val,
                            'string': string_num,
                            'x':      (float(w['x0']) + float(w['x1'])) / 2,
                        })

                    if not staff_notes:
                        continue

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
        print(f"Error parsing {filepath}: {e}")
        return None, None, (4, 4)

    return (all_chords if all_chords else None), tempo_bpm, time_sig


def save_chords_to_musicxml(all_chords, title, output_path, tempo_bpm=None, time_sig=(4, 4)):
    beats, beat_type = time_sig
    root = ET.Element('score-partwise', version='4.0')

    work = ET.SubElement(root, 'work')
    ET.SubElement(work, 'work-title').text = title

    part_list  = ET.SubElement(root, 'part-list')
    score_part = ET.SubElement(part_list, 'score-part', id='P1')
    ET.SubElement(score_part, 'part-name').text = 'Guitar'

    part = ET.SubElement(root, 'part', id='P1')

    measures_chords = []
    cur_measure = []
    for chord in all_chords:
        cur_measure.append(chord)
        if len(cur_measure) == beats:
            measures_chords.append(cur_measure); cur_measure = []
    if cur_measure:
        measures_chords.append(cur_measure)

    open_strings = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}

    for measure_idx, chord_list in enumerate(measures_chords, 1):
        measure = ET.SubElement(part, 'measure', number=str(measure_idx))

        if measure_idx == 1:
            attributes = ET.SubElement(measure, 'attributes')
            ET.SubElement(attributes, 'divisions').text = '1'

            key = ET.SubElement(attributes, 'key')
            ET.SubElement(key, 'fifths').text = '0'

            time_el = ET.SubElement(attributes, 'time')
            ET.SubElement(time_el, 'beats').text = str(beats)
            ET.SubElement(time_el, 'beat-type').text = str(beat_type)

            ET.SubElement(attributes, 'staves').text = '2'

            clef1 = ET.SubElement(attributes, 'clef', number='1')
            ET.SubElement(clef1, 'sign').text = 'G'
            ET.SubElement(clef1, 'line').text = '2'
            ET.SubElement(clef1, 'clef-octave-change').text = '-1'

            clef2 = ET.SubElement(attributes, 'clef', number='2')
            ET.SubElement(clef2, 'sign').text = 'TAB'
            ET.SubElement(clef2, 'line').text = '5'

            sd1 = ET.SubElement(attributes, 'staff-details', number='1')
            ET.SubElement(sd1, 'staff-lines').text = '5'

            sd2 = ET.SubElement(attributes, 'staff-details', number='2')
            ET.SubElement(sd2, 'staff-lines').text = '6'
            for i, (step, octave) in enumerate(
                [('E',2),('A',2),('D',3),('G',3),('B',3),('E',4)], 1
            ):
                tuning = ET.SubElement(sd2, 'staff-tuning', line=str(i))
                ET.SubElement(tuning, 'tuning-step').text = step
                ET.SubElement(tuning, 'tuning-octave').text = str(octave)

            if tempo_bpm:
                direction = ET.SubElement(measure, 'direction', placement='above')
                ET.SubElement(direction, 'direction-type')  # required wrapper
                direction[-1].append(ET.Element('metronome'))
                direction[-1][0].append(ET.Element('beat-unit'))
                direction[-1][0][0].text = 'quarter'
                direction[-1][0].append(ET.Element('per-minute'))
                direction[-1][0][1].text = str(tempo_bpm)
                sound = ET.SubElement(direction, 'sound')
                sound.set('tempo', str(tempo_bpm))

        # Staff 1 – standard notation
        for chord_notes in chord_list:
            for idx, nd in enumerate(chord_notes):
                note = ET.SubElement(measure, 'note')
                if idx > 0: ET.SubElement(note, 'chord')
                midi_num = open_strings.get(nd['string'], 40) + nd['fret']
                step, alter, octave = midi_to_pitch(midi_num)
                pitch = ET.SubElement(note, 'pitch')
                ET.SubElement(pitch, 'step').text = step
                if alter: ET.SubElement(pitch, 'alter').text = str(alter)
                ET.SubElement(pitch, 'octave').text = str(octave)
                ET.SubElement(note, 'duration').text = '1'
                ET.SubElement(note, 'voice').text = '1'
                ET.SubElement(note, 'type').text = 'quarter'
                ET.SubElement(note, 'staff').text = '1'

        backup = ET.SubElement(measure, 'backup')
        ET.SubElement(backup, 'duration').text = str(len(chord_list))

        # Staff 2 – tablature
        for chord_notes in chord_list:
            for idx, nd in enumerate(chord_notes):
                note = ET.SubElement(measure, 'note')
                if idx > 0: ET.SubElement(note, 'chord')
                midi_num = open_strings.get(nd['string'], 40) + nd['fret']
                step, alter, octave = midi_to_pitch(midi_num)
                pitch = ET.SubElement(note, 'pitch')
                ET.SubElement(pitch, 'step').text = step
                if alter: ET.SubElement(pitch, 'alter').text = str(alter)
                ET.SubElement(pitch, 'octave').text = str(octave)
                ET.SubElement(note, 'duration').text = '1'
                ET.SubElement(note, 'voice').text = '5'
                ET.SubElement(note, 'type').text = 'quarter'
                ET.SubElement(note, 'stem').text = 'none'
                ET.SubElement(note, 'staff').text = '2'
                notations = ET.SubElement(note, 'notations')
                technical = ET.SubElement(notations, 'technical')
                ET.SubElement(technical, 'string').text = str(nd['string'])
                ET.SubElement(technical, 'fret').text = str(nd['fret'])

    xml_str    = ET.tostring(root, encoding='utf-8')
    parsed_xml = minidom.parseString(xml_str)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(parsed_xml.toprettyxml(indent='  '))


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Convert verified PDF pieces to MusicXML with standard & tab staves.")
    parser.add_argument('--force', '-f', action='store_true',
                        help='Force regeneration even if XML already exists.')
    parser.add_argument('--piece', '-p', type=str, default=None,
                        help='Process only a specific piece (substring match on Title).')
    args = parser.parse_args()

    if not os.path.exists(XLSX_PATH):
        print(f"Error: {XLSX_PATH} not found."); return

    print(f"Loading {XLSX_PATH}...")
    df = pd.read_excel(XLSX_PATH)

    mask = (df['validated'] == 1) & (df['source'] == 'pdf')
    pdf_pieces = df[mask]
    if args.piece:
        pdf_pieces = pdf_pieces[pdf_pieces['Title'].str.contains(args.piece, case=False, na=False)]
        print(f"Filtered by '{args.piece}': {len(pdf_pieces)} pieces.")
    else:
        print(f"Found {len(pdf_pieces)} validated PDF pieces.")

    success_count = fail_count = skipped_count = 0

    for idx, row in pdf_pieces.iterrows():
        title   = row['Title']
        pdf_rel = row['pdf_path']

        if not isinstance(pdf_rel, str) or not pdf_rel.strip() or pdf_rel.strip().lower() == 'nan':
            print(f"Skipping '{title}': no pdf_path."); continue

        pdf_basename   = os.path.basename(pdf_rel.strip())
        pdf_local_path = os.path.join(BASE_DIR, 'verified_pieces', 'pdf', pdf_basename)
        if not os.path.exists(pdf_local_path):
            pdf_orig = os.path.join(BASE_DIR, pdf_rel.strip())
            if os.path.exists(pdf_orig):
                pdf_local_path = pdf_orig
            else:
                print(f"Warning: PDF for '{title}' not found. Skipping.")
                fail_count += 1; continue

        xml_basename   = os.path.splitext(pdf_basename)[0] + '.musicxml'
        xml_local_path = os.path.join(BASE_DIR, 'verified_pieces', 'pdf', 'xml', xml_basename)
        os.makedirs(os.path.dirname(xml_local_path), exist_ok=True)
        relative_xml   = os.path.join('verified_pieces', 'pdf', 'xml', xml_basename)

        if os.path.exists(xml_local_path) and not args.force:
            if os.path.getsize(xml_local_path) > 100:
                print(f"Skipping '{title}': XML exists (use --force to regenerate).")
                df.at[idx, 'xml_path'] = relative_xml
                df.at[idx, 'status']   = 'found'
                skipped_count += 1; continue

        print(f"Converting '{title}' ({pdf_basename})...")
        try:
            if pdf_basename.lower().endswith('.zip'):
                with tempfile.TemporaryDirectory() as tmp:
                    with zipfile.ZipFile(pdf_local_path, 'r') as zf:
                        zf.extractall(tmp)
                    extracted = []
                    for rd, _, files in os.walk(tmp):
                        for f in files:
                            if f.lower().endswith('.pdf'):
                                extracted.append(os.path.join(rd, f))
                    if not extracted:
                        print(f" -> No PDFs in zip."); fail_count += 1; continue

                    all_chords, tempo, ts = [], None, (4, 4)
                    for pdf_path in sort_pdf_files(extracted):
                        chords, t, s = parse_pdf_to_chords(pdf_path)
                        if chords:
                            all_chords.extend(chords)
                        if t and tempo is None:
                            tempo = t
                        if s != (4, 4) and ts == (4, 4):
                            ts = s
                    if all_chords:
                        save_chords_to_musicxml(all_chords, title, xml_local_path, tempo, ts)
                        df.at[idx, 'xml_path'] = relative_xml
                        df.at[idx, 'status']   = 'found'
                        print(f" -> OK (zip): {xml_basename}  tempo={tempo}  ts={ts}")
                        success_count += 1
                    else:
                        print(f" -> Failed: no chords extracted from zip."); fail_count += 1
            else:
                chords, tempo, ts = parse_pdf_to_chords(pdf_local_path)
                if chords:
                    save_chords_to_musicxml(chords, title, xml_local_path, tempo, ts)
                    df.at[idx, 'xml_path'] = relative_xml
                    df.at[idx, 'status']   = 'found'
                    print(f" -> OK: {xml_basename}  tempo={tempo}  ts={ts}")
                    success_count += 1
                else:
                    print(f" -> Failed: no chords extracted."); fail_count += 1

        except Exception as e:
            print(f" -> Exception for '{title}': {e}"); fail_count += 1

    if success_count > 0 or skipped_count > 0:
        print(f"\nSaving updated spreadsheet...")
        df.to_excel(XLSX_PATH, index=False)

    print(f"\nSummary: converted={success_count}  skipped={skipped_count}  failed={fail_count}")


if __name__ == '__main__':
    main()
