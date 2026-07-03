#!/usr/bin/env python3
"""Combine PDF-extracted pitches/tab (verified_pieces/pdf/xml) with MIDI-derived
rhythm to produce MusicXML with real note durations instead of all-quarter-notes.
"""
import os
import csv
import glob
import argparse
import xml.etree.ElementTree as ET
from xml.dom import minidom
from fractions import Fraction

import pandas as pd
import music21 as m21

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
XLSX_PATH = os.path.join(BASE_DIR, "data", "verified pieces.xlsx")
MIDI_DIR = os.path.join(BASE_DIR, "midi-20260630T194726Z-3-001", "midi")
SKELETON_XML_DIR = os.path.join(BASE_DIR, "verified_pieces", "pdf", "xml")
OUTPUT_XML_DIR = os.path.join(BASE_DIR, "verified_pieces", "pdf", "xml_rhythm")
REPORT_CSV = os.path.join(BASE_DIR, "data", "midi_alignment_report.csv")

OPEN_STRINGS = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}
DIVISIONS = 16  # divisions per quarter note; exactly represents down to 32nd notes

REPORT_FIELDS = [
    'title', 'pdf_basename', 'midi_path', 'coverage_pct',
    'num_pdf_chords', 'num_midi_events', 'status', 'output_path',
]


# ── Duration <-> notated type/dots ──────────────────────────────────────────

def _representable_values():
    bases = [
        (Fraction(4), 'whole'), (Fraction(2), 'half'), (Fraction(1), 'quarter'),
        (Fraction(1, 2), 'eighth'), (Fraction(1, 4), '16th'), (Fraction(1, 8), '32nd'),
    ]
    vals = []
    for base, name in bases:
        for dots in (0, 1, 2):
            val = base * (2 - Fraction(1, 2 ** dots))
            vals.append((val, name, dots))
    return sorted(vals, key=lambda x: -x[0])


REPRESENTABLE = _representable_values()


def exact_type_dots(ql):
    for val, name, dots in REPRESENTABLE:
        if val == ql:
            return name, dots
    return None


def decompose_duration(ql):
    """Greedily split an arbitrary quarterLength into representable (type, dots, value)
    pieces, largest first, meant to be tied together."""
    remaining = ql
    parts = []
    guard = 0
    while remaining > 0 and guard < 12:
        for val, name, dots in REPRESENTABLE:
            if val <= remaining:
                parts.append((name, dots, val))
                remaining -= val
                guard += 1
                break
        else:
            parts.append(('32nd', 0, remaining))
            remaining = Fraction(0)
    return parts


def expand_chord_to_chunks(notes_data, duration_ql):
    exact = exact_type_dots(duration_ql)
    if exact:
        name, dots = exact
        return [{'notes': notes_data, 'type': name, 'dots': dots, 'value': duration_ql,
                 'tie_start': False, 'tie_stop': False}]
    parts = decompose_duration(duration_ql)
    chunks = []
    for i, (name, dots, val) in enumerate(parts):
        chunks.append({
            'notes': notes_data, 'type': name, 'dots': dots, 'value': val,
            'tie_start': i < len(parts) - 1,
            'tie_stop': i > 0,
        })
    return chunks


def group_into_measures(chunk_stream, target_value):
    """Place chunks into measures of `target_value` quarterLengths, splitting (and
    re-decomposing) any chunk that would straddle a measure boundary."""
    measures = []
    cur_measure = []
    cur_total = Fraction(0)
    queue = list(chunk_stream)
    i = 0
    while i < len(queue):
        chunk = queue[i]
        space_left = target_value - cur_total
        if space_left <= 0:
            measures.append(cur_measure)
            cur_measure, cur_total = [], Fraction(0)
            continue
        if chunk['value'] <= space_left:
            cur_measure.append(chunk)
            cur_total += chunk['value']
            i += 1
            continue

        first_parts = decompose_duration(space_left)
        rest_parts = decompose_duration(chunk['value'] - space_left)
        new_chunks = []
        for k, (name, dots, val) in enumerate(first_parts):
            new_chunks.append({
                'notes': chunk['notes'], 'type': name, 'dots': dots, 'value': val,
                'tie_start': True,
                'tie_stop': chunk['tie_stop'] if k == 0 else True,
            })
        for k, (name, dots, val) in enumerate(rest_parts):
            is_last = (k == len(rest_parts) - 1)
            new_chunks.append({
                'notes': chunk['notes'], 'type': name, 'dots': dots, 'value': val,
                'tie_start': chunk['tie_start'] if is_last else True,
                'tie_stop': True,
            })
        queue[i:i + 1] = new_chunks
    if cur_measure:
        measures.append(cur_measure)
    return measures


# ── MIDI chord-event extraction ─────────────────────────────────────────────

def extract_midi_events(midi_path):
    """Returns (events, time_sig) where events is a list of
    (frozenset(midi pitch numbers), duration_quarterLength) sorted by onset,
    and time_sig is (numerator, denominator) from the MIDI meta-event or None."""
    score = m21.converter.parse(midi_path)

    # Extract time signature: pick the one covering the longest span of music.
    # Some MIDIs have a short pickup-measure time sig (e.g. 1/4) at offset 0
    # followed by the real time sig (e.g. 3/4) immediately after — taking the
    # first would give the wrong result for the whole piece.
    time_sig = None
    ts_events = [(float(ts.offset), ts.numerator, ts.denominator)
                 for ts in score.flatten().getElementsByClass(m21.meter.TimeSignature)]
    if ts_events:
        total_len = float(score.flatten().highestTime)
        best_span, best_ts = -1, None
        for i, (off, num, den) in enumerate(ts_events):
            span = (ts_events[i + 1][0] if i + 1 < len(ts_events) else total_len) - off
            if span > best_span:
                best_span, best_ts = span, (num, den)
        time_sig = best_ts

    notes = sorted(score.flatten().notes, key=lambda n: n.offset)

    groups = []
    for n in notes:
        off = round(float(n.offset), 4)
        if groups and groups[-1][0] == off:
            groups[-1][1].append(n)
        else:
            groups.append([off, [n]])

    events = []
    for i, (off, ns) in enumerate(groups):
        pitches = set()
        for n in ns:
            if n.isChord:
                pitches.update(p.midi for p in n.pitches)
            else:
                pitches.add(n.pitch.midi)
        # Use inter-onset interval (IOI) as duration: reflects notated rhythm rather
        # than sounding duration, which is inflated by sustaining bass strings.
        if i + 1 < len(groups):
            dur = Fraction(groups[i + 1][0] - off).limit_denominator(64)
        else:
            dur = max(Fraction(n.quarterLength).limit_denominator(64) for n in ns)
        events.append((frozenset(pitches), dur))
    return events, time_sig


# ── Skeleton MusicXML extraction ────────────────────────────────────────────

def extract_skeleton(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    part = root.find('part')

    title_el = root.find('work/work-title')
    title = title_el.text if title_el is not None else ''

    first_measure = part.find('measure')
    time_el = first_measure.find('attributes/time')
    beats = int(time_el.find('beats').text)
    beat_type = int(time_el.find('beat-type').text)

    tempo_bpm = None
    sound_el = first_measure.find('direction/sound')
    if sound_el is not None and 'tempo' in sound_el.attrib:
        tempo_bpm = int(float(sound_el.attrib['tempo']))

    chords = []
    cur_chord = None
    for measure in part.findall('measure'):
        for note in measure.findall('note'):
            staff_el = note.find('staff')
            if staff_el is None or staff_el.text != '2':
                continue
            technical = note.find('notations/technical')
            if technical is None:
                continue
            nd = {
                'string': int(technical.find('string').text),
                'fret': int(technical.find('fret').text),
            }
            if note.find('chord') is not None and cur_chord is not None:
                cur_chord.append(nd)
            else:
                cur_chord = [nd]
                chords.append(cur_chord)

    return {
        'title': title,
        'chords': chords,
        'tempo_bpm': tempo_bpm,
        'time_sig': (beats, beat_type),
    }


def chord_pitches(chord):
    return frozenset(OPEN_STRINGS.get(nd['string'], 40) + nd['fret'] for nd in chord)


# ── Alignment ────────────────────────────────────────────────────────────────

def _match_cost(pdf_set, midi_set):
    if pdf_set == midi_set:
        return 0.0
    if not pdf_set or not midi_set:
        return 4.0
    # Tremolo / alternating-bass textures: the PDF picks up one plucked note per
    # column, but the MIDI sounds it together with a sustained/re-struck bass note
    # underneath. That's a correct subset relationship, not a wrong note.
    if pdf_set.issubset(midi_set):
        return 0.3 * (len(midi_set) - len(pdf_set))
    if midi_set.issubset(pdf_set):
        return 0.3 * (len(pdf_set) - len(midi_set))
    union = pdf_set | midi_set
    jaccard_dist = 1 - len(pdf_set & midi_set) / len(union)
    return jaccard_dist * 4.0


GAP_COST = 2.0
CONFIDENT_COST_CAP = 1.0  # jaccard_dist < 0.25


def align(pdf_chords, midi_events):
    """Global (Needleman-Wunsch) alignment of PDF chord pitch-sets to MIDI chord
    pitch-sets. Returns (matches: {pdf_idx: midi_idx}, coverage: float)."""
    pdf_sets = [chord_pitches(c) for c in pdf_chords]
    midi_sets = [e[0] for e in midi_events]
    n, m = len(pdf_sets), len(midi_sets)
    if n == 0:
        return {}, 1.0
    if m == 0:
        return {}, 0.0

    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + GAP_COST
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + GAP_COST

    for i in range(1, n + 1):
        pset = pdf_sets[i - 1]
        row, prev_row = dp[i], dp[i - 1]
        for j in range(1, m + 1):
            cost = _match_cost(pset, midi_sets[j - 1])
            row[j] = min(prev_row[j - 1] + cost, prev_row[j] + GAP_COST, row[j - 1] + GAP_COST)

    matches = {}
    i, j = n, m
    while i > 0 and j > 0:
        cost = _match_cost(pdf_sets[i - 1], midi_sets[j - 1])
        if abs(dp[i][j] - (dp[i - 1][j - 1] + cost)) < 1e-9:
            if cost < CONFIDENT_COST_CAP:
                matches[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif abs(dp[i][j] - (dp[i - 1][j] + GAP_COST)) < 1e-9:
            i -= 1
        else:
            j -= 1

    coverage = len(matches) / n
    return matches, coverage


# ── MusicXML writer ──────────────────────────────────────────────────────────

def _write_note(measure_el, chunk_idx, chunk, note_idx, voice, staff_num, with_tab):
    nd = chunk['notes'][note_idx]
    note = ET.SubElement(measure_el, 'note')
    if note_idx > 0:
        ET.SubElement(note, 'chord')

    midi_num = OPEN_STRINGS.get(nd['string'], 40) + nd['fret']
    names = ['C', 'C', 'D', 'D', 'E', 'F', 'F', 'G', 'G', 'A', 'A', 'B']
    alters = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0]
    idx = midi_num % 12
    octave = (midi_num // 12) - 1
    pitch = ET.SubElement(note, 'pitch')
    ET.SubElement(pitch, 'step').text = names[idx]
    if alters[idx]:
        ET.SubElement(pitch, 'alter').text = str(alters[idx])
    ET.SubElement(pitch, 'octave').text = str(octave)

    duration_int = round(chunk['value'] * DIVISIONS)
    ET.SubElement(note, 'duration').text = str(duration_int)

    if chunk['tie_stop']:
        ET.SubElement(note, 'tie', type='stop')
    if chunk['tie_start']:
        ET.SubElement(note, 'tie', type='start')

    ET.SubElement(note, 'voice').text = str(voice)
    ET.SubElement(note, 'type').text = chunk['type']
    for _ in range(chunk['dots']):
        ET.SubElement(note, 'dot')
    if staff_num == 2:
        ET.SubElement(note, 'stem').text = 'none'
    ET.SubElement(note, 'staff').text = str(staff_num)

    if chunk['tie_stop'] or chunk['tie_start'] or with_tab:
        notations = ET.SubElement(note, 'notations')
        if chunk['tie_stop']:
            ET.SubElement(notations, 'tied', type='stop')
        if chunk['tie_start']:
            ET.SubElement(notations, 'tied', type='start')
        if with_tab:
            technical = ET.SubElement(notations, 'technical')
            ET.SubElement(technical, 'string').text = str(nd['string'])
            ET.SubElement(technical, 'fret').text = str(nd['fret'])


def write_rhythm_musicxml(skeleton, measures, output_path):
    beats, beat_type = skeleton['time_sig']
    root = ET.Element('score-partwise', version='4.0')

    work = ET.SubElement(root, 'work')
    ET.SubElement(work, 'work-title').text = skeleton['title']

    part_list = ET.SubElement(root, 'part-list')
    score_part = ET.SubElement(part_list, 'score-part', id='P1')
    ET.SubElement(score_part, 'part-name').text = 'Guitar'

    part = ET.SubElement(root, 'part', id='P1')

    for measure_idx, chunk_list in enumerate(measures, 1):
        measure = ET.SubElement(part, 'measure', number=str(measure_idx))

        if measure_idx == 1:
            attributes = ET.SubElement(measure, 'attributes')
            ET.SubElement(attributes, 'divisions').text = str(DIVISIONS)

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
                [('E', 2), ('A', 2), ('D', 3), ('G', 3), ('B', 3), ('E', 4)], 1
            ):
                tuning = ET.SubElement(sd2, 'staff-tuning', line=str(i))
                ET.SubElement(tuning, 'tuning-step').text = step
                ET.SubElement(tuning, 'tuning-octave').text = str(octave)

            if skeleton['tempo_bpm']:
                direction = ET.SubElement(measure, 'direction', placement='above')
                direction_type = ET.SubElement(direction, 'direction-type')
                metronome = ET.SubElement(direction_type, 'metronome')
                ET.SubElement(metronome, 'beat-unit').text = 'quarter'
                ET.SubElement(metronome, 'per-minute').text = str(skeleton['tempo_bpm'])
                sound = ET.SubElement(direction, 'sound')
                sound.set('tempo', str(skeleton['tempo_bpm']))

        for chunk_idx, chunk in enumerate(chunk_list):
            for note_idx in range(len(chunk['notes'])):
                _write_note(measure, chunk_idx, chunk, note_idx, voice=1, staff_num=1, with_tab=False)

        measure_duration = sum(round(c['value'] * DIVISIONS) for c in chunk_list)
        backup = ET.SubElement(measure, 'backup')
        ET.SubElement(backup, 'duration').text = str(measure_duration)

        for chunk_idx, chunk in enumerate(chunk_list):
            for note_idx in range(len(chunk['notes'])):
                _write_note(measure, chunk_idx, chunk, note_idx, voice=5, staff_num=2, with_tab=True)

    xml_str = ET.tostring(root, encoding='utf-8')
    parsed_xml = minidom.parseString(xml_str)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(parsed_xml.toprettyxml(indent='  '))


# ── Driver ───────────────────────────────────────────────────────────────────

def build_midi_lookup():
    lookup = {}
    for path in glob.glob(os.path.join(MIDI_DIR, '**', '*.mid'), recursive=True):
        stem = os.path.splitext(os.path.basename(path))[0].lower()
        lookup[stem] = path
    return lookup


def load_report(path):
    rows = {}
    if os.path.exists(path):
        with open(path, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                rows[row['pdf_basename']] = row
    return rows


def save_report(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        for row in sorted(rows.values(), key=lambda r: r.get('title', '')):
            writer.writerow(row)


def process_piece(title, pdf_basename, skeleton_path, midi_path, threshold):
    skeleton = extract_skeleton(skeleton_path)
    midi_events, midi_time_sig = extract_midi_events(midi_path)
    # MIDI time-signature meta-event is authoritative: the PDF parser defaults to
    # 4/4 when it can't find text-rendered time-sig digits (e.g. vector-graphic PDFs).
    if midi_time_sig is not None:
        skeleton['time_sig'] = midi_time_sig
    matches, coverage = align(skeleton['chords'], midi_events)
    n = len(skeleton['chords'])

    report_row = {
        'title': title, 'pdf_basename': pdf_basename, 'midi_path': midi_path,
        'coverage_pct': f'{coverage:.3f}', 'num_pdf_chords': n,
        'num_midi_events': len(midi_events), 'status': '', 'output_path': '',
    }

    if coverage < threshold:
        report_row['status'] = 'skipped_low_coverage'
        return report_row, None, skeleton

    durations = [
        midi_events[matches[i]][1] if i in matches else Fraction(1)
        for i in range(n)
    ]
    beats, beat_type = skeleton['time_sig']
    target = beats * Fraction(4, beat_type)

    chunk_stream = []
    for chord, dur in zip(skeleton['chords'], durations):
        chunk_stream.extend(expand_chord_to_chunks(chord, dur))
    measures = group_into_measures(chunk_stream, target)

    report_row['status'] = 'ok'
    return report_row, measures, skeleton


def main():
    parser = argparse.ArgumentParser(
        description="Combine PDF-extracted tab with MIDI rhythm into corrected MusicXML.")
    parser.add_argument('--piece', '-p', type=str, default=None,
                         help='Process only pieces matching this Title substring.')
    parser.add_argument('--force', '-f', action='store_true',
                         help='Reprocess even if output already exists.')
    parser.add_argument('--threshold', type=float, default=0.9,
                         help='Minimum alignment coverage to keep a piece (default 0.9).')
    args = parser.parse_args()

    df = pd.read_excel(XLSX_PATH)
    mask = (df['validated'] == 1) & (df['source'] == 'pdf')
    pieces = df[mask]
    if args.piece:
        pieces = pieces[pieces['Title'].str.contains(args.piece, case=False, na=False)]
        print(f"Filtered by '{args.piece}': {len(pieces)} pieces.")
    else:
        print(f"Found {len(pieces)} validated PDF pieces.")

    midi_lookup = build_midi_lookup()
    os.makedirs(OUTPUT_XML_DIR, exist_ok=True)
    report = load_report(REPORT_CSV)

    counts = {'ok': 0, 'skipped_low_coverage': 0, 'no_midi_match': 0, 'no_skeleton': 0, 'error': 0, 'skipped_exists': 0}

    for _, row in pieces.iterrows():
        title = row['Title']
        pdf_rel = row['pdf_path']
        if not isinstance(pdf_rel, str) or not pdf_rel.strip():
            continue

        pdf_basename = os.path.basename(pdf_rel.strip())
        stem = os.path.splitext(pdf_basename)[0].lower()
        xml_basename = os.path.splitext(pdf_basename)[0] + '.musicxml'
        skeleton_path = os.path.join(SKELETON_XML_DIR, xml_basename)
        output_path = os.path.join(OUTPUT_XML_DIR, xml_basename)

        if not os.path.exists(skeleton_path):
            print(f"Skipping '{title}': no skeleton XML.")
            report[pdf_basename] = {
                'title': title, 'pdf_basename': pdf_basename, 'midi_path': '',
                'coverage_pct': '', 'num_pdf_chords': '', 'num_midi_events': '',
                'status': 'no_skeleton', 'output_path': '',
            }
            counts['no_skeleton'] += 1
            continue

        midi_path = midi_lookup.get(stem)
        if not midi_path:
            print(f"Skipping '{title}': no MIDI match for stem '{stem}'.")
            report[pdf_basename] = {
                'title': title, 'pdf_basename': pdf_basename, 'midi_path': '',
                'coverage_pct': '', 'num_pdf_chords': '', 'num_midi_events': '',
                'status': 'no_midi_match', 'output_path': '',
            }
            counts['no_midi_match'] += 1
            continue

        if os.path.exists(output_path) and not args.force:
            print(f"Skipping '{title}': output exists (use --force to regenerate).")
            counts['skipped_exists'] += 1
            continue

        try:
            report_row, measures, skeleton = process_piece(
                title, pdf_basename, skeleton_path, midi_path, args.threshold)
            if measures is None:
                print(f"'{title}': coverage={float(report_row['coverage_pct']):.1%} -> skipped (below threshold).")
                counts['skipped_low_coverage'] += 1
            else:
                write_rhythm_musicxml(skeleton, measures, output_path)
                report_row['output_path'] = os.path.relpath(output_path, BASE_DIR)
                print(f"'{title}': coverage={float(report_row['coverage_pct']):.1%} -> OK ({xml_basename})")
                counts['ok'] += 1
            report[pdf_basename] = report_row
        except Exception as e:
            print(f" -> Exception for '{title}': {e}")
            counts['error'] += 1

    save_report(report, REPORT_CSV)
    print(f"\nSummary: {counts}")


if __name__ == '__main__':
    main()
