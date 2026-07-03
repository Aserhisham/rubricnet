#!/usr/bin/env python3
import os
import subprocess
import shutil
import pandas as pd
import xml.etree.ElementTree as ET
import copy

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
XLSX_PATH = os.path.join(BASE_DIR, "data", "verified pieces.xlsx")

def detect_musescore():
    """Detects MuseScore executable in the system."""
    # Check flatpak first
    try:
        res = subprocess.run(['flatpak', 'run', 'org.musescore.MuseScore', '--version'], 
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode == 0:
            print("Detected MuseScore via Flatpak")
            return ['flatpak', 'run', 'org.musescore.MuseScore']
    except Exception:
        pass
    
    # Check other executables
    for exe in ['mscore3', 'musescore3', 'mscore', 'musescore']:
        if shutil.which(exe):
            print(f"Detected MuseScore via executable: {exe}")
            return [exe]
            
    return None

def post_process_xml(xml_path):
    """Post-processes MuseScore MusicXML output to:
    1. Force a correct TAB clef for Staff 2.
    2. Sync tuplet tags from Staff 2 to Staff 1 to resolve incomplete measure warnings.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        modified = False
        
        # 1. Look for any clef with number="2"
        for clef in root.findall(".//clef[@number='2']"):
            sign = clef.find("sign")
            if sign is not None:
                if sign.text != "TAB":
                    sign.text = "TAB"
                    modified = True
            else:
                sign = ET.SubElement(clef, "sign")
                sign.text = "TAB"
                modified = True
                
            line = clef.find("line")
            if line is not None:
                if line.text != "5":
                    line.text = "5"
                    modified = True
            else:
                line = ET.SubElement(clef, "line")
                line.text = "5"
                modified = True
                
            coc = clef.find("clef-octave-change")
            if coc is not None:
                clef.remove(coc)
                modified = True

        # 2. Sync tuplets from Staff 2 to Staff 1 (resolving MuseScore's broken export tags on Staff 1)
        for measure in root.findall('.//measure'):
            notes = measure.findall('note')
            s1 = [n for n in notes if n.find('staff') is None or n.find('staff').text == '1']
            s2 = [n for n in notes if n.find('staff') is not None and n.find('staff').text == '2']
            
            if len(s1) != len(s2):
                continue
                
            for n1, n2 in zip(s1, s2):
                t2_elements = n2.findall('.//tuplet')
                notations_1 = n1.find('notations')
                t1_elements = n1.findall('.//tuplet')
                
                t1_types = [t.attrib.get('type') for t in t1_elements]
                t2_types = [t.attrib.get('type') for t in t2_elements]
                
                if t1_types != t2_types:
                    modified = True
                    if notations_1 is None:
                        notations_1 = ET.SubElement(n1, 'notations')
                    
                    # Remove all existing tuplets from notations_1
                    for t1 in list(notations_1.findall('tuplet')):
                        notations_1.remove(t1)
                        
                    # Copy all tuplets from n2 to notations_1
                    for t2 in t2_elements:
                        t2_copy = copy.deepcopy(t2)
                        notations_1.append(t2_copy)
                
        if modified:
            xml_data = ET.tostring(root, encoding='utf-8')
            xml_str = xml_data.decode('utf-8')
            
            # Combine XML declaration + doctype + XML content
            header = '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">\n'
            
            # If the generated xml_str starts with xml declaration, strip it
            if xml_str.startswith('<?xml'):
                idx = xml_str.find('?>')
                if idx != -1:
                    xml_str = xml_str[idx+2:].lstrip()
            
            with open(xml_path, 'w', encoding='utf-8') as f:
                f.write(header + xml_str)
            print(f" -> Post-processed XML (fixed TAB clef & synced tuplets): {os.path.basename(xml_path)}")
    except Exception as e:
        print(f" -> Warning: Failed to post-process XML for {xml_path}: {e}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert verified GP pieces to MusicXML with standard & tab staves.")
    parser.add_argument("--force", "-f", action="store_true", help="Force regeneration of XML files even if they already exist.")
    parser.add_argument("--piece", "-p", type=str, default=None, help="Process only a specific piece (substring match on Title).")
    args = parser.parse_args()

    musescore_cmd = detect_musescore()
    if not musescore_cmd:
        print("Error: MuseScore executable not found. Please install MuseScore.")
        return
        
    if not os.path.exists(XLSX_PATH):
        print(f"Error: {XLSX_PATH} not found.")
        return
        
    print(f"Loading {XLSX_PATH}...")
    df = pd.read_excel(XLSX_PATH)
    
    # Filter for validated dada_gp pieces
    mask = (df['validated'] == 1) & (df['source'] == 'dada_gp')
    dada_pieces = df[mask]
    
    if args.piece:
        dada_pieces = dada_pieces[dada_pieces['Title'].str.contains(args.piece, case=False, na=False)]
        print(f"Filtered by piece '{args.piece}': found {len(dada_pieces)} matching validated dada_gp pieces.")
    else:
        print(f"Found {len(dada_pieces)} validated dada_gp pieces to process.")
        
    success_count = 0
    fail_count = 0
    skipped_count = 0
    
    for idx, row in dada_pieces.iterrows():
        title = row['Title']
        gp_rel = row['gp_path']
        
        if not isinstance(gp_rel, str) or not gp_rel.strip() or gp_rel.strip().lower() == 'nan':
            print(f"Skipping '{title}': no gp_path specified in metadata.")
            continue
            
        # Determine source GP path in verified_pieces/dada/
        gp_basename = os.path.basename(gp_rel.strip())
        gp_local_path = os.path.join(BASE_DIR, "verified_pieces", "dada", gp_basename)
        
        # If it doesn't exist in verified_pieces/dada/, fallback to the original gp_rel (relative to root)
        if not os.path.exists(gp_local_path):
            gp_orig_path = os.path.join(BASE_DIR, gp_rel.strip())
            if os.path.exists(gp_orig_path):
                gp_local_path = gp_orig_path
            else:
                print(f"Warning: GP file for '{title}' not found at '{gp_local_path}' or '{gp_orig_path}'. Skipping.")
                fail_count += 1
                continue
                
        # Define output MusicXML path (in verified_pieces/dada/xml/)
        xml_basename = os.path.splitext(gp_basename)[0] + ".musicxml"
        xml_local_path = os.path.join(BASE_DIR, "verified_pieces", "dada", "xml", xml_basename)
        
        # Ensure target subdirectory exists
        os.makedirs(os.path.dirname(xml_local_path), exist_ok=True)
        
        relative_xml_path = os.path.join("verified_pieces", "dada", "xml", xml_basename)
        
        # Check if already converted
        if os.path.exists(xml_local_path) and not args.force:
            print(f"Skipping conversion for '{title}': MusicXML already exists in subfolder. Use --force to regenerate.")
            df.at[idx, 'xml_path'] = relative_xml_path
            df.at[idx, 'status'] = 'found'
            skipped_count += 1
            continue
            
        print(f"Converting '{title}' ({gp_basename}) to MusicXML...")
        
        # Build command: MuseScore command + --gp-linked + input + -o + output
        cmd = musescore_cmd + ['--gp-linked', gp_local_path, '-o', xml_local_path]
        
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res.returncode == 0 and os.path.exists(xml_local_path):
                # Post-process
                post_process_xml(xml_local_path)
                
                print(f" -> Successfully converted: {xml_basename}")
                df.at[idx, 'xml_path'] = relative_xml_path
                df.at[idx, 'status'] = 'found'
                success_count += 1
            else:
                print(f" -> Failed to convert '{title}'. Return code: {res.returncode}")
                print(f"Stdout:\n{res.stdout}\nStderr:\n{res.stderr}")
                fail_count += 1
        except Exception as e:
            print(f" -> Exception during conversion for '{title}': {e}")
            fail_count += 1
            
    # Save the updated spreadsheet
    if success_count > 0 or skipped_count > 0:
        print(f"Saving updated spreadsheet back to {XLSX_PATH}...")
        df.to_excel(XLSX_PATH, index=False)
        print("Spreadsheet updated successfully.")
        
    print(f"\nSummary:")
    print(f"  Converted: {success_count}")
    print(f"  Skipped (Already Exists): {skipped_count}")
    print(f"  Failed / Missing: {fail_count}")

if __name__ == "__main__":
    main()
