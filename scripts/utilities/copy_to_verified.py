#!/usr/bin/env python3
import os
import shutil
import sys
import glob
import pandas as pd

# Try to import readline for tab-completion
try:
    import readline
except ImportError:
    readline = None

def path_completer(text, state):
    """Simple tab-completer for paths, escaping spaces for readline."""
    text_unescaped = text.replace(r'\ ', ' ')
    matches = glob.glob(text_unescaped + '*')
    
    results = []
    for m in matches:
        m_escaped = m.replace(' ', r'\ ')
        if os.path.isdir(m):
            results.append(m_escaped + '/')
        else:
            results.append(m_escaped)
            
    try:
        return results[state]
    except IndexError:
        return None

def load_path_to_source_mapping():
    """Loads matching CSV metadata to map absolute file paths to their sources."""
    csv_path = 'features/found_pieces.csv'
    path_to_source = {}
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                source = str(row.get('source', '')).strip().lower()
                if not source or source == 'nan':
                    continue
                
                # Map standard columns
                for col in ['pdf_path', 'token_path', 'gp_path', 'xml_path', 'file_path']:
                    p_val = row.get(col)
                    if isinstance(p_val, str) and p_val and p_val != 'nan':
                        abs_p = os.path.abspath(p_val)
                        path_to_source[abs_p] = source
        except Exception as e:
            print(f"Warning: Could not parse metadata CSV {csv_path}: {e}")
    return path_to_source

def determine_subfolder(src_path, path_to_source):
    """Determines the target category subfolder for the copied item."""
    src_abs = os.path.abspath(src_path)
    
    # 1. Direct path lookup from CSV metadata
    source = path_to_source.get(src_abs)
    if source:
        source_map = {
            'dada_gp': 'dada',
            'dada': 'dada',
            'gaps': 'gaps',
            'pdf': 'pdf',
            'mutopia': 'mutopia'
        }
        return source_map.get(source, source)
        
    # 2. Heuristic check based on path keywords
    path_lower = src_abs.lower()
    if 'dada' in path_lower:
        return 'dada'
    if 'gaps' in path_lower:
        return 'gaps'
    if 'mutopia' in path_lower:
        return 'mutopia'
    if 'pdf' in path_lower:
        return 'pdf'
        
    # 3. Interactive fallback menu
    print(f"\nCould not auto-detect category for: {src_path}")
    print("Please select the target category subfolder:")
    print("  [1] dada")
    print("  [2] mutopia")
    print("  [3] pdf")
    print("  [4] gaps")
    print("  [5] Skip / Don't copy")
    
    while True:
        try:
            choice = input("Choice [1-5]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nOperation cancelled.")
            return None
            
        if choice == '1':
            return 'dada'
        elif choice == '2':
            return 'mutopia'
        elif choice == '3':
            return 'pdf'
        elif choice == '4':
            return 'gaps'
        elif choice == '5':
            return None
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    target_dir = os.path.join(base_dir, "verified_pieces")
    
    # Ensure root target directory exists
    if not os.path.exists(target_dir):
        try:
            os.makedirs(target_dir)
            print(f"Created target directory: {target_dir}")
        except Exception as e:
            print(f"Error creating target directory {target_dir}: {e}")
            sys.exit(1)
            
    # Setup readline completion if available
    if readline:
        readline.parse_and_bind("tab: complete")
        readline.set_completer(path_completer)
        delims = readline.get_completer_delims()
        for char in ['/', ' ']:
            if char in delims:
                delims = delims.replace(char, '')
        readline.set_completer_delims(delims)

    # Load source mappings
    print("Loading source mappings from metadata...")
    path_to_source = load_path_to_source_mapping()
    print(f"Loaded {len(path_to_source)} path-to-source mappings.\n")

    print("=" * 50)
    print("      INTERACTIVE VERIFIED PIECES COPY UTILITY")
    print("=" * 50)
    print(f"Target root directory: {target_dir}")
    print("Instructions:")
    print("  - Type the path of a file or directory inside the project.")
    print("  - Press [TAB] to auto-complete paths/filenames.")
    print("  - Items will be automatically categorized into subfolders:")
    print("    (dada, mutopia, pdf, gaps)")
    print("  - Press [Ctrl+C] or type 'exit' / 'quit' to stop.\n")

    try:
        while True:
            try:
                src_input = input("Source path to copy > ").strip()
            except EOFError:
                print("\nExiting...")
                break
                
            if not src_input:
                continue

            if src_input.lower() in ['exit', 'quit']:
                print("Exiting...")
                break

            # Handle escaped spaces from autocompletion
            src_unescaped = src_input.replace(r'\ ', ' ')
            src_path = os.path.abspath(src_unescaped)
            
            if not os.path.exists(src_path):
                print(f"Error: Path '{src_unescaped}' does not exist.")
                continue
            
            # Determine subfolder category
            subfolder = determine_subfolder(src_path, path_to_source)
            if not subfolder:
                print("Skipped copying.")
                print()
                continue
                
            category_dir = os.path.join(target_dir, subfolder)
            if not os.path.exists(category_dir):
                os.makedirs(category_dir)
                print(f"Created category directory: {category_dir}")
                
            dest_path = os.path.join(category_dir, os.path.basename(src_path))
            
            try:
                if os.path.isdir(src_path):
                    if os.path.exists(dest_path):
                        print(f"Warning: Destination '{dest_path}' already exists. Overwriting...")
                        shutil.rmtree(dest_path)
                    shutil.copytree(src_path, dest_path)
                    print(f"Successfully copied directory to: {dest_path}")
                else:
                    if os.path.exists(dest_path):
                        print(f"Warning: Destination file '{dest_path}' already exists. Overwriting...")
                    shutil.copy2(src_path, dest_path)
                    print(f"Successfully copied file to: {dest_path}")
            except Exception as e:
                print(f"Error copying '{src_unescaped}': {e}")
            print() # spacing
            
    except KeyboardInterrupt:
        print("\nExiting... (Ctrl+C pressed)")

if __name__ == "__main__":
    main()
