import pandas as pd
import os
import shutil
import re

def clean_filename(name):
    """Remove characters that are illegal in filenames."""
    if not isinstance(name, str):
        return "Unknown"
    # Replace slashes, colons, etc with dashes
    name = re.sub(r'[\\/*?:"<>|]', "-", name)
    # Remove non-ascii characters for maximum compatibility
    name = name.encode('ascii', 'ignore').decode('ascii')
    return name.strip()

def create_readable_folder():
    csv_path = 'datasets/gaps_v1/gaps_v1_metadata.csv'
    src_dir = 'datasets/gaps_v1/musicxml/'
    dest_dir = 'datasets/gaps_v1/readable_musicxml/'
    
    # Load metadata
    try:
        df = pd.read_csv(csv_path, encoding='latin1')
    except:
        df = pd.read_csv(csv_path, encoding='utf-8')
    
    # Create destination directory
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print(f"Created directory: {dest_dir}")
    
    print("Starting copy and rename process...")
    copied_count = 0
    
    for i, row in df.iterrows():
        scorehash = str(row['scorehash'])
        title = clean_filename(str(row['title']))
        # Extract composer name from 'composers' or 'ComposerName' if available
        composer = clean_filename(str(row.get('composer_name_normalized', 'Unknown')))
        
        src_file = os.path.join(src_dir, f"{scorehash}.xml")
        
        if os.path.exists(src_file):
            # Create a nice name: Composer - Title.xml
            new_name = f"{composer} - {title}.xml"
            
            # Handle potential duplicates by adding the scorehash
            dest_file = os.path.join(dest_dir, new_name)
            if os.path.exists(dest_file):
                new_name = f"{composer} - {title} ({scorehash}).xml"
                dest_file = os.path.join(dest_dir, new_name)
            
            shutil.copy2(src_file, dest_file)
            copied_count += 1
            if copied_count % 50 == 0:
                print(f"Copied {copied_count} files...")
        else:
            print(f"Warning: Source file {src_file} not found for {title}")

    print(f"\nDone! Copied {copied_count} files to {dest_dir}")
    print("You can now browse your pieces with human-readable names!")

if __name__ == "__main__":
    create_readable_folder()
