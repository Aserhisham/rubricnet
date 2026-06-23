import os
import requests
from bs4 import BeautifulSoup
import time
import re

# Base URL for Mutopia searches
BASE_URL = "https://www.mutopiaproject.org/cgibin/make-table.cgi"
MIDI_DIR = "symbolic_data/mutopia_midi"
LY_DIR = "symbolic_data/mutopia_ly"

def download_mutopia_files(composer_id, instrument="Guitar"):
    if not os.path.exists(MIDI_DIR):
        os.makedirs(MIDI_DIR)
    if not os.path.exists(LY_DIR):
        os.makedirs(LY_DIR)
        
    start_at = 0
    total_midi = 0
    total_ly = 0
    
    while True:
        params = {
            "Composer": composer_id,
            "instrument": instrument,
            "startat": start_at
        }
        
        print(f"Scraping {composer_id} starting at {start_at}...")
        response = requests.get(BASE_URL, params=params)
        if response.status_code != 200:
            print(f"Failed to fetch page for {composer_id}")
            break
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all download links for .mid and .ly files
        links = soup.find_all('a', href=re.compile(r'\.(mid|ly)$'))
        
        if not links:
            print(f"No more links found for {composer_id}.")
            break
            
        for link in links:
            url = link['href']
            # If relative URL, make it absolute
            if not url.startswith('http'):
                url = f"https://www.mutopiaproject.org{url}"
                
            extension = url.split('.')[-1]
            filename = url.split('/')[-1]
            
            if extension == 'mid':
                target_dir = MIDI_DIR
            elif extension == 'ly':
                target_dir = LY_DIR
            else:
                continue

            filepath = os.path.join(target_dir, filename)
            
            if os.path.exists(filepath):
                continue
                
            print(f"Downloading {filename}...")
            try:
                res = requests.get(url)
                if res.status_code == 200:
                    with open(filepath, 'wb') as f:
                        f.write(res.content)
                    if extension == 'mid': total_midi += 1
                    else: total_ly += 1
                else:
                    print(f"Failed to download {url}")
            except Exception as e:
                print(f"Error downloading {url}: {e}")
                
            time.sleep(0.3) # Slightly faster but still polite
            
        # Check if there's a "Next 10" or similar link
        next_link = soup.find('a', string=re.compile(r'Next|next'))
        if next_link:
            start_at += 10
        else:
            break
            
    return total_midi, total_ly

if __name__ == "__main__":
    # List of composers to target
    composers = [
        "SorF", 
        "PaganiniN", 
        "GiulianiM", 
        "BachJS", 
        "TarregaF", 
        "CarcassiM", 
        "CarulliF"
    ]
    
    grand_midi = 0
    grand_ly = 0
    for composer in composers:
        m_count, l_count = download_mutopia_files(composer)
        print(f"[{composer}] MIDI: {m_count}, LilyPond: {l_count}")
        grand_midi += m_count
        grand_ly += l_count
        
    print(f"Done! Downloaded {grand_midi} MIDI and {grand_ly} LilyPond files.")
