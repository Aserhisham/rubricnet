import requests
from bs4 import BeautifulSoup
import os
import time
import re

def scrape_mutopia_guitar():
    # Target composers and their Mutopia URL patterns
    # Mutopia pieces for Guitar
    base_url = "https://www.mutopiaproject.org/cgibin/make-table.cgi?Instrument=Guitar"
    
    # We can also filter by composer directly if needed, but listing all guitar works is safer
    # to catch more pieces for our 2335 labels.
    
    download_dir = "symbolic_data/mutopia_guitar"
    os.makedirs(download_dir, exist_ok=True)
    
    print(f"Starting Mutopia Guitar scrape into {download_dir}...")
    
    current_page_url = base_url
    page_num = 1
    
    while current_page_url:
        print(f"Processing Mutopia Page {page_num}...")
        try:
            response = requests.get(current_page_url, timeout=30)
            if response.status_code != 200:
                print(f"Failed to fetch page {page_num}")
                break
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all rows in the results table
            # Mutopia structure: table rows usually contain "MusicXML" download links
            links = soup.find_all('a', href=re.compile(r'\.xml\.zip|\.mxl'))
            
            for link in links:
                file_url = link.get('href')
                if not file_url.startswith('http'):
                    file_url = "https://www.mutopiaproject.org" + file_url
                
                # Filter for MusicXML
                if '.xml.zip' in file_url or '.mxl' in file_url:
                    filename = file_url.split('/')[-1]
                    filepath = os.path.join(download_dir, filename)
                    
                    if not os.path.exists(filepath):
                        print(f"Downloading {filename}...")
                        file_res = requests.get(file_url, timeout=30)
                        with open(filepath, 'wb') as f:
                            f.write(file_res.content)
                        time.sleep(0.5) # Brief pause
                    
            # Check for next page
            next_link = soup.find('a', string=re.compile(r'Next', re.I))
            if next_link:
                path = next_link.get('href')
                if not path.startswith('/'):
                    path = '/' + path
                if not path.startswith('/cgibin/'):
                    path = '/cgibin/' + path if 'make-table.cgi' in path else path
                
                current_page_url = "https://www.mutopiaproject.org" + path
                page_num += 1
            else:
                current_page_url = None
                
        except Exception as e:
            print(f"Error on page {page_num}: {e}")
            break

    print("Mutopia download complete.")

if __name__ == "__main__":
    scrape_mutopia_guitar()
