import requests
from bs4 import BeautifulSoup
import json
import time
import re

def scrape_guitarburst():
    base_url = "https://guitarburst.com/pieces/advanced-search"
    # Basic search params for all pieces (1-20 difficulty, any position)
    params = {
        "min_difficulty": 1,
        "max_difficulty": 20,
        "min_reading": 1,
        "max_reading": 10,
        "position": 15,
        "sort": "title",
        "submit": "Search"
    }
    
    all_pieces = []
    page = 1
    
    print("Starting full dataset extraction...")
    
    while True:
        try:
            current_params = params.copy()
            current_params["page"] = page
            
            response = requests.get(base_url, params=current_params, timeout=30)
            if response.status_code != 200:
                print(f"Error on page {page}: {response.status_code}")
                break
                
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table')
            if not table:
                print("No table found. Reached end?")
                break
                
            rows = table.find_all('tr')[1:] # Skip header
            if not rows:
                break
                
            page_data = []
            for row in rows:
                tds = row.find_all('td')
                if len(tds) >= 6:
                    piece = {
                        "Title": tds[0].text.strip(),
                        "Composer": tds[1].text.strip(),
                        "Difficulty": int(tds[2].text.strip()) if tds[2].text.strip().isdigit() else tds[2].text.strip(),
                        "Reading": int(tds[3].text.strip()) if tds[3].text.strip().isdigit() else tds[3].text.strip(),
                        "Max Position": tds[4].text.strip(),
                        "Era": tds[5].text.strip()
                    }
                    page_data.append(piece)
            
            all_pieces.extend(page_data)
            print(f"Page {page} processed. Total pieces: {len(all_pieces)}")
            
            # Check for next page link
            next_page = soup.find('a', string=re.compile(r'Next|>', re.I))
            pagination_text = soup.get_text()
            # Simple check if current page is the last link in the pagination
            if f"page={page+1}" not in response.text:
                print("No next page link found.")
                break
                
            page += 1
            time.sleep(1) # Be responsible
            
        except Exception as e:
            print(f"Exception on page {page}: {e}")
            break

    output_file = "features/guitarburst_full.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_pieces, f, indent=4, ensure_ascii=False)
    
    print(f"Done! Extracted {len(all_pieces)} pieces to {output_file}")

if __name__ == "__main__":
    scrape_guitarburst()
