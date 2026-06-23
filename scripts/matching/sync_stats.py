import json
import pandas as pd

def sync():
    with open('features/guitarburst_full.json', 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    df.to_csv('features/found_pieces.csv', index=False)
    
    found = df[df['status'] == 'found']
    print(f"Total pieces: {len(df)}")
    print(f"Found pieces: {len(found)}")
    
    # Check top composers for found pieces
    if not found.empty:
        print("\nTop 10 Found Composers:")
        print(found['Composer'].value_counts().head(10))

if __name__ == "__main__":
    sync()
