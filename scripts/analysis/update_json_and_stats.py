import pandas as pd
import json
import os
from collections import Counter

def update_json_and_get_stats():
    csv_path = 'features/found_pieces.csv'
    json_path = 'features/guitarburst_csv.json'

    # Load CSV
    df_found = pd.read_csv(csv_path)
    # Create a set of (Title, Composer) for quick lookup
    # Normalize to avoid mismatch due to quoting or case
    found_set = set()
    for _, row in df_found.iterrows():
        title = str(row['Title']).strip().lower()
        composer = str(row['Composer']).strip().lower()
        found_set.add((title, composer))

    # Load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)

    updated_count = 0
    found_pieces = []
    unfound_counts = Counter()

    for piece in data:
        title = str(piece.get('Title', '')).strip().lower()
        composer = str(piece.get('Composer', '')).strip().lower()
        
        if (title, composer) in found_set:
            piece['status'] = 'found'
            found_pieces.append(piece)
        else:
            piece['status'] = 'not_found'
            unfound_counts[piece.get('Composer', 'Unknown')] += 1
        
    # Save JSON
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Updated {json_path}. Total pieces: {len(data)}")

    # Stats: Top 10 composers by unfound pieces
    print("\nTop 10 Composers by number of unfound pieces:")
    for composer, count in unfound_counts.most_common(10):
        print(f"{composer}: {count}")

    # Stats: Distribution by difficulty for found pieces
    if found_pieces:
        difficulties = [p.get('Difficulty', 0) for p in found_pieces]
        diff_dist = Counter(difficulties)
        print("\nDifficulty Distribution for Found Pieces:")
        for diff in sorted(diff_dist.keys()):
            print(f"Difficulty {diff}: {diff_dist[diff]}")
    else:
        print("\nNo found pieces to show difficulty distribution.")

if __name__ == "__main__":
    update_json_and_get_stats()
