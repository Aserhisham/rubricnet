import json
import collections

def get_stats():
    json_path = 'features/guitarburst_csv.json'

    # Load JSON
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {json_path} not found.")
        return

    unfound_counts = collections.Counter()
    found_difficulties = []

    for piece in data:
        composer = piece.get('Composer', 'Unknown')
        status = piece.get('status', 'not_found')
        
        if status == 'found':
            found_difficulties.append(piece.get('Difficulty', 0))
        else:
            unfound_counts[composer] += 1

    # Output 1: Top 10 composers by number of unfound pieces
    print("--- Top 10 Composers with Unfound Pieces ---")
    top_10_unfound = unfound_counts.most_common(10)
    for i, (composer, count) in enumerate(top_10_unfound, 1):
        print(f"{i}. {composer}: {count} pieces")

    # Output 2: Difficulty distribution for found pieces
    print("\n--- Difficulty Distribution for Found Pieces ---")
    if found_difficulties:
        diff_dist = collections.Counter(found_difficulties)
        sorted_diffs = sorted(diff_dist.keys())
        for diff in sorted_diffs:
            count = diff_dist[diff]
            print(f"Difficulty {diff:2}: {count:3} pieces")
    else:
        print("No pieces marked as 'found' yet.")

if __name__ == "__main__":
    get_stats()
