import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_visual_analysis():
    json_path = 'features/guitarburst_full.json'
    
    # 1. Load Data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    # Ensure numeric types
    df['Difficulty'] = pd.to_numeric(df['Difficulty'], errors='coerce')
    
    # 2. Setup Plotting
    sns.set_theme(style="whitegrid")
    
    # --- Visualization 1: Difficulty Distribution for Found Pieces ---
    found_df = df[df['status'] == 'found']
    if not found_df.empty:
        plt.figure(figsize=(12, 6))
        sns.countplot(data=found_df, x='Difficulty', palette='viridis')
        plt.title('Difficulty Distribution of Found Pieces')
        plt.xlabel('Difficulty Level')
        plt.ylabel('Count')
        plt.savefig('features/found_difficulty_dist.png')
        print(f"Saved: features/found_difficulty_dist.png")
    
    # --- Visualization 2: Top 10 Composers with Unfound Pieces ---
    unfound_df = df[df['status'] == 'not_found']
    top_10_unfound_composers = unfound_df['Composer'].value_counts().head(10).index.tolist()
    
    print("\nTop 10 Composers by number of unfound pieces:")
    print(unfound_df['Composer'].value_counts().head(10))

    # --- Visualization 3: Histograms for Top 10 Composers (Unfound Difficulty) ---
    # We'll create a faceted plot to see the histograms for each of the top 10 composers
    subset_unfound = unfound_df[unfound_df['Composer'].isin(top_10_unfound_composers)]
    
    g = sns.FacetGrid(subset_unfound, col="Composer", col_wrap=5, height=3, sharex=True, sharey=False)
    g.map(sns.histplot, "Difficulty", bins=range(1, 21), kde=False, color='salmon')
    g.set_titles("{col_name}")
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle('Difficulty Distribution for Top 10 Composers (Unfound Pieces)')
    
    plt.savefig('features/unfound_top10_difficulty.png')
    print(f"Saved: features/unfound_top10_difficulty.png")

    # Display basic stats in terminal as requested
    print("\n--- Statistics Summary ---")
    print(f"Total pieces: {len(df)}")
    print(f"Found: {len(found_df)}")
    print(f"Unfound: {len(unfound_df)}")

if __name__ == "__main__":
    generate_visual_analysis()
