import pandas as pd
import json

def generate():
    with open('features/guitarburst_csv.json', 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    df['Difficulty'] = pd.to_numeric(df['Difficulty'], errors='coerce')
    
    # We'll focus on the user's requested 3 first: Bach, Paganini, Sor
    target_composers = ['Paganini', 'Bach', 'Sor']
    mask = df['Composer'].str.contains('|'.join(target_composers), case=False, na=False)
    
    shopping_list = df[mask & (df['status'] == 'not_found')]
    shopping_list = shopping_list[['Composer', 'Title', 'Difficulty']]
    
    shopping_list.to_csv('features/to_find_list.csv', index=False)
    print(f"Created shopping list with {len(shopping_list)} entries.")

if __name__ == "__main__":
    generate()
