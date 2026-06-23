# Project Status: Classical Guitar RubricNet

## 📌 Overview
Transitioning RubricNet (interpretable difficulty estimation) from piano to classical guitar using the GuitarBurst labels matched against GAPS and Dada-GP symbolic datasets.

## ✅ Completed So Far
- **Data Scraping**: 2,335 pieces scraped from [GuitarBurst](http://guitarburst.com/) with difficulty grades (1-20).
- **Symbolic Integration**: Parsed metadata for **GAPS** (~400 MusicXMLs) and **Dada-GP** (~104k Guitar Pro files).
- **Matching Pipeline**:
    - Completed **Standardization**: Canonical composer names mapping (40+ variations resolved).
    - **Dataset Expansion**: Increased verified piece count from 488 to **562** pieces across MIDI, MusicXML, and Guitar Pro formats (after rigorous vetting for false-positive name/opus conflicts).
    - Verified Sources: GAPS, DadaGP, Mutopia, and local PDF collections.
- **Refactoring**: Cleaned workspace. Moved utility scripts to \`scripts/\`, guitar logic to \`guitar/\`, and datasets to \`datasets/\`.

## 🛠 Workspace Structure
- \`rubricnet/\` & \`extractor/\`: Core model/feature logic (Untouchable).
- \`guitar/\`: New guitar-specific logic (\`guitar_features.py\`, \`train_guitar_rubricnet.py\`).
- \`datasets/\`: Symbolic files (GAPS, Dada-GP, Mutopia).
- \`features/\`: Master labels (\`guitarburst_full.json\`) and actual extracted feature data files.
- \`verified_pieces/\`: Validated files copy destination.
- \`notebooks/\`: Jupyter Notebooks for dataset and verified piece analysis.
- \`scripts/\`: Matching, scraping, utilities, and analysis scripts.
  - \`scripts/matching/\`: Symbolic dataset matching and syncing utilities.
  - \`scripts/scraping/\`: Scrapers (Mutopia, GuitarBurst).
  - \`scripts/utilities/\`: Exporting, copying, and cleaning utilities.
  - \`scripts/analysis/\`: Feature analytics and visual plotter scripts.

## 🔜 Next Steps
1. **Vetting**: Review \`features/loose_matches_to_vet.csv\`. Identify correct matches for the ~1,900 "not_found" pieces.
2. **Syncing**: Run \`python3 scripts/matching/sync_datasets.py\` once vetting is done to update the master JSON.
3. **Feature Engineering**: Adapt \`guitar/guitar_features.py\` to calculate barre chords, fretboard shifts, and string jumps using \`pyguitarpro\` (for GP) or \`music21\` (for XML).
4. **Fine-tuning**: Train the RubricNet model using the newly extracted guitar feature vectors.

## ⚠️ Important Details
- **Environment**: Use \`.venv\`.
- **Dada-GP Metadata**: Located at \`datasets/DadaGP-v1.1/_DadaGP_all_metadata.json\`. It is huge; use indexed search logic implemented in scripts.
- **Paths**: Many scripts use relative paths starting from the root. Always run scripts from the project root.
