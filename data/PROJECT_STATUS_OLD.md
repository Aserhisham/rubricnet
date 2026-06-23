# Project Status Report: RubricNet Guitar Adaptation

This document summarizes the progress made in adapting the **RubricNet** architecture for classical guitar difficulty estimation.

## 1. Data Collection & Scraping
*   **Source**: [GuitarBurst](https://guitarburst.com)
*   **Action**: Developed a custom scraper ([extract_guitarburst.py](extract_guitarburst.py)) using `BeautifulSoup`.
*   **Result**: Successfully extracted **2,335 classical guitar pieces** with:
    *   Difficulty Labels (1-20)
    *   Reading Levels (1-10)
    *   Era, Composer, and Title
    *   Technical metadata (e.g., Max Position)
*   **Output**: Stored in [features/guitarburst_full.json](features/guitarburst_full.json).

## 2. Symbolic Dataset Integration (GAPS)
*   **Dataset**: GAPS (Guitar-Aligned Performance Scores).
*   **Action**: Downloaded and extracted ~400 MusicXML files.
*   **Cleanup**: Removed non-MusicXML formats (MIDI, GPX, Syncpoints) to focus on symbolic analysis.
*   **Discovery**: Identified that files were named using unique `scorehash` IDs (e.g., `mvswc.xml`) which map to human-readable titles in [gaps_v1_metadata.csv](gaps_v1/gaps_v1_metadata.csv).

## 3. Data Matching & Organization
*   **Redundancy & Safety**: Created [rename_gaps.py](rename_gaps.py) to generate a **readable** version of the dataset in `gaps_v1/readable_musicxml/` for easier manual browsing.
*   **Automated Matching**: Updated [match_gaps.py](match_gaps.py) with fuzzy string matching (normalized titles and composers).
*   **Current State**: Successfully matched **~238 pieces** from the labels to their corresponding symbolic MusicXML files. These serve as the verified training set.

## 4. Repository Management
*   **Environment**: Configured a Python virtual environment.
*   **Cleanup**: Updated [.gitignore](.gitignore) to exclude large dataset folders (`gaps_v1/`), checkpoints, and training logs, keeping the repository lean.

## Next Steps
1.  **Guitar Feature Extraction**: Adapt the existing RubricNet descriptor logic in `extractor/` to calculate guitar-specific technical difficulty (Barre chords, fretboard position shifts, string jumps).
2.  **Dataset Balancing**: Analyze the distribution of the 238 matched pieces across the 1-20 difficulty scale.
3.  **Model Training**: Begin fine-tuning the RubricNet architecture using the newly extracted guitar features.

---
*Last Updated: May 4, 2026*
