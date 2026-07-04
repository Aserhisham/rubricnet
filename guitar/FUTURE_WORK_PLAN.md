# Future Work Plan: Enhancing RubricNet Interpretability and Performance

This plan outlines theoretical explanations and concrete steps to address class boundary sensitivity, improve technique extraction (specifically barre chords), and introduce localized and context-aware features without violating RubricNet's strict architectural constraints.

---

## 1. Deeper Explanation of Class Boundary Sensitivity

In our current setup, difficulty levels 1–20 are grouped into 8 stratified bins:
* E.g., Class 3 contains Level 8, and Class 4 contains Levels 9–10.
* A piece at Level 8 and a piece at Level 9 are practically identical in difficulty.
* However, in standard classification, predicting Class 4 when the true class is 3 counts as an absolute error ($0\%$ accuracy).

While RubricNet uses a cumulative ordinal loss to penalize distance (an error of 1 class is penalized less than 4 classes), the target labels during training are still hard, discrete boundaries.

### Solution: Ordinal Label Smoothing (OLS)
Instead of representing target labels as a one-hot vector (e.g., `[0, 0, 0, 1, 0, 0, 0, 0]`), we can apply a Gaussian smoothing kernel centered at the true class:
$$\text{Target} = [0.01, 0.04, 0.20, 0.50, 0.20, 0.04, 0.01, 0.00]$$

* **Why it helps:** It tells the model that Class 3 is highly related to Class 2 and Class 4, reducing gradient instability for pieces that lie right on the border of a difficulty bin.
* **Impact on Interpretability:** It has **zero** impact on RubricNet's architecture; the monotonicity constraint is fully preserved.

---

## 2. Technique Extraction: Inferring Barre Chords Without Transcriptions

You are entirely correct that barre chords can be geometrically inferred directly from note configurations, even if the sheet music file does not label them.

### Proposed Barre Chord Inference Algorithm
We can scan every concurrent chord in the data and flag it as a barre chord if:
1. **Fret Coincidence:** The chord contains $\ge 3$ fretted notes (excluding open strings, fret 0).
2. **Same Fret Alignment:** At least 3 of those notes are played on the **same fret** across adjacent strings.
3. **Index Finger Position:** The lowest string in the chord playing that fret index acts as the "root" of the barre.

```python
def is_barre_chord(fretted_notes):
    # fretted_notes is a list of (string, fret) tuples, e.g. [(6, 3), (5, 5), (4, 5), (3, 4), (2, 3), (1, 3)] (G Major)
    frets = [fret for string, fret in fretted_notes if fret > 0]
    if len(frets) < 3:
        return False
    # Count occurrences of each fret
    from collections import Counter
    fret_counts = Counter(frets)
    # If any fret is held down on 3 or more strings, it's highly likely a barre chord
    for fret, count in fret_counts.items():
        if count >= 3:
            return True
    return False
```
By integrating this logic into `guitar/guitar_features.py`, we can bypass missing transcription metadata and extract a highly accurate `inferred_barre_ratio` descriptor.

---

## 3. Local Bottlenecks vs. Global Averaging

A piece's overall grade is usually dictated by its most difficult passages (bottlenecks), not its average state. A piece that is $95\%$ simple but has a $5\%$, high-speed arpeggio section will be graded as highly difficult. 

### Proposed Features: Sliding Window Aggregates
Instead of calculating the mean of a feature across the whole piece, we can implement:
* **Running Window Maxima:** Slide a window of 4 measures (or 10 seconds) across the piece. Calculate the density or stretch in each window, and take the **maximum** or **90th percentile** (`p90`) value.
* **Feature Comparison:**
  * `avg_position_shift` (Global average - context) vs. `p95_position_shift_window` (Peak bottleneck).
  * `avg_chord_stretch` vs. `max_chord_stretch_window`.

---

## 4. Context-Free vs. Context-Aware Physical Proxies

Physical difficulty is highly dependent on time and tempo. We can construct paired descriptors to directly compare context-free physical measures against context-aware equivalents.

| Context-Free Descriptor | Context-Aware Equivalent | Musicological Rationale |
| :--- | :--- | :--- |
| **`avg_chord_stretch`** <br>(fret distance) | **`stretch_velocity`** <br>($\text{stretch} \times \text{tempo\_bps}$) | A wide chord stretch is much harder to execute quickly than slowly. |
| **`avg_position_shift`** <br>(fret shift size) | **`position_shift_speed`** <br>($\frac{\text{shift\_distance}}{\text{duration\_seconds}}$) | Shifting 5 frets over a whole-note rest is trivial; doing it over a sixteenth note is a major bottleneck. |
| **`arpeggio_density`** <br>(notes per second) | **`polyphonic_arpeggio_intensity`** <br>($\text{density} \times \text{avg\_polyphony}$) | Playing fast single-line arpeggios is easier than maintaining multiple voices simultaneously. |

### Empirical Comparison Protocol
We can train two parallel RubricNet models to isolate the impact:
1. **Model A:** Trained on V2 Features + Context-Free Descriptors.
2. **Model B:** Trained on V2 Features + Context-Aware Descriptors.
3. Compare both the **balanced accuracy** and the **descriptor score ranges** to verify if context-awareness leads to stronger monotonic alignments.
