# Classical Guitar Difficulty Descriptors

This document defines the compact set of descriptors used in the RubricNet classical guitar difficulty estimation pipeline. Descriptors are organised into three primary dimensions: Left-Hand Technique, Right-Hand Technique, and Global Score Complexity.

To ensure compatibility across the dataset (where 91% of files are vector PDFs without timing information), temporal/speed features are formulated as event rates (e.g., changes per note) rather than real-world seconds or beats.

---

## 1. Left-Hand (LH) Descriptors

| Descriptor Name | Column / Variable | Pedagogical Meaning | Symbolic Computation / Formula |
| :--- | :--- | :--- | :--- |
| **Barre Ratio** | `barre_ratio` | Measures index-finger pressure & stamina requirements. | $\frac{\text{Count of Barre Chords}}{\text{Total Chords}}$<br><br>A chord is a "barre" if $\ge 3$ strings are pressed at the same fret value (excluding open strings, i.e., fret $> 0$). |
| **Average Chord Stretch** | `avg_chord_stretch` | Measures hand-span extension requirements across frets. | $\frac{1}{N_{chords}} \sum (\text{Max Fret}_i - \text{Min Fret}_i)$<br><br>Calculated for chords with $>1$ note. Min and Max are computed over **fretted notes only** (fret $> 0$) to avoid open strings inflating the span. |
| **Maximum Chord Stretch** | `max_chord_stretch` | Measures the absolute limit of finger extension required. | $\max (\text{Max Fret}_i - \text{Min Fret}_i)$ across the entire piece, where Min and Max are computed over **fretted notes only** (fret $> 0$). |
| **Average Position Shift** | `avg_position_shift` | Measures vertical hand movement along the neck. | $\text{mean}(|\text{avg\_fret}_t - \text{avg\_fret}_{t-1}|)$<br><br>Average fret position of chord $t$ is the mean of all non-open frets in that chord. |
| **Fret Change Rate** | `fret_change_rate` | Measures left-hand finger activity/placement changes per note event. | $\frac{\text{Count of Fret Changes}}{\text{Total Notes} - 1}$<br><br>Frequency at which consecutive note events change fret values. |

---

## 2. Right-Hand (RH) Descriptors

| Descriptor Name | Column / Variable | Pedagogical Meaning | Symbolic Computation / Formula |
| :--- | :--- | :--- | :--- |
| **Arpeggio Density** | `arpeggio_density` | Measures plucking finger coordination (p-i-m-a pattern shifts). | $\frac{\text{Count of String Changes}}{\text{Total Single-Note Transitions}}$<br><br>Percentage of consecutive single-note transitions where the string index changes. |
| **Average String Jump** | `avg_string_jump` | Measures plucking precision across non-adjacent strings. | $\text{mean}(|\text{string}_t - \text{string}_{t-1}|)$ for consecutive notes/chords (using minimum distance between strings if chords). |
| **Maximum String Jump** | `max_string_jump` | Measures the widest string jump required by the plucking hand. | $\max(|\text{string}_t - \text{string}_{t-1}|)$ |
| **Special Technique Ratio** | `special_technique_ratio` | Measures advanced ornamentation and articulation techniques. | $\frac{\text{Count of Ornamentations}}{\text{Total Notes}}$<br><br>Count of techniques (slurs, hammer-ons, pull-offs, slides, tremolo) normalised by total notes (imputed to 0 in PDFs). |

---

## 3. Global Score Complexity (GL) Descriptors

| Descriptor Name | Column / Variable | Pedagogical Meaning | Symbolic Computation / Formula |
| :--- | :--- | :--- | :--- |
| **Average Polyphony** | `avg_polyphony` | Measures vertical density (number of active voices). | $\frac{\text{Total Notes}}{\text{Total Chords/Events}}$<br><br>Average size of chords/events in the piece. |
| **Total Note Count** | `total_notes` | Measures the overall length and density complexity. | Total count of notes in the score. |
| **Tempo** | `tempo_bpm` | Measures speed of execution. | Tempo in BPM (parsed from XML/GP; defaulted to 80 for PDFs). |
