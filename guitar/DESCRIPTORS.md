# Guitar Difficulty Descriptors — Final Set (25)


## Left hand (14 descriptors)

| Descriptor | Definition | ρ |
|---|---|---:|
| `fret_entropy` | Shannon entropy (bits) of the distribution of fret numbers over all notes. High when the piece uses many different frets. | +0.673 |
| `open_string_ratio` | Fraction of all notes played on open strings (fret 0). | −0.491 |
| `max_chord_stretch` | Largest fret span (max − min fretted fret) of any multi-note chord. | +0.442 |
| `max_position_shift_speed_beats` ♩ | Largest value of (position shift ÷ beats to next onset). | +0.439 |
| `max_position_shift` | Largest change in a chord's mean fret between consecutive events. | +0.405 |
| `avg_stretch_velocity_beats` ♩ | Mean of (chord stretch ÷ beats to next onset). | +0.397 |
| `max_avg_chord_stretch_window` ♩ | Maximum, over sliding 16-beat windows, of the mean chord stretch inside the window. | +0.387 |
| `p90_chord_stretch` | 90th percentile of chord fret spans. | +0.365 |
| `barre_ratio` | Fraction of events that are barre chords. A barre is detected when ≥3 notes share one fret across three consecutive strings. | +0.349 |
| `avg_chord_stretch` | Mean fret span over multi-note chords having at least one fretted note. | +0.335 |
| `std_position_shift` | Standard deviation of the mean-fret change between consecutive events. | +0.318 |
| `avg_position_shift_speed_beats` ♩ | Mean of (position shift ÷ beats to next onset). | +0.300 |
| `shift_rate` | Fraction of consecutive events whose mean fret changes by more than 2 frets. | +0.189 |
| `avg_position_shift` | Mean absolute change in a chord's mean fret between consecutive events. | +0.138 |

## Right hand (6 descriptors)

| Descriptor | Definition | ρ |
|---|---|---:|
| `string_entropy` | Shannon entropy (bits) of the distribution of string numbers over all notes. | +0.389 |
| `max_string_jump` | Largest minimal string distance between two consecutive events. | +0.300 |
| `polyphonic_arpeggio_intensity_beats` ♩ | Note density × mean polyphony. Crossing strings is demanding when it happens quickly *and* voices must be sustained. | +0.215 |
| `chord_ratio` | Fraction of events carrying 2 or more simultaneous notes. | −0.031 |
| `avg_string_jump` | Mean minimal string distance between consecutive events. | −0.098 |
| `arpeggio_density` | Among consecutive single-note pairs, the fraction that change string. | −0.132 |

## Global (5 descriptors)

| Descriptor | Definition | ρ |
|---|---|---:|
| `total_notes` | Total number of notes in the piece. | +0.691 |
| `log_total_notes` | log(1 + `total_notes`). | +0.691 |
| `max_note_density_window` ♩ | Maximum notes-per-beat over sliding 16-beat windows. | +0.422 |
| `repetition_ratio` | Fraction of consecutive events that exactly repeat the previous event (same string-and-fret set). | +0.223 |
| `tempo_bpm` | Notated tempo in beats per minute. | −0.110 |
