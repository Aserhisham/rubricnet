# Expert Meeting Sheet — Guitar Difficulty Features

*Prepared 2026-07-17. Context: thesis on interpretable guitar difficulty estimation from notation
(640 pieces, GuitarBurst 1–20 grades, 32 score-derived descriptors, additive rubric model).*

---

## Part 1 — Blind elicitation (do this FIRST, before showing anything below)

Goal: get his mental feature set before ours anchors him. Record every answer verbatim.

1. "When you look at a score/tab and judge how hard a piece is, **what do you look at, in order?**
   Talk me through your first 30 seconds with an unfamiliar piece."
2. "Suppose you had to write down a **checklist of measurable things** — countable or
   computable from the notation alone — that together determine difficulty. What would be on it?"
3. For each item he names: "Is that a *deal-breaker* (single-handedly sets the level) or a
   *contributor*? And what would you measure exactly — the average across the piece, the worst
   passage, or something else?"
4. "What makes two pieces with the **same notes** differ in difficulty?" (fingering choice,
   right-hand pattern, tempo, ...)
5. "What part of difficulty is **invisible in the notation** — things you only discover when
   you physically play it? Roughly what fraction of the total grade is that?"

Afterwards: map his list against ours (Part 2). Items he named that we lack are feature
candidates; items we have that he never mentioned are candidates to question in Part 3.

---

## Part 2 — Our 32 descriptors

ρ = Spearman correlation with the 1–20 difficulty grade on our 640 pieces (sign = direction).
Verdict column: have him mark each row — **K** keep / **X** irrelevant / **M** mismeasured (right idea, wrong formula) / **U** non-monotone ("more" isn't always harder).

### Left hand — position & stretch

| feature |   meaning | ρ | verdict |
|---|---|---|---|
| fret_entropy | how spread-out fret usage is across the whole piece (variety of frets visited) | +0.67 | |
| high_position_ratio | share of notes played at fret 7 or higher | +0.58 | |
| avg_fret | average fret number over all notes | +0.56 | |
| p90_fret | 90th-percentile fret (how high the piece *typically* peaks) | +0.53 | |
| open_string_ratio | share of notes on open strings | −0.49 | |
| max_chord_stretch | largest fret span (max−min fretted) inside any single chord | +0.44 | |
| max_position_shift | largest single jump of hand position (in frets) between consecutive events | +0.41 | |
| p90_chord_stretch | typical "big" chord span (90th percentile) | +0.37 | |
| avg_chord_stretch | average fret span within multi-note chords | +0.33 | |
| std_position_shift | how irregular the position movements are | +0.32 | |
| shift_rate | how often the hand moves more than 2 frets between events | +0.19 | |
| avg_position_shift | average size of position movement between consecutive events | +0.14 | |
| fret_change_rate | how often the fret set changes from one event to the next | −0.14 | |

### Left hand — timing-aware (need note durations)

| feature |   meaning | ρ | verdict |
|---|---|---|---|
| max_position_shift_speed_beats | fastest position shift: frets moved ÷ beats available | +0.44 | |
| avg_stretch_velocity_beats | chord stretch ÷ time available before the next event (stretch under time pressure), averaged | +0.40 | |
| p90_stretch_velocity_beats | same, but the typical "hard" case (90th percentile) | +0.39 | |
| max_avg_chord_stretch_window | worst 16-beat window by average chord stretch (hardest sustained-stretch passage) | +0.39 | |
| p95_position_shift_window | worst 16-beat window by near-maximum position shift | +0.37 | |
| avg_position_shift_speed_beats | average frets-per-beat position movement | +0.30 | |

### Right hand — strings & texture

| feature |   meaning | ρ | verdict |
|---|---|---|---|
| string_entropy | how evenly the 6 strings are used (vs. staying on few strings) | +0.39 | |
| max_string_jump | largest leap between strings from one event to the next | +0.30 | |
| polyphonic_arpeggio_intensity_beats | notes-per-beat × average simultaneous notes (busy polyphonic texture) | +0.21 | |
| arpeggio_density | among single-note→single-note moves, share that change string | −0.13 | |
| avg_string_jump | average string distance between consecutive events | −0.10 | |
| chord_ratio | share of events with 2+ simultaneous notes | −0.03 | |

### Global

| feature |   meaning | ρ | verdict |
|---|---|---|---|
| total_notes / log_total_notes | piece length in notes (raw and log-scaled) | +0.69 | |
| max_note_density_window | busiest 16-beat window, in notes per beat | +0.42 | |
| repetition_ratio | share of consecutive events that repeat the exact same shape | +0.22 | |
| avg_polyphony | average number of simultaneous notes | +0.06 | |
| tempo_bpm | notated tempo | −0.11 | |

---

## Part 3 — Targeted questions

### The dead-features puzzle
These *should* matter but measure near zero: `tempo_bpm` (−0.11), `arpeggio_density` (−0.13),
`fret_change_rate` (−0.14), `avg_string_jump` (−0.10), `chord_ratio` (−0.03), `avg_polyphony` (+0.06).

- [ ] "Is the **concept** irrelevant, or is the **formula** wrong? What would you measure instead?"
- [ ] Specifically tempo: "is tempo meaningless without knowing *what pattern* is played at that tempo?"
- [ ] Specifically arpeggio_density: "does arpeggio *quantity* matter at all, or only the right-hand *pattern type* (p-i-m-a, tremolo, rasgueado)?"

### Monotonicity (our model assumes "more = harder" for every feature)
- [ ] open_string_ratio: more open strings = easier, always? (campanella?)
- [ ] avg_fret / high_position: is high-position playing harder *per se*, or only with shifts?
- [ ] Any feature above he'd call U-shaped or "it depends"? (mark **U** in the table)

### Aggregation: is a piece as hard as its hardest bar?
- [ ] "When you grade a piece: single hardest passage, sustained average demand, or 'hardest
      30 seconds'? How much does a single brutal bar in an otherwise easy piece raise the grade?"
- [ ] "Does *recovery time* around the hard passage matter (endurance)?"

### Measurement validity
- [ ] Chord stretch is in **frets** — but frets narrow up the neck. Should stretch be
      position-weighted? Roughly how (4 frets at pos. II vs pos. IX)?
- [ ] Barre = 3+ same-fret notes on adjacent strings. What does this miss (partial/hinge barres)?
      Does barre **duration/endurance** matter more than barre count?
- [ ] Position shift = change in *average fret*. Is that how you'd define "a shift"?

### Missing features (cross-check against his Part-1 list)
Prompt only if he stalls: slurs/ornaments · tremolo & RH technique types · scordatura/tunings ·
LH finger independence (not just stretch) · sustained bass under moving melody · harmonics ·
rest/recovery structure · key signature / accidentals.
- [ ] For each: "big driver or minor? And is it visible in the notation?"

### Interactions
- [ ] "Name the 2–3 specific combinations where difficulty **multiplies** rather than adds."
      (stretch × speed? barre × shift? high position × no open strings?)
      *(Context: our 5 hand-crafted interaction features all failed statistically — collinear with parents.)*

### Label reality-check (if time)
- [ ] "On a 1–20 scale, how much would two experienced teachers disagree on the same piece?"
- [ ] "Do grades transfer across genres — is a level-10 classical study comparable to a
      level-10 pop fingerstyle arrangement?"
- [ ] "How much can difficulty differ between two *arrangements/editions* of the same piece?"
