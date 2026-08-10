# Descriptor Provenance — Where Each of the 25 Descriptors Came From

This document answers a question the thesis currently answers only in prose: for
every descriptor in the final `V5-pruned-collinear2` set, *what is its source* —
published literature, the domain expert, a standard mathematical construction, or
original design for this thesis — and *where is it implemented*.

The thesis's descriptor chapter (`AIM-thesis/chapters/04-method.tex`, §Descriptor
Engineering) narrates the V1→V5 progression and cites Vélez Vásquez et al. for the
general playability-descriptor idea and Shannon for entropy, but it does not attach
a source to descriptors individually. That is the gap this table closes.

## Origin codes

| Code | Meaning |
|---|---|
| **L** | **Literature analogue.** A descriptor measuring the same construct exists in a published playability/difficulty feature set, though defined for a different repertoire and re-derived here for classical solo guitar. |
| **S** | **Standard construction.** A textbook statistical or information-theoretic measure (Shannon entropy, percentile, standard deviation) applied to a guitar-specific quantity. The measure is standard; the quantity it is applied to is not. |
| **E** | **Expert-elicited.** Named by the domain-expert reviewer, either unprompted in blind elicitation or confirmed on the structured keep/rule-out pass (`guitar/EXPERT_MEETING.md`, session 2026-07-17). |
| **O** | **Original to this thesis.** Designed here, motivated by a correlation audit or by the additive architecture's inability to learn interactions. No published counterpart known. |

Codes combine: a descriptor can be both a literature analogue and expert-confirmed.

## The honest headline

**No descriptor in this set is imported unchanged from a published feature set,
because no published symbolic feature set for classical solo guitar difficulty
exists.** The closest prior work, Vélez Vásquez et al. (2023), targets strummed
rhythm-guitar chord charts in popular music; five of its criteria have genuine
counterparts here (barre usage, finger displacement, chord-shape uncommonness,
right-hand complexity, rhythmic regularity), and those counterparts are marked **L**
below. The remainder — in particular the fretboard-coverage entropies and the entire
rhythm-normalised family — were designed for this thesis. That is a contribution
claim, not a gap, but it must be *stated* per descriptor rather than left implicit.

---

## Left hand (14 descriptors)

| Descriptor | ρ | Origin | Source / justification | Implementation |
|---|---:|:---:|---|---|
| `fret_entropy` | +0.673 | **S**+**O** | Shannon entropy (Shannon 1948) of the fret-usage distribution. The *measure* is standard; applying it to fretboard coverage as a difficulty proxy is original here. Became the single most important left-hand descriptor and absorbed three collinear position descriptors during pruning. | `guitar_features.py:325` |
| `open_string_ratio` | −0.491 | **O** | Original. Encodes the pedagogical fact that open strings cost the left hand nothing; included specifically to test whether the model would learn the expected *negative* sign, which it did. | `guitar_features.py:284` |
| `max_chord_stretch` | +0.442 | **L**+**E** | Chord-shape difficulty / finger displacement (Vásquez et al. 2023). Expert named chord "tightness" unprompted. | `guitar_features.py:790` |
| `max_position_shift_speed_beats` ♩ | +0.439 | **O** | Original. A shift divided by the beats available to execute it — a hand-encoded demand×time interaction the additive model cannot learn. Motivated by the V2 audit finding raw shift distance nearly uninformative (ρ +0.13). | `guitar_features.py:431` |
| `max_position_shift` | +0.405 | **L** | Finger displacement between chords (Vásquez et al. 2023), re-derived as change in mean fret between consecutive events. | `guitar_features.py:315` |
| `avg_stretch_velocity_beats` ♩ | +0.397 | **O** | Original; stretch counterpart of the shift-speed descriptor above. | `calculate_descriptors_v3` |
| `max_avg_chord_stretch_window` ♩ | +0.387 | **O**+**E** | Original windowed "worst-passage" aggregation. Directly matches the expert's described procedure: distil the piece into parts, judge the hardest part — not the average. | `calculate_descriptors_v3` |
| `p90_chord_stretch` | +0.365 | **L**+**S** | Vásquez chord-shape analogue under a percentile aggregation. | `guitar_features.py:299` |
| `barre_ratio` | +0.349 | **L**+**E** | Barre usage is an explicit Vásquez et al. criterion, and the expert named it unprompted. The most directly literature-grounded descriptor in the set. | `guitar_features.py:788` |
| `avg_chord_stretch` | +0.335 | **L**+**E** | As `max_chord_stretch`, mean aggregation. | `guitar_features.py:789` |
| `std_position_shift` | +0.318 | **L**+**S** | Displacement (Vásquez) under a dispersion aggregation. | `guitar_features.py:316` |
| `avg_position_shift_speed_beats` ♩ | +0.300 | **O** | Original; mean counterpart of the max shift-speed descriptor. | `guitar_features.py:430` |
| `shift_rate` | +0.189 | **O** | Original. Fraction of transitions moving more than 2 frets — a thresholded reading of displacement rather than a magnitude. | `guitar_features.py:314` |
| `avg_position_shift` | +0.138 | **L** | The plain Vásquez displacement measure. Retained despite weak individual ρ; §Context-Free vs Context-Aware shows why (it is the context-free parent whose rhythm-normalised children carry the signal). | `guitar_features.py:793` |

## Right hand (6 descriptors)

| Descriptor | ρ | Origin | Source / justification | Implementation |
|---|---:|:---:|---|---|
| `string_entropy` | +0.389 | **S**+**O** | Shannon entropy of the string-usage distribution. As `fret_entropy`: standard measure, original application. | `guitar_features.py:332` |
| `max_string_jump` | +0.300 | **O** | Original. Right-hand analogue of position shift. Vásquez's right-hand criterion is *strumming* complexity, which has no counterpart in classical fingerstyle, so this is not a literature analogue. | `guitar_features.py:819` |
| `polyphonic_arpeggio_intensity_beats` ♩ | +0.215 | **O** | Original. Note density × mean polyphony; designed after the audit showed raw `arpeggio_density` correlates *negatively* with difficulty (easy arpeggio studies), and recovers the intended signal only when scaled by rate and polyphony. | `calculate_descriptors_v3` |
| `chord_ratio` | −0.031 | **O** | Original, generic texture measure. Survived pruning despite near-zero univariate ρ (14th by RF importance, 23rd by Spearman) — a documented case of multivariate usefulness diverging from marginal correlation. | `guitar_features.py:303` |
| `avg_string_jump` | −0.098 | **O** | Original; mean counterpart of `max_string_jump`. | `guitar_features.py:818` |
| `arpeggio_density` | −0.132 | **O** | Original. Retained deliberately as the context-free parent of `polyphonic_arpeggio_intensity_beats`; its negative sign is a genuine finding about easy arpeggio studies, not a defect. | `guitar_features.py:817` |

## Global (5 descriptors)

| Descriptor | ρ | Origin | Source / justification | Implementation |
|---|---:|:---:|---|---|
| `total_notes` | +0.691 | **O** | Score length. Trivially defined, but note that it and its log are the two strongest single descriptors in the set — which is exactly why the single-descriptor control baseline in `guitar/tuned_baselines.py` exists. | `guitar_features.py:825` |
| `log_total_notes` | +0.691 | **S** | Log-compression of the above, standard treatment of a long-tailed count. | `guitar_features.py:278` |
| `max_note_density_window` ♩ | +0.422 | **O**+**E** | Original windowed peak note rate. Note density was the **first** item in the expert's unprompted attention order. | `calculate_descriptors_v3` |
| `repetition_ratio` | +0.223 | **O**+**E** | Original. "Repetition or lack thereof" was the fifth item in the expert's checklist. | `guitar_features.py:339` |
| `tempo_bpm` | −0.110 | **E** | Tempo was the expert's second-named attention item, which is why it survives despite weak ρ and missing values for 14 pieces. Its weakness is itself a finding: notated tempo is unreliable in this corpus. | `guitar_features.py:215` |

♩ = requires recovered rhythm; median-imputed within the training fold when absent.

---

## Provenance summary

| Origin | Count | Descriptors |
|---|---:|---|
| Literature analogue (**L**) | 6 | chord stretch family (3), position shift family (3) |
| Standard construction (**S**) | 5 | both entropies, `log_total_notes`, `p90_chord_stretch`, `std_position_shift` |
| Expert-elicited (**E**) | 7 | `barre_ratio`, chord stretch (avg/max), `max_avg_chord_stretch_window`, `max_note_density_window`, `repetition_ratio`, `tempo_bpm` |
| Original to this thesis (**O**) | 17 | all rhythm-normalised descriptors, both entropies' application, `open_string_ratio`, `shift_rate`, all right-hand descriptors except none |

(Codes overlap, so the column sums exceed 25.)

**Reading:** roughly a quarter of the set has a defensible published analogue, a fifth
is expert-endorsed, and the majority — including every rhythm-aware descriptor, which
is where the V2→V3 performance gain came from — is original. The two strongest
descriptors overall (`fret_entropy`, `total_notes`) are original applications.

## Known weaknesses in this provenance story

1. **Two descriptors were never independently justified**: `avg_string_jump` and
   `chord_ratio` have near-zero univariate correlation, no literature analogue, and no
   expert endorsement. They survive on the empirical grounds that removing weak
   descriptors *hurt* (§Pruning Weak Descriptors). That is a real result, but it means
   their presence rests on an ablation rather than on a reason.
2. **The right-hand dimension is the least grounded.** It is the weakest dimension in
   the ablation (0.197 accuracy alone), has no literature analogue for classical
   fingerstyle, and received the least expert attention. This is the most attackable
   part of the descriptor set and should be acknowledged as such rather than defended.
3. **No descriptor is drawn from a general-purpose symbolic music feature library.**
   Whether such a library would have supplied useful descriptors for free is an open
   question the thesis does not currently address — see `guitar/JSYMBOLIC_STUDY.md`.
