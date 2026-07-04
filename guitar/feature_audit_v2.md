# Feature Audit V2

This audit document evaluates the correlation of both v1 and new v2 features against the raw `Difficulty` target (values 1–20) on the full 716-piece dataset.

## Spearman Correlation Table

| Feature | Spearman $\rho$ | Status |
| --- | --- | --- |
| `total_notes` | +0.6798 | **Kept** |
| `log_total_notes` | +0.6680 | **Kept** |
| `fret_entropy` | +0.6483 | **Kept** |
| `high_position_ratio` | +0.5557 | **Kept** |
| `avg_fret` | +0.5455 | **Kept** |
| `p90_fret` | +0.5348 | **Kept** |
| `open_string_ratio` | -0.4846 | **Kept** |
| `max_chord_stretch` | +0.4410 | **Kept** |
| `max_position_shift` | +0.3918 | **Kept** |
| `string_entropy` | +0.3535 | **Kept** |
| `barre_ratio` | +0.3386 | **Kept** |
| `p90_chord_stretch` | +0.3374 | **Kept** |
| `max_string_jump` | +0.3208 | **Kept** |
| `avg_chord_stretch` | +0.3199 | **Kept** |
| `std_position_shift` | +0.3065 | **Kept** |
| `shift_rate` | +0.2657 | **Kept** |
| `avg_position_shift` | +0.2602 | **Kept** |
| `tempo_bpm` | -0.1590 | **Kept** |
| `repetition_ratio` | +0.1222 | **Kept** |
| `arpeggio_density` | -0.0980 | **Kept** |
| `avg_string_jump` | -0.0645 | **Kept** |
| `special_technique_ratio` | +0.0644 | Dropped (artifact) |
| `chord_ratio` | -0.0526 | **Kept** |
| `avg_string_span` | +0.0471 | Dropped ($|\rho| < 0.05$) |
| `avg_polyphony` | +0.0407 | **Kept** |
| `fret_change_rate` | -0.0402 | **Kept** |
| `unique_shape_rate` | -0.0092 | Dropped ($|\rho| < 0.05$) |

## Decisions & Rationale

- `special_technique_ratio` was unconditionally dropped because it is nonzero for only 29/716 pieces, making it a data-availability artifact rather than a meaningful descriptor.
- `avg_string_span` was dropped due to weak correlation ($|\rho| < 0.05$).
- `unique_shape_rate` was dropped due to weak correlation ($|\rho| < 0.05$).
