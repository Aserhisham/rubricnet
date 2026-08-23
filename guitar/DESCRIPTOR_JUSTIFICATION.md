# Per-Descriptor Justification: Leave-One-Out Ablation (rubricnet)

Each row drops exactly one descriptor from the 25 and retrains. 
**Positive = removing it made the model worse**, i.e. the descriptor earns its place.


Baseline (all 25): accuracy 0.3344 ± 0.042, balanced_accuracy 0.3132 ± 0.040, mae 1.0193 ± 0.066, kendall_tau_b 0.6401 ± 0.029


| Descriptor dropped | ΔAccuracy | ΔBal.Acc | ΔMAE | Δτ-b | Beyond fold noise? |
|---|---:|---:|---:|---:|:---:|
| `chord_ratio` | +0.0156 | +0.0170 | +0.0432 | +0.0099 | no |
| `avg_position_shift` | +0.0156 | +0.0107 | +0.0365 | -0.0041 | no |
| `avg_string_jump` | +0.0120 | +0.0153 | +0.0490 | +0.0170 | no |
| `tempo_bpm` | +0.0109 | +0.0105 | +0.0286 | +0.0090 | no |
| `shift_rate` | +0.0021 | +0.0017 | +0.0182 | +0.0022 | no |
| `arpeggio_density` | -0.0010 | +0.0009 | +0.0094 | -0.0030 | no |

Fold-to-fold std of the baseline is ±0.042 accuracy, so any 
single-descriptor delta smaller than that is inert rather than load-bearing. 
A descriptor being inert is not an argument for removing it -- the blanket-pruning 
experiment (thesis §Pruning Weak Descriptors) showed removing six inert descriptors 
at once *hurt* -- but it does mean its defense is 'harmless and mildly regularising', 
not 'individually predictive', and it should be described that way.
