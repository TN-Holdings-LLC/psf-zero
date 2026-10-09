# H10-FULL

git head ad2f481, uncommitted tracked changes: none; 12 CPUs, Python 3.12.13

- L3 alone: 3.1 s, 113,250 two-qubit gates
- REL: 852.0 s, 113,292 two-qubit gates; decisions {"RESYNTH_STATS": {"selected_original": 2, "not_checked": 2}, "COMPARE_STATS": {"level3_refused": 1, "floor_refused": 1}, "EXACT_STATS": {"checked": 2, "refused_floor": 1, "refused_level3": 1, "not_checkable": 2}, "SKIP_STATS": {}}
- estimates and checks made: 9, 819 s for 3485 s of counted work (ratio 0.23)

| ID | prediction | verdict |
|---|---|---|
| H1 | finishes within 10,800 s | **CONFIRMED** |
| H2 | chooses Qiskit level 3's circuit | **REFUTED** |
| H3 | fewer two-qubit gates than C27's 113,292 | **REFUTED** |
| H4 | estimates' and checks' wall time 0.2-1.0 x their counted work | **CONFIRMED** |
| H5 | the same two-qubit gate count as level 3 alone | **REFUTED** |

| # | caller | function | counted work (s) | wall (s) |
|---|---|---|---|---|
| 2 | _select_resynthesis | excitation_cost | 433.3 | 120.869 |
| 3 | _select_resynthesis | excitation_cost | 437.0 | 108.413 |
| 5 | _select_resynthesis | excitation_cost | 433.3 | 93.852 |
| 6 | _select_resynthesis | excitation_cost | 437.0 | 100.962 |
| 8 | _choose_lazy | hybrid_cost | 351.6 | 146.013 |
| 9 | _choose_lazy | hybrid_cost | 351.6 | 153.099 |
| 10 | _choose_lazy | hybrid_cost | 349.9 | 93.63 |
| 11 | _choose_lazy | _implements | 372.3 | 0.886 |
| 12 | _choose_lazy | _implements | 318.9 | 0.854 |
