# c27_probe (exploratory, nothing predicted)

git head 7299a87, uncommitted tracked changes: none; 11 jobs, 3 at a time

## 1. The parts of a refused call (on WB's output)

| test | instructions | count_ops s | listing s | touched s | refused excitation_cost s |
|---|---|---|---|---|---|
| test_feynman_transpile[hwb10.qasm] | 503690 | 0.008 | 2.917 | 1.163 | 3.027 |
| _hamlib_hamiltonians_transpile[ham_ham_JW-14] | 58031 | 0.0013 | 0.508 | 0.15 | 0.399 |
| transpile[ham_enc_gray_dvalues_4-4-4-4-4-4-4] | 20503 | 0.0003 | 0.172 | 0.051 | 0.172 |
| tonians_transpile[ham_enc_gray_dvalues_8-8-8] | 60157 | 0.0013 | 0.527 | 0.435 | 0.745 |
| _hamlib_hamiltonians_transpile[ham_ham_JW-10] | 13514 | 0.0001 | 0.138 | 0.036 | 0.055 |
| mlib_hamiltonians_transpile[ham_ham_parity10] | 13800 | 0.0003 | 0.172 | 0.041 | 0.089 |

## 2. Who asked for each call, and what was refused

### test_feynman_transpile[hwb10.qasm]

- WB: q2 113292, 100.13 s
  - _select_resynthesis -> excitation_cost: 433.28 s of work, REFUSED
  - _select_resynthesis -> excitation_cost: 436.97 s of work, REFUSED
  - _select_resynthesis -> excitation_cost: 433.28 s of work, REFUSED
  - _select_resynthesis -> excitation_cost: 436.97 s of work, REFUSED
  - _choose_lazy -> hybrid_cost: 351.61 s of work, REFUSED
  - <lambda> -> excitation_cost: 433.28 s of work, REFUSED
  - RESYNTH_STATS: {'not_estimable': 2}
  - COMPARE_STATS: {'not_estimable': 1}
  - WORK_STATS: {'refused': 5, 'refused_excitation_cost': 4, 'refused_hybrid_cost': 1}

### test_hamlib_hamiltonians_transpile[ham_ham_JW-14]

- NB: q2 10856, 129.45 s
  - _select_resynthesis -> excitation_cost: 13.23 s of work, made
  - _select_resynthesis -> excitation_cost: 12.34 s of work, made
  - _select_resynthesis -> _same_action: 25.54 s of work, made
  - _select_resynthesis -> excitation_cost: 13.23 s of work, made
  - _select_resynthesis -> excitation_cost: 12.34 s of work, made
  - _select_resynthesis -> _same_action: 25.54 s of work, made
  - _choose_lazy -> hybrid_cost: 10.22 s of work, made
  - <listcomp> -> hybrid_cost: 10.22 s of work, made
  - <listcomp> -> hybrid_cost: 9.01 s of work, made
  - _choose_lazy -> _implements: 13.71 s of work, made
  - _choose_lazy -> _implements: 8.46 s of work, made
  - RESYNTH_STATS: {'applied': 2, 'selected_resynthesised': 2}
  - COMPARE_STATS: {'level3': 1}
  - EXACT_STATS: {'checked': 4}
  - WORK_STATS: {'made': 11}
- WB: q2 12927, 17.11 s
  - _select_resynthesis -> excitation_cost: 13.23 s of work, REFUSED
  - _select_resynthesis -> excitation_cost: 12.34 s of work, REFUSED
  - _select_resynthesis -> excitation_cost: 13.23 s of work, REFUSED
  - _select_resynthesis -> excitation_cost: 12.34 s of work, REFUSED
  - _choose_lazy -> hybrid_cost: 10.90 s of work, REFUSED
  - <lambda> -> excitation_cost: 13.23 s of work, REFUSED
  - RESYNTH_STATS: {'not_estimable': 2}
  - COMPARE_STATS: {'not_estimable': 1}
  - WORK_STATS: {'refused': 5, 'refused_excitation_cost': 4, 'refused_hybrid_cost': 1}

### b_hamiltonians_transpile[ham_enc_gray_dvalues_4-4-4-4-4-4-4]

- NB: q2 3616, 84.93 s
  - _select_resynthesis -> excitation_cost: 8.25 s of work, made
  - _select_resynthesis -> excitation_cost: 7.40 s of work, made
  - _select_resynthesis -> _same_action: 15.95 s of work, made
  - _select_resynthesis -> excitation_cost: 8.25 s of work, made
  - _select_resynthesis -> excitation_cost: 7.40 s of work, made
  - _select_resynthesis -> _same_action: 15.95 s of work, made
  - _choose_lazy -> hybrid_cost: 6.06 s of work, made
  - <listcomp> -> hybrid_cost: 6.06 s of work, made
  - <listcomp> -> hybrid_cost: 3.03 s of work, made
  - _choose_lazy -> _implements: 8.42 s of work, made
  - _choose_lazy -> _implements: 3.07 s of work, made
  - RESYNTH_STATS: {'applied': 2, 'selected_resynthesised': 2}
  - COMPARE_STATS: {'level3': 1}
  - EXACT_STATS: {'checked': 4}
  - WORK_STATS: {'made': 11}
- WB: q2 4196, 15.1 s
  - _select_resynthesis -> excitation_cost: 8.25 s of work, made
  - _select_resynthesis -> excitation_cost: 7.40 s of work, REFUSED
  - _select_resynthesis -> excitation_cost: 8.25 s of work, REFUSED
  - _select_resynthesis -> excitation_cost: 7.40 s of work, REFUSED
  - _choose_lazy -> hybrid_cost: 6.73 s of work, REFUSED
  - <lambda> -> excitation_cost: 8.25 s of work, REFUSED
  - RESYNTH_STATS: {'not_estimable': 2}
  - COMPARE_STATS: {'not_estimable': 1}
  - WORK_STATS: {'made': 1, 'refused': 4, 'refused_excitation_cost': 3, 'refused_hybrid_cost': 1}

### st_hamlib_hamiltonians_transpile[ham_enc_gray_dvalues_8-8-8]

- NB: q2 11688, 73.61 s
  - _select_resynthesis -> excitation_cost: 4.30 s of work, made
  - _select_resynthesis -> excitation_cost: 4.18 s of work, made
  - _select_resynthesis -> _same_action: 10.55 s of work, made
  - _select_resynthesis -> excitation_cost: 4.30 s of work, made
  - _select_resynthesis -> excitation_cost: 4.18 s of work, made
  - _select_resynthesis -> _same_action: 10.55 s of work, made
  - _choose_lazy -> hybrid_cost: 3.65 s of work, made
  - <listcomp> -> hybrid_cost: 3.65 s of work, made
  - <listcomp> -> hybrid_cost: 3.13 s of work, made
  - _choose_lazy -> _implements: 5.68 s of work, made
  - _choose_lazy -> _implements: 3.76 s of work, made
  - RESYNTH_STATS: {'applied': 2, 'selected_resynthesised': 2}
  - COMPARE_STATS: {'level3': 1}
  - EXACT_STATS: {'checked': 4}
  - WORK_STATS: {'made': 11}
- WB: q2 13261, 26.26 s
  - _select_resynthesis -> excitation_cost: 4.30 s of work, made
  - _select_resynthesis -> excitation_cost: 4.18 s of work, made
  - _select_resynthesis -> _same_action: 10.55 s of work, REFUSED
  - _select_resynthesis -> excitation_cost: 4.30 s of work, REFUSED
  - _select_resynthesis -> excitation_cost: 4.18 s of work, REFUSED
  - _choose_lazy -> hybrid_cost: 3.70 s of work, REFUSED
  - <lambda> -> excitation_cost: 4.30 s of work, REFUSED
  - RESYNTH_STATS: {'kept_original': 1, 'not_estimable': 1}
  - COMPARE_STATS: {'not_estimable': 1}
  - EXACT_STATS: {'checked': 1, 'refused_resynthesis': 1, 'not_checkable': 1}
  - WORK_STATS: {'made': 2, 'refused': 4, 'refused_excitation_cost': 2, 'refused_hybrid_cost': 1, 'refused__same_action': 1}

### test_hamlib_hamiltonians_transpile[ham_ham_JW-10]

- NB: q2 2371, 9.74 s
  - _select_resynthesis -> excitation_cost: 0.81 s of work, made
  - _select_resynthesis -> excitation_cost: 0.83 s of work, made
  - _select_resynthesis -> _same_action: 2.04 s of work, made
  - _select_resynthesis -> excitation_cost: 0.81 s of work, made
  - _select_resynthesis -> excitation_cost: 0.83 s of work, made
  - _select_resynthesis -> _same_action: 2.04 s of work, made
  - _choose_lazy -> hybrid_cost: 0.72 s of work, made
  - <listcomp> -> hybrid_cost: 0.72 s of work, made
  - <listcomp> -> hybrid_cost: 0.63 s of work, made
  - _choose_lazy -> _implements: 1.09 s of work, made
  - _choose_lazy -> _implements: 0.74 s of work, made
  - RESYNTH_STATS: {'applied': 2, 'selected_resynthesised': 2}
  - COMPARE_STATS: {'level3': 1}
  - EXACT_STATS: {'checked': 4}
  - WORK_STATS: {'made': 11}
- WB: q2 2383, 6.35 s
  - _select_resynthesis -> excitation_cost: 0.81 s of work, made
  - _select_resynthesis -> excitation_cost: 0.83 s of work, made
  - _select_resynthesis -> _same_action: 2.04 s of work, made
  - _select_resynthesis -> excitation_cost: 0.81 s of work, made
  - _select_resynthesis -> excitation_cost: 0.83 s of work, made
  - _select_resynthesis -> _same_action: 2.04 s of work, made
  - _choose_lazy -> hybrid_cost: 0.72 s of work, made
  - <listcomp> -> hybrid_cost: 0.72 s of work, made
  - <listcomp> -> hybrid_cost: 0.63 s of work, made
  - _choose_lazy -> _implements: 1.09 s of work, REFUSED
  - _choose_lazy -> _implements: 0.74 s of work, REFUSED
  - <lambda> -> excitation_cost: 0.83 s of work, REFUSED
  - RESYNTH_STATS: {'applied': 2, 'selected_resynthesised': 2}
  - COMPARE_STATS: {'level3_refused': 1, 'floor_refused': 1}
  - EXACT_STATS: {'checked': 4, 'refused_floor': 1, 'refused_level3': 1, 'not_checkable': 2}
  - WORK_STATS: {'made': 9, 'refused': 2, 'refused__implements': 2}

### test_hamlib_hamiltonians_transpile[ham_ham_parity10]

- NB: q2 2367, 10.92 s
  - _select_resynthesis -> excitation_cost: 0.83 s of work, made
  - _select_resynthesis -> excitation_cost: 0.84 s of work, made
  - _select_resynthesis -> _same_action: 2.09 s of work, made
  - _select_resynthesis -> excitation_cost: 0.83 s of work, made
  - _select_resynthesis -> excitation_cost: 0.84 s of work, made
  - _select_resynthesis -> _same_action: 2.09 s of work, made
  - _choose_lazy -> hybrid_cost: 0.73 s of work, made
  - <listcomp> -> hybrid_cost: 0.73 s of work, made
  - <listcomp> -> hybrid_cost: 0.63 s of work, made
  - _choose_lazy -> _implements: 1.12 s of work, made
  - _choose_lazy -> _implements: 0.73 s of work, made
  - RESYNTH_STATS: {'applied': 2, 'selected_resynthesised': 2}
  - COMPARE_STATS: {'level3': 1}
  - EXACT_STATS: {'checked': 4}
  - WORK_STATS: {'made': 11}
- WB: q2 2404, 7.12 s
  - _select_resynthesis -> excitation_cost: 0.83 s of work, made
  - _select_resynthesis -> excitation_cost: 0.84 s of work, made
  - _select_resynthesis -> _same_action: 2.09 s of work, made
  - _select_resynthesis -> excitation_cost: 0.83 s of work, made
  - _select_resynthesis -> excitation_cost: 0.84 s of work, made
  - _select_resynthesis -> _same_action: 2.09 s of work, made
  - _choose_lazy -> hybrid_cost: 0.73 s of work, made
  - <listcomp> -> hybrid_cost: 0.73 s of work, made
  - <listcomp> -> hybrid_cost: 0.63 s of work, made
  - _choose_lazy -> _implements: 1.12 s of work, REFUSED
  - _choose_lazy -> _implements: 0.73 s of work, REFUSED
  - <lambda> -> excitation_cost: 0.84 s of work, REFUSED
  - RESYNTH_STATS: {'applied': 2, 'selected_resynthesised': 2}
  - COMPARE_STATS: {'level3_refused': 1, 'floor_refused': 1}
  - EXACT_STATS: {'checked': 4, 'refused_floor': 1, 'refused_level3': 1, 'not_checkable': 2}
  - WORK_STATS: {'made': 9, 'refused': 2, 'refused__implements': 2}

