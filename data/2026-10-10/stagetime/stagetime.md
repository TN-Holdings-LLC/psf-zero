# STAGETIME (exploratory, nothing predicted)

106 of 106 FakeTorino tests; c33's default call with backend=, timers around its stages.

## warm, small (QK2 warm < 0.5 s): 94 tests; c33 30.45 s summed, Qiskit level 2 7.51 s warm (median per test: c33 115.9 ms, QK2 warm 41.4 ms)

| stage | summed (s) | share of c33's time | median per test (ms) |
|---|---|---|---|
| _unroll_wide | 0.573 | 1.9% | 2.05 |
| _cancel_candidate | 0.532 | 1.7% | 2.75 |
| compile | 6.573 | 21.6% | 7.54 |
| smart_vf2_layout | 1.474 | 4.8% | 10.89 |
| generate_preset_pass_manager | 1.119 | 3.7% | 8.97 |
| pm.run | 18.110 | 59.5% | 60.58 |
| _failed_elements | 0.010 | 0.0% | 0.09 |
| prune_coupling_map | 0.143 | 0.5% | 1.30 |
| _uses_failed | 1.188 | 3.9% | 4.95 |
| other | 0.724 | 2.4% | 1.60 |
| _AbsorbRoutingSwaps (inside pm.run) | 8.979 | 29.5% | 14.49 |
| VF2PostLayout (inside pm.run) | 1.882 | 6.2% | 7.62 |

## warm, large (QK2 warm >= 0.5 s): 12 tests; c33 167.86 s summed, Qiskit level 2 46.06 s warm (median per test: c33 3731.1 ms, QK2 warm 651.6 ms)

| stage | summed (s) | share of c33's time | median per test (ms) |
|---|---|---|---|
| _unroll_wide | 1.714 | 1.0% | 62.19 |
| _cancel_candidate | 2.063 | 1.2% | 58.59 |
| compile | 26.486 | 15.8% | 991.40 |
| smart_vf2_layout | 0.247 | 0.1% | 17.50 |
| generate_preset_pass_manager | 0.316 | 0.2% | 16.99 |
| pm.run | 124.169 | 74.0% | 2260.36 |
| _failed_elements | 0.001 | 0.0% | 0.11 |
| prune_coupling_map | 0.020 | 0.0% | 1.49 |
| _uses_failed | 7.665 | 4.6% | 155.36 |
| other | 5.182 | 3.1% | 93.74 |
| _AbsorbRoutingSwaps (inside pm.run) | 50.551 | 30.1% | 1200.71 |
| VF2PostLayout (inside pm.run) | 0.499 | 0.3% | 32.37 |

## cold, small (QK2 warm < 0.5 s): 94 tests; c33 43.64 s summed, Qiskit level 2 7.51 s warm (median per test: c33 261.2 ms, QK2 warm 41.4 ms)

| stage | summed (s) | share of c33's time | median per test (ms) |
|---|---|---|---|
| _unroll_wide | 0.649 | 1.5% | 2.45 |
| _cancel_candidate | 0.762 | 1.7% | 4.96 |
| compile | 7.015 | 16.1% | 14.99 |
| smart_vf2_layout | 9.896 | 22.7% | 95.15 |
| generate_preset_pass_manager | 2.997 | 6.9% | 25.73 |
| pm.run | 20.048 | 45.9% | 82.11 |
| _failed_elements | 0.009 | 0.0% | 0.08 |
| prune_coupling_map | 0.150 | 0.3% | 1.34 |
| _uses_failed | 1.262 | 2.9% | 4.10 |
| other | 0.855 | 2.0% | 2.13 |
| _AbsorbRoutingSwaps (inside pm.run) | 10.337 | 23.7% | 20.44 |
| VF2PostLayout (inside pm.run) | 1.760 | 4.0% | 8.76 |

## cold, large (QK2 warm >= 0.5 s): 12 tests; c33 174.41 s summed, Qiskit level 2 46.06 s warm (median per test: c33 3858.7 ms, QK2 warm 651.6 ms)

| stage | summed (s) | share of c33's time | median per test (ms) |
|---|---|---|---|
| _unroll_wide | 1.824 | 1.0% | 69.67 |
| _cancel_candidate | 2.231 | 1.3% | 63.54 |
| compile | 25.368 | 14.5% | 947.05 |
| smart_vf2_layout | 1.370 | 0.8% | 104.54 |
| generate_preset_pass_manager | 0.664 | 0.4% | 32.58 |
| pm.run | 129.371 | 74.2% | 2349.64 |
| _failed_elements | 0.001 | 0.0% | 0.10 |
| prune_coupling_map | 0.018 | 0.0% | 1.46 |
| _uses_failed | 7.980 | 4.6% | 158.42 |
| other | 5.583 | 3.2% | 87.14 |
| _AbsorbRoutingSwaps (inside pm.run) | 54.197 | 31.1% | 1395.55 |
| VF2PostLayout (inside pm.run) | 0.505 | 0.3% | 30.34 |

