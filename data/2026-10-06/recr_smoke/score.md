# RECR score (SMOKE)

P0: PASS -- files 6 of 6; compile errors 0; inexact outputs 0; measurement mapping wrong 0; circuit counts ok True

| device | arm | infid (mean) | summed measure error | off-target | failed uses | 2q | median compile s |
|---|---|---|---|---|---|---|---|
| FakeAuckland | R51 | 0.01114 | 0.0379 | 0 | 0 | 15.40 | 0.075 |
| FakeAuckland | R51M | 0.01118 | 0.0359 | 0 | 0 | 15.40 | 0.060 |
| FakeAuckland | C14M | 0.01113 | 0.0356 | 0 | 0 | 15.40 | 0.051 |
| FakeAuckland | A9M | 0.01125 | 0.0379 | 0 | 0 | 15.40 | 0.127 |
| FakeAuckland | A11M | 0.01124 | 0.0357 | 0 | 0 | 15.40 | 0.151 |
| FakeAuckland | L3TM | 0.01139 | 0.0346 | 0 | 0 | 15.40 | 0.007 |
| FakeHanoiV2 | R51 | 0.01243 | 0.0939 | 0 | 0 | 15.40 | 0.105 |
| FakeHanoiV2 | R51M | 0.00946 | 0.0367 | 0 | 0 | 15.40 | 0.068 |
| FakeHanoiV2 | C14M | 0.00968 | 0.0364 | 0 | 0 | 15.40 | 0.088 |
| FakeHanoiV2 | A9M | 0.01247 | 0.0939 | 0 | 0 | 15.40 | 0.285 |
| FakeHanoiV2 | A11M | 0.00949 | 0.0365 | 0 | 0 | 15.40 | 0.275 |
| FakeHanoiV2 | L3TM | 0.00949 | 0.0367 | 0 | 0 | 15.40 | 0.010 |
| FakeTorino | R51 | 0.01491 | 0.1459 | 0 | 0 | 15.80 | 0.169 |
| FakeTorino | R51M | 0.00793 | 0.0618 | 0 | 0 | 15.80 | 0.161 |
| FakeTorino | C14M | 0.00747 | 0.0586 | 0 | 0 | 15.40 | 0.168 |
| FakeTorino | A9M | 0.01753 | 0.1426 | 0 | 0 | 15.40 | 0.587 |
| FakeTorino | A11M | 0.00734 | 0.0586 | 0 | 0 | 15.40 | 0.607 |
| FakeTorino | L3TM | 0.00747 | 0.0586 | 0 | 0 | 15.40 | 0.014 |
| FakeKingston | R51 | 0.01604 | 0.1650 | 0 | 0 | 15.60 | 0.160 |
| FakeKingston | R51M | 0.00269 | 0.0297 | 0 | 0 | 16.30 | 0.148 |
| FakeKingston | C14M | 0.00256 | 0.0284 | 0 | 0 | 15.60 | 0.128 |
| FakeKingston | A9M | 0.01284 | 0.1650 | 0 | 0 | 15.60 | 0.580 |
| FakeKingston | A11M | 0.00257 | 0.0296 | 0 | 0 | 15.60 | 0.648 |
| FakeKingston | L3TM | 0.00259 | 0.0284 | 0 | 0 | 15.60 | 0.013 |
| FakeBrussels | R51 | 0.01788 | 0.1426 | 0 | 0 | 16.00 | 0.128 |
| FakeBrussels | R51M | 0.01308 | 0.0785 | 0 | 0 | 16.00 | 0.129 |
| FakeBrussels | C14M | 0.01273 | 0.0729 | 0 | 0 | 15.90 | 0.103 |
| FakeBrussels | A9M | 0.01194 | 0.1193 | 57 | 0 | 16.00 | 0.434 |
| FakeBrussels | A11M | 0.01240 | 0.0729 | 0 | 0 | 15.90 | 0.358 |
| FakeBrussels | L3TM | 0.01298 | 0.0729 | 0 | 0 | 15.90 | 0.011 |
| FakeOsaka | R51 | 0.01280 | 0.1009 | 0 | 0 | 16.00 | 0.115 |
| FakeOsaka | R51M | 0.01089 | 0.0641 | 0 | 0 | 16.00 | 0.135 |
| FakeOsaka | C14M | 0.01048 | 0.0587 | 0 | 0 | 16.00 | 0.080 |
| FakeOsaka | A9M | 0.01056 | 0.1101 | 50 | 0 | 16.00 | 0.404 |
| FakeOsaka | A11M | 0.01045 | 0.0565 | 0 | 0 | 16.00 | 0.342 |
| FakeOsaka | L3TM | 0.01997 | 0.0359 | 0 | 0 | 15.90 | 0.010 |

## Predictions

- Q1 (compiling with the measurements moves the release's measured qubits to better readout (mean summed measure error R51M / R51 <= 0.85 on both cz devices; refuted >= 1.00 on either)): **CONFIRMED** -- {"FakeTorino": 0.42386211512717537, "FakeKingston": 0.1802575107296137}
- Q2 (c14's readout term never hurts (C14M / R51M infid <= 1.005 and measure error difference <= +0.0005 on every device; refuted infid ratio > 1.02 on any)): **REFUTED** -- {"infid": {"FakeAuckland": 0.9957080955210331, "FakeHanoiV2": 1.022570052188013, "FakeTorino": 0.942808880204697, "FakeKingston": 0.9513567623655468, "FakeBrussels": 0.9733069523544795, "FakeOsaka": 0.9623165566054995}, "meas_err": {"FakeAuckland": -0.00023000000000000104, "FakeHanoiV2": -0.00023999999999998328, "FakeTorino": -0.003222656250000004, "FakeKingston": -0.0013671874999999986, "FakeBrussels": -0.005541992187499997, "FakeOsaka": -0.005369999999999972}}
- Q3 (without measurements c14 is the release (identical in >= 99.9% on every device; refuted < 99% on any)): **CONFIRMED** -- {"FakeAuckland": 1.0, "FakeHanoiV2": 1.0, "FakeTorino": 1.0, "FakeKingston": 1.0, "FakeBrussels": 1.0, "FakeOsaka": 1.0}
- Q4 (a11 keeps every instruction on the target (A11 and A11M off-target 0 on every device; refuted any)): **CONFIRMED** -- {"FakeAuckland": 0, "FakeHanoiV2": 0, "FakeTorino": 0, "FakeKingston": 0, "FakeBrussels": 0, "FakeOsaka": 0}
- Q5 (the a9 defect reproduces on ecr devices (A9M off-target >= 1 on both ecr devices; refuted 0 on both)): **CONFIRMED** -- {"FakeBrussels": 57, "FakeOsaka": 50}
- Q6 (without measurements a11 is a9 on the cx and cz devices (identical in >= 99.9% on each; refuted < 99% on any)): **CONFIRMED** -- {"FakeAuckland": 1.0, "FakeHanoiV2": 1.0, "FakeTorino": 1.0, "FakeKingston": 1.0}
- Q7 (a11 counts readout (A11M / A9M infid <= 0.80 on both cz devices; refuted >= 1.00 on either)): **CONFIRMED** -- {"FakeTorino": 0.41859783489519053, "FakeKingston": 0.20044686456558827}
- Q8 (a11 is level with or ahead of Qiskit level 3 (A11M / L3TM <= 1.00 on >= 5 of 6 devices; refuted > 1.05 on any)): **CONFIRMED** -- {"FakeAuckland": 0.9871380787276949, "FakeHanoiV2": 0.9998178098944868, "FakeTorino": 0.9817929220056129, "FakeKingston": 0.9924994893354805, "FakeBrussels": 0.9552925727451266, "FakeOsaka": 0.5230859467579656}
- Q9 (the release candidate with measurements is level with or ahead of Qiskit level 3 (C14M / L3TM <= 1.00 on >= 5 of 6 devices; refuted > 1.05 on any)): **AMBIGUOUS** -- {"FakeAuckland": 0.977350632085634, "FakeHanoiV2": 1.0198835189624962, "FakeTorino": 1.0000271951320476, "FakeKingston": 0.9888121054230021, "FakeBrussels": 0.9810566973163266, "FakeOsaka": 0.5247263305856653}
- Q10 (both cost nothing (median compile time C14M / R51M on every device and A11M / A9M on the cx and cz devices <= 1.15; refuted > 1.50 on any)): **AMBIGUOUS** -- {"FakeAuckland": 0.8363636363636365, "FakeHanoiV2": 1.2943340691685064, "FakeTorino": 1.0485527544351072, "FakeKingston": 0.8693739424703892, "FakeBrussels": 0.8021722265321953, "FakeOsaka": 0.5907578558225508, "FakeAuckland (A11M/A9M)": 1.1874015748031495, "FakeHanoiV2 (A11M/A9M)": 0.9643609550561798, "FakeTorino (A11M/A9M)": 1.034761864190168, "FakeKingston (A11M/A9M)": 1.116491469929347}

Reported: A9M on the ecr devices is simulated with its off-target gates error-free; its infid there is not comparable.
Reported: check counters by device: {"FakeAuckland": {"exact": {"checked": 100, "refused_resynthesis": 0, "refused_floor": 0, "refused_level3": 0, "not_checkable": 0}, "direction": {"fixed": 0, "fallback": 0}, "l3t_check_a11": {"accepted": 20, "refused": 0, "unavailable": 0}}, "FakeHanoiV2": {"exact": {"checked": 100, "refused_resynthesis": 0, "refused_floor": 0, "refused_level3": 0, "not_checkable": 0}, "direction": {"fixed": 0, "fallback": 0}, "l3t_check_a11": {"accepted": 20, "refused": 0, "unavailable": 0}}, "FakeTorino": {"exact": {"checked": 100, "refused_resynthesis": 0, "refused_floor": 0, "refused_level3": 0, "not_checkable": 0}, "direction": {"fixed": 0, "fallback": 0}, "l3t_check_a11": {"accepted": 20, "refused": 0, "unavailable": 0}}, "FakeKingston": {"exact": {"checked": 100, "refused_resynthesis": 0, "refused_floor": 0, "refused_level3": 0, "not_checkable": 0}, "direction": {"fixed": 0, "fallback": 0}, "l3t_check_a11": {"accepted": 20, "refused": 0, "unavailable": 0}}, "FakeBrussels": {"exact": {"checked": 100, "refused_resynthesis": 0, "refused_floor": 0, "refused_level3": 0, "not_checkable": 0}, "direction": {"fixed": 0, "fallback": 0}, "l3t_check_a11": {"accepted": 20, "refused": 0, "unavailable": 0}}, "FakeOsaka": {"exact": {"checked": 100, "refused_resynthesis": 0, "refused_floor": 0, "refused_level3": 0, "not_checkable": 0}, "direction": {"fixed": 0, "fallback": 0}, "l3t_check_a11": {"accepted": 20, "refused": 0, "unavailable": 0}}}
