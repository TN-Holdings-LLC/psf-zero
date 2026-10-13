# HW3 -- fixed before any real-device step (local; committed before the run, which waits for the owner's go-ahead)

Fixed: 2026-10-13 05:56:13 CEST (Europe/Zurich).

**Seen before this file:**
- HW1 (Addendum 455) and NS1 (Addendum 462): on ibm_kingston, with dynamical decoupling and twirling off, the two
  circuits much shorter than Qiskit's won by 1.20, where a Markovian model of the reported errors and T1/T2 gives
  0.97-0.99.
- An exploratory ASAP variant of NS1, partly run (tfim_n12 still 0.98).
- The workplace's a14 estimate (an idle term that charges dephasing up to the measurement), which ranks HW1's pairs
  better (4 of 6, +0.77).
- HW3's rehearsal on FakeFez (noiseless).
- Not HW2's results (HW2 runs before HW3; if HW3 runs after HW2's results are read, that is stated with the results,
  and these predictions stay as fixed here).

## 1. Why

The hypothesis: without decoupling, a waiting qubit loses coherence faster than the reported T2 says. IBM's reported
T2 is an echo value; free evolution suffers quasi-static noise (T2*) and residual ZZ coupling, both refocused by
decoupling. If so, a shorter schedule wins on hardware for a reason that decoupling removes. The test runs the same
compiled circuits with and without decoupling, on the same schedule, in one job.

## 2. Design (`hw3_real.py`, sha256 in `SHA256SUMS`)

**Device rule:** as HW2: of ibm_fez and ibm_marrakesh, the operational one with fewer pending jobs; neither, nothing
runs. The device's Target at `prepare` is used.

**Pairs** (a shorter and a longer circuit of the same test; which is shorter is read from the durations computed at
`prepare`):

| test | arms |
|---|---|
| tfimP_n12 | c48 with item 83 on (S1) and off (S0): the same gate blocks |
| tfim_n12 | c45 (`level3_candidate=False`) and Qiskit level 3 (HW1's 1.20 win of the shorter) |
| qaoa_n20_p2 | the same two arms (HW1's other 1.20 win) |

**Variants of every compiled circuit:**
- plain: as compiled, as in HW1 and HW2;
- pad: as late as possible on the Target, idle windows filled with delays;
- dd: the same schedule, the circuit's active qubits' idle windows filled with X-X decoupling (Qiskit's
  PadDynamicalDecoupling).

Every circuit is checked (ISA, no failed element, the ideal distribution to 1e-6 by noiseless simulation); a pair
goes only if all six of its circuits pass.

**One job:** 18 circuits, each twice (36 submissions, the second repeat in reverse order), 6,000 shots, the runtime's own decoupling
and twirling off. The rough estimate is 144 s; HW1's 52 submissions (estimate 208 s) used 83 s.

**Measure:** F, the normalised linear XEB (as HW1 and HW2), mean of the two repeats.

## 3. Predictions

Ratio = F(shorter) / F(longer). A pair enters D1 and D2 only if, under pad, the shorter circuit's F exceeds the
longer's by more than the pair's largest repeat difference (time mattered without decoupling). With fewer than two
such pairs, D1 and D2 are not testable.

| ID | prediction |
|---|---|
| D1 | decoupling shrinks the shorter circuit's advantage: \|ln ratio(dd)\| < \|ln ratio(pad)\| on at least 2 of 3 such pairs (all, if fewer than 3) |
| D2 | decoupling raises the longer circuit's F more than the shorter's: F_long(dd)/F_long(pad) > F_short(dd)/F_short(pad), same count |
| D3 | decoupling raises the longer circuit's F on every pair |
| D4 | reported: F(plain) / F(pad) per circuit (the device's own scheduling against explicit ALAP) |
| D5 | reported: whether the shorter circuit still wins with decoupling |

**Reading, fixed now.**
- **D1 and D2 confirmed:** the time sensitivity is mostly a waiting error that decoupling removes. For users who run
  with decoupling, item 83's gain is smaller than without it. PSF-Zero should then state the setting it is measured
  in, and consider inserting decoupling itself.
- **D1 refuted while time mattered:** the time effect is not a waiting error that X-X decoupling refocuses (for
  example leakage, or crosstalk tied to the schedule). Shorter schedules then stay valuable with decoupling on.

**QPU:** about 40-60 s expected, within the remaining allowance after HW2. Runs only with the owner's go-ahead.
