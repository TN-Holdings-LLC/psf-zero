# TIME1: pre-registration (fixed before any real-device step)

**Question.** On a third IBM device, does a compiled circuit's duration pick which of two compilations of the same
input the device runs better? Does it do so better than the two-qubit gate count and ESP do?

- Prototype c49 (Addendum 477) chooses between compiled circuits by duration (item 84).
- Its basis is an observation made after the fact on HW2 (ibm_marrakesh): duration picked the winner on 7 of 7
  pairs.
- Part S of HW2 also supports it: the same gates in a shorter order gave 1.30 and 1.17 times the F.

TIME1 is the pre-registered test of that basis. The owner approved it on 2026-10-14.

**Device rule.**

- ibm_fez if it is operational.
- Otherwise, the operational non-simulator device with at least 127 qubits, excluding ibm_kingston (HW1) and
  ibm_marrakesh (HW2), that has the fewest pending jobs (ties broken by name).

The device's Target at prepare time is used for compiling and for the reported numbers. Its calibration decides
nothing.

**Script.** `time1_real.py` in this kit, fixed by SHA256SUMS. It uses HW2's `hw2_real.py`, fixed by SHA-256 in the
script, for the shared functions: the circuits, the exact distributions and XEB.

**Circuits.** One job, 6,000 shots per circuit per repeat, two repeats. Repeat 2 runs in the reverse order. Dynamical
decoupling and twirling are off.

| class | inputs | arms | pair scored |
|---|---|---|---|
| T1 (order) | TFIM chain from a fixed random product state, n = 12, 16, 20, 4 and 8 second-order Trotter steps, dt 0.3 | c49 with `level3_candidate=False`: item 83 on (S1) and off (S0) | S1 / S0, when the two differ |
| T2 | HW2's seven reading tests | C49 and C48, each the default call with the device | C49 / C48, when the two differ |
| T3 | the same seven | C49 and QK3 (Qiskit level 3, seed 0) | C49 / QK3 |
| control | ghz_n12 | QK3 and C49 | - |

**T1 gate counts.** In T1 the two arms can differ in two-qubit count. Item 84 picks STEINER6's construction by its own
schedule, so with item 83 off a different construction can win. The FakeFez rehearsal gave S1 / S0 two-qubit ratios
of 0.87-1.02. Both counts are reported.

**Checks before submission.** Each circuit must be:

- native to the device (ISA);
- free of any failed element;
- exact by noiseless simulation, within 1e-6, when it has at most 24 active qubits.

A circuit that fails a check is not submitted, and a T1 pair is submitted only whole. The rough QPU estimate (0.5 ms
per shot + 1 s per circuit) must be at most 300 s. From the rehearsal's 70 circuits it is about 280 s; HW2's rate
suggests about 120 s.

**F.** The normalised linear XEB against the exact ideal distribution, as the mean of the two repeats. A pair whose F
is at or below 0 on either side is not scored.

**Predictions.** A ratio is the first arm over the second.

| ID | prediction |
|---|---|
| TA | In T1, S1 has the higher F on at least ceil(5/6 x scored pairs). At least 3 pairs are needed. |
| TB | Over all scored pairs (T1, T2, T3) whose duration ratio is <= 1/1.1 or >= 1.1, the shorter circuit has the higher F on at least 75%. At least 4 such pairs are needed. |
| TC | Over all scored pairs, the rank correlation of log(1 / duration ratio) with log(F ratio) is at least +0.5. It must also be above the rank correlations of log(1 / two-qubit ratio) and of log(ESP ratio). |
| TD | C49's F >= 0.95 x QK3's on at least 6 of the 7 T3 tests. Identical circuits count as met. |
| TE | Control: the ghz_n12 F ratio is within 0.90-1.10. |

**Reading, fixed now.**

- **If TA, TB and TC are all confirmed:** duration as the choice rule is confirmed on a third device. c49 is then
  proposed for release, and the README may say that PSF-Zero chooses by duration, citing TIME1, HW2's part S and the
  after-the-fact HW2 observation.
- **If any of them is refuted:** item 84 is re-examined before any release or README claim. The result is recorded
  either way.
- TD is reported as the practical comparison with Qiskit's default call.
- If TE fails, the job is flagged as unreliable: the control's two circuits should run alike.

**Rehearsal.** FakeFez, noiseless local sampler, with this script as fixed (6,000 shots per repeat); output in
`rehearsal/`.

- 70 circuits; every check was passed.
- Every F is 1 within shot noise, with standard errors of 0.010-0.029. An earlier rehearsal at 3,000 shots gave
  0.015-0.04, which is why the shots were raised.
- The rehearsal's verdicts are meaningless, because there is no noise to separate the arms.

**Not used.** No calibration value chooses anything. ESP enters only as a competing predictor in TC.
