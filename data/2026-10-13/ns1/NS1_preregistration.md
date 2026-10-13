# NS1 -- fixed before the run (local; recorded with the results)

Fixed: 2026-10-13 03:35:22 CEST (Europe/Zurich).

**Question.** HW1 (Addendum 455) found that ESP does not rank PSF-Zero against Qiskit level 3 on ibm_kingston. Does a
standard Markovian noise model built from the same day's calibration predict the device's fidelities and their
ranking? The model is `ns_sim.py` (sha256 0d02400d713e18cdc6076d1c8d6bfc0337993335c54e617c6eb054467135e379); its form
is in its docstring and is not fitted to any device data.
- gate errors as relaxation for the gate's duration plus depolarizing for the rest of the reported error;
- relaxation (T1, T2) while a qubit waits between gates, as late as possible scheduling;
- preparation and readout errors from the calibration;
- XEB estimated by quantum trajectories, with a fixed stopping rule that looks only at the simulation.

**Seen before this file.**
- HW1's results (this is a retrospective test: the data are known, the simulation's output on them is not).
- The model's implementation checks (`test_ns_sim.py`, sha256 d9e0da52fceb579d7002e911c57c0818456aaf79ae28536b27970a85789869cb,
  6 passed: noiseless F = 1; readout-only F against a closed form; depolarizing monotone; waiting costs fidelity;
  no waiting before a qubit's first gate).
- Timing and smoke runs: one noisy trajectory of four HW1 circuits, timed only, no F read; and two circuits compiled
  afresh on FakeKingston (tfim_n12, ba_n20, seed 7, not HW1's).
- A bug in Aer 0.17.2: with the channels appended as QuantumError instructions, some circuits failed ("Kraus is
  empty"); found on HW1's qaoa_n20_p2 PSF circuit, without reading any F. Every channel is now an explicit Kraus
  instruction.

**Run.** `ns_hw1.py` (sha256 233533fae34fe9a77e49b7e2d5999c1d351f186b8af8731d39cb6805169cfe2a): HW1's PSF and QK3
circuits on the seven reading tests of Addendum 455 and the GHZ control (16 circuits, repeat 1's circuit; repeat 2's
is the same circuit), with the calibration CSV of 13:15 CEST. C47 and tfim_n16 are not in NS1. Stopping rule per
circuit: at least 64 trajectories, then until the standard error is <= max(0.01, 0.05 F), 3,000 trajectories or 40
minutes.

**Predictions.**

| ID | prediction | HW1's ESP for comparison |
|---|---|---|
| NS1a | the simulation picks the same winner (PSF or QK3) as the device on >= 6 of the 7 reading tests | 3 of 7 |
| NS1b | rank correlation of the simulated F ratio and the measured F ratio >= 0.6 | -0.18 |
| NS1c | simulated F within 0.5-2 times the measured F on >= 80% of the 16 circuits | -- |
| NS1d | F_real / F_sim per arm (geometric mean): \|ln(PSF / QK3)\| < 0.153 (ESP's gap: F/ESP 0.742 against 0.865) | 0.153 |

**Reading, fixed now.**
- NS1a and NS1b confirmed: the model is a candidate predictor; it is tested prospectively on HW2 (NS2, with HW2's
  circuits and that day's calibration, the code unchanged) before it is used in PSF-Zero.
- NS1a or NS1b refuted: time and the reported error rates are not enough. What decides the device's ranking lies in
  what the model leaves out: crosstalk between simultaneous gates, coherent errors left untwirled, or drift.
- NS1c and NS1d say whether the magnitudes are right and whether PSF-Zero's circuits fall short of the model more
  than Qiskit's.
