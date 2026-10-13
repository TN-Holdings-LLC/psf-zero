# Workplace literature check (request 3 of 2026-10-13): what is published about T2, T2*, residual ZZ and instruction timing on IBM Heron devices, as background for an idle-error hypothesis (2026-10-13)

**Status: desk research.** Sources are linked; where nothing citable was found, it says so. Nothing here was used to
change a14's estimate (whose form was fixed earlier the same day).

## 1. Is the published T2 a Hahn echo (T2E) or a Ramsey (T2*) time?

**Hahn echo.** IBM's "QPU information" guide says, in its description of the calibration data, that "T2 is reported
from a Hahn echo sequence" ([IBM Quantum documentation, QPU information](https://quantum.cloud.ibm.com/docs/en/guides/qpu-information)).
The same guide gives no Ramsey (T2*) value; the dashboard CSV used in Addendum 455 has only "T2 (us)".

Consequence: an echo removes quasi-static dephasing (slow frequency noise, static ZZ shifts from neighbours in a fixed
state). A qubit that simply waits without dynamical decoupling is not echoed, so it dephases on a timescale between
T2* and T2E. An estimate that uses the published T2 for unprotected idle windows is therefore optimistic.

## 2. Typical T2* / T2E ratio

**No Heron-specific, citable number found.**

- General transmon literature: Burnett et al. (2019) report Ramsey T2* close to 2 T1 for most of the time on their
  own (non-IBM) Xmon transmons (1.4 < T2*/T1 < 2.2 in 82% of counts), i.e. qubits whose dephasing is T1-limited
  ([npj Quantum Information 5, 54](https://www.nature.com/articles/s41534-019-0168-5)). Such qubits would show
  T2* close to T2E.
- An individual user's report on unnamed IBM hardware gives T2* of about 2.25 us against an echo T2 of about 37 us
  ([PennyLane forum](https://discuss.pennylane.ai/t/ramsey-and-hahn-echo-coherence-measurements-on-ibm-quantum-hardware-preparing-for-cpmg/9193)).
  This is anecdotal (one qubit, unknown device, possibly a detuned Ramsey fit) and should not be used as a typical
  value.
- The ratio is device- and qubit-dependent and is driven by low-frequency noise (TLS, flux noise on tunable
  elements) and by static ZZ; it should be measured on the device, not assumed.

## 3. Residual ZZ between neighbouring qubits

**No published Heron-specific value found.**

- Heron uses fixed-frequency transmons with tunable couplers; IBM's precursor paper for this architecture reports
  "ZZ < 10 kHz with the coupler in its off state" for its best pair, with CZ gates of 46-136 ns
  ([Stehlik et al., arXiv:2101.07746](https://arxiv.org/pdf/2101.07746); [IBM Research page](https://research.ibm.com/publications/tunable-coupling-architecture-for-fixed-frequency-transmon-superconducting-qubits)).
  Other pairs' residual ZZ is not given there.
- For the earlier fixed-coupling (cross-resonance) IBM devices, a DD survey describes ZZ as "always present" and as
  producing coherent oscillations in idle fidelity without DD ([Ezzell et al., arXiv:2207.03670](https://arxiv.org/pdf/2207.03670)).
  It gives no kHz value for Heron.
- Scale check (not a measurement): a residual ZZ of 10 kHz accumulates a conditional phase of 2 pi x 10 kHz x t;
  over a 20 us idle window that is about 1.3 rad, which is far from negligible. Even 1 kHz gives about 0.13 rad.
  Whether Heron's couplers suppress it to below 1 kHz on every pair is the open question.

## 4. Without DD, are instructions executed as soon as possible or as late as possible?

**Not documented as far as found.**

- Qiskit Runtime's dynamical-decoupling options default to `enable=False` and, when enabled, to
  `scheduling_method="alap"` ([DynamicalDecouplingOptions](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/options-dynamical-decoupling-options.md)).
  The page does not say what timing the hardware uses when DD is off.
- The Runtime scheduling passes page describes ASAP and ALAP analysis passes for users and notes that the scheduler
  does not insert delays on idle qubits in fast-path blocks; it does not state a default for ordinary circuits
  ([qiskit-ibm-runtime scheduling passes](https://quantum.cloud.ibm.com/docs/ja/api/qiskit-ibm-runtime/0.41/transpiler-passes-scheduling)).
- Practical consequence for the next pre-registration: do not rely on an assumed default. Either schedule the
  circuits explicitly before submission (an ALAP or ASAP schedule with explicit delays, so that the executed timing
  is the one the estimate assumes), or state both variants and fix in advance which one is scored.

## 5. What this does and does not support

- **Supported by a primary source:** the published T2 is an echo time, so unprotected idle windows can dephase faster
  than T2 predicts. This is consistent with HW1's finding that the device was more sensitive to time than the T1/T2
  estimate.
- **Plausible, not established for Heron:** static ZZ of order kHz adds a conditional phase over tens of us. No
  Heron number was found; it would have to be measured (for example a Ramsey on one qubit with a neighbour in |0> and
  in |1>), which needs the owner's go-ahead for hardware time.
- **Unknown:** the hardware's default timing without DD.
