# FRAC1: pre-registration (fixed before any real-device step)

**Question.** On the device, do IBM Heron's native fractional gates raise the fidelity of the same compiled circuits?

- The gates are rzz (a ZZ rotation in one CZ-length pulse, 68 ns on ibm_marrakesh) and rx (one 36 ns pulse).
- Prototype c51 (item 86 as revised in c51: every rzz folded into IBM's range) uses them where they shorten the
  circuit. C50-VAL refuted the unrevised item 86 on its V6, and C51-VAL is the offline check of the revision.
- The Target reports rzz errors that IBM's guide says are copies of the CZ errors, so the device's own fidelity of
  rzz is unknown until it is measured.
- The owner approved this kind of test on 2026-10-14. It runs only with the owner's go-ahead for the QPU time.

**Device.** TIME1's device, read from TIME1's manifest (`data/2026-10-14/time1_run/time1_manifest.json`), so that both
tests read one device on one evening. If TIME1 has not prepared, TIME1's device rule applies. The device must offer
rzz when it is loaded with `use_fractional_gates=True`; otherwise nothing is prepared.

**Script.** `frac1_real.py` in this kit, fixed by SHA256SUMS. It uses TIME1's `time1_real.py`, fixed by SHA-256 in the
script, for the circuits, the checks, XEB and the device rule.

**Circuits.** One job on the backend loaded with fractional gates. Circuits without rzz run there too.

| part | inputs | arms |
|---|---|---|
| Ising | TIME1's T1 inputs: TFIM chains from a product state, n = 12, 16, 20, 4 and 8 steps | C49: c49's default call on the Target without fractional gates (as tonight's TIME1 C49 arm) |
| HW2 | HW2's seven reading tests | C51: c51's default call on the Target with fractional gates |
| control | ghz_n12 | C49 and C51 |

- A pair is submitted only where the two circuits differ, and only whole.
- Shots: 4,000 per circuit per repeat, two repeats, the second in reverse order. TIME1 runs first on the same
  evening with 6,000; about 5.5 QPU minutes are left, and the two jobs together are expected at about 3 minutes
  from HW2's rate (0.28 ms per shot).
- Dynamical decoupling and twirling are off. Pauli twirling is not supported with fractional gates.

**Checks before submission.** Each circuit must be:

- native to its Target;
- free of rzz angles outside (0, pi/2];
- free of any failed element;
- exact by noiseless simulation, within 1e-6, when it has at most 24 active qubits.

The rough QPU estimate must be at most 300 s.

**F.** The normalised linear XEB against the exact ideal distribution, as the mean of the two repeats. A pair with F
at or below 0 on either side is not scored.

**Predictions** (ratio = C51 / C49):

| ID | prediction |
|---|---|
| FA | In the Ising part, C51 has the higher F on at least ceil(5/6 x scored pairs). At least 3 pairs are needed. |
| FB | In the Ising part, the geometric mean F ratio is at least 1.10. |
| FC | Over every scored pair whose duration ratio is <= 0.9, C51's F >= C49's on at least 75%. At least 4 such pairs are needed. |
| FD | Reported only: the rank correlation of 1 / duration ratio with F ratio. |
| FE | Control: the ghz_n12 F ratio is within 0.90-1.10. |

**Reading, fixed now.**

- **If FA and FB are confirmed:** item 86 is confirmed on the device. c51 replaces c49 as the release candidate,
  subject to TIME1 as fixed in its prestatement.
- **If either is refuted:** the device's rzz does not deliver the gain its duration promises. Item 86 stays off by
  default, and the result is recorded.
- FC reads the QAOA and Barabasi-Albert pairs, where the expected gain is 10-20% in duration.
- If FE fails, the job is flagged as unreliable.

**Rehearsal.** Run on FakeFez with a fractional copy of its Target (rzz with a CZ's duration, rx with an sx's), using
the noiseless local sampler; the output is in `rehearsal/`.

- It was run at 6,000 shots before the shots were lowered to 4,000. The script is otherwise identical.
- 56 circuits, all checks passed; every C51 rzz angle was in range.
- The Ising circuits went from 2.52 to 1.08 us (n = 20, 4 steps) with half the two-qubit gates.
- The F values show only that the flow works.

**Not used.** No calibration value chooses anything.
