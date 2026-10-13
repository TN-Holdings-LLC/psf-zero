# HW2 -- fixed before any real-device step (local; to be recorded as an Addendum with the results)

Fixed: 2026-10-13 02:25:47 CEST (Europe/Zurich), before any real-device step of HW2.

Written 2026-10-13 before the real-device session; kept on the owner's PC with its sha256 and CEST time, and checked
against them by `91_hw2_prep.sh` and `92_hw2_run.sh` before anything runs.

**Seen before this file:**

- HW1's results on ibm_kingston (Addendum 455), including the exploratory estimate with idling;
- C48-VAL (prototype c48, item 83; offline, fake devices);
- HW2's full rehearsal on FakeFez with a noiseless local simulator (twice: the first showed that only one reading test
  was clearly shorter for PSF-Zero there, so the tfimP tests were added to part R before this file was fixed);
- no real-device data from ibm_fez or ibm_marrakesh.

## 1. Why

HW1 found, on one device and one day, that ESP (gate and readout errors, no time) ranked PSF-Zero against Qiskit
level 3 wrongly more often than not, and that PSF-Zero won where its circuit was also shorter in time. Two things
follow:

- **Part R** asks whether that repeats on a different device. Its findings would go into the README and a paper only
  if they do. The estimate with idling, found after HW1's scores were read, is tested here as a prediction, in a form
  fixed now and not fitted.
- **Part S** asks the question behind item 83 directly. The same blocks of gates are run in two orders, the mover's
  (S0) and item 83's (S1), so that only the duration differs. Does the shorter schedule give the higher fidelity on
  the device?

## 2. Design (`hw2_real.py`, sha256 in `SHA256SUMS`)

**Device rule.** Of ibm_fez and ibm_marrakesh, the one operational when `prepare` runs with fewer pending jobs (a tie
goes to ibm_fez). If neither is operational, nothing is prepared and nothing is submitted. ibm_kingston (HW1's device)
is not used. The device's Target at that moment is used for compiling and for every estimate.

**Part R.** HW1's nine tests, built by the same code with the same seeds, and part S's two tfimP tests:

| test | what |
|---|---|
| qaoa_n12_p2, qaoa_n20_p2 | QAOA on a random 3-regular graph |
| ba_n14, ba_n16, ba_n18, ba_n20 | Benchpress Barabasi-Albert QAOA (three layers) |
| tfim_n12, tfim_n16 | transverse-field Ising from \|0...0>, t = 0.6, Suzuki order 2, two steps |
| ghz_n12 | the control |
| tfimP_n12, tfimP_n16 | as in part S below; added so that R4 has tests where PSF-Zero is clearly shorter (on FakeFez only tfim_n12 was) |

- **Arms:** QK3 = Qiskit level 3, seed 0. PSF = prototype c45 with `level3_candidate=False`, as in HW1.
- **Reading tests:** the seven of Addendum 455's reading. tfim_n16 is left out because XEB exceeded 1 there in HW1; it
  is reported only. GHZ is the control. R4 reads the seven and the two tfimP tests.

**Part S.** Prototype c48 with item 83 off (S0, the mover's order, = c47) and on (S1). Both use
`level3_candidate=False`.

- **Tests:**
  - tfimP_n12 and tfimP_n16: HW1's TFIM after a fixed random layer of ry(theta_q), theta uniform in [0.3 pi,
    0.7 pi], seed "HW2|<test>". From a product state that is not \|0...0>, relaxation does not favour the ideal's
    likely outcomes, so XEB can score it.
  - qaoa_n12_p2 and qaoa_n20_p2.
- **Submission:** a pair is submitted only where S0's and S1's circuits differ, and only if both pass every check.

**Checks before submission (every circuit):**
- the device's gates only;
- no failed element (error >= 0.5);
- the measured distribution equals the ideal one to 1e-6 by noiseless simulation (at most 24 active qubits).

**Estimates, written into the manifest before submission (every circuit):**
- ESP (the summed -ln(1 - error) of every gate and measurement, on the Target);
- duration (every gate as soon as its qubits are free, on the Target's gate durations; measurements left out);
- the idle-aware estimate: ESP x exp(-sum over active qubits of (T - t_first,q) (1/(4 T1,q) + 1/(2 T2,q))), where T
  is the duration, t_first,q is qubit q's first gate, and T1 and T2 come from the Target. A missing T1 or T2 counts as
  1 us.

This is the form of Addendum 455's exploratory estimate, unchanged and not refitted.

**One job:**
- every circuit twice: repeat 1 in one order, repeat 2 in the reverse;
- 6,000 shots each;
- SamplerV2 with dynamical decoupling and twirling off;
- `prepare` refuses a job whose rough QPU estimate (0.5 ms per shot + 1 s per circuit) exceeds 300 s. HW1, with 52
  circuits (estimate 208 s), used 83 QPU seconds; HW2 has about 60.

**Measure:** F, the normalised linear cross-entropy against the ideal distribution, mean of the two repeats (as HW1).
A ratio is not formed where an F is <= 0.

## 3. Predictions

**Part R (PSF / QK3, on the reading tests):**

| ID | prediction |
|---|---|
| R1 | of the tests whose ESP ratio is >= 1.2, PSF's F >= QK3's on at most half (HW1: 1 of 3). Not testable if there are none |
| R2 | the rank correlation of ESP ratio and F ratio is <= 0.3 (HW1: -0.18 on these seven) |
| R3 | the idle-aware estimate picks the same winner as F on at least 5 of 7 (5 of every 7 when fewer can be scored), and its rank correlation with the F ratio is higher than ESP's |
| R4 | on every test (the seven and the two tfimP) whose PSF duration is <= 0.8 x QK3's, PSF's F >= QK3's. Not testable with fewer than two such tests |
| R5 | on the Barabasi-Albert tests, PSF's F <= 1.05 x QK3's on at least 3 of 4 |
| R6 | GHZ control: \|F ratio - 1\| <= 0.10 |

**Replication rule, fixed now.** HW1's finding is replicated if R1, R3 and R4 are all confirmed. Only then do its
statements go into the README or a paper as findings. R2 and R5 are supporting. If replication fails, HW1 stays a
single-device observation and is written so.

**Part S (S1 / S0):**

| ID | prediction |
|---|---|
| S1 | on every tfimP pair submitted, S1's F > S0's F (if both pairs go the other way: refuted; if they split: ambiguous) |
| S2 | tfimP: the geometric mean of the F ratio S1 / S0 is >= 1.05 |
| S3 | reported: the QAOA pairs' duration ratios and F ratios |
| S4 | reported: on each pair, whether ESP and whether the idle-aware estimate picked the device's winner. On the tfimP pairs S1 has slightly more two-qubit gates, so ESP favours S0 and the idle-aware estimate favours S1 |

**Reported:** F / ESP per arm; every repeat difference. A pair whose F difference is smaller than its largest repeat
difference is read as not resolved, whatever the verdict line says.

## 4. Not claimed

- One more device and one more day.
- Fake-device durations are not the device's; the duration used is the one computed from the live Target at
  `prepare`.
- Nothing here measures crosstalk directly: S1 runs more gates at once than S0, so S1 losing would point at crosstalk
  or simultaneous-gate error, not at time.
