# RING1: pre-registration (DRAFT by the workplace, 2026-10-15 02:10 CEST; home fixes it before any real-device step)

> **Status: draft.** Home reviews it, edits it if needed, then fixes it. To fix it: write its SHA-256 and the time
> into `PREREG_SHA256.txt`, and update its line in `SHA256SUMS`. Step 67 stops if `SHA256SUMS` does not match.

**Question.** On a real IBM device, does a ring classifier compiled by PSF-Zero's AI front end classify better than
the same classifier compiled by Qiskit level 3?

- The classifier ring is the in-scope family with the largest gap: on FakeFez the front end's circuits are 0.48-0.59
  times as long as Qiskit level 3's.
- No classifier has been run on a real device yet.
- The front end is a18 in release 2026-10-14.1 (Addenda 489, 492). Its ring path is item 28 (N2's ring formula) and
  item 31 (the second schedule). Every candidate is checked exact (item 30); nothing is searched (item 32).

**Device rule** (TIME1's).

- ibm_fez if it is operational.
- Otherwise, the operational non-simulator device with at least 127 qubits, excluding ibm_kingston and ibm_marrakesh,
  that has the fewest pending jobs (ties broken by name).

Compiling uses the device's Target at prepare time. Its calibration decides nothing.

**Script.** `ring1_real.py` in this kit, fixed by SHA256SUMS. It refuses to run unless these files match the
SHA-256 it fixes:

- the release's `psf_compile.py` (23d4aa4d...) and `benchmarks/psf_ai_compile.py` (a18, 94bd54ee...);
- a15 and a14, which a18 loads;
- TIME1's `time1_real.py` (device rule, stamps);
- HW2's `hw2_real.py` (exact distributions, XEB).

**Model** (`ring1_model.json`, fixed by SHA256SUMS). No training or scikit-learn runs at home.

- The circuit is DEPTH1's / W3's. Each layer is RY(pi/2 x_q), then RY(a) RZ(b) on every qubit, then a CZ ring
  (0,1) ... (n-1,0). A final RY(c) follows on every qubit. The prediction is sign(<Z_0>).
- Data: breast cancer, PCA to n.
- Weights were trained noiselessly by DEPTH1's recipe (Adam, 300 steps; split seed 1, initial seed 1 + 1000 L), then frozen.
- Four configurations: n = 4 and 6, L = 4 and 8 layers.
- 24 test points per configuration: the first 12 of each class in the test split's order.
- Ideal accuracy on these points: 0.875, 0.875, 1.000, 0.958. On the whole test split it is 0.92-0.95.

**Circuits.** One SamplerV2 job, 800 shots per circuit per repeat, two repeats. Repeat 2 runs in the reverse order.
Dynamical decoupling and twirling are off.

| part | inputs | arms |
|---|---|---|
| CLF | 4 configurations x 24 points | A18: a18's `compile_for_model_circuit` with the device's Target, basis cz/rz/sx/x (no fractional gates). QK3: Qiskit level 3 on the same Target, seed 0. |
| control | ghz_n6 | Qiskit level 3, compiled once, submitted twice (CTL-A, CTL-B) |

That is 194 circuits per repeat and 388 in all (310,400 shots).

**Checks before submission.** Each circuit must be:

- native to the device (ISA);
- free of any failed element;
- exact by noiseless simulation, within 1e-6.

The ideal <Z_0> of every point must also equal the model file's. A point is submitted only if both arms pass. The
rough QPU estimate (0.3 ms per shot + 0.1 s per circuit) must be at most 150 s. For 388 circuits it is 132 s;
TIME1's and FRAC1's measured rate (0.27 ms per shot + 0.036 s per circuit) gives about 98 s.

**Measures.** All use both repeats' counts, pooled.

- z = P(0) - P(1) of logical qubit 0.
- A point is correct if sign(z) = y; z = 0 counts half.
- The margin is y z: the correct class's probability minus the other's.
- F is the normalised linear XEB against the exact distribution.

**Predictions.** A ratio is A18 over QK3.

| ID | prediction |
|---|---|
| RA | The pooled margin ratio over all scored points is >= 1.05. Not testable if QK3's pooled margin is <= 0.05, or fewer than 3 configurations are scored. |
| RB | Pooled accuracy: A18 >= QK3. |
| RA2 (reported) | The number of configurations in which A18's margin is higher. |
| RC (reported) | F ratio per configuration, and its geometric mean. |
| RD (reported) | The paired margin difference A18 - QK3 over the points, with its SE. |
| RE (control) | The F ratio of CTL-A to CTL-B is within 0.90-1.10. |

**Reading, fixed now.**

- **RA and RB both confirmed:** the ring formula's shorter circuits classify better on a real device. The README may
  say so for this model and device, citing RING1.
- **RA refuted, RB confirmed:** no loss of accuracy, but no measurable margin gain at this size. Recorded as such.
- **RB refuted:** recorded. a18's ring path is re-examined before any classifier claim.
- If RE fails, the job is flagged as unreliable: the control's two circuits should run alike.

**Known limits** (stated before the run).

- **Readout qubit.** The margin is read on one physical qubit: the one that holds logical qubit 0. This qubit
  differs between the arms. In the rehearsal the qubits were 87 (A18) against 117 and 126 (QK3) for n = 4, and 86
  against 144 for n = 6. So the margin ratio also carries the difference between those qubits' readout and
  relaxation. F (RC) uses every qubit and is less exposed to this. Nothing here corrects for it.
- **Configurations are not independent.** Within one n, each arm puts the circuit on the same qubits in every
  configuration and point. So the four configurations are about two placements per arm. That is why RA is the pooled
  ratio, and the per-configuration count (RA2) is only reported.
- **Accuracy is a weak test.** Only 2 points per configuration have |ideal z| < 0.2. Noise that shrinks z rarely
  changes its sign, so RB is mostly a test of "no loss".

**Rehearsal.** FakeFez, with this script as fixed (SHA-256 1f10b66f...). Output is in `rehearsal/`.

- All 384 classifier circuits and the 4 controls passed every check. Both arms' pairs are complete.
- Durations, A18 against QK3 (means):

  | n | L | A18 | QK3 |
  |---|---|---|---|
  | 4 | 4 | 2.33 µs, 28 two-qubit gates | 3.95 µs, 37 |
  | 4 | 8 | 4.65 µs, 56 | 8.19 µs, 77 |
  | 6 | 4 | 3.63 µs, 60 | 6.98 µs, 70 |
  | 6 | 8 | 7.26 µs, 120 | 15.08 µs, 142 |

- **Noiseless local sampler.** Every F is 1 within shot noise. The margins equal the ideal ones within 0.01. The
  paired difference is +0.0036 (SE 0.0022). The verdicts mean nothing here.
- **FakeFez's noise model** (reported only; nothing was chosen or tuned by it). The simulated copies were scheduled
  with delays, because Aer relaxes idle qubits only on explicit delays.
  - Margin ratio 1.124. A18 was higher in 2 of 4 configurations.
  - F ratios were 1.005, 1.045, 1.091 and 1.366.
  - In n = 4, L = 8, A18's F was higher but its margin lower (0.469 against 0.477): the readout-qubit limit above.
  - Without the delays, the margin ratio was 1.056.
  - A noise model is not the device; this only shows that the job runs and is scored end to end.

**Optional extra arm (not in this draft).** a18 with fractional gates (rzz, rx) is allowed by the instructions as a
separate arm. Leaving it out keeps the estimate under 150 s. Adding it would need fewer points or shots.

**Not used.** No calibration value, ESP, simulation or search chooses anything. The device rule uses only status,
size and queue length.
