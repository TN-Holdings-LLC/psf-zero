# Addendum 167 -- Pre-registration, Stage 1: how robust is the noisy XOR rehearsal across fake IBM backends, and does free (noise-aware) layout beat the rehearsal's pinned layout? (2026-09-25)

> **Imported into the home series as Addendum 167.** Written at the workplace, run on a RunPod pod (RTX 4090), original file `xor-real-device-stage1-preregistration-2026-09-25.md`; body below unchanged. Script hash on file (`xor_prereg_stage1_sweep.py`, normalized SHA-256 `28197bc4...`) re-checked at home: matches.

**Status: pre-registration, locked at the Project save time of this
document.** No Stage-1 run on the real XOR circuit exists at the time of
locking. The only prior data are the existing single-backend rehearsal
results ([`benchmarks/rehearse_result_2.txt`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/rehearse_result_2.txt), reproduced byte-for-byte on a
RunPod RTX 4090 pod on 2026-09-25), quoted as background, not as Stage-1
data. A dry run of the harness on a **stub** circuit was made before locking
(section 7); its outcome values are not Stage-1 data and are not reported.

**Numbering:** the home machine's series had reached Addendum 166 when this
was written. To avoid a collision, this document carries no number; the
official number is assigned when the two records are merged.

## 1. Why this experiment exists

The plan for 2026-09-28 (IBM Quantum free tier restored) is to submit the
trained XOR classifier (Addendum 148, seed 0) to a real device through the
already-tested SamplerV2 path and compare against the noisy rehearsal
(`rehearse_xor_fake127.py`: all four inputs correct, noisy |<Z0>| 0.904 to
0.9185 on FakeBrisbane).

As planned, that comparison has three weaknesses:

1. **It rests on one fake backend.** The real device is not chosen yet and
   will not be FakeBrisbane's calibration snapshot.
2. **The 0.90-0.92 band is mostly shot noise.** At S = 4,000 shots the
   standard error of one <Z0> near |<Z0>| = 0.9 is sqrt((1 - 0.81) / S), about
   0.007. The four rehearsal values span about 0.015, roughly two standard
   errors.
3. **The rehearsal pins the layout.** Reading the repository's
   `route_for_backend` (in `psf_pennylane_gpu_ibm_prototype.py`) shows
   `initial_layout=list(range(qc.num_qubits))` with `optimization_level=1`:
   the circuit always runs on physical qubits 0-3, whatever that day's
   calibration says about them. The function's own comment already says a
   real integration would not pin this.

Stage 1 measures, on many fake backends, how robust correctness is, how much
the result depends on routing seeds, whether shot noise behaves as the
binomial model says (needed to set Stage-2 tolerances honestly), and whether
letting Qiskit choose qubits by calibration beats the pinned layout. Stage 2
(section 6) then fixes the real-device procedure and predictions before
2026-09-28.

## 2. Fixed design

**Circuits.** The four input circuits (00, 01, 10, 11) exactly as
`rehearse_xor_fake127.py` builds them (`build_qiskit`) after `retrain_seed0()`
(constants in that file: N = 4, LAYERS = 3, ITERATIONS = 200, SEED = 0,
LR = 0.1). Built once and reused. **Harness check before any Stage-1 data:**
the exact (statevector) <Z0> of each rebuilt circuit must equal the
rehearsal's +-0.99776 with the correct sign (tolerance 6e-5); otherwise the
script stops.

**Backends (selection rule fixed now; the list is printed by the script, not
chosen by hand).** Every `Fake*` class in the installed
`qiskit_ibm_runtime.fake_provider` that instantiates as a `BackendV2` with at
least 100 qubits and whose `Target` has error values for `measure` and for a
native 2-qubit gate (`ecr`, `cx` or `cz`). Every exclusion is printed with its
reason.

**Two routing arms.**
- **Arm A (as-is):** `route_for_backend(circuit, backend, seed_transpiler=s)`
  from the repository -- pinned to physical qubits 0-3, `optimization_level=1`.
  This is the path 2026-09-28 would use unchanged.
- **Arm B (free layout):** `transpile(circuit, backend=backend,
  optimization_level=3, seed_transpiler=s)` with no `initial_layout`, so
  Qiskit's calibration-aware layout passes choose the physical qubits.

Every routed circuit must pass the repository's `is_isa_compliant`; a failure
stops the run.

**Measurement.** In both arms, only the physical qubit that holds logical
qubit 0 at the end of the routed circuit (`layout.final_index_layout()[0]`)
is measured. This avoids depending on the bit ordering of the other
measurement helpers. In Aer's noise model readout errors are independent per
qubit, so the marginal of qubit 0 is the same whether or not the other qubits
are measured; on real hardware this is not guaranteed (Stage 2 notes it).

**Seeds and shots.** `seed_transpiler` s in {0, 1, 2, 3, 4}; noisy simulation
`seed_simulator` = 0 for all main cells; S = 4,000 shots per circuit.

**Noise model.** `AerSimulator.from_backend(backend)`.

**Harness check C0 (not a prediction; must pass before scoring).** For
FakeBrisbane, arm A, seed 0, each input's noisy <Z0> must lie within
3 * sqrt(2) standard errors of the rehearsal's recorded value (-0.9040,
0.9185, 0.9085, -0.9130). If C0 fails, predictions are not scored and the
mismatch is investigated first.

**Recorded per cell** (backend x arm x seed x input): exact <Z0>, noisy
<Z0>, label, physical qubit measured, its readout error, number of routed
2-qubit gates, `blocks` (PSF-Zero-eligible blocks in the logical circuit),
shots, simulator seed, and environment versions. Margin m = label x noisy
<Z0>; a cell group (backend x arm x seed) has mean margin M over its four
inputs.

## 3. Withdrawn before locking: the calibration-only error-budget predictor (draft P2)

The draft of this document proposed, as its central prediction, a closed-form
error budget (depolarizing shrink factors over the backward light cone of the
measured qubit, plus a readout factor) expected to match the noisy result to
within 0.03. It is **withdrawn before locking**, for two reasons found while
building the harness:

1. **The light-cone construction is structurally biased.** It includes every
   gate that touches a qubit in the growing cone, even when the gate commutes
   with the observable (for example a CX whose control is the measured qubit,
   or a CZ), so it adds qubits to the cone that the observable never actually
   spreads to, and over-counts error. The stub dry run (section 7) made this
   visible; the reason is structural, not a matter of tuning.
2. **On fake backends it cannot test what matters.** Aer's noise model is
   itself built from the same calibration numbers, so any calibration-based
   predictor is checked only against another model of the same data, not
   against hardware.

For 2026-09-28 the predictor will instead be the standard one: an Aer noise
model built from the chosen real device's own calibration on that day
(section 6). P2 is left empty so the other prediction numbers stay as in the
draft.

## 4. Pre-registered predictions

**P1 (correctness is robust).** In every cell group, all four inputs are
classified correctly (m > 0). *Refuted* by any single sign flip.

**P2.** Withdrawn (section 3). Not scored.

**P3 (seed sensitivity is small).** Within each backend x arm, the spread
(max - min) of M across the five `seed_transpiler` values is at most 0.03 for
all backend x arm pairs (*confirmed*). *Refuted* if any backend x arm exceeds
0.06. Otherwise *ambiguous*.

**P4 (shot noise follows the binomial model).** For FakeBrisbane, arm A,
seed 0, repeat the noisy simulation with `seed_simulator` 0 to 49 (50
repeats). For each input i, r_i = (sample SD of <Z0> over the repeats) /
sqrt((1 - mean<Z0>^2) / S). Pooled ratio R = sqrt(mean of r_i^2 over the four
inputs). *Confirmed* if 0.85 <= R <= 1.15; *refuted* if R < 0.7 or R > 1.3;
otherwise *ambiguous*. (With 4 x 49 degrees of freedom the relative standard
error of R is about 0.05, so the confirmed band is about +-3 standard errors.)

**P5 (PSF-Zero is not exercised).** `blocks` = 0 in every cell. Stated so no
Stage-1 or Stage-2 result is read as evidence about PSF-Zero: the XOR circuit
has no block that PSF-Zero would re-synthesize.

**P6 (free layout is at least as good).** For every backend and every seed,
arm B's M is at least arm A's M minus 0.01, **and** arm B beats arm A by more
than 0.01 on at least half of the backends at seed 0 (*confirmed*).
*Refuted* if arm B is worse than arm A by more than 0.03 for any backend and
seed. Otherwise *ambiguous*. If P6 is confirmed, Stage 2 proposes arm B for
2026-09-28; if not, arm A stays unless Stage 2 gives another pre-registered
reason.

## 5. What this does and does not test

It tests the XOR circuit's behaviour under Aer's calibration-based noise
models of many IBM devices. It does **not** test real hardware, where
crosstalk, drift, leakage, coherent errors and idle decoherence (absent here,
since circuits are not scheduled) all add error. No timing is measured. The
GPU is not used (all simulation is CPU Aer).

## 6. Stage 2 (outline only; written and locked after Stage 1 is scored and before 2026-09-28)

- Routing arm for the real run, chosen by the P6 result as stated in P6.
- Device selection rule, fixed in advance: among devices available to the
  account at submission time, the one with the highest predicted M from an
  Aer noise model built from that device's calibration at submission time
  (`AerSimulator.from_backend(real_backend)`, many shots); ties broken by
  shortest queue. The calibration snapshot and the prediction are saved
  together with the job ID before results are read.
- Predictions: all four inputs correct; observed M expected **below** the
  Aer prediction (one-sided), with a tolerance derived from P4's shot-noise
  result and Stage 1's seed spread (P3), and a stated floor below which the
  result counts as worse than the calibration can explain.
- Per the Addendum 150 lesson, the magnitude of <Z0>, not only correctness,
  is scored.
- Measuring only qubit 0 versus all logical qubits on real hardware is a
  choice Stage 2 must fix in advance (it is equivalent in Aer, not
  necessarily on hardware).

## 7. Dry run before locking (methodology, not data)

The harness was run end to end in the workplace sandbox with a **stub**
`rehearse_xor_fake127` module (a hand-written 4-qubit circuit with
|<Z0>| = 0.99776, not the trained XOR classifier) on five and then two fake
backends, to check that the code runs and scores. Changes made because of it:

- Draft P2 withdrawn (section 3).
- P4 redesigned from 10 repeats with per-input thresholds to 50 repeats with
  a pooled ratio, because a power calculation showed the 10-repeat design
  would come out *ambiguous* roughly a third of the time even if the binomial
  model holds exactly.

P3 and P6 thresholds are **unchanged from the draft written before the dry
run**; the stub's outcome values were deliberately not used to adjust them,
and they are not reported here because the stub is not the XOR classifier.

## 8. Files, integrity check and run command

| File | What it is |
|---|---|
| [`xor_prereg_stage1_sweep.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/xor_prereg_stage1_sweep.py) | Stage-1 script (Project: [`psf-zero/benchmarks/xor_prereg_stage1_sweep.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/xor_prereg_stage1_sweep.py); on the pod: `~/pennylane_gpu_mock_test/`, outside the repository) |
| this document | the pre-registered predictions |

Integrity check of the script on the pod (the file is transferred by pasting,
which has corrupted files before): normalized SHA-256 (lines right-stripped,
leading/trailing blank space removed, joined with newlines) must equal
`28197bc43fa70bb06a928da9a84552644d683353bc4eb6cb94e37ba72c45cf0f`.

```
cd ~/pennylane_gpu_mock_test
python -c "import hashlib;print(hashlib.sha256('\n'.join(l.rstrip() for l in open('xor_prereg_stage1_sweep.py',encoding='utf-8').read().strip().splitlines()).encode()).hexdigest())"
python -u xor_prereg_stage1_sweep.py 2>&1 | tee ~/xor_prereg_stage1_run.txt
```

Output: `~/xor_prereg_stage1_2026-09-25.csv` (one row per cell) and the
scoring printed at the end of `~/xor_prereg_stage1_run.txt`. Expected run
time on the order of half an hour (roughly 20 backends; estimated from the
stub dry run, not measured on the pod).

Pre-publication check before any of this leaves the pod: grep the CSV and the
run log for local paths or machine-identifying strings beyond the platform
and CPU columns, per this project's record-keeping rules.
