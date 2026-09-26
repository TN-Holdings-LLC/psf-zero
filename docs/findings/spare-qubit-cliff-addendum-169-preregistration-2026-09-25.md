# Addendum 169 -- Pre-registration, Stage 1b: does putting the measurement into the circuit before layout selection stop the free layout from choosing poor-readout qubits? (2026-09-25)

> **Imported into the home series as Addendum 169.** Written at the workplace, run on a RunPod pod (RTX 4090), original file `xor-real-device-stage1b-preregistration-2026-09-25.md`; body below unchanged. Script hash on file (`xor_prereg_stage1b_sweep.py`, `ec7f173c...`) re-checked at home: matches.

**Status: pre-registration, locked at the Project save time of this
document.** Written after Stage 1 was scored
([`xor-real-device-stage1-results-2026-09-25.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-168-2026-09-25.md), including its correction
note) and before any Stage-1b run on the XOR circuit. A dry run on a stub
circuit was made before locking (section 6); its values are not data.

**Numbering:** no number; assigned when merged with the home record.

## 1. Why this experiment exists

Stage 1 found that the free layout (arm B: `transpile(..., optimization_level=3)`
on a circuit **without** measurement, measurement added afterwards) measured
a poor-readout qubit on FakeKingston (physical qubit 1, readout 0.168,
M 0.656 versus 0.968 pinned) and on FakeBrussels (qubit 28, readout 0.0916,
for three of four inputs), with smaller cases on FakeStrasbourg and
FakeAachen. The pinned layout (arm A, physical qubits 0-3) failed where qubit
0 itself was poor (FakeCusco readout 0.5, FakeTorino 0.167).

Hypothesis (from Stage 1's post-hoc diagnostics): Qiskit's layout scoring
counts the errors of the instructions present in the circuit, so a readout
error only influences the choice if a `measure` is in the circuit when the
layout is chosen. The rehearsal pipeline, and Stage 1's arm B, route first
and measure afterwards.

Stage 1 also showed that in arm B the four inputs can land on different
physical qubits, because each input circuit gets its own layout. Stage 1b
therefore also tests one shared layout for all four inputs, and the full
"simulate candidates with the backend's own noise model, then pick" procedure
planned for 2026-09-28.

## 2. Fixed design

Same circuits, backend selection rule, noise model (`AerSimulator.from_backend`),
shots (4,000) and scoring simulator seed (0) as Stage 1, by importing the
hash-verified Stage-1 script. **One transpiler seed (0) only:** Stage 1 found
the measured qubit and every sampled value identical across five seeds in
both arms for this circuit.

**Arms** (per backend, four inputs each):
- **A:** pinned layout (Stage 1 arm A, repository `route_for_backend`).
- **B:** free layout without measurement (Stage 1 arm B).
- **M1 (measure-aware, per input):** logical qubit 0 is measured into one
  classical bit **before** `transpile(..., optimization_level=3,
  seed_transpiler=0)`. The measured physical qubit is read from the routed
  circuit's single `measure`.
- **M4 (measure-aware, one shared layout):** input 00's measure-aware
  transpile fixes the layout (`initial_index_layout`), and all four inputs are
  transpiled with that `initial_layout` (`optimization_level=3`, seed 0).
- **S (select by simulation):** for each backend, the arm among A, B, M1, M4
  with the highest *predicted* M, where the prediction is the same backend's
  noise model run with an independent simulator seed (1000) and 20,000 shots.
  S is then scored on that arm's main (seed 0, 4,000 shots) result. On fake
  backends the noise model is the ground truth, so S tests the selection
  **procedure**, not how well a calibration predicts real hardware.

**Excluded from scoring, fixed now:** FakeKyoto. Stage 1 found all 144 of its
ECR entries have error 1.0, so every layout is at chance; it is still run and
printed, marked "not scored". All predictions below refer to the remaining
backends.

**Reproducibility check R0 (must pass before scoring):** arms A and B must
reproduce Stage 1's seed-0 `z_noisy` values **exactly** (same code, seeds and
versions). If R0 fails, predictions are not scored.

## 3. Pre-registered predictions

**Q1 (measure-aware layouts avoid poor readout).** In arms M1 and M4, the
measured qubit's readout error is at most 0.05 for every scored backend and
input (*confirmed*). *Refuted* if any exceeds 0.10. Otherwise *ambiguous*.

**Q2 (M1 is not worse than the better of A and B).** For every scored
backend, M(M1) >= max(M(A), M(B)) - 0.02 (*confirmed*). *Refuted* if any
backend has M(M1) < max(M(A), M(B)) - 0.05.

**Q3 (the two Stage-1 failures are fixed).** M(M1) >= 0.90 on both
FakeKingston and FakeBrussels (*confirmed*). *Refuted* if either is below
0.80.

**Q4 (one shared layout costs little).** For every scored backend,
|M(M4) - M(M1)| <= 0.02 (*confirmed*). *Refuted* if any exceeds 0.05.

**Q5 (correctness).** In arms M1 and M4 and in each backend's selected arm,
all four inputs are correct on every scored backend. *Refuted* by any sign
flip.

**Q6 (the selection procedure picks a near-best arm).** For every scored
backend, M(selected arm) >= max over the four arms - 0.02 (*confirmed*).
*Refuted* if any is below max - 0.05.

## 4. What this does and does not test

It tests layout-selection procedures under Aer noise models built from 21
IBM calibration snapshots (20 scored). It does not test real hardware,
calibration drift between the snapshot and a real run, or whether measuring
only logical qubit 0 on hardware behaves like it does in Aer. No timing is
measured; the GPU is not used.

## 5. Use for Stage 2

If Q1-Q3 are confirmed, Stage 2 will use a measure-aware layout (M1 or, if
Q4 is confirmed, M4 for simplicity and consistency across inputs) as the
default candidate, and the Q6 procedure (simulate candidates with the chosen
device's calibration of 2026-09-28, pick the best) as the final choice.
Otherwise Stage 2 falls back to the selection procedure over A and B only.

## 6. Dry run before locking (methodology, not data)

The script was run in the workplace sandbox with the same stub
`rehearse_xor_fake127` module used before Stage 1 (not the XOR classifier),
on FakeBrisbane and FakeTorino, with a stub-generated Stage-1 CSV for R0. It
ran end to end, and R0 reproduced the stub's Stage-1 values exactly (0
mismatches), which supports using exact equality in R0. The thresholds above
were written before the dry run and were not changed after it; the stub's
outcome values are not reported.

## 7. Files, integrity check and run command

| File | What it is |
|---|---|
| [`xor_prereg_stage1b_sweep.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/xor_prereg_stage1b_sweep.py) | Stage-1b script (Project: `psf-zero/benchmarks/`; on the pod: `~/pennylane_gpu_mock_test/`, next to the Stage-1 script it imports) |
| [`xor_prereg_stage1_sweep.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/xor_prereg_stage1_sweep.py) | Stage-1 script, imported unchanged |
| this document | the pre-registered predictions |

Normalized SHA-256 of the Stage-1b script (lines right-stripped, outer blank
space removed, joined with newlines):
`ec7f173cb36fa071b9dbd03de40ead48947edbbc955f6781369c2d8e71105d8e`.
Requires `~/xor_prereg_stage1_2026-09-25.csv` (Stage-1 output) for R0.

```
cd ~/pennylane_gpu_mock_test
python -c "import hashlib;print(hashlib.sha256('\n'.join(l.rstrip() for l in open('xor_prereg_stage1b_sweep.py',encoding='utf-8').read().strip().splitlines()).encode()).hexdigest())"
python -u xor_prereg_stage1b_sweep.py 2>&1 | tee ~/xor_prereg_stage1b_run.txt
```

Output: `~/xor_prereg_stage1b_2026-09-25.csv` (336 rows) and the scoring at
the end of the log. Expected run time about 15-20 minutes (estimated from the
stub dry run, not measured on the pod).
