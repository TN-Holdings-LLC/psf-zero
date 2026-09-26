# Addendum 170 -- Stage 1b results: putting the measurement into the circuit before layout selection fixes the poor-readout failures (2026-09-25)

> **Imported into the home series as Addendum 170.** Written at the workplace, run on a RunPod pod (RTX 4090), original file `xor-real-device-stage1b-results-2026-09-25.md`; body below unchanged. Q1-Q6 and the selection counts re-computed at home from `xor_prereg_stage1b_2026-09-25.csv`: all match, except Q6's minimum, -0.0007 here versus -0.0008 recomputed at home -- a rounding difference around -0.00075; the verdict (bound -0.02) is unaffected.

**Scored against:** [`xor-real-device-stage1b-preregistration-2026-09-25.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-169-preregistration-2026-09-25.md)
(locked in the Project before this run). Thresholds applied exactly as
written. Every verdict below was recomputed from the raw CSV in the
workplace sandbox and matches the script's own scoring. Section 4 is
exploratory.

**Run:** RunPod pod, CPU Aer only, same environment as Stage 1 (Qiskit 2.5.2,
qiskit-aer 0.17.2, qiskit-ibm-runtime 0.50.0, PennyLane 0.45.1),
`python -u xor_prereg_stage1b_sweep.py 2>&1 | tee ~/xor_prereg_stage1b_run.txt`.
336 cells (21 backends x 4 arms x 4 inputs, transpiler seed 0). No timing
measured.

**Script integrity:** the pre-run hash check does not appear in the pasted
log, so it is **not confirmed** at the time of writing. R0 below (exact
reproduction of all 168 arm-A/B values from Stage 1) shows the shared code
path behaved identically, but does not cover the new arms' code. A post-hoc
hash check of the file that ran has been requested.

> **Update (2026-09-25): script integrity confirmed.** The post-hoc check of
> `/root/pennylane_gpu_mock_test/xor_prereg_stage1b_sweep.py` on the pod
> returned the pre-registered normalized SHA-256
> `ec7f173cb36fa071b9dbd03de40ead48947edbbc955f6781369c2d8e71105d8e`. The
> file that ran is the locked script, including the new arms and scoring.

## 1. Checks

- **R0 (arms A and B reproduce Stage 1, seed 0, exactly): 0 mismatches** out
  of 168 values. Passed; predictions scored.
- Backends: the same 21 as Stage 1; FakeKyoto excluded from scoring as
  pre-registered (all 144 ECR errors are 1.0). All four FakeKyoto arms were at
  chance (M 0.000 to 0.004), as expected.

## 2. Scoring (20 scored backends)

| Prediction | Verdict | Numbers |
|---|---|---|
| Q1 measured-qubit readout <= 0.05 in M1 and M4 | CONFIRMED | max 0.0206 (FakeStrasbourg) |
| Q2 M(M1) >= max(M(A), M(B)) - 0.02 | CONFIRMED | min difference -0.0155 (FakeStrasbourg) |
| Q3 M(M1) >= 0.90 on FakeKingston and FakeBrussels | CONFIRMED | 0.9815 and 0.9293 (Stage-1 arm B: 0.656 and 0.832) |
| Q4 \|M(M4) - M(M1)\| <= 0.02 | CONFIRMED | max 0.0017 (FakeCusco) |
| Q5 no sign flip in M1, M4 or the selected arm | CONFIRMED | 0 flips |
| Q6 selected arm >= best arm - 0.02 | CONFIRMED | min difference -0.0007 (FakeAachen) |

All six predictions confirmed.

## 3. Mean margin M per backend (seed 0)

| backend | A (pinned) | B (free, measure added after) | M1 (measure-aware) | M4 (measure-aware, shared layout) | selected |
|---|---:|---:|---:|---:|---|
| FakeAachen | 0.969 | 0.942 | 0.986 | 0.986 | M1 |
| FakeBerlin | 0.942 | 0.962 | 0.977 | 0.977 | M4 |
| FakeBoston | 0.971 | 0.984 | 0.987 | 0.987 | M1 |
| FakeBrisbane | 0.907 | 0.934 | 0.919 | 0.919 | B |
| FakeBrussels | 0.927 | 0.832 | 0.929 | 0.929 | M1 |
| FakeCusco | 0.000 | 0.922 | 0.912 | 0.911 | B |
| FakeFez | 0.918 | 0.969 | 0.970 | 0.970 | M1 |
| FakeKawasaki | 0.902 | 0.935 | 0.958 | 0.958 | M1 |
| FakeKingston | 0.968 | 0.656 | 0.981 | 0.981 | M1 |
| FakeKyiv | 0.950 | 0.936 | 0.963 | 0.963 | M1 |
| FakeKyoto (not scored) | 0.000 | 0.004 | 0.000 | 0.000 | B |
| FakeMarrakesh | 0.961 | 0.962 | 0.981 | 0.981 | M4 |
| FakeMiami | 0.956 | 0.943 | 0.969 | 0.969 | M1 |
| FakeNighthawk | 0.968 | 0.978 | 0.985 | 0.985 | M1 |
| FakeOsaka | 0.918 | 0.941 | 0.958 | 0.958 | M1 |
| FakePittsburgh | 0.942 | 0.982 | 0.985 | 0.985 | M4 |
| FakeQuebec | 0.827 | 0.961 | 0.960 | 0.961 | B |
| FakeSherbrooke | 0.931 | 0.936 | 0.942 | 0.942 | M1 |
| FakeStrasbourg | 0.947 | 0.899 | 0.931 | 0.932 | A |
| FakeTorino | 0.642 | 0.962 | 0.962 | 0.962 | B |
| FakeWashingtonV2 | 0.840 | 0.944 | 0.942 | 0.941 | B |

(From the run log's table; every value checked against the raw CSV, all within rounding.)

## 4. Post-hoc observations (exploratory, not scored)

- **M1 beats both A and B by more than 0.01 on 8 of 20 backends** (Aachen,
  Berlin, Kawasaki, Kingston, Kyiv, Marrakesh, Miami, Osaka). Where it is not
  the best, it is behind by less than 0.016.
- **The measure-aware layout chose a qubit with readout at least as good as
  pinned qubit 0 on 17 of 20 backends.**
- **Shared layout (M4):** every input ran on the same physical qubit on every
  backend. In M1, only FakeCusco put one input (01) on a different qubit.
  M4 costs at most 0.0017, so one layout for all four inputs is essentially
  free here.
- **Selection by simulation** chose M1 on 11 backends, B on 5, M4 on 3 and A
  on 1, and was never more than 0.0007 below the best arm. As pre-registered,
  on fake backends this tests the procedure, not predictive power on hardware.
- **Prediction versus result spread was smaller than shot noise alone
  predicts.** On 255 de-duplicated (backend, qubit, input) cells, the
  standardized difference between the 4,000-shot result (simulator seed 0)
  and the 20,000-shot prediction (seed 1000) had SD 0.76 rather than about 1,
  with no value beyond 3. Not explained here (one candidate is correlation
  between Aer's random streams for different seeds; not checked). It does not
  affect any verdict, and Stage 1's P4 (50 independent repeats, pooled ratio
  0.986) remains the pre-registered shot-noise result. If anything, Stage-2
  tolerances set from the binomial model are conservative.

## 5. Consequences for Stage 2 (2026-09-28), to be fixed in the Stage-2 pre-registration

1. Default candidate: **M4** (measure-aware layout chosen with the
   measurement in the circuit, one layout for all four inputs). It fixed both
   Stage-1 failure types in this test and costs nothing measurable versus M1,
   while keeping all four inputs on the same qubits.
2. Final choice: simulate A, B, M1 and M4 with an Aer noise model built from
   the chosen device's calibration at submission time, and submit the best,
   as in Q6. The calibration snapshot and the predicted values are saved with
   the job IDs before results are read.
3. Measure only logical qubit 0 on hardware, as in all Stage-1/1b arms (the
   design choice these results are about), unless Stage 2 gives a stated
   reason to change it.
4. What these results cannot say: how well a calibration snapshot predicts a
   real device on the day. That is the question 2026-09-28 answers.

## 6. Files

| File | What it is |
|---|---|
| [`psf-zero/data/xor_prereg_stage1b_2026-09-25.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/xor_prereg_stage1b_2026-09-25.csv) | raw data, 336 rows (19,826 bytes as received) |
| [`psf-zero/benchmarks/xor_prereg_stage1b_sweep.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/xor_prereg_stage1b_sweep.py) | the locked script |
| [`xor-real-device-stage1b-preregistration-2026-09-25.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-169-preregistration-2026-09-25.md) | the pre-registration scored here |

Pre-publication grep of the CSV for account names, local paths and host
names: 0 hits.
