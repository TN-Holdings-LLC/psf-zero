# Addendum 168 -- Stage 1 results: the noisy XOR rehearsal across 21 fake IBM backends (2026-09-25)

> **Imported into the home series as Addendum 168.** Written at the workplace, run on a RunPod pod (RTX 4090), original file `xor-real-device-stage1-results-2026-09-25.md`; body below unchanged. Re-checked at home from the raw CSV (`xor_prereg_stage1_2026-09-25.csv`): the 30 sign flips (10 each in FakeCusco A, FakeKyoto A, FakeKyoto B), FakeBrussels B's per-input qubits (16 for input 00, 28 for the other three), FakeCusco A's constant +0.019, the 7 backends whose B layout varies by input, and the 3 backends where B is worse than A by more than 0.03 -- all as stated. The home assistant made the same first-row-only error as this record's own correction note describes: during a chat analysis it called FakeBrussels B "not explained by readout" after looking only at each group's first input. It is explained by readout.

**Scored against:** [`xor-real-device-stage1-preregistration-2026-09-25.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-167-preregistration-2026-09-25.md)
(locked in the Project before this run). Thresholds are applied exactly as
written there. Everything under "Post-hoc diagnostics" was looked at after
seeing the results and is exploratory, not part of the scoring.

**Run:** RunPod pod (the GPU was not used; all simulation is CPU Aer),
`python -u xor_prereg_stage1_sweep.py 2>&1 | tee ~/xor_prereg_stage1_run.txt`.
Script integrity on the pod checked against the pre-registered normalized
SHA-256. Environment printed by the script: Linux-6.8.0-64-generic-x86_64,
Python 3.12.3, Qiskit 2.5.2, qiskit-aer 0.17.2, qiskit-ibm-runtime 0.50.0,
PennyLane 0.45.1. 840 cells (21 backends x 2 arms x 5 seeds x 4 inputs). No
timing measured.

## 1. Harness checks

- Circuit check: exact <Z0> = [-0.99776, 0.99776, 0.99776, -0.99776],
  `blocks` = [0, 0, 0, 0]. Passed.
- Backends: 21 included (FakeAachen, Berlin, Boston, Brisbane, Brussels,
  Cusco, Fez, Kawasaki, Kingston, Kyiv, Kyoto, Marrakesh, Miami, Nighthawk,
  Osaka, Pittsburgh, Quebec, Sherbrooke, Strasbourg, Torino, WashingtonV2);
  48 excluded (47 below 100 qubits, `FakeProviderForBackendV2` not a
  backend). The package warns that FakeNighthawk's properties "are not
  intended to represent typical nighthawk error values".
- C0 (FakeBrisbane, arm A, seed 0 versus the recorded rehearsal):

| input | harness | rehearsal | tolerance | |
|---|---:|---:|---:|---|
| 00 | -0.9000 | -0.9040 | 0.0287 | OK |
| 01 | 0.9105 | 0.9185 | 0.0265 | OK |
| 10 | 0.9065 | 0.9085 | 0.0280 | OK |
| 11 | -0.9095 | -0.9130 | 0.0274 | OK |

C0 passed, so the predictions were scored.

## 2. Scoring

| Prediction | Verdict | Numbers |
|---|---|---|
| P1 all four inputs correct in every cell group | **REFUTED** | 30 sign flips |
| P2 | withdrawn before locking | not scored |
| P3 seed spread <= 0.03 | CONFIRMED (vacuously, see below) | max spread 0.0000 |
| P4 shot noise binomial, pooled ratio in [0.85, 1.15] | CONFIRMED | pooled ratio 0.986 (per input 1.025, 1.020, 0.958, 0.940) |
| P5 `blocks` = 0 everywhere | CONFIRMED | 0 nonzero cells |
| P6 free layout never worse by > 0.01, better on >= half | **REFUTED** | min (B - A) = -0.3124; share better by > 0.01 = 0.524 |

**P3 is confirmed only vacuously.** For every backend and both arms, the
five `seed_transpiler` values gave the same M to three decimals: for this
small circuit the seed did not change the routed circuit at all. The result
shows that seeds do not matter *for this circuit*, not that seed sensitivity
is small in general.

## 3. Mean margin M per backend (identical for all five seeds)

| backend | 2q gate | arm A (pinned 0-3) | arm B (free layout) | B - A |
|---|---|---:|---:|---:|
| FakeAachen | cz | 0.969 | 0.942 | -0.027 |
| FakeBerlin | cz | 0.942 | 0.962 | +0.020 |
| FakeBoston | cz | 0.971 | 0.985 | +0.014 |
| FakeBrisbane | ecr | 0.907 | 0.934 | +0.027 |
| FakeBrussels | ecr | 0.927 | 0.832 | -0.095 |
| FakeCusco | ecr | 0.000 | 0.922 | +0.922 |
| FakeFez | cz | 0.918 | 0.969 | +0.051 |
| FakeKawasaki | ecr | 0.901 | 0.935 | +0.034 |
| FakeKingston | cz | 0.968 | 0.656 | -0.312 |
| FakeKyiv | ecr | 0.950 | 0.936 | -0.014 |
| FakeKyoto | ecr | 0.000 | 0.004 | +0.004 |
| FakeMarrakesh | cz | 0.961 | 0.962 | +0.001 |
| FakeMiami | cz | 0.956 | 0.943 | -0.013 |
| FakeNighthawk | cz | 0.968 | 0.978 | +0.010 |
| FakeOsaka | ecr | 0.918 | 0.941 | +0.023 |
| FakePittsburgh | cz | 0.942 | 0.982 | +0.040 |
| FakeQuebec | ecr | 0.827 | 0.961 | +0.134 |
| FakeSherbrooke | ecr | 0.930 | 0.936 | +0.006 |
| FakeStrasbourg | ecr | 0.947 | 0.899 | -0.048 |
| FakeTorino | cz | 0.642 | 0.962 | +0.320 |
| FakeWashingtonV2 | cx | 0.840 | 0.944 | +0.104 |

(Seed-0 values read from the CSV and rounded to three decimals; B - A
computed from these rounded values. Every cell has 9 routed 2-qubit gates in both arms: the arms
differ only in which physical qubits are used.)

## 4. Post-hoc diagnostics (exploratory, not scored)

Checked after the results, by reading the CSV's `q0_physical` and
`readout_error` columns on the pod and the packaged fake-backend calibration
data (qiskit-ibm-runtime 0.50.0) in the workplace sandbox.

**Where the 30 sign flips are.** Every cell group except three has M >= 0.642
and seed-identical values, so flips can only occur in FakeCusco arm A,
FakeKyoto arm A and FakeKyoto arm B, all at chance level (M 0.000 to 0.004).
With identical seeds, 30 flips = 5 seeds x 6 flips among those 12 input
cells, about half, as expected at chance. (Deduced from the per-group
values; to be confirmed by counting in the CSV.)

**FakeCusco, arm A:** the pinned layout measures physical qubit 0, whose
readout error is **0.5** in the snapshot, i.e. a coin flip. Arm B used
qubit 2 (readout 0.0073) and reached 0.922.

**FakeKyoto, both arms:** in the packaged snapshot **all 144 ECR entries
have error exactly 1.0**, so every 2-qubit gate is fully depolarizing and
every layout fails. The snapshot is unusable as a device model. The
pre-registered selection rule (error values present) did not exclude it,
because the values are present, they are just 1.0.

**FakeTorino, arm A:** qubit 0 readout error 0.167; M 0.642 is close to what
readout alone would do (a factor 1 - 2 x 0.167 = 0.67 on about 0.96).

**FakeKingston, arm B:** the free layout put logical qubit 0 on physical
qubit 1, readout error **0.168**, giving M 0.656; arm A measured qubit 0
(readout 0.0095) and reached 0.968. Likely reason (hypothesis, not tested):
the circuit given to `transpile` had no measurement, so the measured qubit's
readout error did not enter layout scoring. Qiskit's `VF2PostLayout` in
strict-direction mode scores the circuit's own instructions against the
Target; a readout error only counts if a `measure` is in the circuit. The
rehearsal pipeline also routes first and adds measurement afterwards.

**Smaller arm-B losses:** FakeBrussels (-0.095), FakeStrasbourg (-0.048) and
FakeAachen (-0.027) also measured a qubit with higher readout error than arm
A's qubit 0 (0.0186 vs 0.0146, 0.028 vs 0.016, 0.0236 vs 0.0077). For
Brussels that difference (about 0.008 in expected margin) does not explain a
0.095 loss; the rest is unexplained here (gate errors on the chosen qubits
were not inspected).

**"Faulty" flags do not catch this.** `properties().faulty_qubits()` returned
an empty list for FakeCusco, FakeKyoto, FakeTorino, FakeKingston and
FakeBrussels. A check for flagged faulty qubits would not have warned about
any of the failures above; the error values themselves have to be checked.

## 5. What this means for 2026-09-28 (input to Stage 2, not a scored claim)

1. Neither fixed routing is safe. The pinned layout failed badly on 3 of 21
   backends (Cusco, Kyoto, Torino); the free layout on 2 (Kyoto, Kingston).
2. Correctness is robust to gradual noise (all groups with M >= 0.64 were
   fully correct) but collapses to chance when a single bad element sits in
   the circuit (readout 0.5, or 2-qubit error 1.0). Addendum 150's "accuracy
   is very robust to noise" holds for gradual noise, not for broken parts.
3. Before submitting, the chosen qubits' calibration must be checked directly
   (readout and 2-qubit errors), and candidate layouts compared by simulating
   them with an Aer noise model built from that day's calibration.
4. Measurement should be inside the circuit before layout selection, so the
   layout pass sees the readout error (to be tested; see the proposed Stage 1b).
5. Shot noise follows the binomial model (P4): at 4,000 shots the standard
   error per input is about 0.007 near |<Z0>| = 0.9, and Stage-2 tolerances
   can be set from that.

## 6. Files

| File | What it is |
|---|---|
| [`xor_prereg_stage1_2026-09-25.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/xor_prereg_stage1_2026-09-25.csv) | raw data, 840 rows (on the pod; to be added to `data/` after download) |
| `xor_prereg_stage1_run.txt` | run log (on the pod) |
| [`xor_prereg_stage1_sweep.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/xor_prereg_stage1_sweep.py) | the locked script |
| [`xor-real-device-stage1-preregistration-2026-09-25.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-167-preregistration-2026-09-25.md) | the pre-registration scored here |

Pre-publication check: the CSV's environment columns contain only the
platform string and `x86_64`; no local paths or account names were written by
the script. To be re-checked by grep when the file is downloaded.

---

> **Correction, 2026-09-25 -- two statements above were based on too narrow a
> look at the data.** Found when the full CSV (840 rows) was downloaded from
> the pod and every number above was recomputed from the raw rows. The text
> above is left as written; this note supersedes the two points below.
>
> 1. *"For Brussels that difference ... does not explain a 0.095 loss; the
>    rest is unexplained here."* -- **Wrong.** The earlier check printed only
>    the first input's row per backend and arm. In arm B each input circuit is
>    transpiled separately, and on FakeBrussels input 00 was placed on
>    physical qubit 16 (readout 0.0186, <Z0> -0.943) but inputs 01, 10 and 11
>    on physical qubit **28 (readout 0.0916)**, giving |<Z0>| 0.78 to 0.80.
>    That readout error accounts for the loss (a factor 1 - 2 x 0.0916 = 0.82
>    on about 0.97). FakeBrussels is therefore a second case of the same
>    pattern as FakeKingston: the free layout measured a qubit with poor
>    readout. FakeStrasbourg (qubit 47, readout 0.028, all inputs) and
>    FakeAachen (qubit 134, readout 0.0236, all inputs) fit the same pattern at
>    a smaller size.
> 2. *"the seed did not change the routed circuit at all"* (P3) -- stronger
>    than what was checked. What the CSV shows is that, for every backend, arm
>    and input, **the measured physical qubit and the sampled <Z0> were
>    identical across all five seeds**. Identical sampled values under the same
>    simulator seed strongly suggest identical routed circuits, but the
>    circuits themselves were not compared.
>
> Also observed in the full CSV (not stated above): in arm B, different inputs
> can land on different physical qubits on the same backend (FakeBrussels,
> Cusco, Kawasaki, Kyiv, Kyoto, Quebec, Sherbrooke), because each input
> circuit gets its own layout. On 2026-09-28 the four inputs may therefore run
> on different qubits unless one layout is fixed for all four.
>
> Recomputed from the raw CSV (matches the script's own scoring): 840 rows,
> 21 backends; sign flips = 30, exactly 10 each in FakeCusco arm A, FakeKyoto
> arm A and FakeKyoto arm B (this confirms the deduction in section 4); P3
> max spread 0.0; P6 min (B - A) = -0.3124, share better by > 0.01 at seed 0 =
> 0.524, backends where B is worse by > 0.03: FakeBrussels, FakeKingston,
> FakeStrasbourg; `blocks` nonzero in 0 cells; 9 routed 2-qubit gates in every
> cell. In FakeCusco arm A all four inputs read <Z0> = +0.019 (readout 0.5
> makes the outcome independent of the state), so the two label -1 inputs
> flip and the two label +1 inputs are "correct" by chance.
>
> The CSV is now in the Project as [`psf-zero/data/xor_prereg_stage1_2026-09-25.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/xor_prereg_stage1_2026-09-25.csv)
> (143,823 bytes as received). Pre-publication grep for account names, local
> paths and host names: 0 hits.
