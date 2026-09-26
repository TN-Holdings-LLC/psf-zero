# Addendum 172 -- Stage 1c results: NOT SCORED (R0 failed); exploratory readout of PSF-Zero versus Qiskit on circuits where PSF-Zero acts (2026-09-25)

> **Imported into the home series as Addendum 172.** Written at the workplace, run on a RunPod pod (RTX 4090), original file `xor-real-device-stage1c-results-2026-09-25.md`; body below unchanged. Re-computed at home from `xor_prereg_stage1c_2026-09-25.csv`: P and Z structurally identical in 105/105 cells, |dTVD| > 0.002 in 8 cells (max 0.0043), P smaller than A in both size and depth in 40/105, mean TVD_A - TVD_P = +0.00003, and the per-arm mean sizes and depths -- all as stated.

**Official outcome: the pre-registered predictions C1-C7 are NOT scored.**
The pre-registration ([`xor-real-device-stage1c-preregistration-2026-09-25.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-171-preregistration-2026-09-25.md))
made exact replication of Addendum 156 (R0) a condition for scoring, and R0
failed. The cause is identified (section 2) and is a design error in the
pre-registration, not a property of the compilers. Everything in sections 3
and 4 is **exploratory**: the locked thresholds are shown for reference as
"would be" verdicts, not as pre-registered results.

**Run:** RunPod pod, `python -u xor_prereg_stage1c_sweep.py 2>&1 | tee
~/xor_prereg_stage1c_run.txt`. All PSF-Zero blocks passed the real
`lightning.gpu` check (RTX 4090) and the operator-equivalence check. 440 rows
(22 backends x 5 tapes x 4 arms). CSV recomputed in the workplace sandbox.
No timing measured.

## 1. R0 result

Addendum 156's path on FakeManilaV2 reproduced the repository CSV exactly in
**every structural field** for all 10 (tape, arm) pairs: routed 2-qubit
gates 6, depth 23 (arm A) and 16 (PSF-Zero), size 84 and 56, PSF-Zero
fallbacks 0. It did **not** reproduce `tvd_noisy` or `tvd_ideal` (20
mismatches). Since `tvd_ideal` also differs, the difference lies in the
circuits' content (the exact output distribution), not in noise sampling.

## 2. Cause of the R0 failure (identified; a design error)

[`data/compare_with_without_psf_2026-09-24.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/compare_with_without_psf_2026-09-24.csv) (Addendum 156) was produced
on the evening of 2026-09-24, **before** the fix to the 2-qubit
`QubitUnitary` bit order in `tape_to_qiskit` (Addenda 164-166 in the home
record). The repository on the pod contains the fixed conversion, so the
same tapes now become circuits whose 2-qubit unitaries have the correct,
different qubit order: same gate counts, different distributions.

Evidence: in the pre-lock sandbox dry run, which used the workplace copy of
the prototypes (the pre-fix conversion), `tvd_ideal` reproduced the
repository CSV on all five tapes (0.014531, 0.019912, 0.021032, 0.023834,
0.021823). On the pod, with the fixed conversion, it does not (0.016263,
0.022804, 0.018694, 0.020174, 0.021909).

Choosing a reference produced by pre-fix code as an exact-replication gate
was a mistake in the Stage-1c pre-registration. It does not bear on the
compression question (both versions synthesize random 2-qubit unitaries,
and the structural numbers match exactly), but the protocol said "not
scored", so it is not scored. A formal re-test would need a new
pre-registration with a reference regenerated from the fixed code.

## 3. Exploratory readout (105 cells per arm; FakeKyoto excluded)

| Check (locked threshold) | Observed | "Would be" |
|---|---|---|
| C1 PSF-Zero fallbacks = 0 | 0 on all tapes | confirmed |
| C2 routed 2q = 6 in A, Z, P | 6 in every cell | confirmed |
| C3 routed 2q = 6 in T (Qiskit opt 3) | 6 in every cell | confirmed |
| C4 P = Z: size and depth equal, \|dTVD\| <= 0.002 | size, depth and 1q count equal in 105/105; \|dTVD\| > 0.002 in 8 cells (max 0.0043, mean 0.0005) | refuted by the TVD clause |
| C5 P smaller than A in size and depth | in 40 of 105 | refuted |
| C6 mean(TVD_A - TVD_P) <= 0.01 | +0.00003; P better in 47 of 105 | confirmed |
| C7 XOR untouched by PSF-Zero | 0 blocks and identical circuit, 4 of 4 | confirmed |

(C7 does not depend on R0; it is listed here only because the protocol
suspends all scoring.)

**Size and depth, mean over tapes, by native 2-qubit gate** (every backend
gave the same numbers on all five tapes):

| backends | A size / depth | Z | P (PSF-Zero) | T (Qiskit opt 3) |
|---|---|---|---|---|
| cx (FakeManilaV2, FakeWashingtonV2) | 84 / 23 | 56 / 16 | 56 / 16 | 56 / 16 |
| cz (10 Heron-class) | 76 / 22 | 72 / 23 | 72 / 23 | **64 / 19** |
| ecr, group 1 (Brisbane, Osaka, Quebec, Strasbourg) | 84 / 23 | 74 / 21 | 74 / 21 | 80 / 23 |
| ecr, group 2 (Brussels, Kyiv) | 75 / 23 | 72 / 21 | 72 / 21 | 80 / 23 |
| ecr, group 3 (Cusco, Kawasaki, Sherbrooke) | **66 / 19** | 70 / 19 | 70 / 19 | 80 / 23 |

Mean over all 105 cells: size A 76.8, Z 70.6, P 70.6, T 70.1; depth A 22.0,
Z 21.2, P 21.2, T 20.4. Arm T's size was no larger than any of A, Z, P in
60 of 105 cells.

**Noisy score (mean TVD, lower is better):** cx backends A 0.0342, Z 0.0322,
P 0.0318, T 0.0308; cz backends 0.0126, 0.0126, 0.0127, 0.0123; ecr backends
0.0260, 0.0262, 0.0264, 0.0251. T had a lower TVD than P in 61 of 105 cells
(mean difference 0.0008).

## 4. What this suggests (exploratory; consistent with Addenda 156, 157 and 159)

1. **"Size 56, depth 16, 0 fallbacks" reproduces**, but only on backends
   whose native 2-qubit gate is CX, and Qiskit reaches exactly the same
   numbers there both with `euler_basis="ZSX"` and with its full
   `optimization_level=3` pipeline.
2. **PSF-Zero's circuits are structurally identical to Qiskit ZSX's** (same
   size, depth and single-qubit count in all 105 cells), as Addendum 159
   reported. Small TVD differences in 8 cells suggest the gate parameters are
   not bit-identical; that was not investigated.
3. **No synthesizer reduces the 2-qubit count** (6 everywhere), and
   **Qiskit's own full pipeline inserts no SWAPs** on this workload. The
   "Qiskit bloats the circuit with SWAPs" hypothesis finds no support here.
4. **On the 2026-09-28 device class (CZ and ECR backends), PSF-Zero is not
   consistently smaller than Qiskit's default decomposer** (larger depth on
   all 10 CZ backends, larger size on three ECR backends), and Qiskit's full
   pipeline is the smallest on CZ backends.
5. **No noisy-score advantage** for PSF-Zero over any Qiskit arm.
6. The XOR circuit is untouched by PSF-Zero (C7), so no 2026-09-28 result can
   be attributed to PSF-Zero's compression.

These points are exploratory under this protocol. They agree with the
existing record rather than overturn it.

## 5. Files

| File | What it is |
|---|---|
| [`psf-zero/data/xor_prereg_stage1c_2026-09-25.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/xor_prereg_stage1c_2026-09-25.csv) | raw data, 440 rows (23,611 bytes as received) |
| [`psf-zero/benchmarks/xor_prereg_stage1c_sweep.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/xor_prereg_stage1c_sweep.py) | the locked script (the pod run's hash check was not shown; not confirmed -- see the update below) |
| [`xor-real-device-stage1c-preregistration-2026-09-25.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-171-preregistration-2026-09-25.md) | the pre-registration |

Pre-publication grep of the CSV: 0 hits.

> **Update (2026-09-25): script integrity confirmed.** A post-hoc check of
> `/root/pennylane_gpu_mock_test/xor_prereg_stage1c_sweep.py` on the pod
> returned the pre-registered normalized SHA-256
> `4d342727ffafddf635a30488bee618155431be077c75a42cac6b7eb36f7bf195`. The file
> that ran is the locked script. (This does not change the outcome: R0 still
> failed and C1-C7 remain unscored.)
