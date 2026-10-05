# Workplace results: QML-2 "DEPTH", stage 1 (pre-registration `depth1-preregistration-2026-10-05.md`). P0 FAILED on a too-strict exactness threshold, so nothing is scored. Described: depth stops paying in margin at L = 2-4 under noise, but accuracy hardly moves (noise costs at most 2 test points up to 286 two-qubit gates); compilers change margin, and on FakeTorino 4-qubit rings by up to 21%, but never accuracy by 2 test points. The scorer's gate reads NO-GO, so no GPU stage is proposed on this evidence (2026-10-05)

**Status: results of the workplace pre-registration of 2026-10-05.**

- **Lock:** Project save time of the pre-registration, before the run (started 02:27 UTC).
- **Scoring:** by the locked `depth_eval.py score`, and re-checked by `depth_verify.py`.
  - `depth_verify.py` was written after the lock, during the run. When it was written, only the fine-tuning log
    lines of DEP and FT0 had been seen, and no deployment output.
  - It agrees on P0 and on every number below.
- **Setting:** workplace sandbox, 2 processes; 4 + 24 + 4 jobs, 106 minutes (02:27-04:13 UTC); 13,392 deployed
  circuits.
- **Provenance:** every file records `depth_eval.py` normalized SHA-256 `82488d96…` and c12 `1dc9b9b1…`, as locked.

## 1. P0 failed: no prediction is scored

**What failed.** Every compiled circuit's noiseless z was required to be within 1e-6 of the logical z.

- **How many rows exceeded it:** 2 of 13,392, by 1.03e-5 and 8.7e-6.
- **Which rows:** both are L3T (Qiskit level 3 with the Target and `approximation_degree=1.0`), on FakeTorino, D38,
  n = 6, L = 16 (286-289 two-qubit gates).
- **The other arms:** RPSF max 2.4e-13; C12 max 2.3e-11.
- **Also checked:** the reduced simulation equalled the whole-device one exactly on all 288 checked circuits.

**The cause is the threshold, not the harness.**

- Qiskit's exact path specialises each two-qubit block to an average gate fidelity of 1 - 1e-9. Amplitude errors of
  about 3e-5 per block are therefore allowed.
- Over about 300 blocks, a z deviation of 1e-5 is within that budget. As a state infidelity it is of order 1e-10, far
  below B17's 1e-6 line.
- A threshold of 1e-6 on z was too strict for deep L3T circuits. This is a design error of the pre-registration, as
  in Addendum 301.
- **The lesson:** state the exactness check as a state infidelity (1e-6), not as an absolute z difference.

**What follows from the pre-registration.** Nothing below P0 is scored. The verdict lines the locked score printed
are not results:

| H1 | H2 | H3 | H4 | H5 | H6 | Gate |
|---|---|---|---|---|---|---|
| CONFIRMED | CONFIRMED | AMBIGUOUS | CONFIRMED | AMBIGUOUS | AMBIGUOUS | NO-GO |

The 2 rows change L3T's cell means by less than 1e-6. The description below is therefore not affected, but it is not
a scored test.

## 2. Descriptive results

The full table (accuracy, shot accuracy, margin, flip rate and two-qubit count per dataset, n, L, arm and device) is
in `score.md`.

**Margin under noise peaks early.** On FakeAuckland, n = 6:

| dataset | best margin (L) | margin at L = 16 | ideal margin at L = 16 |
|---|---|---|---|
| BC | 0.564-0.568 (L = 2) | 0.153-0.155 | 0.644 |
| D38 | 0.604-0.614 (L = 2) | 0.130-0.135 | 0.674 |

Noise removes about 77-80% of the margin at L = 16, against 9-11% at L = 2.

**Accuracy hardly moves.**

- **What noise costs:** shot-based accuracy (4,000 shots, readout) minus ideal accuracy, on FakeAuckland, n = 6, is
  at most 1.8 test points (2 without shots) in any cell, at L = 2 as at L = 16. In D38, L = 16, the noisy
  accuracy is even above the ideal.
- **Prediction flips** (noisy sign different from the noiseless sign): at most 2.8% in any cell. Pooled: FakeAuckland
  0.43%, FakeTorino 0.36-0.39%.
- **H2's "CONFIRMED" line is confounded.** On BC, n = 6, shot accuracy is 0.914-0.917 at L = 16 against 0.947 at
  L = 12. But the noiseless model itself drops from 0.956 to 0.930 between those depths, which is one training run's
  variation. The noise accounts for only about 1.5-1.8 of the 3-4 points. As a statement about noise, H2 would not
  have held.

**Compilers change the margin, not the accuracy.**

| device | C12 - RPSF pooled margin | C12 - L3T pooled margin |
|---|---|---|
| FakeAuckland | +0.0049 | +0.0062 |
| FakeTorino | +0.0187 | +0.0016 |

- **Where the gain comes from:** almost all of FakeTorino's gain is the 4-qubit ring. There the guarded call routes
  with 20% more two-qubit gates (188 against 157 at L = 16), which the recommended call and L3T avoid. Margin at
  L = 16: RPSF 0.368 / 0.357 against C12 0.447 / 0.431 (BC / D38), that is +21%.
- **6-qubit rings on FakeTorino:** C12 kept the guarded call's circuit, so the two arms are identical.
- **Accuracy:** in no cell does the shot accuracy differ between C12 and RPSF by 2 test points. This is why the gate
  reads NO-GO.

**Fine-tuning (BC, n = 6, FakeAuckland, C12; two seeds)** — change in deployed margin from DEP:

| L | FTN (through the noise) | FT0 (noiseless control) |
|---|---|---|
| 4 | +0.006 | -0.011 |
| 12 | -0.002 | -0.037 |

- With 40 SPSA steps at a = 0.3, noiseless fine-tuning lowered the deployed margin. Fine-tuning through the noise
  did not: relative to the control, +0.035 at L = 12.
- But FTN did not raise the margin over θ*. Accuracy changed by at most 1 test point either way.

## 3. Exploratory (not pre-registered; written after the run): fewer shots

The recorded noisy z values were re-sampled at fewer shots (200 repetitions, each point's readout error, the same
random numbers in every arm). Script: `depth_shots_explore.py`.

Pooled shot accuracy over datasets, n and L >= 8:

| device, n | 32 shots: RPSF / C12 / L3T | 100 shots | 4,000 shots |
|---|---|---|---|
| FakeAuckland, 4 | 0.889 / 0.893 / 0.887 | 0.935 / 0.936 / 0.935 | 0.946 / 0.948 / 0.947 |
| FakeAuckland, 6 | 0.840 / 0.841 / 0.841 | 0.913 / 0.914 / 0.913 | 0.948 / 0.947 / 0.948 |
| FakeTorino, 4 | **0.922 / 0.934 / 0.934** | 0.945 / 0.947 / 0.947 | 0.952 / 0.952 / 0.952 |
| FakeTorino, 6 | 0.899 / 0.899 / **0.914** | 0.939 / 0.939 / 0.942 | 0.953 / 0.953 / 0.953 |

**Reading:**

- The extra margin turns into accuracy only when shots are scarce: up to 1.2-1.5 points at 32 shots.
- The shot budget itself matters far more than the compiler: 5-11 points between 32 and 4,000 shots.

## 4. Reading

- **For the owner's question** ("does the AI get smarter through these circuits?"):
  - Up to 6 qubits and 286 two-qubit gates on these fake devices, the classifier's accuracy is robust to noise.
    Making it deeper keeps paying in noiseless accuracy roughly up to L = 8-12.
  - Noise eats its confidence (margin) from L = 4 on, and by L = 16 most of it.
- **What compilers buy here:** confidence, not correctness.
  - The gain is large where routing differs (Heron 4-rings: +21% margin).
  - It is small elsewhere.
  - It becomes accuracy only in a shot-starved regime.
- **Against L3T:** C12 is level (+0.002 to +0.006 pooled margin).
- **Fine-tuning through the noise** avoided the damage that the same amount of noiseless fine-tuning did. It did not
  improve on θ*.
- **Stage 2 (GPU):** the scorer's gate reads NO-GO, and it is unscored only because of the threshold error. The data
  support not renting a GPU for more of the same question.

## 5. What would be worth doing next (proposals, not run)

1. **A shot-budget test, pre-registered:** fixed total shots per prediction (32-256). The compiler's margin should then
   show in accuracy. Cheap on CPU, and it reuses this design.
2. **Noise-aware training from scratch** (not fine-tuning), with a loss that rewards margin under noise. FTN's
   damage-free behaviour hints at something, but 40 steps did not show a gain.
3. **For stage 2 to be worth a GPU:** a circuit regime where noise flips predictions at full shots. On this evidence
   that needs either noisier devices (ecr, Addendum 293) or several hundred two-qubit gates beyond L = 16. A cheap CPU
   check first: n = 4-6 at L = 24-32.

## 6. What this does not establish

- Hardware.
- More than 6 logical qubits.
- Other models.
- One noiseless training per cell, two fine-tuning seeds.

## 7. Files

| folder | contents |
|---|---|
| `scored/` | `env.txt`, `progress.txt`, logs, 4 train, 24 deploy and 4 fine-tune JSON files, `score.md`, `score_log.txt` |
| scripts | `depth_verify.py` (output in `verify.txt`), `depth_shots_explore.py` (output in `shots_explore.txt`) |
