# Workplace pre-registration: QML-2 "DEPTH", stage 1. How does a quantum classifier trained on real data behave on a noisy fake device as it is made deeper, how much do compilers move that, and does fine-tuning through the noise help? (2026-10-05)

**Status: pre-registration, locked at the Project save time of this document**, before the scored run.

- **Setting:** workplace sandbox (2 CPUs), Python 3.11.15, Qiskit 2.5.2, qiskit-aer 0.17.2, scikit-learn 1.8.0,
  Rust core 2026-09-29.1. Fake devices and Aer noise only: no IBM, no hardware, no GPU.
- **Owner's go-ahead:** 2026-10-05 ("始めてください"), after the design `qml2-design-2026-10-05.md`.
- **Times** are sandbox times and are not compared across machines.

## 1. Why

Earlier quantum-classifier tests found the same thing at 4 qubits and shallow depth (Addenda 291 and 297; workplace
`qml-results-2026-10-05.md`): compilers change the margin by a few per cent, but not the accuracy. Accuracy moved
only on borderline inputs.

This stage asks the question at the scale where noise starts to bind:

- whether depth stops paying under noise;
- whether better compilation moves that limit;
- whether fine-tuning through the compiler and the noise recovers anything.

It uses real data and the current compilers.

## 2. Design (`depth_eval.py`, `run_depth.sh`)

**Data.** Bundled with scikit-learn, so nothing is downloaded.

- **BC:** breast_cancer, 569 samples.
- **D38:** digits, classes 3 against 8, 357 samples.
- **Preparation:** an 80/20 stratified split with **split seed 1**. Features are standardised, reduced by PCA to n
  components and scaled to [-1, 1] on the training set.
- **Test sets:** 114 points (BC) and 72 points (D38).

**Model.** n qubits, L layers.

- **Each layer:** RY(π/2 · x_q) on every qubit, then RY(a) RZ(b) on every qubit, then a CZ ring
  (0,1), (1,2), ..., (n-1,0).
- **After the layers:** RY(c) on every qubit.
- **Output:** z = <Z_0>. The prediction is sign(z) and the loss is MSE(y, z).
- **Routing:** heavy-hex has no 4-cycle or 6-cycle, so the ring always needs routing.

**Sizes:** n ∈ {4, 6}; L ∈ {1, 2, 4, 8, 12, 16}.

**Training (noiseless).** Adam with learning rate 0.05, batch 64 and 300 steps, on exact gradients. The gradients
come from an adjoint method in a numpy statevector. They were checked against parameter shift (3.9e-16) and the
forward pass against Qiskit's Statevector (3.3e-16). The init seed is 1 + 1000·L.

**Arms:**

| arm | what it is |
|---|---|
| RPSF | psf_compile with `target` and `placement_refine=True`: 2026-10-02.2's call, guarded paths only. The file is the c12 candidate; c12 is identical to 2026-10-04.1 on this call |
| C12 | candidate psf_compile 2026-10-05.c12 (home, Addendum 342 pending) with the recommended call: + `final_resynthesis="select"`, `compare_level3`, `compare_floor`, `candidate_score="hybrid"`. If c12 is not adopted, this arm reads as 2026-10-04.1's call made exact |
| L3T | Qiskit level 3 with the Target, `approximation_degree=1.0` |

**Devices:** FakeAuckland (cx) and FakeTorino (cz).

**Deployment (every test point × L × arm × device):**

- Every circuit is compiled and checked noiselessly: its z must equal the logical z.
- It is then simulated with `NoiseModel.from_backend` restricted to its touched qubits (the same QuantumError
  objects, renumbered), with Aer's density matrix. z is read exactly.
- **Score adjustments:** the readout error of the physical qubit that carries logical qubit 0 (Aer's asymmetric
  probabilities), and 4,000 shots × 20 repetitions. The random numbers depend on (dataset, n, device, L, point)
  only, so they are the same in every arm.

**Fine-tuning (BC, n = 6, FakeAuckland, arm C12, L ∈ {4, 12}, seeds 1 and 2):**

- **Start:** the noiselessly trained θ*.
- **Two variants, 40 SPSA steps each** (batch 16; a = 0.3, c = 0.1, A = 5; α = 0.602, γ = 0.101; the same random
  numbers in both):
  - FTN: the loss is computed through the compiler and the noisy device (exact z);
  - FT0: the noiseless loss (a control for "more optimisation").
- **Evaluation:** DEP (θ* as is), FT0 and FTN are each deployed on the test set as above.

**Size:** 4 training jobs, 24 deployment jobs and 4 fine-tuning jobs, run in parallel as 2 processes.

## 3. Predictions (scored only by `depth_eval.py score`)

**P0, harness.** All of these must hold, or nothing below is scored:

- 24 of 24 deployment files and 4 fine-tuning files;
- every compiled circuit's noiseless z within 1e-6 of the logical z;
- the reduced simulation equals the whole-device simulation (Aer's own noise model, `save_density_matrix`) within
  1e-9 on the first two points of every (L, file).

| ID | Prediction | CONFIRMED | REFUTED (otherwise AMBIGUOUS) |
|---|---|---|---|
| H1 | depth stops paying in margin (FakeAuckland, n = 6) | for both datasets and every arm, the deployed margin at L = 16 is below 0.8 × the best margin over L | for any dataset and arm, L = 16 has the largest margin |
| H2 | ... and in accuracy (FakeAuckland, n = 6, shot-based with readout) | for at least one dataset, in every arm, the shot accuracy at L = 16 is at least 2 test points below the best over L | for every dataset and arm, L = 16 is the best (or tied best) |
| H3 | the recommended compiler keeps more margin than the guarded call | pooled over datasets, n and L, C12 - RPSF mean margin >= +0.005 on both devices | < 0 on either device |
| H4 | C12 is level with error-aware Qiskit | pooled \|C12 - L3T\| margin <= 0.01 on both devices | C12 < L3T - 0.02 on either device |
| H5 | C12 flips no more predictions than RPSF | pooled exact flip rate (noisy sign ≠ noiseless sign) C12 <= RPSF on both devices | C12 > RPSF + 0.01 on either device |
| H6 | fine-tuning through the noise helps where noise binds | at L = 12, mean over seeds of FTN - DEP deployed margin >= +0.02, and larger than FT0 - DEP | FTN - DEP < -0.01 |

**Gate for stage 2 (GPU), computed by the scorer.**

- **GO** if (H1 or H2 is CONFIRMED) **and** (H3 is CONFIRMED, or some cell has |shot accuracy C12 - RPSF| of at least
  2 test points).
- **NO-GO** otherwise. Then no GPU stage is proposed on this evidence.

**Reported without prediction:**

- the full table (accuracy, shot accuracy, margin, flip rate and two-qubit count, per dataset, n, L, arm and
  device);
- FakeTorino for H1 and H2;
- n = 4;
- the fine-tuning results at L = 4;
- compile times.

**Expectations, and what was known when writing them** (disclosed):

- **H1 is expected.** The workplace pilot (BC, split seed 0, n = 6, L up to 8; FakeAuckland) and the dry run (split
  seed 2) both showed the deployed margin peaking at L = 2-4 and falling at L = 8-12. It is the least informative
  item.
- **H2 is the open question.** In the pilot the accuracy still rose up to L = 8 although the margin halved. Whether the
  margin's fall at L = 12-16, together with shots and readout, turns into lost test points is not known.
- **H3-H5:**
  - The pilot showed C12 and L3T about 0.01 above RPSF in margin at L = 2-4, and level at L = 1 and 8.
  - In the dry run (12 points, L ∈ {1, 4, 12}) C12 - RPSF was +0.007 (FakeAuckland) and +0.015 (FakeTorino), and
    C12 - L3T was +0.006 on both.
  - So H3's bound of +0.005 may well come out AMBIGUOUS on FakeAuckland.
- **H6 is the most uncertain.** In the dry run (3 steps, a = 0.1) nothing moved (FTN - DEP -0.0003).
- **A5/C12 and the simulator.** The estimate inside C12 shares the simulator's physics, so H3 and H4 favour C12 by
  construction (Addenda 286-339).

## 4. What this will not establish

- **Hardware.**
- **More than 6 logical qubits, or touched-qubit counts above about 8.** That is stage 2.
- **Other models, data encodings or optimisers.**
- **Reliability over training seeds.** There is one noiseless training per (dataset, n, L) and two fine-tuning seeds.

## 5. Development (disclosed)

**Pilot** (`pilot_depth.py`; BC, split seed 0, init seed 0, n = 6, L ∈ {1, 2, 4, 8}; arms RPSF, C12, L3T):

- **FakeAuckland**, all 12 cells:

  | L | ideal acc / margin | RPSF | C12 | L3T |
  |---|---|---|---|---|
  | 1 | 0.868 / 0.399 | 0.860 / 0.389 | 0.860 / 0.388 | 0.860 / 0.389 |
  | 2 | 0.886 / 0.571 | 0.886 / 0.504 | 0.886 / 0.514 | 0.886 / 0.514 |
  | 4 | 0.921 / 0.632 | 0.912 / 0.483 | 0.921 / 0.493 | 0.921 / 0.500 |
  | 8 | 0.930 / 0.635 | 0.930 / 0.336 | 0.930 / 0.336 | 0.930 / 0.333 |

- **FakeTorino:** only L = 1 for RPSF and C12 (both 0.868 / 0.388). The rest was stopped: whole-device noise models
  made it about 7 times slower, which led to the restricted noise model used here.
- **An Aer behaviour found while developing:** `save_expectation_value` on a whole-device compiled circuit gave a
  wrong value (-0.014 against -0.277). The pilot and this harness use `save_density_matrix`, which agrees with the
  exact value. P0 checks the restricted simulation against the whole-device one.

**Dry run** (`--dry`: split seed 2, init seed 2, 12 test points, L ∈ {1, 4, 12}, 60 training steps, 3 fine-tuning
steps; 664 s): the harness ran end to end.

- **P0 passed:** compiled noiseless against logical 1.7e-14; reduced against whole-device 0 on 144 circuits.
- **Its verdict lines (not results):** H1 CONFIRMED, H2 AMBIGUOUS, H3-H5 CONFIRMED, H6 AMBIGUOUS, GATE GO.

**Changed after the dry run, before the lock:** the fine-tuning step size a, from 0.1 to 0.3, because 0.1 left θ
essentially unchanged in 3 steps.

**Not seen:** no scored data (split seed 1) was compiled, simulated or trained on before the lock.

## 6. Locked files (normalized SHA-256)

| file | bytes | normalized SHA-256 |
|---|---|---|
| `depth_eval.py` | 27,934 | `82488d96144bb1c88f69676a0e6a642c22d4756a47d8a1ad8b07be5327c796ac` |
| `run_depth.sh` | 1,960 | `b7422bc96c8001d902fc7d476692e8ce4d79f927ab4ed710624df89115bec2d7` |
| `psf_compile.py` (candidate 2026-10-05.c12, from `exact_stage_v1`) | | `1dc9b9b1d5b08118c33cad81bfc4be4905b5e593a3c60abab6caa56835685ab1` |
| `psf_smart_layout.py` (2026-10-01.1, from `main`) | | `624e8f8a00e1635a1ee3bc77b5b0f41bd69a94022e214d679b86cc66cc1cf241` |

**Normalization:** CRLF to LF, trailing whitespace stripped from each line, trailing blank lines dropped, lines
joined with "\n", no final newline.

**Run:**

```
PAR=2 bash run_depth.sh <out> <dir>/psf_compile.py     (PYTHONPATH must contain the Rust core)
```
