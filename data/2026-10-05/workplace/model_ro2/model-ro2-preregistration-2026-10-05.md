# Workplace pre-registration: MODEL-RO2. Re-test of candidate AI front end psf_ai_compile 2026-10-05.a10 (readout of measured qubits in the state-aware estimate) on fresh circuits, with the exactness check corrected (2026-10-05)

**Status: pre-registration, locked at the Project save time of this document**, before the scored run.

- **Setting:** workplace sandbox (2 CPUs), Qiskit 2.5.2, Aer 0.17.2, core 2026-09-29.1. Fake devices and Aer noise
  only.
- **Owner's go-ahead:** 2026-10-05 ("a10 の再試験").

## 1. Why a re-test

**What went wrong in MODEL-RO** (`model-ro-results-2026-10-05.md`). Its P0 compared the noiseless total variation with
1e-6. Total variation is first order in the amplitude error, so one A10 output failed at 4.8e-6, although its state
infidelity was 3e-11 (the polish tolerance of the front end). Nothing was scored. The descriptive results favoured a10
strongly on the Heron devices.

**What this test changes, and only this:**

1. **P0's exactness check is a state infidelity <= 1e-6**, using READOUT's locked `readout_eval.state_infid` on the
   compiled circuit with its measurements removed. Total variation is still recorded, but only reported.
2. **Fresh circuits**, because the model-written set has been seen.

**Unchanged:** the candidate, the arms, the devices, the metric, M1-M5 with their thresholds, and the scorer's logic.

## 2. Circuits (`ai10_eval2.circuits`)

**Families:** synthetic, in the style of the model-written set. They use its tasks' states, written with explicit
gates, with 2-qubit `unitary` blocks as PennyLane tapes arrive.

| family | what it is | n |
|---|---|---|
| W | cascade construction | 3, 4, 5 |
| GHZ | | 3, 4, 5, 6 |
| DICKE | all weight-2 strings, StatePreparation | 4 |
| QFT | of a random basis state | 3, 4, 5 |
| RSP | random real state, StatePreparation | 3, 4 |
| ENT | a depth-2 brick of Haar 2-qubit unitaries | 3, 4, 5 |

**Construction of each circuit:**

1. Transpile to [cx, u] at level 1.
2. Consolidate a seeded random half of its 2-qubit blocks into `unitary` gates.
3. Add a seeded random single-qubit layer at the end.
4. Add `measure_all()`.

**Size:** 200 circuits, spread evenly over the 16 (family, n) cells (13 or 12 each), with seed base 71,000,000.
They have 3-6 qubits and about 7 two-qubit operations each.

**The dry run** used seed base 71,500,000 (12 circuits). No scored circuit was generated or compiled before the
lock.

## 3. Arms, devices, metric (as in MODEL-RO)

**Arms:**

| arm | what it is |
|---|---|
| A9 | a9 |
| A10 | the candidate |
| L3TM | Qiskit level 3 with the Target, `approximation_degree=1.0` |

A9 and A10 both have candidate psf_compile c13 underneath, with the device Target.

**Devices:** FakeTorino, FakeKingston, FakeAuckland.

**Metric:** the classical infidelity 1 - (Σ √(p q))² between:

- p, the ideal distribution;
- q, the noisy distribution: Aer density matrix with the restricted noise model, then each measured qubit's readout
  assignment probabilities.

## 4. Predictions (scored only by `ai10_eval2.py score`; identical to MODEL-RO's)

**P0.** All of these must hold, or nothing below is scored:

- 3 device files with 200 circuits each;
- every output's state infidelity (measurements removed) <= 1e-6;
- clbit j measures logical j's final qubit in every output.

| ID | CONFIRMED | REFUTED (otherwise AMBIGUOUS) |
|---|---|---|
| M1 | A10/A9 mean summed measure error <= 0.80 on both Heron devices and <= 1 everywhere | > 1 on any device |
| M2 | A10/A9 mean classical infidelity <= 1.00 everywhere and <= 0.95 on both Heron devices | > 1.01 on any device |
| M3 | A10/L3TM classical infidelity <= 1.00 on every device | > 1.05 on any device |
| M4 | per circuit A10 <= A9 in >= 80% of circuits on both Heron devices | < 60% on either |
| M5 | median compile time A10 <= 1.2 × A9 on every device | > 2 × on any device |

**Expectations, and what was seen** (disclosed):

- MODEL-RO's descriptive results on the model-written set: A10/A9 0.51 / 0.30 / 0.98; A10/L3TM 0.99 / 0.97 / 0.94;
  per circuit 88% / 84%.
- This test's dry run (12 synthetic circuits): P0 passed (state infidelity max 3.8e-15). M1-M5 printed CONFIRMED.
- **M3 is the least certain:** on the model-written set, A10 was within 1.3-2.6% of level 3 on the Heron devices.

## 5. What this will not establish

- Hardware.
- More than 8 qubits.
- The vLLM loop itself.

## 6. Locked files (normalized SHA-256)

| file | normalized SHA-256 |
|---|---|
| `ai10_eval2.py` | `12bba73653b607c1cd6bfad99c7b184e10af2ca6262a8e5531e9b5d665a8e299` |
| `run_ai10b.sh` | `2e0c4681d1d4480c8b90276ae47834751fc20ca54ea9ff3122a324e830f0c52f` |
| `psf_ai_compile.py` (candidate a10) | `1af633d5f91a639a4894a092b5d4603c738cb51584d69d8068bfce9a0303e480` |
| `psf_ai_compile_a9.py` (a9) | `f6df9f7a28c8fac8e72116cf138cbc30781b1d48b5557d445e53484c19674b54` |
| `psf_compile.py` (candidate c13) | `edaafc2c927496238fd5d81e45e2da3d46cc874c49209bdadfd640d41bb87cb2` |
| `depth_eval.py` (for `Noisy`) | `82488d96144bb1c88f69676a0e6a642c22d4756a47d8a1ad8b07be5327c796ac` |
| `readout_eval.py` (for `state_infid`; READOUT, locked) | `9d83426dd28ecddff67c555656d2ef3c5e0b9b43ee5a2d4eb5bc7204e8c6f1d0` |
