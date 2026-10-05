# Workplace pre-registration: MODEL-RO. Does candidate AI front end psf_ai_compile 2026-10-05.a10 (readout of measured qubits in the state-aware estimate, item 15) make sampled model-written circuits more faithful than a9? (2026-10-05)

**Status: pre-registration, locked at the Project save time of this document**, before the scored run.

- **Setting:** workplace sandbox (2 CPUs), Qiskit 2.5.2, Aer 0.17.2, core 2026-09-29.1. Fake devices and Aer noise
  only.
- **Companion test:** READOUT (`readout-preregistration-2026-10-05.md`) tests the same blind spot in the release (c13,
  item 40).

## 1. Why, and the candidate

**The blind spot.** a9's state-aware estimate (`_state_weights` and `_score_weights`, from a4) skips `measure`. As a
result:

- neither the choice among candidates nor the state-aware re-placement sees readout error;
- yet model-written circuits are sampled, so their measured qubits' readout enters every result.

**a10, item 15:**

- Each measured circuit qubit is recorded once, with weight 1.
- It is charged the Target's measure error of the physical qubit it is mapped to.
- Without measurements, the estimate and the output are exactly a9's (tested).

**Tests** (`test_a10_readout.py`, 9 cases, all passed in the sandbox, 14 s). They check:

- the versions;
- that a10 equals a9 instruction by instruction on unmeasured circuits (synthetic GHZ and W, FakeTorino and
  FakeAuckland);
- that the estimate changes by exactly the summed measure error;
- that measured outputs are exact (noiseless total variation 1e-6) and that clbit j measures logical j's final
  qubit, on 3 devices;
- that on FakeTorino a10 is never worse in summed readout and better on at least one of 4 synthetic circuits.

## 2. Design (`ai10_eval.py`, `run_ai10.sh`)

**Set:** the 153 model-written circuits of Addendum 285 (`data/2026-10-02/ai6/outputs/model_circuits.qpy`, raw
SHA-256 `3c012e4c…`; 3-6 qubits; no measurements of their own), each with `measure_all()`. The set was used in
Addenda 285 and 312-316.

**Arms:**

| arm | what it is |
|---|---|
| A9 | a9 (`exact_stage_v1`) |
| A10 | the candidate |
| L3TM | Qiskit level 3 with the Target, `approximation_degree=1.0` |

A9 and A10 both have candidate psf_compile c13 underneath and get the device Target.

**Devices:** FakeTorino, FakeKingston, FakeAuckland.

**Metric:** the classical infidelity 1 - (Σ_x √(p_x q_x))² between:

- p, the ideal distribution of the measured logical qubits;
- q, the noisy distribution, computed as follows:
  - Aer density matrix of the compiled circuit without its measurements;
  - the noise model restricted to the touched qubits (DEPTH's `Noisy`);
  - the marginal on the measured physical qubits, in clbit order;
  - each qubit's readout assignment probabilities (Aer's, asymmetric).

**Also recorded:** the summed measure error, the two-qubit count and the compile time.

## 3. Predictions (scored only by `ai10_eval.py score`)

**P0.** All of these must hold, or nothing below is scored:

- 3 device files with 153 circuits each;
- the noiseless distribution of every arm's output within total variation 1e-6 of the ideal;
- clbit j measures logical j's final qubit in every output.

| ID | Prediction | CONFIRMED | REFUTED (otherwise AMBIGUOUS) |
|---|---|---|---|
| M1 | a10 puts the measured qubits on better readout | A10/A9 mean summed measure error <= 0.80 on both Heron devices and <= 1 everywhere | > 1 on any device |
| M2 | the sampled distribution is more faithful | A10/A9 mean classical infidelity <= 1.00 everywhere and <= 0.95 on both Heron devices | > 1.01 on any device |
| M3 | a10 is at or ahead of error-aware Qiskit | A10/L3TM <= 1.00 on every device | > 1.05 on any device |
| M4 | broadly, not by a few circuits | per circuit A10 <= A9 in >= 80% of circuits on both Heron devices | < 60% on either |
| M5 | no extra time | median compile time A10 <= 1.2 × A9 on every device | > 2 × on any device |

**Expectations, and what was seen** (disclosed):

- **Smoke check before the tests were written:** circuits 0, 40 and 120 of this set, with `measure_all()`, were
  compiled by a9 and a10 on FakeTorino and FakeKingston. Only the summed measure error and the two-qubit count were
  looked at:
  - a10 lowered the summed measure error in all six (for example 0.223 → 0.061, 0.239 → 0.030);
  - the two-qubit count rose by 1 in one of them.
  - No distribution was simulated.
- **Dry run** (6 synthetic GHZ circuits, not from this set):
  - P0 passed;
  - M1, M2, M4 and M5 CONFIRMED; M3 AMBIGUOUS (A10/L3TM 0.976 / 1.005 / 1.036).
- **M3 is the open one.** Level 3 counts readout through its averaged error map (Addendum 308), so on these small
  circuits it may already find the same qubits.
- **A10's estimate and the simulator share their physics** (as in Addenda 286-339). M2 favours A10 by construction to
  that extent. The readout model (asymmetric assignment errors) is the simulator's own.

## 4. What this will not establish

- Hardware.
- Circuits wider than 8 qubits: there a10 calls the release, so c13's item 40 applies instead.
- The vLLM loop itself: pass rates, and cost.

## 5. Locked files (normalized SHA-256)

| file | normalized SHA-256 |
|---|---|
| `ai10_eval.py` | `9e04390ed477a9e4d03f67bbe1c6e50712ed669d92bad299fe83a3fdc8536477` |
| `run_ai10.sh` | `240b06b2a813776c226df158f418e1931b2c4be9013819e36329dd528e81a4df` |
| `psf_ai_compile.py` (candidate a10) | `1af633d5f91a639a4894a092b5d4603c738cb51584d69d8068bfce9a0303e480` |
| `psf_ai_compile_a9.py` (a9, `exact_stage_v1`) | `f6df9f7a28c8fac8e72116cf138cbc30781b1d48b5557d445e53484c19674b54` |
| `psf_compile.py` (candidate c13) | `edaafc2c927496238fd5d81e45e2da3d46cc874c49209bdadfd640d41bb87cb2` |
| `depth_eval.py` (DEPTH, for `Noisy`) | `82488d96144bb1c88f69676a0e6a642c22d4756a47d8a1ad8b07be5327c796ac` |
| `test_a10_readout.py` | `28704cede24a162a79c3fb212b9a71d3612380d57080542c4552064f735fceb4` |
