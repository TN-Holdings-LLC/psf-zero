# Workplace pre-registration: READOUT. Does compiling a quantum classifier with its final measurement, and candidate psf_compile 2026-10-05.c13 (readout of measured qubits in hybrid_cost, item 40), put its output on a better-readout qubit and buy margin and few-shot accuracy? (2026-10-05)

**Status: pre-registration, locked at the Project save time of this document**, before the scored run.

- **Setting:** workplace sandbox (2 CPUs), Qiskit 2.5.2, Aer 0.17.2, core 2026-09-29.1. Fake devices and Aer noise
  only.
- **Owner's instruction (2026-10-05 afternoon):** improvements, code and tests based on the day's results.

## 1. Why

DEPTH stage 1 (`depth1-results-2026-10-05.md`) recorded the readout error of the qubit that carries the classifier's
output. On FakeTorino it was a mean of:

| arm | readout error |
|---|---|
| guarded call (RPSF) | 0.034 (max 0.146) |
| recommended call (C12) | 0.025 |
| Qiskit level 3 (L3T) | 0.011 |

Home's Addendum 297 had found the same ("the PSF-Zero layout does not consider readout error").

Two causes were seen in the code and on development data:

1. **The circuits were compiled without measurements**, so item 33's re-placement (`VF2PostLayout`, which scores
   `measure` errors when they are present) had nothing to see. On development data (split seed 2, BC, FakeTorino,
   n = 6, L = 12), compiling the same circuits with the measurement moved the output qubit's readout error from
   0.146 to 0.006.
2. **`hybrid_cost` (items 37-38) skips `measure`**, so the choice among the release's, the floor-placed and level 3's
   circuits never sees readout. On development data (FakeKingston, n = 6, L = 4) adding it moved the choice from
   0.0149 to 0.0079.

## 2. The candidate (c13, changelog item 40)

- **Base:** candidate c12 (`exact_stage_v1`).
- **What it adds:** `readout_cost(circ, target)`, the sum of the Target's measure error over the measured qubits
  (each qubit once). `hybrid_cost` adds it to its estimate.
- **Circuits without measurements:** they get c12's estimate unchanged, so their output is identical (tested).

**Tests:** `test_c13_readout.py`, 11 cases, all passed in the sandbox in 17.5 s. They check:

- the versions;
- `readout_cost`, including that each qubit is counted once and that it is 0 without measurements;
- that `hybrid_cost` changes by exactly `readout_cost`;
- that c13 equals c12 instruction by instruction without measurements, on 3 devices × 3 circuits;
- that measured outputs are exact (state infidelity) and measure the qubit where logical 0 ends, on 3 devices;
- that on FakeKingston 6-qubit rings c13 is never worse than c12 in output readout and better on at least one.

## 3. Design (`readout_eval.py`, `run_readout.sh`)

**Classifier, data and training:** those of `depth_eval.py` (DEPTH, unchanged and imported). New split seed 3 and
init seed 3 + 1000·L.

| item | values |
|---|---|
| datasets | BC, D38 |
| n | 4, 6 |
| L | 4, 12 |
| test points | the first 40 per dataset |

**Arms** (the recommended call: `placement_refine`, `final_resynthesis="select"`, `compare_level3`,
`compare_floor`, `candidate_score="hybrid"`):

| arm | what it is |
|---|---|
| C12 | c12, circuit without measurement (as in DEPTH) |
| C13 | c13, circuit without measurement. Compared with C12 instruction by instruction, not simulated |
| C12M | c12, circuit with `measure(0 -> c0)` |
| C13M | c13, the same |
| L3TM | Qiskit level 3 with the Target and `approximation_degree=1.0`, with the measurement |

**Devices:** FakeTorino, FakeKingston (Heron, uneven readout) and FakeAuckland (cx, control).

**Metrics:**

- the Target's measure error of the output qubit;
- the effective margin y·(z (1 - e01 - e10) - (e01 - e10)), with z from the restricted-noise density-matrix
  simulation of DEPTH and Aer's assignment probabilities;
- the shot accuracy at 32 shots (200 repetitions) and at 4,000 shots (20 repetitions), with the same random numbers in
  every arm.

## 4. Predictions (scored only by `readout_eval.py score`)

**P0.** All of these must hold, or nothing below is scored:

- 3 of 3 device files;
- every output of every simulated arm exact as a **state infidelity <= 1e-6** (DEPTH's lesson: not an absolute z
  difference);
- in every measured arm, the measured qubit is the final position of logical 0.

| ID | Prediction | CONFIRMED | REFUTED (otherwise AMBIGUOUS) |
|---|---|---|---|
| R1 | compiling with the measurement moves the output to a better-readout qubit | FakeTorino mean measure error C12M / C12 <= 0.70 | >= 1.00 |
| R2 | c13's readout term never makes it worse, and somewhere makes it better | C13M - C12M mean measure error <= 0 on every device and < 0 on at least one | > +0.001 on any device |
| R3 | together they buy margin where readout is uneven | FakeTorino effective margin C13M - C12 >= +0.01 | < 0 |
| R4 | C13M is level with error-aware Qiskit | effective margin C13M - L3TM >= -0.01 on every device | < -0.02 on any device |
| R5 | without measurements nothing changes | C13 identical to C12 in 100% of circuits | < 99.9% |
| R6 | it stays cheap | median compile time C13M <= 1.2 × C12M on every device | > 2 × on any device |
| R7 | it shows in few-shot accuracy | FakeTorino 32-shot accuracy C13M - C12 >= 0 | < -0.01 |

**Expectations** (development data and the dry run were seen; disclosed):

- **R1 and R3 are expected** from the development data.
- **R2 may be AMBIGUOUS:** c13 changes the choice only where the floor-placed or level 3's circuit has better readout
  and a nearly equal noise estimate.
- **R7:** 40 test points per dataset is a small set, and the 32-shot difference may be within its noise.

## 5. Development (disclosed)

**Exploration** (`explore_readout.py`, development split seed 2, 12 points; arms C12, C12M, C13M, L3TM; FakeTorino and
FakeKingston, n 4 and 6, L 4 and 12): the figures quoted in section 1.

**Dry run** (`DRY=1`: split seed 2, 6 points, 40 training steps; 6 minutes):

- **P0 passed:** state infidelity max 7.9e-15; measured qubit = final logical 0 everywhere.
- **Verdict lines R1-R7 (not results):** all CONFIRMED.
  - FakeTorino measure error: C12 0.048, C12M 0.0084, C13M 0.0084, L3TM 0.0085.
  - FakeKingston: C12M 0.0101, C13M 0.0093.

**Changed after the dry run, before the lock:** test points from 60 to 40 per dataset, for time only.

**Not seen:** no circuit of split seed 3 was compiled or trained on before the lock.

## 6. Locked files (normalized SHA-256)

| file | normalized SHA-256 |
|---|---|
| `readout_eval.py` | `9d83426dd28ecddff67c555656d2ef3c5e0b9b43ee5a2d4eb5bc7204e8c6f1d0` |
| `run_readout.sh` | `836624e344dc8ae3567f9f7c2ef71c033a67209a84dcdf7853f21066c4dc89d7` |
| `depth_eval.py` (DEPTH, unchanged) | `82488d96144bb1c88f69676a0e6a642c22d4756a47d8a1ad8b07be5327c796ac` |
| `psf_compile.py` (candidate c13) | `edaafc2c927496238fd5d81e45e2da3d46cc874c49209bdadfd640d41bb87cb2` |
| `test_c13_readout.py` | `e6313b79b3687dd0e32ae72ce04a1cc4b15defc6d22dd52082d26875559f23f9` |
| c12 `psf_compile.py` (`exact_stage_v1`) | `1dc9b9b1d5b08118c33cad81bfc4be4905b5e593a3c60abab6caa56835685ab1` |
| `psf_smart_layout.py` (2026-10-01.1) | `624e8f8a00e1635a1ee3bc77b5b0f41bd69a94022e214d679b86cc66cc1cf241` |
