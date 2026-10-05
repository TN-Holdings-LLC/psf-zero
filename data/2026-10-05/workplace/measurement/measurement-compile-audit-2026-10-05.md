# Workplace audit: which harnesses compile circuits without the measurements they are scored with? A helper (`measured_compile.py`, tests 3/3) and the proposed wording for the README (2026-10-05)

**Status: audit of the repository at `5cfa7c3` (read only), plus a helper with tests.** Nothing in the repository was
changed, and locked scripts are left as they are.

## 1. Why

**READOUT** (`readout-results-2026-10-05.md`): compiling a classifier with its final measurement moved its output on
FakeTorino from a qubit with readout error 0.048 to one with 0.008. With the helper below, on the n = 6, L = 12 case,
it moved from 0.146 to 0.006.

**MODEL-RO2** (`model-ro2-results-2026-10-05.md`): the AI front end needs the measurements too (a10, item 15).

PSF-Zero's re-placement (release item 33), Qiskit's VF2PostLayout and the candidate estimates (c13 item 40, a10 item
15) all see readout only through `measure` instructions in the circuit.

## 2. The harnesses (repository `5cfa7c3`)

| harness | circuits compiled with measurements? | readout in the score? | consequence |
|---|---|---|---|
| `qml_home_eval.py` (Addenda 290-291) | no (`build()` has none) | no (exact z from the density matrix) | none for its metric |
| `qml_home2_eval.py` (Addenda 296-297) | no | **yes**: the output qubit's readout error and shots | **readout-blind placement entered the shot results.** 297's "readout placement" finding (0.048 against 0.011 on FakeTorino) is this effect. The L3T arm was not blind because its layout stage averages in readout (Addendum 308) |
| `gap_eval.py`, `hold*_eval.py` | no | no (state fidelity of the final-layout qubits) | none for their metric |
| `e2e_vllm_psf_v11.py` | no | no (state fidelity of the model's tape, `qml.state()`) | none for its metric, but **any sampled scoring or hardware run** of the loop should compile with measurements |
| workplace `depth_eval.py` (DEPTH, 2026-10-05) | no | **yes** (readout + shots) | the same as `qml_home2_eval`; fixed in `readout_eval.py` |

`qml_home*`, `gap` and `hold*` already call `remove_final_measurements` before reading the density matrix, so
compiling with measurements would not disturb their reading.

## 3. The helper (`measured_compile.py`)

| function | what it does |
|---|---|
| `with_measurements(qc, measured=None)` | adds the measurements, one classical bit per measured logical qubit (default: all). A circuit that already measures is returned as is |
| `compile_measured(compile_fn, qc, measured)` | compiles the circuit with them |
| `measured_physical_qubits(out)` | the physical qubit behind each classical bit, in clbit order |
| `without_final_measurements(out)` | removes them again for exact reading, keeping the layout |

**Tests** (`test_measured_compile.py`, 3 cases, all passed in 7 s):

- the order of the measurements, and that adding them twice changes nothing;
- that the measured qubit is logical 0's final position and that the layout survives removing the measurements;
- that on FakeTorino the output qubit's readout error is no higher with measurements (0.0063 against 0.1458 without).

## 4. Proposed changes (for home; adoption is the owner's)

1. **README and the usage notes:**

   > When a circuit's outcomes will be sampled (or read out with readout error), compile it **with** its final
   > measurements. PSF-Zero's placement steps and its candidate estimates see readout error only through `measure`
   > instructions. Without them, the measured qubits are placed readout-blind.

2. **New harnesses** that score shots or readout should compile `with_measurements(qc, measured)`. They should then read
   z or probabilities from `without_final_measurements(out)`, at `measured_physical_qubits(out)`.
3. **The locked harnesses stay as they are.** A re-run of a `qml_home2`-type test with measurements would need its own
   pre-registration.
4. **The vLLM loop (v11):** no change for its present metric (state fidelity). Before any sampled or hardware scoring,
   compile with `measure_all()` (a11 then also places readout-aware).
