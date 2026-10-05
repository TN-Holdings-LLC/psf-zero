# Workplace results: MODEL-RO2 (pre-registration `model-ro2-preregistration-2026-10-05.md`). P0 passed; all five confirmed. On 200 fresh model-style circuits, candidate AI front end a10 cuts the classical infidelity of the sampled distribution to 0.35× of a9's on FakeTorino and 0.11× on FakeKingston (FakeAuckland 0.98×), level with or ahead of Qiskit level 3 on every device, at no extra compile time (2026-10-05)

**Status: results of the workplace pre-registration of 2026-10-05**, a re-test of MODEL-RO with the exactness check
corrected.

- **Lock:** Project save time, before the run (started 05:32 UTC).
- **Scoring:** by the locked `ai10_eval2.py score`.
- **Arithmetic re-check:** `ai10b_verify.py`, adapted from MODEL-RO's checker. It agrees on every number. It is not
  blind: the adaptation was made after the locked score was seen.
- **Setting:** workplace sandbox, 2 processes; 200 circuits × 3 devices × 3 arms; 11 minutes.

## 1. Verdicts

| ID | Verdict | Numbers |
|---|---|---|
| P0 | **PASS** | 3 × 200 circuits; max state infidelity 4.0e-15 (<= 1e-6); clbit j measures logical j's final qubit in every output |
| M1 | **CONFIRMED** | A10/A9 summed measure error: FakeTorino 0.387, FakeKingston 0.180, FakeAuckland 0.949 |
| M2 | **CONFIRMED** | A10/A9 classical infidelity: 0.349, 0.108, 0.977 |
| M3 | **CONFIRMED** | A10/L3TM: 0.975, 0.966, 0.870 |
| M4 | **CONFIRMED** | per circuit A10 <= A9: 93.5% (FakeTorino), 97.5% (FakeKingston); FakeAuckland 94.0% (reported) |
| M5 | **CONFIRMED** | median compile time A10/A9 at most 1.002 |

## 2. Numbers

| device | arm | classical infidelity | summed measure error | 2q | median compile s |
|---|---|---|---|---|---|
| FakeTorino | A9 | 0.01335 | 0.1279 | 9.69 | 0.856 |
| FakeTorino | A10 | **0.00466** | 0.0496 | 9.70 | 0.827 |
| FakeTorino | L3TM | 0.00477 | 0.0500 | 9.73 | 0.020 |
| FakeKingston | A9 | 0.01558 | 0.1474 | 10.04 | 1.116 |
| FakeKingston | A10 | **0.00168** | 0.0266 | 10.04 | 1.118 |
| FakeKingston | L3TM | 0.00174 | 0.0246 | 10.21 | 0.020 |
| FakeAuckland | A9 | 0.00585 | 0.0355 | 9.54 | 0.318 |
| FakeAuckland | A10 | **0.00571** | 0.0336 | 9.54 | 0.314 |
| FakeAuckland | L3TM | 0.00656 | 0.0317 | 9.74 | 0.013 |

## 3. Reading

- **Confirmed on fresh circuits:** the blind spot found in the morning, and a10's fix, carry over.
  - On the Heron devices, whose readout is uneven, most of a sampled circuit's error was readout. a9 placed the
    measured qubits where the summed measure error was 0.13-0.15; a10 places them at 0.03-0.05.
  - The output distribution's infidelity falls by 65% (FakeTorino) and 89% (FakeKingston).
- **Per circuit the gain is broad** (a10 better or equal in 94-98% of circuits), and it costs nothing in compile time.
- **Against Qiskit level 3:**
  - a10 is level or ahead on all three devices: 0.97-0.98 on Heron, 0.87 on FakeAuckland.
  - It keeps its slightly lower two-qubit count.
  - Level 3 already counts readout through its averaged error map (Addendum 308); a10 now does too, with its
    state-aware gate estimate on top.
- **MODEL-RO's descriptive picture holds,** with a larger effect here. The synthetic set measures more qubits per
  circuit than the model-written set (whose infidelities, 0.017-0.062, included more gate error).
- **Limits:**
  - Fake devices, and Aer's own readout model (asymmetric assignment errors from the same snapshot). The estimate and
    the simulator share that model.
  - On hardware, readout drifts and is correlated across qubits.
  - Synthetic circuits, not new model output.

## 4. Consequences (adoption is the owner's)

- **a10 (item 15)** is supported as the AI front end's next candidate. It sits on top of a9, with c13 underneath.
  Without measurements it is a9 exactly (tested).
- **For the vLLM loop:** model-written circuits are sampled, so this is the setting a10 targets. The next check would
  be the harness v11 with a10, on a pod run (owner's decision).

## 5. Files (`improve/model_ro2/` in the handoff)

`env.txt`, `progress.txt`, logs, the three `ai10b_<device>.json`, `score.md`, `score_log.txt`, `verify.txt`; `dry/`.
