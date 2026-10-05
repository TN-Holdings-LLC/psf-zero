# Workplace results: READOUT (pre-registration `readout-preregistration-2026-10-05.md`). All seven confirmed: compiling the classifier with its measurement moves its output off a bad-readout qubit (FakeTorino: measure error 0.048 → 0.0084), c13's readout term helps where the measurement alone does not (FakeKingston 0.0101 → 0.0084), and together they raise the effective margin by 0.034 on FakeTorino, level with Qiskit level 3, at no extra compile time (2026-10-05)

**Status: results of the workplace pre-registration of 2026-10-05.**

- **Lock:** Project save time, before the run (started 04:32 UTC).
- **Scoring:** by the locked `readout_eval.py score`, and re-checked by `readout_verify.py`. That script was written
  after the lock and before any scored output was read. It agrees on every number.
- **Setting:** workplace sandbox, 2 processes; 960 rows (320 per device), 5 compiles each; 38 minutes.
- **Provenance:** every file records the locked hashes (`readout_eval.py` `9d83426d…`, c12 `1dc9b9b1…`, c13
  `edaafc2c…`, `depth_eval.py` `82488d96…`).

## 1. Verdicts

| ID | Verdict | Numbers |
|---|---|---|
| P0 | **PASS** | 3/3 devices; max state infidelity 1.1e-9 (<= 1e-6); measured qubit = final logical 0 in every measured arm |
| R1 | **CONFIRMED** | FakeTorino measure error of the output qubit, C12M / C12 = 0.0084 / 0.0484 = 0.173 |
| R2 | **CONFIRMED** | C13M - C12M: FakeTorino 0.0000, FakeKingston -0.0017, FakeAuckland -0.00002 |
| R3 | **CONFIRMED** | FakeTorino effective margin C13M - C12 = +0.0344 |
| R4 | **CONFIRMED** | C13M - L3TM effective margin: FakeTorino +0.0066, FakeKingston +0.0005, FakeAuckland +0.0094 |
| R5 | **CONFIRMED** | without measurements C13 = C12 in 960 of 960 circuits |
| R6 | **CONFIRMED** | median compile time C13M / C12M at most 0.998 |
| R7 | **CONFIRMED** | FakeTorino 32-shot accuracy C13M - C12 = +0.0096 |

## 2. Numbers

| device | arm | measure error of the output qubit | effective margin | acc 32 shots | acc 4,000 shots | 2q | median compile s |
|---|---|---|---|---|---|---|---|
| FakeTorino | C12 | 0.0484 | 0.4908 | 0.9325 | 0.9623 | 114.0 | 0.640 |
| FakeTorino | C12M | 0.0084 | 0.5252 | 0.9421 | 0.9625 | 114.0 | 0.639 |
| FakeTorino | C13M | 0.0084 | 0.5252 | 0.9421 | 0.9625 | 114.0 | 0.638 |
| FakeTorino | L3TM | 0.0086 | 0.5186 | 0.9414 | 0.9625 | 109.2 | 0.029 |
| FakeKingston | C12 | 0.0117 | 0.5992 | 0.9483 | 0.9625 | 109.5 | 0.675 |
| FakeKingston | C12M | 0.0101 | 0.6021 | 0.9485 | 0.9625 | 109.5 | 0.676 |
| FakeKingston | C13M | 0.0084 | 0.6039 | 0.9487 | 0.9625 | 109.5 | 0.662 |
| FakeKingston | L3TM | 0.0083 | 0.6034 | 0.9486 | 0.9625 | 109.5 | 0.031 |
| FakeAuckland | C12 | 0.0075 | 0.4254 | 0.9137 | 0.9625 | 109.4 | 0.427 |
| FakeAuckland | C12M | 0.0066 | 0.4270 | 0.9141 | 0.9625 | 109.5 | 0.423 |
| FakeAuckland | C13M | 0.0066 | 0.4270 | 0.9141 | 0.9625 | 109.5 | 0.422 |
| FakeAuckland | L3TM | 0.0070 | 0.4176 | 0.9120 | 0.9625 | 109.5 | 0.024 |

**Which qubit carries the output.** The choice depends on the circuit's structure (n, L), not on the data, so each
device has only 4 structural cases. On FakeTorino, n = 6, L = 12:

- C12 puts the output on qubit 12 (measure error 0.146);
- C12M and C13M put it on qubit 37 (0.006);
- L3TM puts it on qubit 11 (0.008).

That one case carries most of FakeTorino's effect. On FakeKingston, n = 6, L = 4, c13 moves part of the circuits
from qubit 4 (0.0149) to qubit 2 (0.0079); c12 does not, even with the measurement.

## 3. Reading

- **The larger fix is in how the compiler is called: compile with the final measurement.**
  - Without it, PSF-Zero's placement steps never see readout.
  - With it, item 33's re-placement scores the `measure` error, as Qiskit's VF2PostLayout does.
  - FakeTorino's output qubit went from a mean 0.048 to 0.0084 with c12 itself.
- **c13's readout term covers the case the re-placement does not reach:** the choice among the release's, the
  floor-placed and level 3's circuits (FakeKingston, -17%).
  - Without measurements, nothing changes (960/960 identical). The compile time is the same.
- **What it buys:**
  - +0.034 effective margin on FakeTorino (+7%);
  - +1 point of 32-shot accuracy;
  - nothing at 4,000 shots, where accuracy is saturated (as in DEPTH).
- **Level with Qiskit level 3**, which counts readout through its averaged error map (Addendum 308).
- **Limits:**
  - fake devices only;
  - 4 structural cases per device;
  - the effect depends on how uneven a device's readout is (FakeTorino p90 0.11, FakeAuckland 0.0125).

## 4. Consequences (proposals; adoption is the owner's)

1. **Documentation and harnesses:** compile circuits with their final measurements when outcomes will be sampled. The
   DEPTH harness did not, and neither do the QML harnesses at home (they read z from the density matrix). For vLLM's
   model-written circuits, MODEL-RO tests the AI front end.
2. **c13 (item 40) as a candidate on top of c12:** it changes nothing without measurements, and helps on uneven-readout
   devices.

## 5. Files (`readout/` in the handoff)

`env.txt`, `progress.txt`, logs, the three `readout_<device>.json`, `score.md`, `score_log.txt`, `verify.txt`; the dry
run in `dry/`.
