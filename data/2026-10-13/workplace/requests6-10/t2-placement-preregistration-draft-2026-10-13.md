# Draft pre-registration (request 7 of 2026-10-13): a T2-aware placement score for PSF-Zero (on c48), with its weight derived from physics, and predictions for FakeFez, FakeMarrakesh, FakeKingston and FakeTorino

**Status: design and predictions only; nothing implemented or run.** Written at the workplace on 2026-10-13 (CEST).
The changelog item number is left to the home session (provisionally "item T2P"). Device statistics below were read
from the fake devices' Targets before writing the predictions; no compile was made.

## 1. Why

BA diagnosis (`a14-hw1-seven-pairs-2026-10-13.md`, section 4): on HW1 the PSF arms that lost kept a qubit with a low
T2 (qubit 88, 20 us) in the circuit from its first gate to the end. Placement by ESP cannot see this: ESP charges gate
and readout errors only. On HW1's within-test data the state-blind exposure, sum over used qubits of window / T2,
had the best rank correlation with the device (+0.58), but it was found after the data.

## 2. The score (fixed here, no free weight)

For a candidate circuit and a placement p (logical qubit q -> device qubit p(q)):

    cost(p) = sum over gates and measurements of -ln(1 - e)                      (ESP's cost, unchanged)
            + sum over used logical qubits q of I_q * ( 1 / (6 T1_p(q)) + 1 / (3 T2_p(q)) )

- **I_q**, the idle time of q: from q's first gate to the end of the circuit (as soon as possible on the Target's
  durations), minus the durations of q's own gates. The window is the exposure window of HW2's P4; the gate time is
  removed because the reported gate errors already contain the decoherence during the gate.
- **Why these coefficients.** The reported gate errors are average gate infidelities (randomized benchmarking). The
  average gate infidelity of thermal relaxation for a time t, to first order, is t (1/(6 T1) + 1/(3 T2)) (the average
  over pure states of the first-order loss t/T1 [(1-z)/2 - (x^2+y^2)/4] + t/T_phi (1-z^2)/2, with 1/T_phi =
  1/T2 - 1/(2 T1)). The idle term is therefore in the same unit as every ESP term, and no weight is chosen.
- T2 is capped at 2 T1; a missing T1 or T2 counts as 1 us (as HW2).
- **Variant, declared now as a separate arm (not a tuning):** T2P-half, the same with T2 / 2 in the idle term only
  (the unechoed-dephasing hypothesis H-T2*, as HW2-PLUS's P2).

**What one unit costs.** At the device medians (read from the Targets):

| device | median T1 / T2 (us) | median CZ error | 1 us idle at the medians | one exposure unit (idle = T2) |
|---|---|---|---|---|
| FakeFez | 145 / 88 | 0.0038 | 4.9e-3 = 1.3 CZ | 0.43 = 114 CZ |
| FakeMarrakesh | 197 / 118 | 0.0032 | 3.7e-3 = 1.1 CZ | 0.43 = 134 CZ |
| FakeKingston | 283 / 144 | 0.0018 | 2.9e-3 = 1.6 CZ | 0.42 = 232 CZ |
| FakeTorino | 185 / 141 | 0.0041 | 3.3e-3 = 0.8 CZ | 0.46 = 113 CZ |

So a microsecond of waiting on a typical qubit costs about one CZ, and on a qubit with T2 = 20 us about five to nine.
About 20% of the qubits on each of the four devices have T2 below half the device median (0.200-0.212).

## 3. Where it applies in c48

Everywhere c48 compares placements or candidates by ESP: the placement by the device's errors (items 60, 75), the
STEINER6-or-usual choice (item 69), the level-3 candidate (item 80) and the near-failed comparison (item 74). The
schedule (I_q) is computed once per candidate on its own placement and reused for its re-placements (Heron's gate
durations are uniform per gate type); the final comparison between candidates re-schedules each placed candidate.
Without a Target, nothing changes.

## 4. Test design (offline, fake devices; to be fixed with the implementation before any run)

- **Devices:** FakeFez, FakeMarrakesh, FakeKingston, FakeTorino.
- **Circuits:** HW2's part R tests built by HW2's code (nine tests and the two tfimP tests) and 20 new ones not used in
  any development: Barabasi-Albert QAOA, three layers, n = 14, 16, 18, 20, five seeds each from a fixed new seed range.
- **Arms:** C48 (as it is), T2P, T2P-half, and Qiskit level 3 with the Target (reference).
- **Recorded per circuit, all on the fake Target:** ESP (product form), exposure X = sum window / T2, the idle term,
  the combined cost, two-qubit count, duration, used qubits below the device's 10th percentile of T2, exactness (at
  most 24 active qubits), failed elements, compile time, the same output on a second call.

## 5. Predictions (written before any implementation)

| ID | prediction | confirmed | refuted |
|---|---|---|---|
| V0 | every T2P and T2P-half output exact (state infidelity <= 1e-6 where checkable), on target, no failed element, the same on a second call | all | any failure |
| V1 | combined cost: T2P <= C48 on each device | on >= 95% of circuits | on < 85% |
| V2 | exposure: geometric mean T2P / C48 over the BA circuits | <= 0.90 on every device | > 0.97 on any device |
| V3 | ESP: geometric mean T2P / C48 over all circuits | >= 0.97 on every device, and no circuit below 0.90 | < 0.95 on any device |
| V4 | used qubits below the device's T2 10th percentile, summed per device | T2P <= half of C48's on every device where C48 uses any | T2P >= C48's on any such device |
| V5 | two-qubit count and duration, median T2P / C48 | within 0.95-1.05 | outside 0.90-1.10 |
| V6 | compile time, median T2P / C48 | <= 1.2 | > 1.5 |
| V7 | reported: T2P-half against T2P (exposure, ESP); T2P and C48 against Qiskit level 3 (ESP, exposure, combined cost) | - | - |

Anything between the confirmed and refuted bounds is ambiguous.

**Reading, fixed now.** These are compile-time quantities on fake Targets; they say what the score does, not whether
it helps on hardware. If V0-V4 are confirmed, a hardware test is proposed separately (pre-registered, the owner's
go-ahead): T2P against C48 on the BA tests where HW1 lost, prediction "T2P's F >= C48's on at least 3 of 4".

## 6. Risks stated in advance

- Avoiding low-T2 qubits can force longer routes on a heavy-hex lattice: V3 and V5 watch for it.
- T2 on the fake Targets is a snapshot; on hardware T2 fluctuates by tens of per cent within a day, so the placement
  should use the calibration of the day (as HW2 does).
- The state-blind idle term charges a qubit fully even when it waits in |0>. a14's state-aware term does not; on HW1
  the state-blind exposure ranked better, which is why the state-blind form is proposed here. HW2-PLUS's Q3 tests that
  comparison on new data.
