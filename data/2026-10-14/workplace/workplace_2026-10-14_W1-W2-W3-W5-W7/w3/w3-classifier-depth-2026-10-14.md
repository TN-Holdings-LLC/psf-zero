# W3 (2026-10-14): the data re-uploading classifier at 1-16 layers -- does c48's duration advantage over Qiskit level 3 grow with depth?

**Status: offline calculation, done.** Workplace, 2026-10-14 (times from `TZ=Europe/Zurich date`). Script
`w3_depth.py`, rows in `run/w3_rows.jsonl`, log `run.log`. Measured only correctness, CZ count and duration; no ESP,
no search, one compile per cell (`seed_transpiler=0`).

## Setup

- **Model:** DEPTH / DEPTH-R's classifier (Addenda 345, 399), built by `depth_eval.circuit()` unchanged: per layer
  RY(pi/2 x_q), RY(a) RZ(b) on every qubit, then a CZ ring (0,1), ..., (n-1,0); a final RY layer. n = 4 and 6,
  L = 1..16. (The project's DEPTH runs build this model in Qiskit; its PennyLane classifiers in parts X and Stage 2
  are 2-4 qubits and not layered this way.)
- **Angles:** one data point and one parameter vector per (n, L), drawn once from `default_rng(20261014)`; not
  trained and not varied. Generic angles avoid accidental cancellations; trained angles are not expected to change
  the structure.
- **Arms:** c48 as shipped (`compile_for_hardware(qc, backend=b, seed_transpiler=0)`; its item 80 may return Qiskit
  level 3's circuit by ESP), **PSF-C** (c48 with `level3_candidate=False, near_failed_error=None`: no estimate-based
  choice), **QK3** (`generate_preset_pass_manager(3, b, seed_transpiler=0)`). All qubits measured.
- **Devices:** FakeFez and FakeTorino (DEPTH used FakeTorino and FakeAuckland; FakeAuckland is a cx device and not a
  Heron, so FakeFez takes its place).
- **Correctness:** total variation distance between the compiled and logical outcome distributions (Aer
  statevector): at most 1e-14 in all 192 compiles; every output ISA.

## Result: duration ratio PSF-C / QK3 by number of layers

| device, n | L=1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FakeFez, 4 | 1.03 | 0.95 | 1.00 | 0.86 | 0.99 | 0.84 | 1.00 | 0.83 | 0.99 | 0.83 | 1.00 | 0.81 | 1.00 | 0.81 | 1.00 | 0.80 |
| FakeFez, 6 | 0.55 | 1.01 | 0.99 | 1.00 | 1.01 | 1.00 | 1.01 | 1.00 | 1.05 | 0.66 | 1.00 | 0.65 | 1.00 | 0.64 | 1.01 | 0.63 |
| FakeTorino, 4 | 0.93 | 0.92 | 1.00 | 1.18 | 0.99 | 1.00 | 1.00 | 0.99 | 1.00 | 1.00 | 1.01 | 1.02 | 1.01 | 1.02 | 1.02 | 1.01 |
| FakeTorino, 6 | 0.95 | 1.04 | 1.02 | 1.03 | 1.01 | 1.00 | 1.03 | 1.05 | 1.04 | 0.84 | 1.01 | 0.80 | 1.02 | 0.76 | 1.01 | 0.68 |

Examples (CZ, duration):

| device, n, L | PSF-C | QK3 |
|---|---|---|
| FakeFez, 4, 15 / 16 | 147, 15.7 us / 120, 13.4 us | 147, 15.7 us / 157, 16.8 us |
| FakeFez, 6, 15 / 16 | 267, 28.3 us / 278, 19.1 us | 267, 28.1 us / 286, 30.4 us |
| FakeTorino, 4, 15 / 16 | 147, 14.7 us / 157, 15.7 us | 147, 14.4 us / 157, 15.5 us |
| FakeTorino, 6, 15 / 16 | 269, 26.3 us / 274, 18.9 us | 269, 26.0 us / 289, 27.8 us |

## Reading

- **The gap does not open with depth in general.** At an odd number of layers c48 and Qiskit level 3 are level on
  both devices and both sizes (0.99-1.05 from L = 3 on). On FakeTorino with 4 qubits they are level at every depth
  from L = 5.
- **It opens only at an even number of layers, and not everywhere:**
  - FakeFez, n = 4: from 0.95 (L = 2) gradually to 0.80 (L = 16); the absolute gap grows from 0.1 to 3.3 us, with
    fewer CZ (120 against 157 at L = 16).
  - n = 6 on both devices: level up to L = 8, then a step at L = 10 (0.66 FakeFez, 0.84 FakeTorino) and a slow
    decline to 0.63 / 0.68 at L = 16; there the CZ counts are close (278 vs 286, 274 vs 289), so the gain is in the
    schedule, not the gate count. The absolute gap grows from 6.4 to 11.4 us (FakeFez).
- The parity pattern is consistent with c48's mirrored layers (item 82: every second layer ends where the first
  began), but that was not isolated here; the step at L = 10 for n = 6 points to a size condition inside c48 that
  was not identified.
- **c48 as shipped:** item 80 returned "Qiskit level 3 kept" in 51 of 64 compiles, yet its durations equal PSF-C's
  where PSF-C is shorter (it re-runs level 3 on PSF's own circuit); where PSF-C is slightly longer (FakeTorino, odd
  L), the shipped call matches QK3. Ratios for the shipped call are in the rows file.

**Answer to W3's question:** the difference does not grow with the number of layers as such. It appears, and then
grows, only for an even number of layers (FakeFez at 4 qubits from L = 2; 6 qubits on both devices from L = 10).
Per the request, this is reported and the work stops here; whether a hardware test on even-layer circuits is worth
pre-registering is home's decision.
