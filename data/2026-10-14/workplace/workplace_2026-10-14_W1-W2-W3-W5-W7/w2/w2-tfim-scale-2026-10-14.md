# W2 (2026-10-14): TFIM chains at scale -- c48 against Qiskit level 3 given the Pauli evolution, and against Qiskit level 3 given a hand-written brickwork circuit

**Status: offline calculation, done (192 compiles, 02:01-02:23 CEST).** Workplace, 2026-10-14; times from
`TZ=Europe/Zurich date`. Script `w2_tfim.py`, rows `full/w2_rows.jsonl`, table `w2_table.md` (from `w2_table.py`),
log `full.log`. Measured: correctness, CZ count, duration only. No ESP column, no search, one compile per cell
(`seed_transpiler=0`).

## Setup

- **Model:** H = sum Z_i Z_{i+1} + sum X_i on an open chain (J = h = 1), time 0.6, second-order Suzuki-Trotter with 2,
  4, 8, 10 steps, from |0...0>, all qubits measured (HW1/HW2's tfim test, scaled). n = 12, 16, 24, 32, 50, 100.
- **Arms:**
  - **PSF:** c48 as shipped, `compile_for_hardware(qc, backend=b, seed_transpiler=0)` (item 83 on).
  - **PSF-C:** c48 with `level3_candidate=False, near_failed_error=None`.
  - **QK3:** `generate_preset_pass_manager(3, b, seed_transpiler=0)` on the PauliEvolutionGate circuit.
  - **QK3-BW:** the same pass manager on a hand-written brickwork circuit with the same unitary: RZZ(dt/2) layer, then
    RX(dt) / RZZ(dt) alternating, ending RZZ(dt/2); each RZZ layer even edges then odd edges. Equality of outcomes
    with the Pauli-evolution circuit was checked at n = 12.
- **Duration:** as soon as possible on the Target's durations, measurements left out.
- **Correctness:** n <= 24 total variation distance of the exact outcome distributions; n > 24 the largest difference
  of <Z_i> and <Z_i Z_{i+1}> along the chain (Aer MPS). Largest error over all 192 compiles 3.7e-9; every output ISA.
- **Devices:** FakeFez, FakeMarrakesh.

## Result (CZ, duration in us; full table in `w2_table.md`)

| device | n | steps | PSF | QK3 | QK3-BW | PSF / QK3 time | PSF / QK3-BW time |
|---|---|---|---|---|---|---|---|
| FakeFez | 12 | 2 | 66, 1.49 | 82, 9.86 | 66, 1.49 | 0.15 | 1.00 |
| FakeFez | 12 | 10 | 242, 5.52 | 402, 48.3 | 242, 5.30 | 0.11 | 1.04 |
| FakeFez | 50 | 10 | 1078, 5.52 | 1922, 231 | 1078, 5.30 | 0.024 | 1.04 |
| FakeFez | 100 | 8 | 1782, 4.70 | 3138, 377 | 1782, 4.70 | 0.012 | 1.00 |
| FakeFez | 100 | 10 | **3922, 471** | 3922, 471 | 2178, 5.74 | 1.00 | **82** |
| FakeMarrakesh | 50 | 10 | 1078, 6.26 | 1922, 237 | 1078, 5.93 | 0.026 | 1.06 |
| FakeMarrakesh | 100 | 2 | **696, 6.74** | 786, 96.6 | 594, 1.66 | 0.070 | **4.1** |
| FakeMarrakesh | 100 | 4 | **1065, 8.11** | 1570, 193 | 990, 2.76 | 0.042 | **2.9** |
| FakeMarrakesh | 100 | 8 | **1871, 15.9** | 3138, 385 | 1782, 4.87 | 0.041 | **3.3** |
| FakeMarrakesh | 100 | 10 | **3922, 483** | 3922, 482 | 2178, 5.93 | 1.00 | **81** |

## Reading

1. **Against Qiskit level 3 given the Pauli evolution, PSF is far shorter** wherever c48's construction applies:
   0.012-0.16 of QK3's duration, with 0.55-0.89 of its CZ. The ratio improves with n because QK3's duration grows
   with n while PSF's does not.
2. **"Shorter however it is written" does not hold.** A hand-written brickwork circuit given to the same Qiskit
   level 3 has **exactly PSF's CZ count in 43 of 48 cells** and a duration within 0.93-1.06 of PSF's for every n <= 50
   on both devices (sometimes shorter, sometimes longer). PSF's advantage over QK3 on TFIM is the order in which
   Qiskit lays out the evolution's terms, not something a user cannot write by hand.
3. **PSF's duration is independent of n up to 50** (FakeFez: 1.49-1.58 us at 2 steps, 5.52 us at 10 steps), and so is
   QK3-BW's. It is not independent at n = 100:
   - **n = 100, 10 steps, both devices:** c48 falls back to its usual path, because the evolution has more than 2000
     terms (item 70's size limit; `STEINER_STATS["too_large"]`), and returns essentially QK3's circuit (3922 CZ,
     471-483 us). The brickwork circuit gives 2178 CZ, 5.7-5.9 us: 82x shorter.
   - **n = 100 on FakeMarrakesh, 2-8 steps:** STEINER6 runs on the map without the failed couplers (13 failed couplers,
     26 directed entries, on that fake device), but its 100-qubit placement is not a clean path (696 against 594 CZ at 2
     steps), and its duration is 2.9-4.1x the brickwork circuit's, which Qiskit level 3 placed on an unbroken
     100-qubit path. On FakeFez (fewer failed elements) PSF and brickwork are equal at n = 100 up to 8 steps.
4. **Item 80 (ESP choice of Qiskit level 3):** reported "level 3 kept" in 44 of 48 cells, yet PSF's output equals
   PSF-C's in 47 of 48 (the exception differs by 0.03 us): on these circuits it re-runs level 3 on PSF's own circuit
   and changes nothing measurable.

## Limits

Fake devices' calibrations; one model (open-chain TFIM); durations from the Target's gate lengths with an as-soon-as-
possible schedule (the device's own timing is what HW3's D4 / `scheduler_timing` would show). At n = 100 the
correctness check is on one- and two-point Z correlations, not the full distribution.
