# Workplace -> home, 2026-10-14: W1, W2, W3, W5, W7 (revised requests)

Packed 2026-10-14 02:25 CEST (`TZ=Europe/Zurich date`). Offline only: no IBM connection, no push. Common rule
followed: no search, no sweeps, no seed variation, no choice by ESP or simulation; measured only correctness, CZ count,
duration. Scripts run from the psf-zero repository root (bundle 0614ddc), Rust core 2026-09-29.1, qiskit 2.5.2,
qiskit-aer 0.17.2, qiskit-ibm-runtime 0.50.0.

| request | folder | result in one line |
|---|---|---|
| W2 (high) | `w2/` | c48 is 0.012-0.16x Qiskit L3's duration on TFIM, but a hand-written brickwork circuit given to Qiskit L3 has the same CZ (43/48 cells) and duration within 0.93-1.06 up to n = 50; at n = 100 PSF loses (10 steps: size limit, 82x; FakeMarrakesh 2-8 steps: 2.9-4.1x) |
| W3 (high) | `w3/` | the gap does not grow with depth in general: level at odd L; it opens only at even L (FakeFez n=4 to 0.80; n=6 both devices to 0.63/0.68 from L=10) |
| W5 (medium) | `w5/` | cutting failed qubits/couplers first leaves 148 connected qubits; with it all 27 SCALE1 circuits compile (incl. the two refused 144-qubit ones), 0 failed elements, deterministic, correct. Structure-only placement equals QK3 on chains; on non-chains it needs an exact-embedding step (rustworkx VF2, first match, no score) and still loses where routing is needed |
| W7 (medium) | `w7/` | inventory: item 83 is the only feature with direct hardware support; the estimate-only features in the default call are 33/60, 64, 69, 74, 75's score, 78, 80 |
| W1 (low) | `w1/` | (1) pre-registered hardware compiler comparison: not found; (2) same-gate reordering raising hardware F: partly known (Murali 2020, Liu 2026), no Trotter/Pauli case; (3) cliff: partly known (VF2PostLayout runtime in Qiskit 2.1-2.3), no spare-qubit dependence documented |

Check: `python check_intake.py`.
