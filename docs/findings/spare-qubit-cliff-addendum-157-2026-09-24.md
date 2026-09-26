# Addendum 157 -- With vs without PSF-Zero in the connection: identical CX count and no fidelity difference (all four predictions hold); an unregistered 30% depth / 33% size reduction, most likely the same decomposer-configuration effect as Addenda 114-116 rather than anything PSF-Zero-specific (2026-09-24 night)

**Pre-registered in**:
`spare-qubit-cliff-addendum-156-preregistration-2026-09-24.md`. Raw
output observed as text in the conversation.

## 0. In one line

PSF-Zero's own block synthesizer was plugged into the connection for the
first time and ran on all 10 of its blocks (2 per tape, 5 tapes) with zero fallbacks, every block
verified on real `lightning.gpu` (worst 7.34e-13). Against Qiskit's
`TwoQubitBasisDecomposer` as used by the prototype: same routed CX count
(6 vs 6) on all 5 tapes, noisy TVD differing by 0.0014-0.0074 in no
consistent direction, identical noiseless TVD. Unregistered: PSF-Zero's
routed circuits were shallower (depth 16 vs 23) and smaller (56 vs 84
gates) on every tape.

## 1. Results

Routing to FakeManilaV2 (optimization_level=1, pinned layout); SamplerV2
local testing mode, 4000 shots, simulator seed 42.

| tape | arm | block CX | routed CX | depth | size | GPU diff | TVD noisy | TVD ideal | synth ms (median) | fallbacks |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | A without | 6 | 6 | 23 | 84 | 5.94e-15 | 0.1333 | 0.0145 | 0.074 | -- |
| 0 | B with PSF | 6 | 6 | 16 | 56 | 7.34e-13 | 0.1298 | 0.0145 | 0.614 | 0 |
| 1 | A without | 6 | 6 | 23 | 84 | 6.94e-15 | 0.0649 | 0.0199 | 0.081 | -- |
| 1 | B with PSF | 6 | 6 | 16 | 56 | 7.99e-15 | 0.0635 | 0.0199 | 0.575 | 0 |
| 2 | A without | 6 | 6 | 23 | 84 | 5.11e-15 | 0.1398 | 0.0210 | 0.070 | -- |
| 2 | B with PSF | 6 | 6 | 16 | 56 | 8.66e-15 | 0.1441 | 0.0210 | 0.600 | 0 |
| 3 | A without | 6 | 6 | 23 | 84 | 3.61e-15 | 0.0740 | 0.0238 | 0.072 | -- |
| 3 | B with PSF | 6 | 6 | 16 | 56 | 2.28e-14 | 0.0666 | 0.0238 | 0.782 | 0 |
| 4 | A without | 6 | 6 | 23 | 84 | 1.22e-15 | 0.0593 | 0.0218 | 0.068 | -- |
| 4 | B with PSF | 6 | 6 | 16 | 56 | 4.16e-15 | 0.0643 | 0.0218 | 0.608 | 0 |

## 2. Scoring

**P1 (both arms correct, zero fallbacks) -- CONFIRMED.**
**P2 (same routed CX count) -- CONFIRMED**, 6 vs 6 on every tape.
**P3 (noisy |TVD_A - TVD_B| < 0.05) -- CONFIRMED**, 0.0014-0.0074; B lower on
tapes 0, 1, 3, higher on 2, 4.
**P4 (noiseless TVD < 0.1) -- CONFIRMED**, at most 0.0238, identical
between arms per tape (equivalent unitaries, same simulator seed).

## 3. Unregistered observations

- **Depth 23 -> 16 and size 84 -> 56 with PSF-Zero, on every tape.** Not
  attributed to PSF-Zero itself. Arm A is the prototype's stand-in,
  `TwoQubitBasisDecomposer(CXGate())` with no `euler_basis` set -- the same
  unconfigured decomposer Addendum 114 traced PSF-Zero's own former excess
  single-qubit pulses to, and Addendum 116 fixed with `euler_basis="ZSX"`.
  Arm B's CX-basis entangling core is built with that fixed, ZSX-configured
  decomposer. The difference is therefore most likely the same
  configuration effect, and a ZSX-configured Qiskit decomposer would likely
  close most or all of it. Not tested here.
- **No fidelity gain from the smaller circuit.** The removed gates are
  single-qubit gates; CX and readout errors, which dominate the device
  snapshot's noise, are unchanged (same CX count, same measured qubits).
- **Synthesis time**: PSF-Zero's block synthesizer took about 8x longer per
  block in this path (median ~0.6 ms vs ~0.07 ms). Plausible cause: with
  `entangling_basis="cx"`, the CX core of each block is produced by a
  Qiskit decomposition on top of the Rust core's Cartan decomposition, and
  random blocks never hit the core cache. 10 blocks per arm (n_blocks=2
  per tape in the CSV); no timing claim is made.

## 4. What this means

PSF-Zero works as a drop-in synthesizer in this connection, correctly and
without fallback. At this scale it does not change what matters on the
device (CX count, noisy fidelity). This is consistent with, not a
departure from, this project's standing conclusion: PSF-Zero's own
established advantage is compile speed at the saturated large-scale layout
cliff, which a 4-qubit circuit on a 5-qubit device does not reach.

## 5. Files

| File | What it is |
|---|---|
| `compare_with_without_psf.py` | this run's script (Qiskit-level path, Addendum 156 Section 1a) |
| `compare_with_without_psf_2026-09-24.csv` | raw results, 10 rows |
