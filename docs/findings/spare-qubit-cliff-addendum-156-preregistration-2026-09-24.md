# Addendum 156 -- Pre-registration: the same PennyLane -> GPU-verified synthesis -> routing -> SamplerV2 connection, with and without PSF-Zero doing the synthesis (2026-09-24 night)

**Status: pre-registration only. No measurement has been run.**

## 1. Why this experiment exists

Every connection test so far (Addenda 152-155) used
`reference_cpu_synthesize` -- Qiskit's own `TwoQubitBasisDecomposer` -- as a
stand-in synthesizer. **No PSF-Zero code ran anywhere in that chain.** This
experiment swaps the synthesis step for PSF-Zero's own block synthesizer
(`psf_compile.SU4GeodesicPSFSynthesizer`, Rust-core Cartan decomposition,
`entangling_basis="cx"`) and runs the identical chain both ways.

## 1a. A design problem found before running anything

The prototype connection (`psf_pennylane_gpu_transform`) collapses every
synthesized block back into a single 4x4 `unitary` instruction before
converting to PennyLane and on to routing -- deliberately, per its own
comments, to sidestep a PennyLane/Qiskit gate-convention problem. Routing
(`transpile`) then re-synthesizes those matrices with Qiskit's own
decomposer. **So in that chain, the synthesizer's gate sequence never
reaches the device: Arm A and Arm B would produce the same routed circuit
by construction, and any "no difference" result would say nothing about
PSF-Zero.** Found by reading the splice step before running; the design
below uses a Qiskit-level path instead.

## 2. Design

- **Arm A ("without")**: `reference_cpu_synthesize` (Qiskit
  `TwoQubitBasisDecomposer(CXGate())`), unchanged from Addenda 152-155.
- **Arm B ("with")**: `SU4GeodesicPSFSynthesizer(GeodesicPSFHyper(
  entangling_basis="cx", on_unsupported="raise"), verify=True)`. With
  `on_unsupported="raise"`, any block the core cannot handle raises instead
  of silently falling back to Qiskit's decomposer -- so a completed Arm B run
  means every block really was synthesized by PSF-Zero's core.
- Both arms: tape -> Qiskit circuit -> `collect_and_consolidate` (the
  prototype's own block collection) -> each consolidated block synthesized
  by the arm's synthesizer and checked on real `lightning.gpu` (fixed
  version, Addendum 154) -> the synthesized GATES composed directly into
  the output circuit (not collapsed back to a matrix) -> routing to
  `FakeManilaV2` at `optimization_level=1` (which does not re-synthesize
  two-qubit blocks) -> submission through the real `SamplerV2` in local testing
  mode, to `FakeManilaV2` (noisy) and to `AerSimulator` (noiseless), 4000
  shots, fixed simulator seed.
- 5 tapes (two random 2-qubit blocks each, seeds 0-4), identical across
  arms.
- Recorded per tape and arm: two-qubit gate count of the synthesized blocks themselves (before routing); worst GPU check difference; routed two-qubit
  gate count, depth and size; noisy and noiseless TVD against the routed
  circuit's exact distribution; per-block synthesis time (reported, no
  claim attached -- 10 blocks per arm is far too few for a timing result).

## 3. Pre-registered predictions

**P1 (both arms correct).** Every block in both arms passes the real-GPU
check (difference < 1e-6), and Arm B completes with zero fallbacks.

**P2 (same gate count).** The routed two-qubit gate count is identical
between arms on every tape. Basis: a generic SU(4) block needs 3 CX by
either method, and Addenda 116-117 found PSF-Zero's CX-basis output matches
Qiskit's own re-compile after the decomposer fix.

**P3 (no meaningful fidelity difference at this scale).** Noisy TVD differs
between arms by less than 0.05 on every tape. PSF-Zero's own established
advantage (compile speed at the saturated large-scale layout cliff) does
not apply to a 4-qubit circuit on a 5-qubit device.

**P4 (noiseless sanity).** Noiseless TVD < 0.1 for both arms on every
tape.

If Arm B shows consistently fewer gates or lower noisy TVD, that is a new
finding, reported as such; if it shows more or higher, that is reported
equally.

## 4. What this cannot establish

- Anything about the large-scale cliff, where PSF-Zero's own advantage
  lies.
- Real hardware (2026-09-28).
- Timing, from 10 blocks per arm.
