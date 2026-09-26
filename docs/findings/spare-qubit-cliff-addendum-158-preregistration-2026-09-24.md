# Addendum 158 -- Pre-registration: is Addendum 157's 30% depth reduction PSF-Zero-specific, or just the euler_basis="ZSX" configuration? A third arm settles it (2026-09-24 night)

**Status: pre-registration only. No measurement has been run.**

## 1. Why this experiment exists

Addendum 157 found PSF-Zero's routed circuits shallower (depth 16 vs 23)
and smaller (56 vs 84 gates) than the prototype's stand-in,
`TwoQubitBasisDecomposer(CXGate())` with no `euler_basis`, on all 5 tapes,
and attributed this -- without testing it -- to the same unconfigured-
decomposer effect Addendum 114 found and Addendum 116 fixed inside
PSF-Zero's own CX path with `euler_basis="ZSX"`. This adds the missing arm.

## 2. Design

Identical to Addendum 156/157 (same 5 tapes, Qiskit-level path, real
`lightning.gpu` check per block, FakeManilaV2 routing at
`optimization_level=1`, SamplerV2 local testing mode, 4000 shots, seed 42),
with three arms:

- **A**: `TwoQubitBasisDecomposer(CXGate())` (unconfigured; Addendum 157's
  Arm A)
- **B**: PSF-Zero, `SU4GeodesicPSFSynthesizer`, `entangling_basis="cx"`,
  `on_unsupported="raise"` (Addendum 157's Arm B)
- **C (new)**: `TwoQubitBasisDecomposer(CXGate(), euler_basis="ZSX")` --
  exactly the decomposer configuration PSF-Zero's own CX path uses

## 3. Pre-registered predictions

**P1 (the main one).** C's routed depth and size equal B's (16 and 56) on
every tape. **If C's depth or size differs from B's on any tape, the
reduction is at least partly PSF-Zero-specific, and Addendum 157 Section 3's
attribution is wrong.**

**P2.** C's routed CX count equals A's and B's (6) on every tape.

**P3.** C's noisy TVD is within 0.05 of B's on every tape.

**P4.** A and B reproduce Addendum 157's own figures exactly (same depth,
size and TVD values), confirming the run is comparable to the previous one.

## 4. What this cannot establish

Unchanged from Addendum 156: nothing about the large-scale cliff, real
hardware, or timing.
