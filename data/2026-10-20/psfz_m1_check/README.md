# psfz M1: the three improvements of 2026-10-20, built and checked (exploratory; not pre-registered)

Built at home on 2026-10-20 (about 03:30-04:15 CEST) from the workplace proposal
`docs/design/psfz_improvements_proposal_2026-10-20.md`, in the cloud workspace (one CPU core per run). Same harnesses
and seed (20261019) as `data/2026-10-19/`. Raw rows: `template_check.json`, `twin_check_pure_rust.json`,
`twin_check_lapack.json`.

**§1, templates: class-Z runs.** A run whose slots all turn into rotations about ±z, after conjugation by the fixed
factors on either side, is G · rz(φ) or rz(φ) · G for every value, with φ = Σ ± θ_slot. It is written in the short
form that G's fixed ZYZ angle t allows (0, 1 or 2 pulses), and bind sets one rz to base + φ. The right-hand case is the
workplace's rule. The left-hand case was added at home after the first check left TFIM runs `rx · F` in the general
form. Both read only the slots' gate names and the fixed factors; the left case is tried only when the right case fails.

| family (FakeFez / FakeTorino) | sx / cz, template 2026-10-19 | sx / cz, template now | sx / cz, r4 at the same values |
|---|---|---|---|
| TFIM n 6, 3 steps | 81 / 30 | **69 / 30** | 69 / 30 |
| TFIM n 10, 4 steps | 189 / 72 | **159 / 72** | 159 / 72 |
| TFIM n 16, 3 steps | 236 / 90 | **204 / 90** | 204 / 90 |
| ring classifier n 12, L 8 / 16 | 216 / 96, 408 / 192 | 216 / 96, 408 / 192 | 216 / 96, 408 / 192 |

Fixed shape 14 of 14, exact to 2.4e-16 in the measured distribution, bound angles equal the direct build (≤ 4.4e-16),
bind 0.03-0.39 ms. `compile` (r4): 96 of 96 outputs identical. Unit tests 17 of 17.

**§2 and §3, twin: one split per CZ-rotation-CZ block, and the bound counted per block.** Two CZ on one edge with
only single-qubit operations on one end and only rz on the other between them are one two-site gate W, applied with
one SVD. Each two-site gate's operator Schmidt rank is computed once, from its fixed 4 x 4 matrix, and the bound of a
cut is the sum of ceil(log2 rank) over the gates on it, capped by the half-widths. The expectation values were also
reduced from order chi^4 to chi^3 per site. That change was not in the proposal; it became the bottleneck once the
SVD was faster.

| TFIM chain, FakeFez | bound now | bound 2026-10-19 (CZ count) | reached | simulate, pure Rust (2026-10-19 → now) | simulate, `lapack` (2026-10-19 → now) |
|---|---|---|---|---|---|
| n 50, 6 steps | 2^6 | 2^12 | 64 | 26.0 → 7.4 s | 5.9 → 0.45 s |
| n 80, 6 steps | 2^6 | 2^12 | 64 | 42.3 → 11.2 s | 11.6 → 0.71 s |
| n 8-20, 2-4 steps | 2^steps (capped) | 2^(2 steps) | equal to the bound | ≤ 0.06 s | ≤ 0.01 s |

<Z> within 3.5e-15 of the state vector up to 20 qubits; dropped weight ≤ 2.4e-23.

- **Vanity figure:** the 80-qubit, 6-step chain in 0.71 s on one core (60 x faster than the pure-Rust prototype of
  2026-10-19), with the bound equal to the bond dimension reached in every case.
- **Essential figures:**
  - The bound is tight on these TFIM chains, but it is only an upper bound in general.
  - Templates now cost no more sx than r4 on TFIM, but only for runs whose shortness holds for every value. A run
    that r4 shortens because of particular values (an angle of zero) stays general, by design.
  - Chains only, noiseless; the LAPACK build used the workspace's single-threaded system library.
- Gate 1's T3 (IR draft) and the twin now count the same thing for TFIM: one rank-2 block per rzz, so 2^(k · steps)
  is an upper bound of the twin's block count. A record that passes T3 is never refused by the twin's bound.
