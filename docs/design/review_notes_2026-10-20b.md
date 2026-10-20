# Home review of the workplace documents W1-W3 of 2026-10-20 (about 03:40 CEST)

Reviewed at home the same morning. Checked by hand or by a short computation where stated; one measurement was made
(the twin's time against the bond dimension, §W2). Nothing else was run. The owner's decisions of the same morning
apply: certificate ε = 1e-12; inside shared types, outside HTTP (REST/JSON) first and a thin MCP wrapper second;
order S first, P0 and P1 in parallel.

## W1. `W1_psf_cert_spec_2026-10-20.md` (certificate)

**Checked and correct.**

- Gate widths: rz z^{±1}; rx, ry (z ± z^{-1}) / 2; p, u1 1 or z²; u3 width 2 in each variable. Amplitudes span 2 d_k.
- A probability is real, so its exponents are symmetric: a in [m, m + W] gives |a|² in [-W, W], a span of 2W = 4 d_k.
  The difference of two such polynomials spans 4 max(d_k^T, d_k^R), not the sum. Correct.
- Schwartz-Zippel with individual degrees on the grid of G-th roots of unity: P[miss] ≤ Σ deg_k / G. With 588 slots,
  D = 2,352, G = 2^32: 5.48e-7 per binding, 1.64e-19 for k = 3. Recomputed.
- With ε = 1e-12 (decided), k = 3 meets ε for every D ≤ G · ε^(1/3) ≈ 4.3e5, that is up to about 107,000 slots. v0's
  CZ and sx caps keep templates far below that, so k = 3 can be fixed for v0.
- τ ≈ 4.4e-11 at n 12, L 16 recomputed; the observed 2.4e-16 is 5 orders below it.

**One correction needed: the seed (§5).**

- seed = SHA-256(template ‖ reference ‖ "psf-cert/0") makes the bindings a fixed function of the polynomial under
  test. For a given defective template they either catch it or they do not; (D / G)^k is then not a probability of
  anything, only a heuristic (it holds if SHA-256 behaves like a random oracle with respect to the defect).
- Proposed fix: commit first, then draw. The template and reference hashes are recorded, then a fresh 256-bit nonce
  is drawn from the operating system and recorded in the certificate; seed = SHA-256(template ‖ reference ‖ nonce).
  The nonce is independent of the template, so the bound is a true probability, and anyone can still reproduce the
  bindings from the certificate.
- Small: crates.io is not reachable from the cloud workspace. ChaCha20 and SHA-256 are each about 100 lines of pure
  Rust and need no crate; or the stream can be SHA-256 in counter mode.

**One addition proposed: a causal-cone stage (A') before stage B.**

- The v0 records read local outcomes (ring classifier: one qubit; TFIM: ⟨Z⟩ per site). A marginal on a set of
  measured qubits depends only on the gates in its backward causal cone in that circuit's own gate list. The cone of
  T and the cone of R are each computed from their own gate lists, and each marginal is computed exactly by a state
  vector on its cone.
- The polynomial argument holds unchanged for marginals (a sum of |a|² terms).
- A TFIM chain gains about 2 qubits per side per brick layer, so the cone is about 4 · steps + 1 wide: about 25
  qubits at 6 steps, whatever n is. An 80-qubit, 6-step chain becomes about 80 cone state vectors of at most 25 qubits.
- This uses no MPS code, so the circularity of stage B (§6) does not arise. Stage B is then needed only for joint
  distributions over wide sets, or cones wider than n_SV.
- Cost has to be measured: a 25-qubit cone is 512 MB and seconds per binding; many cones overlap and could share work.
- The certificate must then say what it covers: "the marginals the record reads out", not the joint distribution.

**Stage A's cost at n_SV = 24.** The 10 s budget holds at n 12. At 24 qubits, a binding is 2^12 times more work per
gate, minutes rather than seconds. So n_SV = 24 is a memory limit, not a budget; the budget holds up to about 16-18
qubits.

**My view on W1 §9** (the owner decides):

1. ε: decided (1e-12). k = 3 and G = 2^32 meet it for all v0 sizes (above).
2. Stage C templates: refuse (`CERT_NO_BOUND`), as rule 3 says. With stage A', stage C should be rare in v0.
3. n_SV 24 as the memory limit; χ_cert see W2 below (1,024 is too slow).
4. δ_fail 1e-6 and s = 10: reasonable. The sentence "any real rewrite defect seen so far is ≥ 1e-3" has no cited
   record; it should cite one or be removed.
5. Stage B's residual risk: prefer stage A' for local readouts; stage B only where A' does not apply, with the
   residual risk stated.
6. Certificates for `circuit` output: yes (deterministic check at the given values plus C2).

## W2. `W2_psf_twin_noise_rings_2026-10-20.md` (twin with noise and rings)

**Checked and correct.**

- A one-site operator, unitary or not, cannot raise a Schmidt rank; renormalising does not change it. The 15-Pauli
  channel sampled as a product of two local Paulis is the same channel and raises no bond.
- The monomial block rule: each basis state of a follows one path, so the block is Σ_{a0} c |a_end⟩⟨a0| ⊗ V_{a0}, rank
  ≤ 2. Every noise operator in the model (Paulis, K0, K1, phase flip) is monomial. Correct.
- Folded order 0, n−1, 1, n−2, …: every cut is crossed by exactly 2 ring edges (checked at the ends and the middle of
  the fold), so B_c = min(2^(c+1), 2^(n−c−1), 2^(2s)). Correct.
- Moving T3 after the build, as one function on the built artifact, meets the morning's condition literally. Agree.

**The block rule needs one sentence about other two-site operations.** The rule speaks of operations on site a or b,
but a two-site operation on another edge that touches a or b (CZ(b, c), crosstalk RZZ(a', a)) can fall inside the
block. The rule still holds if it is stated as: *every operation touching a during the block is monomial on a or
diagonal in a's basis*; the other two-site operations then act inside V_{a0}, on one side of the cut, and the rank
across the cut (a, b) stays ≤ 2. Today's `plan_ops` in `mps.rs` is stricter: any two-site operation touching a or b
ends the block. That is valid but looser under crosstalk; the rule should be written down one way and the code made to
match.

**Crosstalk as TOQB really plans it.** `toqb/s2r.py` `with_crosstalk` adds, before each sx or x on q, RZZ with every
*active device neighbour* of q (and RZ for idle ones). These pairs come from the device's coupling map, not from the
circuit's edges. In a snake placement on heavy-hex, active neighbours seem to be only the chain neighbours, but that is
placement-dependent; `bond_bound` must read the scheduled plan, as W2 §3 says. Proposal for M3: the twin reads traj2's
plan (the steps list, exported as JSON). Then both engines run the same plan, and the agreement test compares engines,
not two schedulers.

**χ_max = 1,024 is too slow. Measured** (`psfz twin` with `lapack`, one core, cloud workspace, noiseless TFIM chains
on FakeFez, seed 20261020; exploratory):

| n, steps | bound = bond reached | two-site gates | simulate |
|---|---|---|---|
| 16, 6 | 64 | 90 | 0.05 s |
| 18, 7 | 128 | 119 | 0.40 s |
| 18, 8 | 256 | 136 | 2.04 s |
| 20, 9 | 512 | 171 | 11.08 s |

- About 5 × per doubling of χ. Extrapolated, not measured: χ = 1,024 is about a minute per trajectory.
- With 8,192 trajectories, χ = 512 is about 25 core-hours per circuit and χ = 1,024 about 140 core-hours.
- So the noisy twin's χ_max should be set by time: 128 (0.4 s per trajectory, about 55 core-minutes for 8,192) or 256
  (about 4.6 core-hours). χ_max = 1,024 only for single noiseless runs.

**The agreement list (W2 §4) contains cases the twin would refuse or could not finish.**

- Ring n 24, L 8: B = min(…, 2^16) is 2^12 = 4,096 in the middle, above any χ_max. Ring n 20, L 8: 1,024.
- Proposal: choose the list by the bound before running, every case B ≤ 256. For example, TFIM n 10, 16, 24 at 2
  steps; rings n 12 (L 2, 8), n 16 (L 2), n 20 and 24 (L 2).
- Count: the list as written is 11 circuits × 2 bindings × 2 models = 44 comparisons, not about 28. False-failure
  chance 44 × 6.3e-5 ≈ 0.003. The gate (4 se) still fits.

**My view on W2 §6:** 1 T3 after the build, one function (agree). 2 Folded order (agree). 3 χ_max by time: noisy
128-256, noiseless single runs up to 1,024. 4 List chosen by the bound, B ≤ 256, written down before the run. 5 Exact
P(1) per trajectory (agree; it matches traj2).

## W3. `W3_N2_adversarial_set_2026-10-20.md` (adversarial set N2)

**Checked.**

- The IR draft has **19** refusal codes (counted). The "20" in instruction 5 and ARCHITECTURE §8 was my miscount at
  home; there is no twentieth code. `CERT_NO_BOUND` (W1) is a later stop, not a gate-1 code, and belongs in
  ARCHITECTURE §8's list of stops.
- R14 and K5: CZ 2 · 23 · 10 = 460 and 2 · 23 · 8 = 368; bounds 2^10 = 1,024 > 256 and 2^8 = 256 (inclusive).
  R12: 12 · 64 = 768 > 500. K1 and K2 (96 / 216, 192 / 408) match the M1 template check. R11: n 100 is within the
  schema's maximum (100) and above FakeFez's longest chain (80-89), so T1 fires.
- `NUMBER_NOT_FINITE` is reachable with psfz's own JSON parser: Rust's `"1e999".parse::<f64>()` returns `inf` (tested),
  and psfz's `json.rs` parses numbers that way. With a finiteness check before the schema, R03 reaches the code.

**Agree with §0's corrections:** version and finiteness before the schema; evenness and n ≥ 12 for rings moved to R2;
an optional twin request field; the tighter device pattern (relative local paths, no `..`, no `://`).

**One addition.** Fix N2 (SHA-256 and CEST time) before gate 1's code is written, not only before a model sees it.
Then the gate is built to the specification, not to the test set, as with the owner's pre-registrations.

**My view on W3 §4:** 1 Adopt §0. 2 Keep `NUMBER_NOT_FINITE` (reachable, informative). 3 No twentieth code; the count
is 19. 4 The decline record: yes. Stop E relies on a person noticing the mismatch, which is the weakest of the stops.
5 Gate 1 enforces the ranges whatever the decoder does; decoder enforcement is optional. 6 F4 accepted, labelled
synthetic and test-only. 7 The owner fixes the set, before gate-1 code exists.

## Owner decisions (2026-10-20, 03:44 CEST)

- **W1 §5, the binding seed:** drawn after the template is fixed. The template and reference hashes are recorded
  first, then a fresh 256-bit nonce from the operating system; seed = SHA-256(template ‖ reference ‖ nonce), and the
  nonce is written into the certificate.
- **W2 §6.3, χ_max of the noisy twin:** 128 to start (about 0.4 s per trajectory at that bond, measured above).
- **W3 §4.4, the decline record:** adopted. `{"ir": "psf-intent/0", "decline": "out_of_family"}` ends out-of-family
  requests at the model's own record (stop D) instead of the echo (stop E).
- **W1 §9.2, stage C templates:** stopped (`CERT_NO_BOUND`) (03:47 CEST).
- **W3 §0:** all four corrections adopted (03:47 CEST); applied in `IR_v0_spec_r2_2026-10-20.md`.
- **N2 is fixed before gate-1 code is written** (03:47 CEST). Candidate: `data/2026-10-20/n2/`.
