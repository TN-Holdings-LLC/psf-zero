# W1. psf-cert: specification of the certificate (proposal, 2026-10-20)

**Status.** Written at the workplace on 2026-10-20 (03:30 CEST). This is a proposal for the owner, and nothing was
run. It answers instruction 5, W1.

**Basis:**

- design memo §6;
- ARCHITECTURE v0.1 §3, §4 and §8;
- psfz's code as read on 2026-10-20: `compile.rs` step 6 drops the last rz before a measurement; `template.rs` bind.

## 1. What is certified

psfz drops diagonal operations that only precede measurements, so the template's unitary is not the reference's. The
certificate therefore compares **the joint distribution of the measured bits**:

    Δ(x | θ) = p_T(x | θ) − p_R(x | θ)      for every outcome x of the measured classical bits

- **T** is the template.
- **R** is the reference circuit, built from the IR gate by gate with no rewriting, by an implementation independent of
  psfz's rewrites (§6).
- **θ** are the slot values.
- **"Correct"** means Δ(x | θ) = 0 for all x and all θ.

**The certificate is split in two, because of how a template works.** The emitted angles are closed-form but
non-polynomial functions of θ (ZYZ angles). The polynomial argument (§2) therefore applies to the template's
*structure*: its gate list, with each run's matrix taken as the product of its factors. That is the "factor
semantics", and it equals the emitted run up to a global phase whenever bind is right.

| part | what it shows | how |
|---|---|---|
| C1. Structure | factor-semantics T and R give the same measured distribution for all θ | random bindings on a grid, with the Schwartz-Zippel bound (§2) |
| C2. Bind | each emitted run, rz(a) sx rz(b) sx rz(c) at bind(θ), equals the product of its factors up to a phase | (i) bind's closed form is unit-tested as psfz's other rewrites are; (ii) at every certificate binding, every run's 2 × 2 identity is checked numerically and the largest deviation is reported |

A run's phase is a global phase of the whole state, so it does not change any probability. C1 and C2 together give
the emitted circuit's distribution.

## 2. The bound (C1), stated exactly

**Variables.** z_k = e^{iθ_k/2} for each slot parameter k. A slot gate with several parameters (u3) gives one variable
per parameter.

**Degree of one gate.** In every entry of a standard parametrised gate, z_k appears with exponents in a range of
width w = 2:

- rz: z_k^{−1} or z_k;
- rx, ry: (z_k ± z_k^{−1}) / 2;
- p, u1: 1 or z_k²;
- u3: width 2 in each of its three variables.

**Amplitudes.** ⟨y | U(θ) | 0⟩ is a Laurent polynomial in which z_k's exponents span at most W_k = Σ (over the gates
that use k) w = 2 d_k, where d_k is the number of gates parameter k enters.

**Probabilities: home's reading is correct.**

- On the unit circle, conj(z_k) = z_k^{−1}, so conj(a(z)) is again a Laurent polynomial with the same span.
- p(y | θ) = a · conj(a) therefore spans 2 W_k = 4 d_k in z_k: twice the amplitude's degree.
- A marginal over the unmeasured qubits is a sum of such terms, with the same span.

**The difference.**

    deg_k(Δ) ≤ 4 · max(d_k^T, d_k^R)       (not the sum: Δ is a difference, not a product)

- In factor semantics each slot is one factor of one run, so d_k^T = 1. In the reference, d_k^R is the number of gates
  using k, normally 1. So deg_k(Δ) ≤ 4.
- Multiplying by Π z_k^{2 max(d_k^T, d_k^R)} gives an ordinary polynomial with the same zeros on the circle and
  individual degree ≤ deg_k.

**The bound.** Each z_k is drawn independently and uniformly from the G-th roots of unity, that is
θ_k = 4π j_k / G. If the structure is wrong, some outcome x* has Δ(x* | ·) ≢ 0. Then (Schwartz-Zippel, with
individual degrees, on the grid S^m with |S| = G):

    P[ one binding misses ] ≤ D / G,      with  D = Σ_k deg_k(Δ) ≤ 4 Σ_k max(d_k^T, d_k^R)
    P[ k independent bindings all miss ] ≤ (D / G)^k

- **"Miss" means** every outcome agrees at that binding. All outcomes are compared at once, so it is enough that one
  outcome's polynomial is nonzero.
- **Example** (n = 12, L = 16; 588 slots, each used once): D = 2,352. With G = 2^32: 5.5e-7 per binding, 1.7e-19 for
  k = 3. These are the memo's numbers; for distributions the degree count gives the same value.

**What the bound does not cover** (stated in the certificate):

- **It is about exact values.** A difference polynomial that is nonzero but stays below the numerical tolerance τ (§4)
  at every sampled binding is not distinguished from rounding. The certificate states "exact identity except with
  probability ≤ (D / G)^k, or a defect smaller than τ at every sampled binding". A general lower bound on |Δ| at random
  points (anti-concentration) is not claimed.
- **Bind (C2) is a closed form,** proved by unit tests and checked at the k bindings, not bounded by this argument.

## 3. Stages by width

| stage | when | method | bound | basis |
|---|---|---|---|---|
| A. State vector | active qubits ≤ n_SV (proposal: 24; 256 MB in complex128) | full distribution of the measured bits for T and R at each binding | (D / G)^k as in §2 | exact polynomial argument; evaluation exact up to rounding (§4) |
| B. MPS (instruction's option 2) | chains, and rings in W2's folded order, with W2's bond bound ≤ χ_cert (proposal: 1,024) for both T and R | distribution of the measured bits from the MPS of T and of R; splits drop only singular values below 1e-12 × the largest, and the dropped weight is added to the tolerance (§4) | (D / G)^k, conditional on the MPS being exact to the reported dropped weight | same polynomial argument; the circularity guard is in §6 |
| C. Rules only (instruction's option 3) | anything beyond A and B | none at run time: the closed forms of every rewrite used (psfz r1-r5) are unit-tested | **no probabilistic bound**; the certificate says so | the same as psfz r1-r4 today |

**Option 1 (comparing a random subset of outcomes) is not proposed as a stage.** A defect can change the
probabilities of only a few outcomes among 2^m. A uniformly drawn subset then finds it with a probability as small
as (subset size) / 2^m. No useful bound holds in the worst case. It can serve as a smoke test inside stage B.

**Stage C and ARCHITECTURE rule 3.** Rule 3 says no circuit leaves without a certificate whose bound is at most ε.
Under it, a stage-C template cannot leave. Proposal: stop it with a new stop `CERT_NO_BOUND`. Whether rule 3 or a
labelled release applies is an owner point (§9).

## 4. Numerical tolerance: telling rounding from a real mismatch

**Error model.** u = 2^−53 ≈ 1.1e-16 (complex128). Each gate applied to a state vector (a 2 × 2 or diagonal 4 × 4
matrix-vector step, with matrix entries from closed forms or trigonometric functions) adds at most about 8 u to
‖ψ̃ − ψ‖₂. Unitary steps do not amplify earlier errors, so after g gates ‖ψ̃ − ψ‖₂ ≤ 8 u g to first order. For any
set of outcomes, |‖Pψ̃‖² − ‖Pψ‖²| ≤ 2 ‖ψ̃ − ψ‖ + ‖ψ̃ − ψ‖².

**Thresholds** (fixed formulas; s = 10 is a safety factor):

    τ      = s · 16 u · (g_T + g_R)                          stage A
    τ_MPS  = τ + 2 · (dropped weight of T + dropped weight of R)     stage B
    τ_run  = s · 8 u · (factors in the run + 5)              C2, per run, after aligning the global phase
    δ_fail = 1e-6                                            proposal; any real rewrite defect seen so far is ≥ 1e-3

**Verdict per certificate:**

- **PASS:** max |Δ| ≤ τ (or τ_MPS) at every binding, and every run is within τ_run.
- **FAIL** (`CERT_FAILED`, a bug, not a refusal): any |Δ| ≥ δ_fail, or a run deviation ≥ δ_fail.
- **INCONCLUSIVE:** anything between.
  - The same bindings are recomputed once with compensated (double-double) accumulation.
  - If the result is still in the zone, the certificate stays INCONCLUSIVE, which counts as not certified under
    rule 3.

**Example, n = 12, L = 16.** g_T is about 1,900 and g_R about 600, so τ ≈ 10 × 16 × 1.1e-16 × 2,500 ≈ 4.4e-11. The
template check of 2026-10-19 saw at most 2.4e-16. That leaves about 5 orders of magnitude between rounding and τ,
and 5 more between τ and δ_fail.

## 5. Bindings: drawn reproducibly, not chosen

- **Seed.** seed = SHA-256(template_sha256 ‖ reference_sha256 ‖ "psf-cert/0"). The grid indices j_k come from
  ChaCha20 with that seed. Anyone can reproduce the bindings, and the builder cannot pick them.
- **Angles.** θ_k = 4π j_k / G, computed in double. They are not exactly on the grid; that rounding is part of the
  error model (it moves each probability by about u · D) and is covered by τ.
- **k is fixed per certificate** (proposal: k = 3). One more binding is drawn only on INCONCLUSIVE.

## 6. The circularity guard (stage B, and the reference in general)

- **The reference R** is built from the IR gate by gate by a separate module. It uses none of psfz's rewriting code.
  Its gate matrices come from one table, which is unit-tested against Qiskit's matrices.
- **Stage B simulates T and R with psf-twin's MPS,** whose own correctness is established separately:
  - against the state vector up to 20 qubits (2026-10-19: 6.5e-15);
  - against traj2 (W2 §4).

  A bug shared by the MPS and psfz could still cancel. The guard is that stage B runs only on MPS code paths that are
  also exercised in stage A sizes, and the certificate records the MPS version.
- **Residual risk** (stated in stage B certificates): a defect in the MPS that appears only above n_SV and that
  affects T and R identically. It cannot be excluded by this method.

## 7. The certificate artifact (ARCHITECTURE §3, in detail)

```json
{
  "format": "psf-cert/0",
  "verdict": "PASS | FAIL | INCONCLUSIVE",
  "stage": "A-statevector | B-mps | C-rules",
  "compared": "joint distribution of measured bits",
  "measured_qubits": [0],
  "inputs": {"template_sha256": "...", "reference_sha256": "...", "intent_sha256": "...",
             "device_sha256": "...", "psfz_version": "...", "cert_version": "...", "twin_version": "..."},
  "structure": {
    "slots": 588, "variables": 588,
    "degree": {"rule": "deg_k <= 2 * max(W_k^T, W_k^R), W = sum of gate widths (2 per standard gate)",
               "per_variable_max": 4, "sum_D": 2352},
    "grid": {"G": 4294967296, "variable": "z_k = exp(i theta_k / 2)", "theta": "4 pi j / G"},
    "bindings": {"k": 3, "seed_sha256": "...", "prng": "chacha20", "extra_on_inconclusive": 0},
    "failure_bound": {"per_binding": 5.48e-7, "total": 1.64e-19, "formula": "(D / G)^k"}
  },
  "bind_check": {"runs": 216, "max_dev": 0.0, "tau_run_max": 0.0},
  "tolerance": {"u": 1.11e-16, "s": 10, "g_T": 1900, "g_R": 600, "tau": 4.4e-11, "delta_fail": 1e-6,
                "mps_dropped_weight": null},
  "observed": {"max_abs_diff": 0.0, "per_binding": [0.0, 0.0, 0.0]},
  "limits_stated": ["exact identity except with probability <= total, or a defect below tau at every sampled binding",
                    "bind proved by unit-tested closed forms and checked at the k bindings"],
  "timing_s": {"build_R": 0.0, "simulate": 0.0, "total": 0.0},
  "produced": "ISO time with CEST"
}
```

The numbers above are placeholders except the example's D, G and bounds.

## 8. Cost (estimate, not measured)

For n = 12, L = 16, stage A:

- **Per binding:** T has about 1,900 and R about 600 single- or two-qubit steps on 2^12 amplitudes. That is about
  2,500 × 2^12 × 8 ≈ 8e7 floating-point operations, about 0.1 s single-threaded in Rust.
- **k = 3, both circuits, plus the run checks:** about 0.3-0.5 s.
- **Python (numpy) harness:** a few seconds.

Within the 10 s budget. Stage B costs as psf-twin at the bound's bond dimension (W2 §5).

## 9. Points for the owner to decide

1. **ε of rule 3,** and whether (D / G)^k ≤ ε with k = 3 and G = 2^32 is accepted (1.7e-19 at n = 12, L = 16).
2. **Stage C templates.** Refused under rule 3 (the proposed stop `CERT_NO_BOUND`), or released with a labelled
   "rules-only" certificate.
3. **n_SV** (proposal 24) and **χ_cert** (proposal 1,024) for stage B.
4. **δ_fail = 1e-6 and s = 10.**
5. **Stage B's residual risk** (§6): accepted as stated, or stage B limited to sizes where an independent simulator
   (for example Qiskit Aer's MPS) re-checks a sample.
6. **Whether `circuit` output (bound angles) needs a certificate too.** Proposal: yes. The same C1 at the given values
   plus C2 is a deterministic check with no bound needed. The polynomial bound only matters for templates.
