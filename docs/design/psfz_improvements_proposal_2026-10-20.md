# psfz: three improvements by closed-form rules (proposal, 2026-10-20)

**Status.** Written at the workplace, 2026-10-20, about 01:45 CEST. These are proposals for the owner. Nothing was
built or run here; the sources were read from `workplace_pack_2026-10-20.zip`: `rust/psfz/src/compile.rs`,
`template.rs`, `mps.rs`, and the check notes of 2026-10-19. Every rule below reads the structure of the input or
fixed numbers of the template. None reads a slot's value, scores alternatives or searches.

---

## 1. Templates: runs that are diagonal for every value, recognised from the structure

**The cause of TFIM's extra sx (from the code).**

- psfz writes ZZ(θ) as H · CZ · RX(θ) · CZ · H, with the basis changes on one qubit (compile.rs, step 5).
- In a TFIM chain the field rotation rx(h·dt) is a slot. It therefore sits in a run between fixed factors that
  include those H's.
- H · rx(β) · H = rz(β): the run is diagonal for every β. r4 sees a diagonal matrix at the given values and writes
  one rz. The template writes the general form rz sx rz sx rz, which costs 2 sx.
- The 16-19 % (12 / 30 / 32 more sx at n = 6 / 10 / 16) equals 2 sx in each of 6 / 15 / 16 runs. This is consistent
  with the explanation but not counted run by run here.

**The rule (class Z).** A run with slots has factors in time order F_0, S_1, F_1, …, S_m, F_m. The F are fixed
matrices (possibly the identity). Each S_k is a slot whose gate is a rotation about a fixed axis n_k:

- rz, p and u1 about z (p and u1 up to a phase);
- rx about x;
- ry about y.

u, u2 and u3 have no single axis: a run containing one stays general.

    A_k = F_(k-1) ··· F_0           (the fixed product before slot k)
    G   = F_m ··· F_0               (all fixed factors)
    M(θ) = G · T_m(θ_m) ··· T_1(θ_1),   with  T_k(θ) = A_k^† S_k(θ) A_k

The identity is exact; it inserts A_k A_k^† between the factors. T_k is a rotation by θ_k about the axis
a_k = R(A_k)ᵀ n_k, where R(A) is A's rotation in SO(3).

- **Class Z holds** when every a_k is ±z: |a_k · ẑ| ≥ 1 − TOL, with TOL = 1e-10 as `is_diag` uses. Then
  T_m ··· T_1 = rz(φ), with φ = Σ_k sign(a_k · ẑ) θ_k. φ is affine in the slots.
- **Writing the run.** With G's fixed ZYZ angles (p_G, t_G, l_G), M(θ) = rz(p_G) · ry(t_G) · rz(l_G + φ). The form
  depends on t_G, which is fixed:

| t_G (fixed) | written as | sx / x |
|---|---|---|
| 0 | rz(p_G + l_G + φ) | 0 / 0 |
| π | rz(l_G + φ + π) x rz(p_G) | 0 / 1 |
| π/2 | rz(l_G + φ − π/2) sx rz(p_G + π/2) | 1 / 0 |
| otherwise | the general form (as now) | 2 / 0 |

- **Before a measurement,** the last rz is dropped as now. A t_G = 0 run before a measurement disappears entirely.
- **Otherwise the run is general,** unchanged.

**Why this keeps S1-S3.**

- The class reads only the slots' gate names and the fixed factors. Fixed two-qubit angles are template structure,
  since they are not slots in r5. So the gate list still depends only on the IR and the device (S1).
- Each output angle is affine in φ, a closed form (S2).
- The slot map records φ's signs, so bind evaluates them (S3).

**Where it must also be used.** The scheduling rule "a pending run with a slot counts as two pulses" (step 5, `pick`)
becomes "a pending run counts as its class's pulses". These are 0, 1 or 2, computed on the run so far by the same
rule, so they are still structural. This can move basis changes compared with today. It must be checked (below), not
assumed neutral.

**Checks to run (home, CPU):**

- unit tests: H rx H is class Z; S rx S† (axis y) is not; rx rz rx is not; a chain of rz is;
- the S1 harness at three random value sets;
- exactness at random bindings against the input, below 1e-15 as before;
- sx and cz against r4 on the same 6 TFIM and 8 ring-classifier inputs. Expected: TFIM sx equal to r4's at generic
  values; ring classifier unchanged (216 / 96, 408 / 192);
- `compile` (r4) outputs unchanged (96 / 96).

---

## 2. MPS twin: one rzz as one two-site gate, one split

**Pattern** (read from the compiled gate list; structural). Two CZ on the same edge (a, b) of the chain, with only
these operations on a and b between them:

- single-qubit operations on b;
- nothing on a, or only diagonal operations on a (rz commutes with CZ).

The pair and the operations between them become one two-site gate:

    W = (D_a ⊗ I) · CZ · (I ⊗ U_b) · CZ = (D_a ⊗ I) · ( |0⟩⟨0| ⊗ U_b + |1⟩⟨1| ⊗ Z U_b Z )

- U_b is the product of b's operations between the two CZ; D_a is that of a's diagonal operations.
- **W has operator Schmidt rank at most 2.** It is 1 when U_b commutes or anticommutes with Z, that is when U_b is
  diagonal or anti-diagonal.
- psfz's ZZ form produces exactly this pattern: H · CZ · RX · CZ · H with RX on b.

**Applying W.** Contract W with the two-site tensor (χ_l × 2 × 2 × χ_r), then do one SVD of the (2χ_l) × (2χ_r)
matrix, dropping singular values below 1e-12 of the largest, as now.

- The new bond is at most 2 × the old one, which is the final value.
- Today each rzz costs two splits, the second at up to twice the final bond dimension (home's note of 2026-10-19).

**Expected effect** (estimate, not measured): the split that now runs at twice the final dimension runs at the final
one. That is about 2³ = 8 × fewer operations for that split, and one split fewer per rzz. The SVD dominates at
bond dimension 64 (26-42 s pure Rust, 5.9-11.6 s with LAPACK).

**Exactness.**

- W is a 4 × 4 product of the same matrices, so the state is the same up to rounding.
- Checks:
  - ⟨Z⟩ against the state vector up to 20 qubits (existing harness);
  - the dropped weight;
  - the reached bond dimension against the bound of §3.

**Alternative (owner's choice; not recommended for the twin).** Simulate at the compiler's ZZ level, applying
exp(−iθ ZZ/2) as one diagonal two-site gate. This is cheaper still, but it simulates the intermediate form, not the
gate list that is run. The recommendation is the pattern on the output, so the twin simulates exactly what is sent;
with noise added later, the operations between the CZ carry their own trajectory noise inside U_b.

---

## 3. The bond-dimension bound counted from the input's gates

**Formula.** The chain's sites are in chain order, and cut i lies between sites i and i + 1.

    χ_i ≤ min( 2^(i+1),  2^(n−i−1),  Π_{blocks b on edge (i, i+1)} r(b) )

- **r(b), the operator Schmidt rank of block b:** the number of singular values above 1e-12 × the largest, of b's
  4 × 4 matrix realigned as (a a′) × (b b′). That is a fixed 4 × 4 computation, not a search.
- **A block** is a maximal run of two-qubit operations on edge (i, i + 1), taken with the single-qubit operations
  between them on those two qubits. Local operations do not change the rank, so merging them is allowed, and
  r(b) ≤ min(4, product of the parts' ranks).
- **Only edge (i, i + 1) counts for cut i.** On a chain, no other two-qubit operation crosses that cut, and
  single-qubit operations never change a bond.

**Ranks of the input gates:**

- rzz(θ): 2, or 1 when θ ≡ 0 (mod π);
- cz, cx, cp(λ ≠ 0), crz, rxx, ryy: 2;
- a general block or swap: up to 4.

If an angle is a slot (not in r5, where two-qubit angles are fixed), the structural value is used: 2 for rzz.

**Why counting from the input is valid.**

- Each input block and its compiled form (for example ZZ as CZ · RX · CZ with basis changes) differ only by
  single-qubit operations, so they have the same rank.
- When compile's ordering merges input operations on the same pair, the merged rank is at most the product. The input
  count stays an upper bound.
- **It bounds the bond at block boundaries.** Inside a block applied gate by gate (two CZ today), the intermediate
  bond can exceed it. With §2 a block is one gate, so the bound also holds at every split.

**Expected.**

- A TFIM chain with s first-order steps (one rzz per edge per step) gets the bound 2^min(s, i+1, n−i−1). That is
  4 / 16 / 64 at s = 2 / 4 / 6, equal to the bond dimension reached in the check of 2026-10-19, instead of today's
  2^(2s).
- The refusal rule is unchanged: a bound above CHI_MAX is refused before the run. Runs that the CZ count wrongly
  refused would now be admitted.

**Checks:**

- the table of 2026-10-19 rerun, with the bound column equal to the reached column;
- one input with a general two-qubit block (rank 4) and one with rzz(π) (rank 1), to test both ends.

---

## Vanity and essential figures (when built and checked)

| | figure |
|---|---|
| Vanity | "TFIM templates now cost no more sx than r4"; "the bound is exact" |
| Essential | §1 removes only the diagonal-for-every-value runs. Runs whose shortening r4 owes to particular values (an angle equal to zero, for example) stay general in a template, by design. §2's speed-up is an estimate until timed. §3 is an upper bound that is tight on TFIM chains; it is not proven tight in general |
