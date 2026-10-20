# W2. psf-twin with noise and rings: design (proposal, 2026-10-20)

**Status.** Written at the workplace, 2026-10-20 (about 03:40 CEST). This is a proposal for the owner, and nothing
was run. It answers instruction 5, W2.

**Basis:**

- the MPS prototype (`psfz twin`; `mps.rs`);
- the psfz improvement proposal of 2026-10-20 (§2: one rzz as one two-site gate; §3: bound from the gates);
- TOQB's noise model as `toqb/cudaq_engine.py` describes traj2's plans:
  - "g1": rz, sx / x with over-rotation, depolarizing, phase flip, amplitude damping;
  - "cz": CZ with 2-qubit depolarizing, then phase flip and amplitude damping on each qubit;
  - "zz": static crosstalk RZZ;
  - "relax": idle phase flip and amplitude damping;
  - readout applied to ⟨Z⟩.

## 1. Noise by trajectories, and why local noise does not raise a bond

**Method.** As in traj2, each trajectory samples one Kraus operator (or one Pauli) per noise point and applies it to
the MPS, then renormalises. A trajectory reports its exact P(1) on the measured qubit, so its per-trajectory value has
the same meaning as traj2's.

**Claim: local operations never increase a bond dimension.** For any operator K on one site (unitary or not),
rank_cut(K ⊗ I · ψ) ≤ rank_cut(ψ) for every cut, because applying K to one side of a Schmidt decomposition cannot add
terms. The noise channels:

| noise point | what a trajectory applies | local? | effect on the bound |
|---|---|---|---|
| depolarizing (1 qubit) | I, X, Y or Z on the site | yes | none |
| phase flip | I or Z | yes | none |
| amplitude damping | K0 = diag(1, √(1 − g)) or K1 = √g · \|0⟩⟨1\|, then renormalise | yes (non-unitary) | none (K1 can lower a rank) |
| 2-qubit depolarizing after CZ | one of 16 Paulis P_a ⊗ P_b, with probabilities 1 − 15 dep/16 and dep/16 | **a product of two local Paulis**: apply P_a on site a, then P_b on site b | none |
| over-rotation | part of the sx / x gate matrix | yes | none |
| readout | applied to ⟨Z⟩ after the run | classical | none |
| ZZ crosstalk | RZZ(θ) on a coupled pair | **no: two sites** | counted (§3) |

- **Home's two questions, answered.** The bound holds unchanged under 1-qubit noise. The 15-Pauli channel, sampled as
  one product Pauli per trajectory, is exactly the same channel as traj2's and changes no bond.
- **Normalisation.** Amplitude damping needs the branch probabilities ‖K_j ψ‖². With the orthogonality centre moved to
  the site, each costs O(χ² d), as in standard MPS trajectory codes. The centre is moved as the gates already require,
  so there is no extra SVD.

## 2. One rule for "how much a block of operations can raise a bond"

**Monomial operators.** A single-site operator is *monomial* when each row and each column has at most one nonzero
entry. These are:

- rz and x;
- every Pauli;
- K0 and K1 of amplitude damping;
- phase flip.

sx, rx(θ), ry(θ), H and general u are not monomial.

**Block rule (rank ≤ 2).** Take a sequence of operations on one edge (a, b) in which:

- every two-site operation is diagonal (CZ, ZZ crosstalk, cp, the diagonal rzz);
- every operation on one of the two sites (say a) is monomial;
- operations on the other site (b) are arbitrary.

The whole sequence has operator Schmidt rank at most 2 across (a, b).

**Proof sketch.** A monomial operator maps each computational basis state of a to a multiple of one basis state.
Each of a's two input states therefore follows one deterministic path, and a diagonal two-site gate acts on b
conditioned on a's current basis state. So the block equals Σ_{a0 ∈ {0, 1}} c_{a0} |a_end(a0)⟩⟨a0| ⊗ V_{a0}: two terms.

**What it covers.**

- psfz's ZZ form, H · CZ · RX(θ) · CZ · H: H and RX are on b, nothing is on a.
- ZZ crosstalk next to a CZ.
- Every noise jump in between.

All of these are one block of rank ≤ 2, whatever the trajectory's jumps are, because every noise operator in the
model is monomial. The bound is therefore the same for every trajectory and is decided before the run.

**Blocks are cut greedily, in time order.** A block on edge (a, b) continues while both hold:

- its two-site operations are diagonal;
- at least one endpoint has seen only monomial operations since the block began.

The first operation that breaks this starts a new block. Non-monomial operations on both endpoints, such as the sx
of a ring classifier's layer, end every block.

- **Rank of a block that breaks the rule** (a non-diagonal two-site gate such as rxx): the rank of its noiseless
  4 × 4 matrix, realigned (singular values above 1e-12 × the largest). If noise points fall inside such a block, the
  rank is min(4, product of the parts' ranks).
- **This is a fixed partition, not a search for the best one.** Any partition gives a valid bound; greedy is simply
  the declared one.

## 3. The single bound (for gate 1's T3 and the twin's refusal)

Let π be the MPS site order (§5): the identity for a chain, the folded order for a ring. Cut c separates the sites
at positions 0 … c from the rest. Block β on qubits (a, b) crosses cut c when min(π(a), π(b)) ≤ c < max(π(a), π(b)).

    B_c = min( 2^(c+1),  2^(n−c−1),  Π_{blocks β crossing c} r(β) )
    B   = max_c B_c                          refuse when B > χ_max   (TWIN_BOND_LIMIT)

with r(β) = 2 for blocks under §2's rule, and the realigned rank (≤ 4) otherwise.

**Proposal: one function, run on the built artifact.**

- `bond_bound(ops, noise_plan, order)` is evaluated on the built template (or circuit) and its noise plan.
- **This moves T3 after psfz's build** (milliseconds). Gate 1's T3 and the twin's admission check then call the same
  function on the same input. That meets home's condition literally: the same formula and the same input.
- A pre-build T3 on the IR alone cannot see psfz's ZZ form, the schedule's crosstalk points or the noise plan, so it
  could only be a looser formula.
- **An IR-only condition may still be kept as an early refusal,** but only one that never refuses what
  `bond_bound` would admit. For a TFIM chain without crosstalk this is 2^min(c+1, n−c−1, k · steps), and it is exact
  there.
- **Alternative for the owner:** keep T3 in gate 1 on the IR and accept that the twin may refuse later. That is two
  formulas, which is what home asked to avoid.

**Crosstalk in the count.**

- Each ZZ step on a coupled pair is a diagonal two-site operation.
- When it falls in the same greedy block as the edge's CZ (adjacent in time, one side monomial), it costs nothing
  extra. Otherwise it opens a block of rank 2 on its pair.
- Pairs that are not neighbours in the site order cross every cut between them.
- Which pairs and how many steps there are comes from the noise plan (traj2's "zz" steps), so the count is exact for
  that plan.

## 4. Agreement test against traj2 (same form as `check_cudaq`)

- **Circuits** (fixed list, written down before the run):
  - TFIM chains at n = 10, 16 and 24 (2 steps);
  - ring classifiers at n = 12, 16, 20 and 24 (L = 2 and 8);
  - 2 random slot bindings each, seeded from the circuit's SHA-256.
- **Noise models:** the twin's model (device file) and the stand-in's (its Marrakesh-based model with over-rotation
  and ZZ). These are the two models `check_cudaq` used.
- **Statistic:** for each circuit and model, ⟨Z⟩ on the measured qubit from T = 8,192 trajectories in each engine.
  |d| / se, with se² = Var_twin / T + Var_traj2 / T, using each engine's per-trajectory sample variance.
- **Gate:** every |d| / se < 4. About 28 comparisons, so the chance of a false failure is about 28 × 6.3e-5 ≈ 0.002.
- **Plus the exact noiseless check:** the twin's P(1) against the state vector, below 1e-12, up to 20 qubits.
- **Recorded:**
  - the bound B and the reached bond dimension per cut (they must satisfy reached ≤ B);
  - the dropped weight;
  - times for both engines.

## 5. Rings: three ways, compared

n qubits form a ring; L layers, each with one CZ block per ring edge (the ring classifier).

| | 1. Folded order (proposed) | 2. Periodic MPS | 3. SWAP the closing edge |
|---|---|---|---|
| site order | 0, n−1, 1, n−2, 2, … : every ring edge spans MPS distance 1 or 2 | the ring itself; tensors in a closed loop | the chain order; the closing CZ is carried by swaps |
| bound | each cut is crossed by exactly 2 ring edges, so B_c = min(2^(c+1), 2^(n−c−1), 2^(2 s)), with s blocks per edge (s = L for the classifier) | across a bipartition into two arcs, 2 bonds are cut: rank ≤ χ_1 χ_2 | each chain cut is crossed by its chain edge and by the closing edge in transit: 2^(2 s) after each layer, and a factor 2 more while a qubit is in transit |
| how a distance-2 gate is applied | contract the 3 sites, apply the gate on the outer two, split twice; the middle cut is crossed and counted, so no step exceeds B | all gates are nearest neighbour | 2(n−2) swaps per closing gate (rank 4 each, exact) |
| splits per layer | about 2n | about n, but on the loop | about 3n |
| cost per split | O((2χ)³) with χ ≤ B | expectation values O(n χ⁵); no exact canonical form, so dropping small singular values has no error bound | O((2χ)³), with χ up to 2B in transit |
| exactness | as for chains: only singular values below 1e-12 dropped; dropped weight reported | not exact in the same sense (no orthonormal gauge) | exact, but transient bonds exceed B |
| code | reuses the chain MPS; needs the 3-site update | new code | reuses the chain MPS |

**Proposal: option 1.** It keeps the chain code, its exactness and the single bound of §3. The ring's cost shows in
the formula (2s instead of s per cut), as it must: a ring state has twice the entanglement across a cut that a chain
has.

**Example.** A ring classifier with n = 12 and L = 16 has B = min(2^(c+1), 2^(11−c), 2^32), which is at most 2^6 = 64
in the middle. That is a state-vector-sized case, so stage A of W1 applies anyway. Rings matter for the twin at
n ≥ 26 or so, where B is at most 2^min(13, 2L).

**Cost formula** for one trajectory (option 1):

    time ≈ Σ over two-site blocks of c_svd · (2 χ_c)³ + (local operations, negligible)

χ_c ≤ B_c is the bond at the block's cut, and c_svd is the SVD's constant: pure Rust, or the `lapack` feature. With
noise, multiply by T. Trajectories are independent and run in parallel on the pod's cores.

## 6. Points for the owner to decide

1. **T3 moves after the build** (one function on the built artifact), or stays in gate 1 on the IR (two formulas).
2. **The ring method:** folded order (proposal), periodic MPS, or swaps.
3. **χ_max** for the twin (proposal: 1,024 on the pod; it can be lower on a workstation).
4. **The agreement test's circuit list and gate** (4 se, T = 8,192), as in `check_cudaq`.
5. **Whether the twin reports per-trajectory exact P(1)** (as traj2) or shots (as CUDA-Q). Proposal: exact P(1), to
   match traj2's estimator.
