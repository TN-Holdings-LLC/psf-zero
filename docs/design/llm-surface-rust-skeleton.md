# Design memo: an LLM on the surface, PSF-Zero in Rust as the skeleton (concept, 2026-10-19)

**Status: concept. Nothing here is built, run or pre-registered.** Written at home on 2026-10-19 (CEST) at the
owner's request ("the surface is an LLM, but the skeleton, down to the bones, is PSF in Rust, so that hallucination
cannot happen"). Values marked `[ ]` are open. Each phase below is to be pre-registered before it is scored.

## 1. The idea in one paragraph

The language model never writes a gate. It writes an **intent**: a small typed record (the intent IR, §3) that names
a circuit family from PSF-Zero's scope and its parameters. Rust checks the record (gate 1, §4), builds the circuit by
closed forms on the device's lattice (psfz, §5), and attaches a certificate that the built circuit is the circuit the
intent describes (§6). Everything the user receives (circuit, template, numbers) comes from Rust or from the twin,
never from the model's text. What the model can still get wrong is *which* intent the user meant; that is shown back
to the user in a canonical form (§7) and is the one error this design cannot remove.

**Where this comes from.** The vLLM x PSF-Zero line (Addenda 263-270, 2026-09-30) had the model write circuits that
PSF-Zero then compiled. This design inverts that division of labour: the model chooses, Rust constructs.

## 2. What is claimed, and what is not

**Claimed (testable, §9):** no circuit leaves the system unless it was built by Rust from an intent that passed gate 1,
and its certificate holds. Invalid gates, wrong decompositions, out-of-scope circuits and model-invented numbers
therefore cannot reach the output, by construction, not by filtering.

**Not claimed:**

- that the model understands the user (intent errors remain: §7);
- anything outside the families of §3 (refused with a reason, never approximated);
- that the circuit performs well on a device (that is the twin's and the device's measurement, as in TOQB).

## 3. The intent IR (v0)

A closed set of families: the places where PSF-Zero builds by formula and has won on devices (README, "Scope",
2026-10-14). One record per request. Draft:

```json
{
  "ir": "psf-intent/0",
  "family": "ring_classifier",
  "params": {"n": 12, "layers": 8, "encoding": "ry", "encoding_scale": 1.5708, "readout": "z0"},
  "output": "template",
  "device": "file:target.json"
}
```

| family | parameters (types and ranges `[ ]`) | builds |
|---|---|---|
| `ring_classifier` | n (even, >= 12 to embed on heavy-hex), layers, encoding, readout | TOQB's model circuit: ry / rz per qubit, CZ around the ring |
| `tfim_chain` | n, J, h, dt, steps, Trotter order (1 or 2) | Trotter steps of the transverse-field Ising chain |
| `heisenberg_chain` | n, Jx, Jy, Jz, dt, steps | the same for XYZ couplings |
| `rzz_layer` | graph (chain or even ring), angles as slots | fractional rzz natives where the device offers them |

- **Angles are slots, not numbers,** when `output` is `template`: the IR names which parameters are trainable.
- **The device enters only as a Target file** (as psfz's device JSON today). The model is never shown calibration data.

## 4. Gate 1: checks before anything is built (Rust)

1. **Syntax:** the record parses against the schema. With constrained decoding (§7) this cannot fail in practice,
   but it is checked anyway.
2. **Ranges:** every parameter in its declared range.
3. **Scope, decided by formula:** a chain of length n fits if the longest working row-and-bridge path of the device
   (psfz r4's exact dynamic programme) has at least n qubits; an even ring fits if a band of hexagons with working
   boundary of length n exists in psfz's fixed candidate order. No VF2, no search, no score.
4. **Failed elements:** excluded by psfz's existing rule (error >= 0.5).

A record that fails any check is **refused** with a code and a reason (`OUT_OF_SCOPE_GRAPH`, `RING_TOO_SHORT`,
`PARAM_RANGE`, `NO_WORKING_PLACEMENT`, ...). There is no fallback: no search, no hand-off to another compiler, no
second try by the model.

## 5. psfz r5: templates whose shape never depends on the angles

**Found on 2026-10-19 (home, FakeFez, 12-qubit ring classifier):** psfz r4 compiles one circuit per call in 24-39 ms
(L 8 / L 16), and its output's gate list changes with the angle values (the count of rz differs between data
points), so the build-once, bind-many pattern of training loops (`compile_template`, 3.6-4.7 ms per bind in Python)
is not available in Rust, and GPU simulators lose their batching by structure. r5's requirements:

- **S1. Fixed shape.** The output's gate list (names and qubits) depends only on the IR and the device, never on slot
  values. Every single-qubit run is written in the general form rz(a) sx rz(b) sx rz(c), even when a shorter form
  exists for particular values; a rotation equal to zero keeps its slot.
- **S2. Closed-form bind.** Each output angle is a closed-form function of the slots it depends on: for a run of
  single-qubit rotations, the product of their SU(2) matrices converted to Z-SX-Z Euler angles; for an rzz slot, an
  affine map. `bind(slots)` evaluates these functions only: O(number of runs), no compilation.
- **S3. Slot map in the output:** for every output angle, the slots it depends on and its function, so that a
  simulator (traj2, CUDA-Q) can rebind without psfz.
- **S4. Speed target:** bind at most `[ ]` ms for n = 12, L = 16 (the Python template's 4.7 ms as the reference).

## 6. The certificate: equivalence by random bindings, with a stated bound

For a template T and the intent's reference circuit R (built from the IR directly, gate by gate, with no rewriting),
let M(θ) = U_T(θ) U_R(θ)^†. T is correct up to a global phase exactly when every off-diagonal entry of M and every
difference M_ii − M_00 vanishes for all θ. After substituting z_k = e^{iθ_k/2}, each of these is a Laurent polynomial
whose degree in z_k is at most 4 d_k, where d_k is the number of gates slot k enters (2 d_k from each of U_T and U_R).
If one of them is not identically zero, a random binding with each z_k drawn uniformly from the G-th roots of unity
makes it vanish with probability at most Σ_k 4 d_k / G (Schwartz–Zippel on a grid). With k independent bindings that
all agree, the bound is that value to the power k.

- **Example:** n = 12, L = 16, 588 slots each used once (d_k = 1), G = 2^32: one binding gives <= 5.5e-7; three
  bindings <= 1.7e-19.
- **Check per binding:** M computed in full for up to about 13 active qubits (a 2^13 x 2^13 product on one GPU).
  Beyond that, the check is on random input states (U_T ψ against U_R ψ up to a phase), which adds the usual
  random-state argument to the bound `[ ]`; beyond about 24 active qubits, the certificate is the unit-tested closed
  form of each rewrite (as psfz r1-r4 today), and it says so.
- **This is a computation with a stated bound, not a search,** and its cost is a few simulations per template, not per
  binding.

## 7. The surface: the model, and what it may say

- **Constrained decoding** against the IR's JSON Schema (for example vLLM's guided decoding), so the model's output is
  always a parseable record. Any open model can sit here; it is replaceable.
- **Inputs to the model:** the user's request and the IR's schema with each family's one-line description. No device
  data, no gate set.
- **Outputs of the model:** the IR record, and one sentence restating the request. **No numbers:** durations, gate
  counts, fidelities and costs are filled in by Rust and the twin, and the model's sentence is not allowed to contain
  digits other than the IR's own parameters (checked).
- **Echo:** Rust renders the accepted IR as a canonical spec ("12-qubit ring classifier, 8 layers, ry encoding, read
  qubit 0; template with 588 slots; device file sha256 ..."). The user confirms it before anything runs on a device.

## 8. Hallucination types and where each is stopped

| type | example | stopped by | strength |
|---|---|---|---|
| syntax | a gate that does not exist, qubit 200 on a 156-qubit device | constrained decoding, gate 1 | by construction |
| physics | a wrong decomposition, a wrong angle convention | the model does not decompose; Rust builds; certificate (§6) | by construction (bound stated) |
| scope | "any graph works" | gate 1 refuses with a reason | by construction |
| numbers | "this reaches 99 % fidelity" | the model may not state numbers; Rust and the twin do | by construction (checked) |
| intent | the user asked for a TFIM chain, the model chose a ring classifier | the echo and the user's confirmation | **reduced, not removed** |

**Vanity figure:** "zero hallucinated circuits". **Essential figure:** how often the chosen intent differed from what
the user meant, and how many of those the echo caught.

## 9. How it would be tested (skeleton of a pre-registration)

- **Sets (written before any model is run, `[ ]` by whom):** N1, 100 natural-language requests with their gold IR;
  N2, 50 adversarial requests (out of scope, invalid, attempts to make the model emit gates or numbers).
- **Primary (pass / fail):** number of circuits emitted that did not come from Rust with a holding certificate: must
  be 0 over N1 and N2. One is a failure of the design, not of the model.
- **Secondary:** refusals on N2 (target: all 50, with the right code); IR exact match on N1 (reported, no threshold
  `[ ]`); intent errors on N1 and how many the echo showed (reported).
- **Comparison (the vLLM line's design, as a baseline):** the same N1 and N2 with the model writing circuits that
  PSF-Zero compiles; count of invalid or wrong circuits there.

## 10. Phases

| phase | deliverable | where | first measurement |
|---|---|---|---|
| P0 | IR v0: Rust types, JSON Schema, gate 1 with refusal codes | home (CPU) | unit tests; N2's refusals without any model |
| P1 | psfz r5: fixed-shape templates, closed-form bind, slot map | home (CPU) | S1 on every family; bind time (S4); exactness against r4 at random bindings |
| P2 | certificate (§6) | home GPU or pod | the bound and the check on all of P1's templates |
| P3 | surface: one open model with constrained decoding | pod | N1 / N2 as in §9 |
| P4 | templates into TOQB's loop and CUDA-Q (X22's 2 x 2) | pod | time per loop with psfz r5 templates against Python templates |

## 11. Open decisions for the owner

- The families of IR v0 (the four above, or fewer).
- Ring classifiers need n >= 12 on heavy-hex to use psfz; at n = 12 Qiskit level 2 also finds the ring with no SWAP
  (2026-10-19, FakeFez, L 8: 96 CZ on 12 qubits for both). The compression gap to Qiskit then lies in compile time,
  bind time and single-qubit counts, not in two-qubit gates. Which comparison should TOQB show for this design `[ ]`.
- Whether P3 uses the model of the vLLM line (gpt-oss-120b) or a smaller one `[ ]`.
- Who writes N1 and N2 `[ ]` (they must be fixed before any model sees them).
