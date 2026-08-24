"""

r0_psf_zero_transform.py — Corrected Version

=============================================

R0-PSF-Zero: a PennyLane transform that replaces fixed two-qubit unitary

blocks in a tape with their KAK/Cartan decomposition (local gates + an

Ising XX/YY/ZZ canonical core), using the already-verified psf_zero_core

Rust extension.

The pasted version was a pure mock, and a more dangerous one than the

earlier Qiskit/QGL versions in this same pipeline: `_rust_optimize_true_kak`

returned a FIXED circuit (`k1=k2=k3=k4=I`, `c=[0.1,0.2,0.3]`) regardless of

the actual captured 2-qubit matrix `U`. Verified directly by running it:

- Applying the transform to `RX, RY, CNOT, RX` changed the circuit's

  expectation value from 0.573003 (untransformed) to -0.341170

  (transformed) -- a completely different answer.

- Swapping the CNOT for a SWAP gate and re-running produced the exact same

  transformed output as the CNOT case, bit-for-bit -- the transform ignored

  which gate it was "optimizing" entirely.

- The demo's own printed gradients ([-0.8895, -0.0732, -0.9164]) bore no

  resemblance to the true circuit's gradients ([0.0514, -0.5868, -0.5247])

  -- not just in magnitude but in which parameter matters most. Used inside

  a real VQE/QML training loop, this would optimize toward a different,

  arbitrary circuit while reporting "fully operational".

This version calls the real Rust core and was checked by direct unitary

reconstruction (not just against a downstream loss number): the recovered

circuit's matrix matches an arbitrary random 2-qubit target to a Frobenius

norm difference of ~3e-15, and a full torch forward+backward pass through

the transform matches an untransformed reference circuit's loss and

gradients to float64 precision (see the demo at the bottom).

"""

import warnings

import numpy as np

import pennylane as qml

from pennylane.tape import QuantumTape

# The pasted version's comment for the "production" call was

# `psf_zero_core_rs.kak_decompose(U)` returning `k1, k2, c, k3, k4` (four

# SU(2) matrices + 3 angles) -- that function/signature doesn't exist. The

# real, compiled entry point is `batch_decompose`, which already returns

# the local factors as Euler-angle triples directly (so the pasted file's

# separate `_su2_to_euler` matrix->angle conversion step isn't needed at all

# once you call the real function).

from psf_zero_core import batch_decompose

def _rust_kak_ops(U: np.ndarray, wires: list):

    """Real KAK decomposition of a 4x4 unitary `U` into native PennyLane

    ops on `wires`. Returns None if `U` sits at a Weyl-chamber degeneracy

    (CNOT, SWAP, iSWAP, Identity, ...) that psf_zero_core cannot resolve --

    a known, already-documented limitation of the underlying method (same

    as in the Rust core's own NOTES.md and the Qiskit integration's

    `SynthesisDegenerateGateError`), not something fixed here.

    """

    U = np.asarray(U, dtype=complex)

    u_r = U.real.tolist()

    u_i = U.imag.tolist()

    try:

        angles, k1, k2, global_phase = batch_decompose([u_r], [u_i])[0]

    except Exception:

        return None

    t0, t1, t2, t3 = angles

    c1 = (t0 + t1) / 2.0

    c2 = (t1 + t3) / 2.0

    c3 = (t0 + t3) / 2.0

    w0, w1 = wires[0], wires[1]

    # Each (k)[i] triple is (phi, theta, lam) for M = Rz(phi) Ry(theta) Rz(lam).

    # PennyLane's qml.Rot(a, b, c) == Rz(c) Ry(b) Rz(a), so this single call

    # reproduces M directly with Rot(lam, theta, phi) -- verified against

    # the explicit matrix product, no separate append-order juggling needed

    # (unlike the Qiskit integration, which had to build M from three

    # separate rz/ry/rz appends in reverse order).

    #

    # Wire mapping: PennyLane is *big-endian* (wires[0] is the first/left

    # kron factor) -- verified directly (`RX` on wire 0 matches

    # kron(RX, I), not kron(I, RX)) -- the OPPOSITE of Qiskit's

    # little-endian convention. So k2[0]/k1[0] ("the left factor of

    # kron(k*l, k*r)" in the math this decomposition uses) go on wires[0]

    # with NO swap, unlike the Qiskit version where that same factor had to

    # go on qubit 1.

    ops = [

        qml.Rot(k2[0][2], k2[0][1], k2[0][0], wires=w0),

        qml.Rot(k2[1][2], k2[1][1], k2[1][0], wires=w1),

        # IsingXX/YY/ZZ(t) = exp(-i t/2 P⊗P) -- verified against

        # scipy.linalg.expm -- same convention as Qiskit's RXX/RYY/RZZ, so

        # realizing exp(i*c*P⊗P) needs t = -2c, same sign fix as before.

        qml.IsingXX(-2 * c1, wires=wires),

        qml.IsingYY(-2 * c2, wires=wires),

        qml.IsingZZ(-2 * c3, wires=wires),

        qml.Rot(k1[0][2], k1[0][1], k1[0][0], wires=w0),

        qml.Rot(k1[1][2], k1[1][1], k1[1][0], wires=w1),

        # qml.GlobalPhase(phi) implements multiplication by exp(-i*phi);

        # verified by direct full-matrix reconstruction against the target

        # (norm diff ~3e-15) that the correct sign here is the negation of

        # the Rust core's returned `global_phase`.

        qml.GlobalPhase(-global_phase, wires=wires),

    ]

    return ops

@qml.transforms.transform

def r0_psf_zero_transform(tape: QuantumTape):

    """

    R0-PSF-Zero Transform — Real Analytical Edition

    Intercepts fixed (non-trainable) 2-qubit unitary blocks and replaces

    them with their real KAK decomposition via the Rust core. Two cases

    are now handled explicitly instead of silently mishandled:

    - A block at a Weyl-chamber degeneracy (CNOT, SWAP, iSWAP, Identity):

      kept as the original gate, with a warning, rather than crashing or

      (as the pasted version did for every gate) silently substituting an

      unrelated fixed circuit.

    - A *trainable* 2-qubit block (e.g. a `QubitUnitary` built from

      `requires_grad` torch parameters, as you'd want for compiling a

      variational two-qubit ansatz block): kept unchanged, with a warning,

      rather than detached from the gradient graph. This transform's

      Rust-computed decomposition angles are plain floats with no

      connection back to the block's original parameters, so "fully

      transparent to Autograd" (the pasted docstring's claim) only actually

      holds for fixed blocks. Making this differentiable through a

      trainable block would need a custom torch.autograd.Function wrapping

      the KAK map with an analytically-derived Jacobian -- out of scope for

      this pass; silently pretending it works would be worse than leaving

      it alone.

    """

    new_ops = []

    for op in tape.operations:

        if len(op.wires) == 2 and op.has_matrix:

            U = op.matrix()

            if getattr(U, "requires_grad", False):

                warnings.warn(

                    f"{op.name} on wires {list(op.wires)} is a trainable two-qubit "

                    "block; this transform's Rust-based KAK decomposition is not "

                    "differentiable through, so it is left untouched rather than "

                    "silently detaching it from the gradient graph.",

                    UserWarning,

                )

                new_ops.append(op)

                continue

            optimized_ops = _rust_kak_ops(np.asarray(U, dtype=complex), list(op.wires))

            if optimized_ops is None:

                warnings.warn(

                    f"{op.name} on wires {list(op.wires)} sits at a Weyl-chamber "

                    "degeneracy; psf_zero_core cannot decompose it, keeping the "

                    "original gate unchanged.",

                    UserWarning,

                )

                new_ops.append(op)

            else:

                new_ops.extend(optimized_ops)

        else:

            new_ops.append(op)

    new_tape = QuantumTape(new_ops, tape.measurements)

    def null_postprocessing(results):

        return results[0]

    return [new_tape], null_postprocessing

# =====================================================================

# Usage Example & Verification

# =====================================================================

if __name__ == "__main__":

    import torch

    from scipy.stats import unitary_group

    print("=== R0-PSF-Zero x PennyLane -- Corrected Analytical Engine ===\n")

    dev = qml.device("default.qubit", wires=2)

    # --- Case 1: CNOT, as in the original demo. CNOT sits exactly at a

    # Weyl-chamber degeneracy, so it's correctly kept unchanged (with a

    # warning) instead of being silently replaced.

    @qml.qnode(dev, interface="torch", diff_method="backprop")

    def true_circuit(params):

        qml.RX(params[0], wires=0)

        qml.RY(params[1], wires=1)

        qml.CNOT(wires=[0, 1])

        qml.RX(params[2], wires=0)

        return qml.expval(qml.PauliZ(0))

    @qml.qnode(dev, interface="torch", diff_method="backprop")

    @r0_psf_zero_transform

    def r0_circuit_cnot(params):

        qml.RX(params[0], wires=0)

        qml.RY(params[1], wires=1)

        qml.CNOT(wires=[0, 1])

        qml.RX(params[2], wires=0)

        return qml.expval(qml.PauliZ(0))

    params = torch.tensor([0.8, -0.5, 1.2], requires_grad=True)

    ref_params = torch.tensor([0.8, -0.5, 1.2], requires_grad=True)

    loss = r0_circuit_cnot(params)

    ref_loss = true_circuit(ref_params)

    loss.backward()

    ref_loss.backward()

    print(f"CNOT case -- transformed loss: {loss.item():.6f} vs untransformed: {ref_loss.item():.6f}")

    print(f"CNOT case -- transformed grad: {params.grad} vs untransformed: {ref_params.grad}\n")

    # --- Case 2: a generic, non-degenerate fixed two-qubit block. This is

    # what the transform can actually decompose, and now really does.

    rng = np.random.default_rng(21)

    U = unitary_group.rvs(4, random_state=rng)

    @qml.qnode(dev, interface="torch", diff_method="backprop")

    def true_circuit_block(params):

        qml.RX(params[0], wires=0)

        qml.RY(params[1], wires=1)

        qml.QubitUnitary(U, wires=[0, 1])

        qml.RX(params[2], wires=0)

        return qml.expval(qml.PauliZ(0))

    @qml.qnode(dev, interface="torch", diff_method="backprop")

    @r0_psf_zero_transform

    def r0_circuit_block(params):

        qml.RX(params[0], wires=0)

        qml.RY(params[1], wires=1)

        qml.QubitUnitary(U, wires=[0, 1])

        qml.RX(params[2], wires=0)

        return qml.expval(qml.PauliZ(0))

    p1 = torch.tensor([0.8, -0.5, 1.2], requires_grad=True)

    p2 = torch.tensor([0.8, -0.5, 1.2], requires_grad=True)

    l1 = true_circuit_block(p1)

    l2 = r0_circuit_block(p2)

    l1.backward()

    l2.backward()

    print(f"Random-block case -- transformed loss: {l2.item():.12f} vs untransformed: {l1.item():.12f}")

    print(f"Random-block case -- transformed grad: {p2.grad}")

    print(f"Random-block case -- untransformed grad: {p1.grad}")

    print(f"Random-block case -- max |grad diff|: {(p1.grad - p2.grad).abs().max().item():.3e}")
