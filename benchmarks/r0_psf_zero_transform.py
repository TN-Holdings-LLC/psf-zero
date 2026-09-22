"""r0_psf_zero_transform.py -- PSF-Zero's Rust KAK core as a PennyLane transform.

VERSION: 2026-09-21

Replaces fixed (non-trainable) two-qubit `qml.QubitUnitary` blocks with an
exact Cartan (KAK) decomposition computed by the compiled Rust core
(`psf_zero_core`), emitted as native PennyLane operations:

    Rot, Rot, IsingXX, IsingYY, IsingZZ, Rot, Rot, GlobalPhase

Status -- read before relying on this
-------------------------------------
Verified (2026-09-21, numpy only, no PennyLane or psf_zero_core needed):
  The gate assembly below reproduces U exactly from the Rust core's own
  documented contract
      U = e^{i*phase} (e1l kron e1r) N(c1,c2,c3) (e2l kron e2r),
      each local == Rz(phi) Ry(theta) Rz(lam),
      c1=(t0+t1)/2, c2=(t1+t3)/2, c3=(t0+t3)/2   (lib.rs cartan_from_angles)
  over 2,000 random instances, worst ||circuit - U|| = 2.96e-15, with the
  Euler-angle extraction ported line-for-line from lib.rs. Two negative
  controls confirm the check has teeth: passing Rot arguments in the Rust
  order instead of reversed gives an error of 3.96; dropping GlobalPhase
  gives 4.00.

Verified end-to-end (2026-09-21, PennyLane 0.42.3 + built psf_zero_core,
Python 3.10, `python r0_psf_zero_transform.py`, autograd interface,
default.qubit):
  - Random fixed QubitUnitary: loss matches the untransformed circuit to
    1.89e-15, gradient to 2.00e-15; the block was actually replaced by
    Rot, Rot, IsingXX, IsingYY, IsingZZ, Rot, Rot, GlobalPhase.
  - Native CNOT: left untouched (loss and gradient differences exactly 0).
  - Direct matrix reconstruction including global phase: 6.35e-15.
  - Three QubitUnitary blocks mixed with CNOT/RX/RY: loss matches to
    1.55e-15, gradient to 2.22e-15; exactly one core call carried all three
    blocks; CNOT untouched; order preserved (28 operations).
  This confirms the four PennyLane conventions below against the installed
  PennyLane itself, not only its documentation.

Still untested:
  - torch and JAX interfaces (only autograd was run).

Environment note: pennylane-qiskit must NOT be installed into this project's
environment. No release pip tried (0.45.0 back to 0.42.0) accepts Qiskit
2.5.x -- 0.45.0 caps it at <= 2.3.0 -- so pip resolves the conflict by
downgrading Qiskit. On 2026-09-21 this replaced Qiskit 2.5.2, the version
every PSF-Zero benchmark and both papers were measured on, with 1.2.4 (it
was restored and confirmed with `pip check`). Use a separate virtual
environment if PennyLane on Qiskit backends is needed.

PennyLane conventions relied on:
  qml.Rot(a, b, c)     == RZ(c) RY(b) RZ(a)        (argument order reversed
                                                    relative to the Rust triple)
  qml.IsingPP(t)       == exp(-i t/2 P kron P)      (so exp(i c P kron P) needs t=-2c)
  qml.GlobalPhase(p)   == exp(-i p) I              (so the Rust phase is negated)
  wires[0]             == the LEFT kron factor     (big-endian; the opposite of
                                                    Qiskit, so no qubit swap here)

Changes in this revision
------------------------
1. Base: the direct-triple version ("Corrected Version"). The "(v2)" variant,
   which rebuilt each Rust triple into a 2x2 matrix and re-extracted Euler
   angles with its own `su2_to_euler`, was also verified correct (worst error
   4.24e-15) but performs a redundant round trip; `batch_decompose` already
   returns Euler triples, confirmed by reading lib.rs directly.
2. Only `qml.QubitUnitary` two-qubit blocks are decomposed. The current Rust
   core handles CNOT, SWAP, iSWAP and the identity correctly (lib.rs,
   `CartanError::DegenerateWeylPoint` docstring), so the previous "any
   two-qubit op with a matrix" filter would expand an already-native gate --
   one CNOT, CZ, or IsingXX -- into eight operations. That increases gate
   count, the opposite of this pass's purpose. Earlier comments assuming CNOT
   is "kept unchanged because it is degenerate" described an older core.
3. Errors are no longer swallowed. The previous `except Exception: return
   None` reported every failure -- including genuine bugs such as a wrong
   input shape -- as "degenerate gate, kept unchanged". This now uses
   `batch_decompose_checked`, which reports each item's failure by
   `CartanError` variant name; only those named decomposition failures keep
   the original gate (with a warning naming the variant). Anything else
   propagates.
4. One core call per tape, not per gate. All eligible blocks are sent to the
   Rust core in a single batch (which releases the GIL for the whole batch);
   one failing block no longer affects the others.

Known limitations
-----------------
- Trainable blocks are left untouched (with a warning). The Rust angles are
  plain floats with no autograd link back to the block's parameters, so
  replacing a trainable block would silently detach it from the gradient
  graph. Detection uses the `requires_grad` attribute (torch / autograd);
  JAX-traced inputs are untested.
- Gate-count benefit is circuit-dependent: a single isolated QubitUnitary
  becomes eight operations. The pass pays off where a QubitUnitary stands in
  for a longer native sequence (e.g. consolidated blocks), matching the
  trade-off stated in the main PSF-Zero README.
"""
from __future__ import annotations

import warnings

import numpy as np
import pennylane as qml
from pennylane.tape import QuantumTape

from psf_zero_core import batch_decompose_checked

# Operation names this pass decomposes. Kept as a module constant (rather
# than a transform argument) so the default behaviour needs no extra API.
DECOMPOSE_OP_NAMES = ("QubitUnitary",)


def _is_trainable(matrix) -> bool:
    return bool(getattr(matrix, "requires_grad", False))


def _kak_ops(result, wires) -> list:
    """Native PennyLane ops for one successful `batch_decompose_checked` item.

    `result` is `(angles, k1, k2, global_phase)`: `angles` are the core's four
    raw magic-basis angles (t0, t1, t2, t3), not Cartan coefficients; `k1`
    and `k2` are `[left_triple, right_triple]` ZYZ triples (phi, theta, lam)
    with `local == Rz(phi) Ry(theta) Rz(lam)`; `k2` is applied first in time.
    """
    angles, k1, k2, global_phase = result
    t0, t1, _t2, t3 = angles
    c1 = (t0 + t1) / 2.0
    c2 = (t1 + t3) / 2.0
    c3 = (t0 + t3) / 2.0
    w0, w1 = wires[0], wires[1]
    return [
        # Rot(lam, theta, phi) == Rz(phi) Ry(theta) Rz(lam): reversed order.
        qml.Rot(k2[0][2], k2[0][1], k2[0][0], wires=w0),
        qml.Rot(k2[1][2], k2[1][1], k2[1][0], wires=w1),
        qml.IsingXX(-2.0 * c1, wires=wires),
        qml.IsingYY(-2.0 * c2, wires=wires),
        qml.IsingZZ(-2.0 * c3, wires=wires),
        qml.Rot(k1[0][2], k1[0][1], k1[0][0], wires=w0),
        qml.Rot(k1[1][2], k1[1][1], k1[1][0], wires=w1),
        qml.GlobalPhase(-global_phase, wires=wires),
    ]


@qml.transforms.transform
def r0_psf_zero_transform(tape: QuantumTape):
    """Replace fixed two-qubit QubitUnitary blocks with their exact KAK
    decomposition from psf_zero_core. See the module docstring for scope,
    verification status, and limitations."""
    # Pass 1: find eligible blocks, preserving their position in the tape.
    eligible_index = []
    batch_r, batch_i = [], []
    for idx, op in enumerate(tape.operations):
        if op.name not in DECOMPOSE_OP_NAMES or len(op.wires) != 2:
            continue
        matrix = op.matrix()
        if _is_trainable(matrix):
            warnings.warn(
                f"{op.name} on wires {list(op.wires)} is trainable; the Rust "
                "decomposition has no autograd link to its parameters, so it is "
                "left unchanged rather than detached from the gradient graph.",
                UserWarning,
            )
            continue
        u = np.asarray(matrix, dtype=complex)
        eligible_index.append(idx)
        batch_r.append(u.real.tolist())
        batch_i.append(u.imag.tolist())

    # Pass 2: one core call for the whole tape. Non-Cartan errors propagate.
    replacements = {}
    if eligible_index:
        outcomes = batch_decompose_checked(batch_r, batch_i)
        for idx, (result, error_name) in zip(eligible_index, outcomes):
            op = tape.operations[idx]
            if result is None:
                warnings.warn(
                    f"{op.name} on wires {list(op.wires)}: psf_zero_core reported "
                    f"CartanError.{error_name}; keeping the original gate.",
                    UserWarning,
                )
                continue
            replacements[idx] = _kak_ops(result, list(op.wires))

    # Pass 3: rebuild the tape in the original order.
    new_ops = []
    for idx, op in enumerate(tape.operations):
        new_ops.extend(replacements.get(idx, [op]))
    new_tape = QuantumTape(new_ops, tape.measurements, shots=tape.shots)

    def postprocessing(results):
        return results[0]

    return [new_tape], postprocessing


# =====================================================================
# Self-test: run `python r0_psf_zero_transform.py` in a real environment.
# Every check prints PASS/FAIL against an explicit tolerance rather than
# printing numbers for a human to compare by eye.
# =====================================================================
if __name__ == "__main__":
    # Uses PennyLane's bundled autograd interface (pennylane.numpy + qml.grad),
    # so no extra dependency such as torch is needed to run the self-test.
    from pennylane import numpy as pnp
    from scipy.stats import unitary_group

    TOL = 1e-9
    failures = 0

    def check(label, ok, detail=""):
        global failures
        failures += 0 if ok else 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")

    dev = qml.device("default.qubit", wires=2)
    U = unitary_group.rvs(4, random_state=np.random.default_rng(21))

    def make_body(use_unitary):
        def body(params):
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=1)
            if use_unitary:
                qml.QubitUnitary(U, wires=[0, 1])
            else:
                qml.CNOT(wires=[0, 1])
            qml.RX(params[2], wires=0)
            return qml.expval(qml.PauliZ(0))
        return body

    def compare(label, use_unitary):
        ref = qml.QNode(make_body(use_unitary), dev, interface="autograd", diff_method="backprop")
        new = r0_psf_zero_transform(ref)
        params = pnp.array([0.8, -0.5, 1.2], requires_grad=True)
        dl = abs(float(ref(params)) - float(new(params)))
        dg = float(np.max(np.abs(np.asarray(qml.grad(ref)(params)) - np.asarray(qml.grad(new)(params)))))
        print(f"{label}:")
        check("loss matches untransformed", dl < TOL, f"|diff|={dl:.2e}")
        check("gradient matches untransformed", dg < TOL, f"max|diff|={dg:.2e}")
        return new, params

    print("=== r0_psf_zero_transform self-test ===\n")

    # 1. A generic fixed QubitUnitary: must be decomposed AND stay exact.
    new, params = compare("Case 1 -- random fixed QubitUnitary", use_unitary=True)
    try:
        tape = qml.workflow.construct_tape(new)(params)
        names = [op.name for op in tape.operations]
        check("QubitUnitary was actually replaced", "QubitUnitary" not in names and "IsingXX" in names,
              f"ops={names}")
    except Exception as exc:  # API differs across PennyLane versions
        print(f"  [SKIP] replacement check (construct_tape unavailable: {type(exc).__name__}); "
              "Case 3 below still checks the decomposition itself")

    # 2. A native CNOT: must be left alone (not expanded into eight ops).
    compare("Case 2 -- native CNOT (should be untouched)", use_unitary=False)

    # 3. The full matrix, checked directly with no circuit around it.
    ops = _kak_ops(batch_decompose_checked([U.real.tolist()], [U.imag.tolist()])[0][0], [0, 1])
    W = np.eye(4, dtype=complex)
    for op in ops:
        W = qml.matrix(op, wire_order=[0, 1]) @ W
    err = np.linalg.norm(W - U)
    print("Case 3 -- direct matrix reconstruction:")
    check("||circuit - U|| < 1e-9 (global phase included)", err < TOL, f"err={err:.2e}")

    # 4. Several QubitUnitary blocks mixed with native gates, against the
    #    REAL core: exactness, one batched core call, order preserved.
    U1, U2, U3 = (unitary_group.rvs(4, random_state=np.random.default_rng(s)) for s in (1, 2, 3))

    def multi_body(params):
        qml.RX(params[0], wires=0)
        qml.QubitUnitary(U1, wires=[0, 1])
        qml.CNOT(wires=[0, 1])
        qml.QubitUnitary(U2, wires=[0, 1])
        qml.RY(params[1], wires=1)
        qml.QubitUnitary(U3, wires=[0, 1])
        qml.RX(params[2], wires=0)
        return qml.expval(qml.PauliZ(0))

    print("Case 4 -- three QubitUnitary blocks mixed with CNOT/RX/RY:")
    ref = qml.QNode(multi_body, dev, interface="autograd", diff_method="backprop")
    new = r0_psf_zero_transform(ref)
    params = pnp.array([0.8, -0.5, 1.2], requires_grad=True)
    dl = abs(float(ref(params)) - float(new(params)))
    dg = float(np.max(np.abs(np.asarray(qml.grad(ref)(params)) - np.asarray(qml.grad(new)(params)))))
    check("loss matches untransformed", dl < TOL, f"|diff|={dl:.2e}")
    check("gradient matches untransformed", dg < TOL, f"max|diff|={dg:.2e}")

    # Apply the transform directly to a tape (no QNode machinery involved),
    # counting calls into the real core by wrapping the module-level name
    # the transform looks up at call time.
    calls = []
    _real_core = batch_decompose_checked

    def _counting_core(batch_r, batch_i):
        calls.append(len(batch_r))
        return _real_core(batch_r, batch_i)

    globals()["batch_decompose_checked"] = _counting_core
    try:
        tape = QuantumTape(
            [qml.RX(0.8, wires=0), qml.QubitUnitary(U1, wires=[0, 1]), qml.CNOT(wires=[0, 1]),
             qml.QubitUnitary(U2, wires=[0, 1]), qml.RY(-0.5, wires=1),
             qml.QubitUnitary(U3, wires=[0, 1]), qml.RX(1.2, wires=0)],
            [qml.expval(qml.PauliZ(0))],
        )
        (out_tape,), _ = r0_psf_zero_transform(tape)
    finally:
        globals()["batch_decompose_checked"] = _real_core
    names = [op.name for op in out_tape.operations]
    check("exactly one core call carrying all three blocks", calls == [3], f"calls={calls}")
    check("no QubitUnitary left, three decompositions present",
          names.count("QubitUnitary") == 0 and names.count("IsingXX") == 3)
    check("CNOT untouched and order preserved",
          names.count("CNOT") == 1 and names[0] == "RX" and names[9] == "CNOT"
          and names[18] == "RY" and names[-1] == "RX", f"len={len(names)}")

    print(f"\n{'ALL CHECKS PASSED' if failures == 0 else f'{failures} CHECK(S) FAILED'}")
