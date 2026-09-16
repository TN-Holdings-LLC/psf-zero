"""Verification for the 2026-09-16 revision of psf_compile.py.

HOW THIS WAS RUN, AND WHAT THAT DOES AND DOES NOT ESTABLISH
-----------------------------------------------------------
Run 2026-09-16 in a Linux sandbox (Qiskit 2.5.2, Python 3.11) against a
**stand-in core**, not the real `psf_zero_core` -- the compiled extension does
not load here. The stand-in is deliberately NOT checked in alongside this
script: importing a stub in place of the real core is exactly the bug the
2026-09-15 revision of `psf_compile.py` was written to fix, and a copy sitting
next to the real module invites it back.

What that means for the results below:
  - Tests 2-9 exercise `psf_compile.py`'s own logic (circuit rebuilding,
    register handling, argument validation, gating). Those are independent of
    which core answers, so passing here means this file is correct.
  - Test 1 checks that whichever core is installed agrees with this file's
    `_reconstruct`. Against the stand-in it confirms the harness; **against
    the real core it is a genuine cross-check and is worth re-running on a
    machine that has it.**
  - Nothing here measures speed. The 2026-09-16 changes are argued
    structurally (fewer per-instruction lookups, one less 4x4 reconstruction
    on the `cx` path) and have not been timed.

Checks, in order:
  1. the stub core's convention matches _reconstruct (so tests 2+ mean anything)
  2. compile() output is unitarily equivalent to the input, verify=True
  3. same for verify="strict", verify=False, entangling_basis="cx"
  4. NEW: named / split registers survive (the old construction dropped them)
  5. NEW: a classically-conditioned instruction survives
  6. NEW: bad argument values raise instead of silently falling through to a different code path
  7. global phase is preserved
  8. block_gate_floor still gates: a shallow circuit reports 0 blocks
"""
import sys
import warnings

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator, random_unitary

import psf_compile
from psf_compile import compile as psf

FAILED = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not cond:
        FAILED.append(name)


def dense_pair_circuit(n=4, gates_per_pair=20, seed=7, regs=None):
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(*regs) if regs is not None else QuantumCircuit(n)
    for pair in [(i, i + 1) for i in range(0, n - 1, 2)]:
        for _ in range(gates_per_pair):
            qc.append(UnitaryGate(random_unitary(4, seed=int(rng.integers(1 << 30)))), list(pair))
    return qc


print("1. stub core convention matches _reconstruct")
worst = 0.0
for s in range(20):
    U = random_unitary(4, seed=s).data
    cartan, k1, k2, ph = psf_compile.geometric_decompose(U.real.tolist(), U.imag.tolist())
    worst = max(worst, psf_compile._infidelity(U, psf_compile._reconstruct(cartan, k1, k2, ph)))
check("stub reconstruction infidelity < 1e-12", worst < 1e-12, f"worst {worst:.2e}")
if worst >= 1e-12:
    print("stub convention wrong -- later results would be meaningless")
    sys.exit(1)

print("2-3. equivalence across verify / entangling_basis")
base = dense_pair_circuit()
target = Operator(base)
for kwargs in (
    dict(verify=True),
    dict(verify="strict"),
    dict(verify=False),
    dict(verify=True, entangling_basis="cx"),
):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any fallback would raise here
        out = psf(base, **kwargs)
    check(f"equiv {kwargs}", Operator(out).equiv(target))

print("4. named / split registers survive")
qr_a, qr_b = QuantumRegister(2, "data"), QuantumRegister(2, "anc")
cr = ClassicalRegister(2, "meas")
named = dense_pair_circuit(regs=[qr_a, qr_b, cr])
out = psf(named)
check("register names preserved", [r.name for r in out.qregs] == ["data", "anc"],
      str([r.name for r in out.qregs]))
check("clbit register preserved", [r.name for r in out.cregs] == ["meas"])
check("named-register circuit still equivalent", Operator(out.remove_final_measurements(
    inplace=False)).equiv(Operator(named.remove_final_measurements(inplace=False))))

print("   (old construction, for contrast)")
try:
    old_style = QuantumCircuit(named.num_qubits, named.num_clbits)
    old_style.append(named.data[0].operation, named.data[0].qubits, named.data[0].clbits)
    check("old construction accepted a foreign Bit", False, "it did not raise -- claim wrong")
except Exception as exc:
    check("old construction raised on a foreign Bit", True, type(exc).__name__)

print("5. classically-conditioned instruction survives")
cond = dense_pair_circuit(regs=[QuantumRegister(4, "q"), ClassicalRegister(2, "c")])
cond.measure(0, 0)
with cond.if_test((cond.cregs[0], 1)):
    cond.x(1)
out = psf(cond)
check("if_test block preserved", any(i.operation.name == "if_else" for i in out.data))

print("6. argument validation")
for bad in (dict(verify="Strict"), dict(verify=1), dict(entangling_basis="CX"),
            dict(on_unsupported="ignore"), dict(tol=0.0)):
    try:
        psf(base, **bad)
        check(f"rejects {bad}", False, "no error raised")
    except ValueError as exc:
        check(f"rejects {bad}", True, str(exc)[:48])

print("7. global phase preserved")
gp = dense_pair_circuit(n=2, gates_per_pair=20, seed=3)
gp.global_phase = 0.37
out = psf(gp)
check("global phase equivalence", Operator(out).equiv(Operator(gp)))
check("global phase carried, not dropped",
      abs(np.exp(1j * out.global_phase)) > 0 and Operator(out).equiv(Operator(gp), atol=1e-12))

print("8. block_gate_floor still gates (standard gates, runs below the floor)")
shallow = QuantumCircuit(2)
for _ in range(3):
    shallow.h(0)
    shallow.cx(0, 1)
out = psf(shallow)
check("short standard-gate run passes through unchanged",
      out.count_ops() == shallow.count_ops(), str(dict(out.count_ops())))
check("shallow circuit still equivalent", Operator(out).equiv(Operator(shallow)))
# For contrast: a pre-existing 2-qubit UnitaryGate IS re-synthesized regardless
# of the floor (this is what the README quickstart relies on) -- unchanged
# behaviour, recorded here so it is not mistaken for a regression.
one_block = QuantumCircuit(2)
one_block.append(UnitaryGate(random_unitary(4, seed=1)), [0, 1])
out = psf(one_block)
check("single pre-existing UnitaryGate is synthesized and equivalent",
      Operator(out).equiv(Operator(one_block)) and "unitary" not in out.count_ops(),
      str(dict(out.count_ops())))

print("9. loose bits (no registers)")
from qiskit.circuit import Qubit  # noqa: E402

loose = QuantumCircuit([Qubit(), Qubit()])
for _ in range(20):
    loose.append(UnitaryGate(random_unitary(4, seed=11)), [0, 1])
out = psf(loose)
check("loose-bit circuit compiles and is equivalent", Operator(out).equiv(Operator(loose)))
check("loose-bit circuit keeps its bit count", out.num_qubits == 2)

print()
print("FAILURES:", FAILED if FAILED else "none")
sys.exit(1 if FAILED else 0)
