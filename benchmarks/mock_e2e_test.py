"""End-to-end test of r0_psf_zero_transform's control flow with stand-ins for
PennyLane and psf_zero_core (neither is installed here). Stand-in gates use
PennyLane's documented matrix conventions; the stand-in core returns the Rust
contract's own factors for matrices it built itself."""
import sys, types, warnings
import numpy as np
from scipy.linalg import expm

def Rz(t): return np.array([[np.exp(-1j*t/2), 0], [0, np.exp(1j*t/2)]])
def Ry(t): return np.array([[np.cos(t/2), -np.sin(t/2)], [np.sin(t/2), np.cos(t/2)]])
X = np.array([[0, 1], [1, 0]]); Y = np.array([[0, -1j], [1j, 0]]); Z = np.diag([1, -1]); I2 = np.eye(2)

# ---------------- stand-in pennylane ----------------
class Op:
    def __init__(self, name, wires, mat2or4):
        self.name, self.wires, self._m = name, list(wires) if isinstance(wires, (list, tuple)) else [wires], mat2or4
    def matrix(self): return self._m
    def full(self):  # 4x4 on wires [0,1], big-endian
        m = np.asarray(self._m)
        if m.shape == (4, 4): return m
        return np.kron(m, I2) if self.wires == [0] else np.kron(I2, m)

class _M(np.ndarray): pass
pl = types.ModuleType("pennylane")
pl.Rot = lambda a, b, c, wires: Op("Rot", wires, Rz(c) @ Ry(b) @ Rz(a))
pl.IsingXX = lambda t, wires: Op("IsingXX", wires, expm(-1j*t/2*np.kron(X, X)))
pl.IsingYY = lambda t, wires: Op("IsingYY", wires, expm(-1j*t/2*np.kron(Y, Y)))
pl.IsingZZ = lambda t, wires: Op("IsingZZ", wires, expm(-1j*t/2*np.kron(Z, Z)))
pl.GlobalPhase = lambda p, wires: Op("GlobalPhase", wires, np.exp(-1j*p)*np.eye(4))
pl.QubitUnitary = lambda U, wires: Op("QubitUnitary", wires, U)
pl.CNOT = lambda wires: Op("CNOT", wires, np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex))
pl.RX = lambda t, wires: Op("RX", wires, expm(-1j*t/2*X))
pl.transforms = types.SimpleNamespace(transform=lambda f: f)
class QuantumTape:
    def __init__(self, ops, measurements, shots=None):
        self.operations, self.measurements, self.shots = list(ops), measurements, shots
tape_mod = types.ModuleType("pennylane.tape"); tape_mod.QuantumTape = QuantumTape
sys.modules["pennylane"] = pl; sys.modules["pennylane.tape"] = tape_mod

# ---------------- stand-in psf_zero_core ----------------
def rust_euler(m):  # ported from lib.rs su2_to_euler_zyz
    a, b = m[0, 0], m[0, 1]
    theta = 2*np.arctan2(abs(b), abs(a))
    if theta < 1e-12: return (0.0, 0.0, (-2*np.angle(a)) % (4*np.pi))
    pr = np.pi - np.angle(a) - np.angle(b); lr = np.angle(b) - np.angle(a) - np.pi
    phi = pr % (2*np.pi); return (phi, theta, (lr + (pr - phi)) % (4*np.pi))

REGISTRY, CALLS = {}, {"n": 0, "sizes": []}
def build_known_U(rng, tag):
    def su2():
        v = rng.normal(size=4); v /= np.linalg.norm(v); w, x, y, z = v
        return np.array([[w+1j*z, y+1j*x], [-y+1j*x, w-1j*z]])
    e1l, e1r, e2l, e2r = su2(), su2(), su2(), su2()
    c1, c2, c3 = rng.uniform(-np.pi/4, np.pi/4, 3); ph = rng.uniform(-np.pi, np.pi)
    U = np.exp(1j*ph)*np.kron(e1l, e1r) @ expm(1j*(c1*np.kron(X,X)+c2*np.kron(Y,Y)+c3*np.kron(Z,Z))) @ np.kron(e2l, e2r)
    angles = (c1-c2+c3, c1+c2-c3, 0.0, -c1+c2+c3)
    REGISTRY[np.round(U, 10).tobytes()] = (angles, [list(rust_euler(e1l)), list(rust_euler(e1r))],
                                           [list(rust_euler(e2l)), list(rust_euler(e2r))], ph)
    return U

DEGENERATE = {}
RAISE_NON_CARTAN = {"on": False}
def batch_decompose_checked(br, bi):
    CALLS["n"] += 1; CALLS["sizes"].append(len(br))
    if RAISE_NON_CARTAN["on"]: raise TypeError("simulated non-Cartan failure (e.g. bad input shape)")
    out = []
    for r, i in zip(br, bi):
        key = np.round(np.array(r) + 1j*np.array(i), 10).tobytes()
        if key in DEGENERATE: out.append((None, "DegenerateWeylPoint"))
        else: out.append((REGISTRY[key], None))
    return out
core = types.ModuleType("psf_zero_core"); core.batch_decompose_checked = batch_decompose_checked
sys.modules["psf_zero_core"] = core

import importlib.util
spec = importlib.util.spec_from_file_location("r0", "r0_psf_zero_transform.py")
r0 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r0)

def unitary(ops):
    W = np.eye(4, dtype=complex)
    for op in ops: W = op.full() @ W
    return W

fails = 0
def check(label, ok, detail=""):
    global fails; fails += 0 if ok else 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}  {detail}")

rng = np.random.default_rng(3)
U1, U3 = build_known_U(rng, 1), build_known_U(rng, 3)
Udeg = build_known_U(rng, 2); DEGENERATE[np.round(Udeg, 10).tobytes()] = True

print("Test A -- mixed tape: RX, QubitUnitary, CNOT, degenerate QubitUnitary, QubitUnitary")
ops_in = [pl.RX(0.7, 0), pl.QubitUnitary(U1, [0, 1]), pl.CNOT([0, 1]), pl.QubitUnitary(Udeg, [0, 1]),
          pl.QubitUnitary(U3, [0, 1])]
tape = QuantumTape(ops_in, ["m"], shots=None)
CALLS.update(n=0, sizes=[])
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    (new_tape,), post = r0.r0_psf_zero_transform(tape)
names = [o.name for o in new_tape.operations]
check("exactly one core call for the whole tape", CALLS["n"] == 1, f"calls={CALLS['n']}")
check("batch contained the 3 QubitUnitary blocks only", CALLS["sizes"] == [3], f"sizes={CALLS['sizes']}")
check("CNOT left untouched", names.count("CNOT") == 1)
check("degenerate block kept, with a warning naming the variant",
      names.count("QubitUnitary") == 1 and any("DegenerateWeylPoint" in str(x.message) for x in w))
check("order preserved (RX first, CNOT between the two decompositions)",
      names[0] == "RX" and names[9] == "CNOT" and names[10] == "QubitUnitary", f"len={len(names)}")
err = np.linalg.norm(unitary(new_tape.operations) - unitary(ops_in))
check("whole-tape unitary unchanged", err < 1e-12, f"err={err:.2e}")
check("measurements and postprocessing preserved", new_tape.measurements == ["m"] and post(["r"]) == "r")

print("Test B -- trainable QubitUnitary is left alone, not sent to the core")
class Trainable(np.ndarray): requires_grad = True
tr = U1.view(Trainable)
CALLS.update(n=0, sizes=[])
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    (nt,), _ = r0.r0_psf_zero_transform(QuantumTape([pl.QubitUnitary(tr, [0, 1])], []))
check("left unchanged", [o.name for o in nt.operations] == ["QubitUnitary"])
check("core not called", CALLS["n"] == 0)
check("warned as trainable", any("trainable" in str(x.message) for x in w))

print("Test C -- a non-Cartan core failure propagates instead of being swallowed")
RAISE_NON_CARTAN["on"] = True
try:
    r0.r0_psf_zero_transform(QuantumTape([pl.QubitUnitary(U1, [0, 1])], []))
    check("TypeError propagated", False, "was swallowed")
except TypeError:
    check("TypeError propagated", True)
RAISE_NON_CARTAN["on"] = False

print("Test D -- tape with nothing eligible makes no core call")
CALLS.update(n=0, sizes=[])
(nt,), _ = r0.r0_psf_zero_transform(QuantumTape([pl.RX(0.1, 0), pl.CNOT([0, 1])], []))
check("no core call", CALLS["n"] == 0)
check("ops unchanged", [o.name for o in nt.operations] == ["RX", "CNOT"])

print(f"\n{'ALL PASSED' if fails == 0 else f'{fails} FAILED'}")
