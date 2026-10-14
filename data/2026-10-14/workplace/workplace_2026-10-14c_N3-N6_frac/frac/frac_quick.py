"""frac_quick.py -- quick look (2026-10-14): what do (1) rx as one pulse, (2) rzz with its sign folded into a Pauli
frame, (3) native rzz in N1's brickwork / N2's ring, do to circuit duration? FakeFez durations: cz per coupler, rzz =
that coupler's cz length (as on ibm_marrakesh, where rzz = cz = 68 ns), rx = sx length (one pulse), rz = 0.
Correctness: exact outcome distribution against the logical circuit (Aer statevector)."""
import math
import os
import sys
import warnings

import numpy as np

warnings.simplefilter("ignore")
sys.path[:0] = [os.path.join(os.getcwd(), "data", "2026-10-14", "workplace", "workplace_2026-10-14_W1-W2-W3-W5-W7", "w2"), os.path.dirname(os.path.abspath(__file__))]
import w2_tfim as W2  # noqa: E402

from qiskit import QuantumCircuit  # noqa: E402
from qiskit.transpiler import PassManager  # noqa: E402
from qiskit.transpiler.passes import Optimize1qGatesDecomposition  # noqa: E402
from qiskit_ibm_runtime.fake_provider import FakeFez  # noqa: E402

B = FakeFez()
T = B.target
SX = T["sx"][(0,)].duration


def cz_len(a, b):
    p = T["cz"].get((a, b)) or T["cz"].get((b, a))
    return p.duration


def duration(qc):
    clock = {}
    for ins in qc.data:
        nm = ins.operation.name
        if nm in ("barrier", "measure"):
            continue
        qs = [qc.find_bit(q).index for q in ins.qubits]
        if nm in ("cz", "rzz"):
            d = cz_len(*qs)
        elif nm == "rz":
            d = 0.0
        elif nm in ("sx", "x", "rx"):
            d = SX
        else:
            raise ValueError(nm)
        s = max(clock.get(q, 0.0) for q in qs)
        for q in qs:
            clock[q] = s + d
    return max(clock.values()) * 1e6


def two_q(qc):
    return sum(1 for i in qc.data if len(i.qubits) == 2)


def one_q_basis(qc, basis):
    return PassManager([Optimize1qGatesDecomposition(basis=basis)]).run(qc)


def line(n):
    """a straight run of n qubits on FakeFez's first row (0-15), continued down the bridge (as W5's snake)"""
    snake = list(range(0, 16)) + [19] + list(range(35, 20, -1)) + [36] + list(range(41, 56))
    return snake[:n]


# ------------------------------------------------------------------ frames

def lower_zz(qc_logical, frame=True):
    """logical circuit with rzz(any angle), rx/ry/rz/h/x/sx and cz -> physical-gate circuit using rzz only with
    0 < theta <= pi/2. Angles reduced mod 2pi; |theta| > pi/2 folded by Z x Z (two free rz(pi)). A negative angle:
    with frame=True one x is placed before it (merged into the single-qubit run before it, when there is one) and the
    x after it is kept in a Pauli X frame (it conjugates later 1q gates, flips later rzz signs, turns into rz(pi) through
    cz, and is undone by one x per qubit before measurement, which a classical bit flip could replace); with
    frame=False two explicit x gates are placed around the rzz."""
    n = qc_logical.num_qubits
    out = QuantumCircuit(n, qc_logical.num_clbits)
    f = [0] * n
    for ins in qc_logical.data:
        nm = ins.operation.name
        qs = [qc_logical.find_bit(q).index for q in ins.qubits]
        if nm == "barrier":
            continue
        if nm == "measure":
            q = qs[0]
            if f[q]:
                out.x(q)
                f[q] = 0
            out.measure(q, qc_logical.find_bit(ins.clbits[0]).index)
            continue
        if nm == "rzz":
            a, b = qs
            th = float(ins.operation.params[0])
            th = (th + math.pi) % (2 * math.pi) - math.pi            # (-pi, pi]
            if abs(th) > math.pi / 2:                                # rzz(th) = rzz(th - sign pi) . (Z x Z)
                out.rz(math.pi, a)
                out.rz(math.pi, b)
                th -= math.copysign(math.pi, th)
            if abs(th) < 1e-12:
                continue
            eff = th * (-1) ** (f[a] ^ f[b])
            if eff < 0:
                if frame:                   # x before (merges into the 1q run before it), the x after kept in the frame
                    out.x(a)
                    f[a] ^= 1
                    out.rzz(-eff, a, b)
                else:
                    out.x(a)
                    out.rzz(-eff, a, b)
                    out.x(a)
            else:
                out.rzz(eff, a, b)
            continue
        if nm == "cz":
            a, b = qs
            out.cz(a, b)
            if f[a]:
                out.rz(math.pi, b)
            if f[b]:
                out.rz(math.pi, a)
            continue
        # single-qubit gate: physical = X^f G X^f (merged into the 1q runs afterwards)
        q = qs[0]
        if f[q]:
            out.x(q)
        out.append(ins.operation, [q])
        if f[q]:
            out.x(q)
    return out


# ------------------------------------------------------------------ circuits

def tfim_logical(n, steps, J=1.0, h=1.0, t=0.6):
    return W2.tfim_brickwork(n, steps)          # rzz / rx brickwork, merged half layers (W2's QK3-BW input)


def heis_logical(n, steps, t=0.5):
    """XXX-type chain: rxx, ryy, rzz on even then odd edges, per step (first order)."""
    dt = t / steps
    qc = QuantumCircuit(n)
    for q in range(0, n, 2):
        qc.x(q)
    for _ in range(steps):
        for s in (0, 1):
            for q in range(s, n - 1, 2):
                qc.rxx(2 * dt, q, q + 1)
                qc.ryy(2 * dt, q, q + 1)
                qc.rzz(2 * dt, q, q + 1)
    qc.measure_all()
    return qc


def expand_xx_yy(qc):
    """rxx(t) = (H x H) rzz(t) (H x H); ryy(t) = (Rx(pi/2) x Rx(pi/2)) rzz(t) (Rx(-pi/2) x Rx(-pi/2))"""
    out = QuantumCircuit(*qc.qregs, *qc.cregs)
    for ins in qc.data:
        nm = ins.operation.name
        qs = ins.qubits
        if nm == "rxx":
            out.h(qs)
            out.rzz(ins.operation.params[0], *qs)
            out.h(qs)
        elif nm == "swap":                    # three cx, each cx = (I x H) cz (I x H)
            a, b = qs
            for c, t in ((a, b), (b, a), (a, b)):
                out.h(t)
                out.cz(c, t)
                out.h(t)
        elif nm == "ryy":
            out.rx(math.pi / 2, qs)
            out.rzz(ins.operation.params[0], *qs)
            out.rx(-math.pi / 2, qs)
        else:
            out.append(ins.operation, qs, ins.clbits)
    return out


def to_cz(qc):
    """rzz(t) = cx . rz(t) . cx with cx = (I x H) cz (I x H)"""
    out = QuantumCircuit(*qc.qregs, *qc.cregs)
    for ins in qc.data:
        if ins.operation.name == "rzz":
            a, b = ins.qubits
            th = ins.operation.params[0]
            out.h(b)
            out.cz(a, b)
            out.h(b)
            out.rz(th, b)
            out.h(b)
            out.cz(a, b)
            out.h(b)
        else:
            out.append(ins.operation, ins.qubits, ins.clbits)
    return out


def place(qc, phys, N=156):
    out = QuantumCircuit(N, qc.num_clbits)
    for ins in qc.data:
        out.append(ins.operation, [phys[qc.find_bit(q).index] for q in ins.qubits],
                   [qc.find_bit(c).index for c in ins.clbits])
    return out


def classifier_n2(n, L, seed=20261014):
    import importlib.util
    spec = importlib.util.spec_from_file_location("depth_eval", os.path.join(os.getcwd(), "data", "2026-10-05", "workplace", "depth1", "depth_eval.py"))
    D = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(D)
    import n2_ring as N2R
    rng = np.random.default_rng(seed)
    ang = {(nn, LL): (rng.uniform(-1, 1, nn), rng.uniform(-np.pi, np.pi, D.n_params(nn, LL)))
           for nn in (4, 6) for LL in range(1, 17)}
    x, th = ang[(n, L)]
    logical = D.circuit(x, th, n, L)
    logical.measure_all()
    built = N2R.build(x, th, n, L, list(range(n)), n)   # the ring formula on a line 0..n-1 (logical wires)
    return logical, built


def arms(logical_gates, ref_logical, phys):
    """durations: CZ+sx (today), CZ+rx1 (1), rzz naive + rx1 (3 without 2), rzz + frame + rx1 (2+3)"""
    res = {}
    ref = W2.probs(ref_logical)
    variants = {
        "cz_sx": one_q_basis(to_cz(expand_xx_yy(logical_gates)), ["rz", "sx", "x"]),
        "cz_rx": one_q_basis(to_cz(expand_xx_yy(logical_gates)), ["rz", "rx"]),
        "rzz_naive_rx": one_q_basis(lower_zz(expand_xx_yy(logical_gates), frame=False), ["rz", "rx"]),
        "rzz_frame_rx": one_q_basis(lower_zz(expand_xx_yy(logical_gates), frame=True), ["rz", "rx"]),
    }
    for k, c in variants.items():
        pc = place(c, phys)
        err = W2.tvd(W2.probs(pc), ref)
        res[k] = (duration(pc), two_q(pc), err)
    return res


def main():
    rows = []
    cases = []
    for n, s in ((12, 2), (12, 10), (16, 4), (32, 10)):
        q = tfim_logical(n, s)
        cases.append((f"tfim n={n} steps={s}", q, q, n))
    for n, s in ((12, 2), (12, 6)):
        q = heis_logical(n, s)
        cases.append((f"heis n={n} steps={s}", q, q, n))
    for n, L in ((4, 8), (4, 16), (6, 8), (6, 16)):
        lg, built = classifier_n2(n, L)
        cases.append((f"classifier(N2) n={n} L={L}", built, lg, n))
    print("| circuit | today: CZ + 2-pulse 1q | (1) CZ + 1-pulse rx | (3) rzz + rx, signs by explicit x | (2)+(3) rzz + rx, signs in a Pauli frame |")
    print("|---|---|---|---|---|")
    for name, gates, ref, n in cases:
        if n > 16:
            r = {}
            for k, c in {"cz_sx": one_q_basis(to_cz(expand_xx_yy(gates)), ["rz", "sx", "x"]),
                         "cz_rx": one_q_basis(to_cz(expand_xx_yy(gates)), ["rz", "rx"]),
                         "rzz_naive_rx": one_q_basis(lower_zz(expand_xx_yy(gates), frame=False), ["rz", "rx"]),
                         "rzz_frame_rx": one_q_basis(lower_zz(expand_xx_yy(gates), frame=True), ["rz", "rx"])}.items():
                pc = place(c, line(n))
                r[k] = (duration(pc), two_q(pc), float("nan"))
        else:
            r = arms(gates, ref, line(n))
        base = r["cz_sx"][0]
        cells = " | ".join(f"{r[k][0]:.2f} us ({r[k][0] / base:.2f}), 2q {r[k][1]}, err {r[k][2]:.0e}"
                           for k in ("cz_sx", "cz_rx", "rzz_naive_rx", "rzz_frame_rx"))
        print(f"| {name} | {cells} |", flush=True)


if __name__ == "__main__":
    main()
