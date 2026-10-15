"""W2 probe: c60's chain/graph outputs against a native emission (item 99's machinery, no reorder) of the logical
circuit they hand to Qiskit, on the same fixed layout. FakeFez T and F, C58-VAL's 12 in-scope tests."""
# Path note (2026-10-15c): the absolute workplace paths of the original run are replaced by the 2026-10-15b
# delivery inside bundle q; run from the repository root. Nothing else is changed.
import sys, os, contextlib, io, runpy, shutil, tempfile, warnings, time
warnings.simplefilter("ignore")
_tmp = tempfile.mkdtemp() + "/c55"
sys.argv = ["c55_val", _tmp]
g = runpy.run_path(os.path.join("benchmarks", "c55_val.py"), run_name="imp")
M = g["load"](os.path.join(os.path.join("data", "2026-10-15", "workplace", "workplace_2026-10-15b_W1-W2", "w30"), "c60", "psf_compile.py"), "psf_compile"); M.WARN_WITHOUT_TARGET = False
from qiskit import QuantumCircuit
real = M.generate_preset_pass_manager
CAP = {}
class W:
    def __init__(self, pm, kw): self.pm, self.kw = pm, kw
    def run(self, c):
        CAP["circ"], CAP["kw"] = c, self.kw
        return self.pm.run(c)
def fake(*a, **kw):
    return W(real(*a, **kw), kw)
M.generate_preset_pass_manager = fake

import numpy as np
def drop_diag_before_measure(ops, meas):
    mq = {q for q, _ in meas}
    changed = True
    while changed:
        changed = False
        last = {}
        for k, (nm, qs, op, dg) in enumerate(ops):
            for q in qs: last[q] = k
        for k in sorted(set(last.values()), reverse=True):
            nm, qs, op, dg = ops[k]
            if dg and all(q in mq and last[q] == k for q in qs):
                del ops[k]; changed = True; break
    return ops

class Blk:
    def __init__(self, a, b): self.a, self.b, self.U = a, b, np.eye(4, dtype=complex); self.n2 = 0
def m4(op, qs, a, b):
    m = np.asarray(op.to_matrix(), dtype=complex)
    if len(qs) == 1:
        return np.kron(np.eye(2), m) if qs[0] == a else np.kron(m, np.eye(2))
    if qs == (a, b): return m
    P = np.eye(4)[[0, 2, 1, 3]]
    return P @ m @ P

def emit(circ, layout, target, blocks=False):
    got = M._n99_read(circ)
    if got is None: return None, "not read"
    ops, meas = got
    if blocks:
        ops = drop_diag_before_measure(list(ops), meas)
        seq, opn = [], {}
        for nm, qs, op, dg in ops:
            if len(qs) == 1:
                B = opn.get(qs[0])
                if B is not None: B.U = m4(op, qs, B.a, B.b) @ B.U
                else: seq.append(("1", nm, qs, op, dg))
                continue
            a, b = qs
            B = opn.get(a)
            if B is not None and B is opn.get(b) and {B.a, B.b} == {a, b}:
                B.U = m4(op, qs, B.a, B.b) @ B.U; B.n2 += 1; continue
            for q in qs:
                if opn.get(q) is not None:
                    C = opn[q]; opn.pop(C.a, None); opn.pop(C.b, None)
            B = Blk(a, b); B.U = m4(op, qs, a, b) @ B.U; B.n2 = 1
            opn[a] = opn[b] = B; seq.append(("B", B))
        from qiskit.circuit.library import UnitaryGate
        new = []
        for it in seq:
            if it[0] == "1": new.append(it[1:])
            else:
                B = it[1]
                U = B.U
                dg = bool(np.all(np.abs(U - np.diag(np.diag(U))) < 1e-12))
                new.append(("unitary", (B.a, B.b), UnitaryGate(U, check_input=False), dg))
        ops = new
    frac = "rzz" in target.operation_names; use_rx = "rx" in target.operation_names
    N = target.num_qubits
    out = QuantumCircuit(N)
    for cr in circ.cregs: out.add_register(cr)
    one = (1+0j, 0j, 0j, 1+0j); pending = {}
    phys = layout
    def flush(q):
        U = pending.pop(q, None)
        if U is not None:
            cls, ang = M._n99_run_class(U, use_rx); M._n99_emit_run(out, phys[q], cls, ang, use_rx)
    def two(name, a, b):
        pa, pb = phys[a], phys[b]
        if (pa, pb) not in target[name]: pa, pb = pb, pa
        return pa, pb
    for nm, qs, op, dg in ops:
        if len(qs) == 1:
            pending[qs[0]] = M._n99_mul(M._n99_t(op), pending.get(qs[0], one)); continue
        steps = M._n99_steps(nm, qs, op, dg, frac)
        for st in steps:
            if st[0] == "1":
                pending[st[1]] = M._n99_mul(st[2], pending.get(st[1], one)); continue
            flush(st[1]); flush(st[2])
            if st[0] == "cz": out.cz(*two("cz", st[1], st[2]))
            else: out.rzz(st[3], *two("rzz", st[1], st[2]))
    for q in sorted(pending): flush(q)
    for q, cb in meas: out.measure(phys[q], cb)
    return out, None

IN = ("tfim_n12", "tfim_n16", "tfimP_n12", "tfimP_n16", "tfim_n20_r3", "heis_n12_r2", "tfim_n40_r2", "hhI_n16", "hhI_n20", "ring12_tfim", "hhtree_heis_n16", "ghz_n12")
t, ft = g["device"]("FakeFez")
for name in IN:
    qc = g["build"](name)
    for lab, tg in (("T", t), ("F", ft)):
        CAP.clear()
        b = dict(M.FORMULA97_STATS)
        s = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            o = M.compile_for_hardware(qc, target=tg, entangling_basis="cx", layout_search=True, seed_transpiler=0)
        secs = time.perf_counter() - s
        path = next((k for k in ("chain", "graph", "native") if M.FORMULA97_STATS[k] > b[k]), "?")
        mo = g["metrics"](o, tg)
        if path == "native" or "circ" not in CAP:
            print(f"{name:16s} {lab} {path}: (native or nothing captured) dur {mo['dur_us']:.3f}"); continue
        lay = CAP["kw"]["initial_layout"]
        e, err = emit(CAP["circ"], lay, tg, blocks=(lab == "T"))
        if e is None:
            print(name, lab, path, err); continue
        me = g["metrics"](e, tg)
        ex = g["exactness"](qc, e) if me["active"] <= 20 else None
        print(f"{name:16s} {lab} {path:5s} c60 {mo['dur_us']:6.3f} us 2q {mo['twoq']:4d} | native {me['dur_us']:6.3f} us 2q {me['twoq']:4d} exact {ex} ({secs*1000:.0f} ms)", flush=True)
