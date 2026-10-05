"""diag_resynth_1005.py -- exploratory diagnosis, written after the EXACT scored run (Addendum 343) was seen.

Finding to explain: in part X, c12 refused item 35's re-synthesis exactly 128 times on each cx device (0 on the cz
devices, and 0 on all HOLD6 F circuits of part Y). Are these refusals of wrong re-syntheses (Qiskit #17057), or
refusals of exact ones (a false alarm of `_same_action`)?

For every call of `_final_resynthesis` that reaches the item-39 check, this records:
  - `_same_action`'s verdict and the largest state infidelity it computed (its own seeds and code path);
  - an independent check: the process infidelity between the two circuits, from Qiskit's Operator on the touched
    qubits.
Nothing is scored. Run from the repository root (commit ae1e946 or later, with the patches in place):
    python diag_resynth_1005.py [devices...]          default: FakeAuckland FakeTorino
"""
import argparse
import collections
import os
import sys

import numpy as np

REPO = os.getcwd()
sys.path.insert(0, os.path.join(REPO, "benchmarks"))
import exact_eval as ev  # noqa: E402  (the locked harness: same loader, generator and call)


def touched_operator(circ, act):
    """Operator of `circ` restricted to the physical qubits `act` (barrier, measure, delay skipped)."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    m = {p: j for j, p in enumerate(act)}
    red = QuantumCircuit(len(act))
    for ins in circ.data:
        if ins.operation.name in ("barrier", "measure", "delay"):
            continue
        red.append(ins.operation, [m[circ.find_bit(q).index] for q in ins.qubits])
    return Operator(red).data


def same_action_value(c12, ref, new):
    """The largest state infidelity `_same_action` computes over its seeds (its code, without the threshold)."""
    a, b = c12._ops_of(ref), c12._ops_of(new)
    if a is None or b is None:
        return float("nan")
    touched = sorted({i for _, q in a + b for i in q})
    pos = {p: j for j, p in enumerate(touched)}
    worst = 0.0
    for seed in c12.EXACT_SEEDS:
        rng = np.random.default_rng(seed)
        psi = np.array(1.0 + 0j)
        for _ in touched:
            psi = np.multiply.outer(psi, c12._product_state(rng))
        x, y = c12._apply_ops(psi, a, pos), c12._apply_ops(psi, b, pos)
        worst = max(worst, 1.0 - abs(np.vdot(x.ravel(), y.ravel())) ** 2)
    return worst


def main():
    p = argparse.ArgumentParser()
    p.add_argument("devices", nargs="*", default=["FakeAuckland", "FakeTorino"])
    devices = p.parse_args().devices
    _, _, c12, _, _, _ = ev.load(argparse.Namespace(repo=REPO), "rel")
    from qiskit_ibm_runtime import fake_provider

    rec, state = [], {"in_resynth": False, "cell": None, "n": None, "which": None}
    orig_same, orig_resynth = c12._same_action, c12._final_resynthesis

    def same_wrap(ref, new, tol=c12.EXACT_TOL):
        ok = orig_same(ref, new, tol)
        if state["in_resynth"]:
            act = sorted({ref.find_bit(q).index for i in ref.data for q in i.qubits}
                         | {new.find_bit(q).index for i in new.data for q in i.qubits})
            if len(act) <= 12:
                U, V = touched_operator(ref, act), touched_operator(new, act)
                proc = float(1 - abs(np.trace(U.conj().T @ V) / U.shape[0]) ** 2)
            else:
                proc = float("nan")
            ops_ref = collections.Counter(i.operation.name for i in ref.data)
            ops_new = collections.Counter(i.operation.name for i in new.data)
            rec.append(dict(device=state["device"], cell=state["cell"], n=state["n"], ok=ok,
                            sa=same_action_value(c12, ref, new), proc=proc, nact=len(act),
                            ref_ops=dict(ops_ref), new_ops=dict(ops_new),
                            gp=(float(ref.global_phase), float(new.global_phase))))
        return ok

    def resynth_wrap(out, target, max_error):
        state["in_resynth"] = True
        try:
            return orig_resynth(out, target, max_error)
        finally:
            state["in_resynth"] = False

    c12._same_action, c12._final_resynthesis = same_wrap, resynth_wrap
    for dev in devices:
        tgt = getattr(fake_provider, dev)().target
        cm = tgt.build_coupling_map()
        nat = [g for g in ("cx", "cz", "rz", "sx", "x") if g in tgt.operation_names]
        base = dict(coupling_map=cm, basis_gates=nat, entangling_basis="cx", layout_search=True, seed_transpiler=0,
                    target=tgt)
        for cell, prm, qc in ev.x_circuits(False):
            state.update(device=dev, cell=cell, n=prm["n"])
            ev.quiet(c12.compile_for_hardware, qc, **ev.FULL, **base)

    print("calls of the item-39 check inside _final_resynthesis, by device and cell:")
    print("  device        cell  n  calls  refused  refused&proc<=1e-10 (false)  refused&proc>1e-6 (real)")
    keys = sorted({(r["device"], r["cell"], r["n"]) for r in rec})
    for k in keys:
        rs = [r for r in rec if (r["device"], r["cell"], r["n"]) == k]
        ref = [r for r in rs if not r["ok"]]
        print("  %-13s %-4s %2d  %5d  %7d  %27d  %24d" % (k + (len(rs), len(ref),
              sum(r["proc"] <= 1e-10 for r in ref), sum(r["proc"] > 1e-6 for r in ref))))
    acc = [r for r in rec if r["ok"]]
    print("accepted: %d, max process infidelity %.1e" % (len(acc), max([r["proc"] for r in acc] or [0])))
    bad = [r for r in rec if not r["ok"]]
    if bad:
        r = bad[0]
        print("first refusal:", {k: r[k] for k in ("device", "cell", "n", "sa", "proc", "nact", "gp")})
        print("  ops before:", r["ref_ops"])
        print("  ops after: ", r["new_ops"])
        print("refusals: _same_action value min %.1e median %.1e max %.1e; process infidelity min %.1e max %.1e"
              % (min(x["sa"] for x in bad), float(np.median([x["sa"] for x in bad])), max(x["sa"] for x in bad),
                 min(x["proc"] for x in bad), max(x["proc"] for x in bad)))


if __name__ == "__main__":
    main()
