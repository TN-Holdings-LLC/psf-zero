"""indep_check_v10_heldout.py -- independent re-check (numpy only, no PennyLane/Qiskit) of the v10 evaluation outputs.

The held-out targets are rebuilt here from their definitions in the v10 pre-registration (section 2), not taken from the
harness. All five targets are symmetric under qubit relabelling within each group (the singlet only up to a global sign),
so the result does not depend on the harness's bit-order convention. Only the parameter parser and gate aliases are
reused from e2e_vllm_psf_v6.py.

    python indep_check_v10_heldout.py OUT     (OUT = the unpacked v10eval_outputs_0930.zip)
"""
import glob, itertools, json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e2e_vllm_psf_v6 import ALIASES, _safe_eval  # parameter parsing only

X = np.array([[0, 1], [1, 0]]); Y = np.array([[0, -1j], [1j, 0]]); Z = np.diag([1, -1]); H = np.array([[1, 1], [1, -1]]) / math.sqrt(2)
S = np.diag([1, 1j]); T = np.diag([1, np.exp(1j * math.pi / 4)])
def rx(t): return np.array([[math.cos(t / 2), -1j * math.sin(t / 2)], [-1j * math.sin(t / 2), math.cos(t / 2)]])
def ry(t): return np.array([[math.cos(t / 2), -math.sin(t / 2)], [math.sin(t / 2), math.cos(t / 2)]])
def rz(t): return np.diag([np.exp(-1j * t / 2), np.exp(1j * t / 2)])
def p1(t): return np.diag([1, np.exp(1j * t)])
ONE = {"h": H, "x": X, "y": Y, "z": Z, "s": S, "sdg": S.conj().T, "t": T, "tdg": T.conj().T}


def apply1(psi, n, q, U):
    psi = psi.reshape((2,) * n)
    return np.moveaxis(np.tensordot(U, psi, axes=([1], [q])), 0, q).reshape(-1)


def apply_c(psi, n, c, t, U):
    psi = psi.reshape((2,) * n).copy(); idx = [slice(None)] * n; idx[c] = 1
    sub = psi[tuple(idx)]; tt = t if t < c else t - 1
    psi[tuple(idx)] = np.moveaxis(np.tensordot(U, sub, axes=([1], [tt])), 0, tt)
    return psi.reshape(-1)


def simulate(gates, n):
    psi = np.zeros(2 ** n, complex); psi[0] = 1
    for g in gates:
        nm = str(g["name"]).lower().strip(); nm = ALIASES.get(nm, nm); q = [int(x) for x in g["qubits"]]
        p = g.get("params") or []; p = p if isinstance(p, list) else [p]; p = [_safe_eval(v) for v in p]
        if nm in ONE: psi = apply1(psi, n, q[0], ONE[nm])
        elif nm in ("rx", "ry", "rz"): psi = apply1(psi, n, q[0], {"rx": rx, "ry": ry, "rz": rz}[nm](p[0]))
        elif nm == "cx": psi = apply_c(psi, n, q[0], q[1], X)
        elif nm == "cz": psi = apply_c(psi, n, q[0], q[1], Z)
        elif nm == "cy": psi = apply_c(psi, n, q[0], q[1], Y)
        elif nm == "cry": psi = apply_c(psi, n, q[0], q[1], ry(p[0]))
        elif nm == "crz": psi = apply_c(psi, n, q[0], q[1], rz(p[0]))
        elif nm == "crx": psi = apply_c(psi, n, q[0], q[1], rx(p[0]))
        elif nm == "cp": psi = apply_c(psi, n, q[0], q[1], p1(p[0]))
        elif nm == "swap": psi = np.swapaxes(psi.reshape((2,) * n), q[0], q[1]).reshape(-1)
        else: raise ValueError(f"gate {nm}")
    return psi


def basis(n, ones):
    v = np.zeros(2 ** n, complex)
    for s in ones: v[int(s, 2)] += 1
    return v / np.linalg.norm(v)


def targets():
    w4 = basis(4, ["0001", "0010", "0100", "1000"])
    d42 = basis(4, ["".join("1" if i in c else "0" for i in range(4)) for c in itertools.combinations(range(4), 2)])
    g3i = np.zeros(8, complex); g3i[0] = 1; g3i[7] = 1j; g3i /= math.sqrt(2)
    sing = np.array([0, 1, -1, 0], complex) / math.sqrt(2); s3 = np.kron(np.kron(sing, sing), sing)
    ghz3 = basis(3, ["000", "111"])
    return {"w4": (4, w4), "dicke42": (4, d42), "ghz3i": (3, g3i), "singlet3": (6, s3),
            "fill27g9": (27, [(list(range(3 * k, 3 * k + 3)), ghz3) for k in range(9)])}


def main(root):
    tg = targets(); rows = []; skipped = 0
    for f in sorted(glob.glob(os.path.join(root, "v*", "run*", "*", "rounds.jsonl"))):
        arm, run, task = f.split(os.sep)[-4:-1]
        n, t = tg[task]
        for r in map(json.loads, open(f, encoding="utf-8")):
            if not r.get("spec") or r.get("fidelity_logical") is None:
                continue
            gates = r["spec"]["gates"]
            if task == "fill27g9":
                gid = {q: k for k, (w, _) in enumerate(t) for q in w}
                if any(len({gid[int(q)] for q in g["qubits"]}) > 1 for g in gates):
                    skipped += 1; continue
                F = 1.0
                for k, (w, gv) in enumerate(t):
                    loc = {q: i for i, q in enumerate(w)}
                    sub = [dict(g, qubits=[loc[int(q)] for q in g["qubits"]]) for g in gates if gid[int(g["qubits"][0])] == k]
                    F *= abs(np.vdot(gv, simulate(sub, 3))) ** 2
            else:
                F = abs(np.vdot(t, simulate(gates, n))) ** 2
            rows.append((arm, run, task, r["round"], F, r["fidelity_logical"], r.get("fidelity_compiled")))
    worst = max(abs(a - b) for *_, a, b, _ in rows)
    dis = sum((a >= 0.9999) != ((c or 0) >= 0.9999) for *_, a, b, c in rows)
    print(f"circuits {len(rows)}  max |F_indep - F_recorded| {worst:.2e}  solved disagreements {dis}  "
          f"fill27g9 cross-group skipped {skipped}")
    solved = {}
    for arm, run, task, _, a, _, _ in rows:
        solved[(arm, run, task)] = solved.get((arm, run, task), False) or a >= 0.9999
    for arm in ("v9", "v10"):
        per = {tk: int(sum(v for (a, _, t2), v in solved.items() if a == arm and t2 == tk)) for tk in tg}
        print(f"{arm}: solved (independent, logical) {sum(per.values())}/15  {per}")


if __name__ == "__main__":
    main(sys.argv[1])
