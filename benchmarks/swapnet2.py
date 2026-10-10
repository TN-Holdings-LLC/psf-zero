"""swapnet2.py -- SWAPNET2 (2026-10-10; swapnet.py with single-qubit Z terms applied as rz before the network,
since Qiskit's router takes two-qubit terms only, and identity terms dropped (a global phase); exploratory, nothing predicted): routing by construction instead of search, for
the circuits where a construction exists. A HamLib test whose Hamiltonian has only Z and ZZ terms (all terms
commute, so their order is free) is routed by a line swap network (Qiskit's Commuting2qGateRouter with
SwapStrategy.from_line) on a path of the device that avoids its failed elements, instead of by Sabre's search.

Arms, on BP-FINAL's HamLib FakeTorino tests, on FakeTorino and FakeKingston, each test and arm in its own process:
  SN1  the swap network, then Qiskit level 1 on the Target with the path as the layout and no routing
  SNC  the same, with PSF-Zero's compression (psf_compile.compile: each two-qubit block, ZZ and SWAP together,
       synthesised exactly) before the translation
A test that is not "Z and ZZ only", or with no path, is recorded as not eligible. The routed circuit is checked
exactly before the translation: on 64 random basis states its phases equal the Hamiltonian's up to one global phase,
and its swaps move every qubit to the position its measurement is read from (the check is exact for diagonal
circuits at any width). Recorded: time, two-qubit count, ESP and operations on failed elements as in esp_ft.py.
The report compares them with ESP-FT's QK2 and QK3 and ESP-C32's C32D on the same tests.

    cd <psf-zero repository>
    python <this file> run --bp <benchpress clone> --out DIR [--par 4] [--smoke]
    python <this file> report --out DIR
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import random
import subprocess
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = os.path.abspath(os.environ.get("PSF_ZERO_REPO", os.getcwd()))
HERE = os.path.join(REPO, "benchmarks")
sys.path[:0] = [os.path.join(REPO, "data", "2026-10-05", "workplace", "depth1"), HERE, REPO]
C32 = os.path.join(REPO, "patches", "psf_compile_c32_2026-10-10", "psf_compile.py")
ARMS = ("SN1", "SNC")
DEVICES = ("FakeTorino", "FakeKingston")
FAILED = 0.5
TIMEOUT, BUDGET = 900, 3600
PATH_BUDGET_S = 2.0


def tests(bp):
    import bp_final as F
    return [t for t in F.population(bp) if t[0] == "HamLib, FakeTorino"]


def device(name, bp_backend):
    if name == "FakeTorino":
        return bp_backend
    from qiskit_ibm_runtime.fake_provider import FakeKingston
    return FakeKingston()


def esp_of(out, target):
    lg, n_failed, dead = 0.0, 0, False
    for ins in out.data:
        if ins.operation.name in ("barrier", "delay"):
            continue
        qargs = tuple(out.find_bit(q).index for q in ins.qubits)
        try:
            props = target[ins.operation.name][qargs]
        except KeyError:
            continue
        e = props.error if props is not None and props.error is not None else 0.0
        n_failed += e >= FAILED
        if e >= 1.0:
            dead = True
        else:
            lg += math.log10(1.0 - e)
    return (None if dead else round(lg, 6)), int(n_failed)


def diagonal_evolution(qc):
    """(the PauliEvolutionGate, its qubit indices, the final measurements as (qubit, clbit)) when qc is one evolution
    of a Hamiltonian with Z and ZZ terms only, followed by measurements at most; else a reason (str)."""
    evo, meas = None, []
    for ins in qc.data:
        name = ins.operation.name
        if name == "barrier":
            continue
        if name == "PauliEvolution" and evo is None and not meas:
            evo = ins
        elif name == "measure":
            meas.append((qc.find_bit(ins.qubits[0]).index, qc.find_bit(ins.clbits[0]).index))
        else:
            return f"instruction {name}"
    if evo is None:
        return "no PauliEvolutionGate"
    op = evo.operation.operator
    if isinstance(op, list):
        return "a list of operators"
    for label in op.paulis.to_labels():
        if set(label) - {"I", "Z"}:
            return "a term with X or Y"
        if sum(ch == "Z" for ch in label) > 2:
            return "a Z term of weight > 2"
    return evo, [qc.find_bit(q).index for q in evo.qubits], meas


def best_path(n, edges_ok, err, budget_s=PATH_BUDGET_S):
    """A simple path of n physical qubits on the working couplers, with the smallest summed two-qubit error found by
    a depth-first search within the budget; None if none is found."""
    adj = {}
    for a, b in edges_ok:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    best, t0 = None, time.perf_counter()
    for start in sorted(adj, key=lambda v: (len(adj[v]), v)):
        if time.perf_counter() - t0 > budget_s:
            break
        stack, path, seen = [iter(sorted(adj[start], key=lambda w: err.get((start, w), 0.0)))], [start], {start}
        steps = 0
        while stack and steps < 200_000:
            steps += 1
            if len(path) == n:
                cost = sum(err.get((path[i], path[i + 1]), 0.0) for i in range(n - 1))
                if best is None or cost < best[0]:
                    best = (cost, list(path))
                break
            nxt = next(stack[-1], None)
            if nxt is None:
                stack.pop()
                seen.discard(path.pop())
                continue
            if nxt in seen:
                continue
            path.append(nxt)
            seen.add(nxt)
            stack.append(iter(sorted(adj[nxt] - seen, key=lambda w: err.get((nxt, w), 0.0))))
    return best


def phase_of(label_terms, time_, bits, qubits):
    """-t * sum_k c_k * prod_{Z in label_k} (-1)^bit, with label character j from the right on qubits[j]."""
    s = 0.0
    for label, c in label_terms:
        sign = 1
        for j, ch in enumerate(reversed(label)):
            if ch == "Z" and bits[qubits[j]]:
                sign = -sign
        s += c * sign
    return -time_ * s


def check_exact(evo_terms, t, n, routed, meas_pos):
    """Item-39-style check, exact for diagonal circuits: on 64 random basis states, the routed circuit's phase minus
    the Hamiltonian's is one constant, and every logical qubit ends at meas_pos[i]."""
    rng = random.Random(611)
    diffs = []
    for _ in range(64):
        x = [rng.randint(0, 1) for _ in range(n)]
        want = phase_of(evo_terms, t, x, list(range(n)))
        at = list(range(n))  # at[p]: the logical qubit at line position p
        bits = list(x)       # bits[p]: the bit at line position p
        got = 0.0
        for ins in routed.data:
            q = [routed.find_bit(b).index for b in ins.qubits]
            name = ins.operation.name
            if name == "swap":
                a, b = q
                at[a], at[b] = at[b], at[a]
                bits[a], bits[b] = bits[b], bits[a]
            elif name == "PauliEvolution":
                op = ins.operation.operator
                terms = [(lb, float(complex(c).real)) for lb, c in op.to_list()]
                got += phase_of(terms, float(ins.operation.time), bits, q)
            elif name in ("rz", "rzz"):  # exp(-i theta/2 Z...) on its qubits
                got += phase_of([("Z" * len(q), 1.0)], float(ins.operation.params[0]) / 2, bits, q)
            elif name in ("barrier",):
                continue
            else:
                return f"unexpected {name} in the routed circuit"
        if any(at[meas_pos[i]] != i for i in range(n)):
            return "a qubit is not where it is measured"
        diffs.append((got - want) % (2 * math.pi))
    spread = max(min(abs(d - diffs[0]), 2 * math.pi - abs(d - diffs[0])) for d in diffs)
    return True if spread < 1e-9 else f"phases differ by up to {spread:.3g}"


def one(a):
    warnings.simplefilter("ignore")
    stratum, tid, kind, arg = json.loads(a.job)
    import bp_mock as B
    import core_fix_c2_eval as H
    from qiskit import QuantumCircuit, transpile
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import (Commuting2qGateRouter,
                                                                            FindCommutingPauliEvolutions,
                                                                            SwapStrategy)
    qc, bp_backend = B.build(a.bp, kind, arg)
    backend = device(a.device, bp_backend)
    target = backend.target
    rec = dict(stratum=stratum, test=tid, device=a.device, arm=a.arm, input_qubits=qc.num_qubits)
    d = diagonal_evolution(qc)
    if isinstance(d, str):
        print(json.dumps(dict(rec, eligible=False, why=d)), flush=True)
        return
    evo, eq, meas = d
    n = len(eq)
    if eq != list(range(n)) or qc.num_qubits != n:
        print(json.dumps(dict(rec, eligible=False, why="the evolution is not on all qubits in order")), flush=True)
        return
    H.load_module(os.path.join(HERE, "psf_smart_layout.py"), "psf_smart_layout")
    pc = H.load_module(C32, "psf_compile")
    t0 = time.perf_counter()
    edges, fq = pc._failed_elements(target, FAILED)
    cm = pc.prune_coupling_map(target.build_coupling_map(), target, FAILED)
    g2 = next(g for g in ("cz", "ecr", "cx") if g in target.operation_names)
    err = {}
    for qa, p in target[g2].items():
        if qa is not None and p is not None and p.error is not None:
            err[tuple(qa)] = err[tuple(qa)[::-1]] = p.error
    und = {tuple(sorted(e)) for e in cm.get_edges() if not (set(e) & fq)}
    found = best_path(n, und, err)
    if found is None:
        print(json.dumps(dict(rec, eligible=False, why=f"no path of {n} working qubits found",
                              t=round(time.perf_counter() - t0, 3))), flush=True)
        return
    path = found[1]
    # route the evolution alone on the line 0..n-1, then measure each logical qubit where it ends
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp
    t_evo = float(evo.operation.time)
    one_q, two_q = [], []
    for lb, c in evo.operation.operator.to_list():
        w = [j for j, ch in enumerate(reversed(lb)) if ch == "Z"]
        if len(w) == 1:
            one_q.append((w[0], float(complex(c).real)))
        elif len(w) == 2:
            two_q.append((lb, float(complex(c).real)))
    body = QuantumCircuit(n)
    for q, c in one_q:  # exp(-i t c Z) = rz(2 t c); every term commutes, so their order is free
        body.rz(2 * t_evo * c, q)
    if two_q:
        body.append(PauliEvolutionGate(SparsePauliOp.from_list(two_q), time=t_evo), list(range(n)))
    routed = PassManager([FindCommutingPauliEvolutions(),
                          Commuting2qGateRouter(SwapStrategy.from_line(list(range(n))))]).run(body)
    at = list(range(n))
    for ins in routed.data:
        if ins.operation.name == "swap":
            x, y = [routed.find_bit(b).index for b in ins.qubits]
            at[x], at[y] = at[y], at[x]
    pos = {lq: p for p, lq in enumerate(at)}
    meas_pos = [pos[i] for i in range(n)]
    terms = [(lb, float(complex(c).real)) for lb, c in evo.operation.operator.to_list() if set(lb) != {"I"}]
    exact = check_exact(terms, t_evo, n, routed, meas_pos)
    full = QuantumCircuit(n, qc.num_clbits)
    full.compose(routed, range(n), inplace=True)
    for q, c in meas:
        full.measure(pos[q], c)
    if a.arm == "SNC":
        with contextlib.redirect_stdout(io.StringIO()):
            full = pc.compile(full, entangling_basis="cx")
    out = transpile(full, target=target, initial_layout=path, routing_method="none", optimization_level=1,
                    seed_transpiler=0)
    rec["t"] = round(time.perf_counter() - t0, 3)
    rec.update(eligible=True, n=n, terms=len(terms), swaps=sum(1 for i in routed.data if i.operation.name == "swap"),
               exact=exact, path_error=round(found[0], 6))
    rec["q2"] = sum(1 for i in out.data if len(i.qubits) == 2 and i.operation.name not in ("barrier", "delay"))
    rec["esp"], rec["on_failed"] = esp_of(out, target)
    print(json.dumps(rec), flush=True)


def git(*args):
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True).stdout.strip()


def run(a):
    if os.path.exists(a.out):
        sys.exit(f"STOP: {a.out} exists")
    ts = tests(a.bp)
    if a.smoke:
        ts = [t for t in ts if "tsp_prob-lin105_Ncity-7_enc-unary" in t[1]][:1] or ts[:1]
    jobs = sorted((hashlib.sha256(("SWAPNET2|" + t[1] + d + x).encode()).hexdigest(), t, d, x)
                  for t in ts for d in DEVICES for x in ARMS)
    os.makedirs(a.out)
    path = os.path.join(a.out, "swapnet2.jsonl")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(dict(meta=dict(start_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                           git_head=git("rev-parse", "--short", "HEAD"),
                                           dirty_tracked=git("status", "--porcelain", "--untracked-files=no"),
                                           tests=len(ts), jobs=len(jobs), par=a.par))) + "\n")
    print(f"SWAPNET2: {len(ts)} tests, {len(jobs)} jobs, par {a.par}", flush=True)
    t_start = time.perf_counter()

    def go(job):
        _, t, d, x = job
        base = dict(stratum=t[0], test=t[1], device=d, arm=x)
        if time.perf_counter() - t_start > BUDGET:
            return dict(base, error="not started (budget)")
        try:
            p = subprocess.run([sys.executable, os.path.abspath(__file__), "one", "--bp", a.bp, "--arm", x,
                                "--device", d, "--job", json.dumps(list(t))], capture_output=True, text=True,
                               timeout=TIMEOUT, cwd=REPO, env=dict(os.environ, PSF_ZERO_REPO=REPO))
            lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
            return json.loads(lines[-1]) if lines else dict(base, error=(p.stderr or "")[-400:], exit=p.returncode)
        except subprocess.TimeoutExpired:
            return dict(base, error=f"timeout {TIMEOUT} s")

    n = 0
    with ThreadPoolExecutor(a.par) as ex, open(path, "a", encoding="utf-8", newline="\n") as fh:
        for f in as_completed([ex.submit(go, j) for j in jobs]):
            r = f.result()
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            n += 1
            msg = (r["error"][:60] if "error" in r else
                   (f"{r['t']} s, q2 {r['q2']}, exact {r['exact']}, esp {r['esp']}" if r.get("eligible")
                    else f"not eligible: {r['why']}"))
            print(f"[{n}/{len(jobs)} {time.perf_counter() - t_start:5.0f} s] {r['device'][4:]:8s} {r['arm']} "
                  f"{r['test'][-42:]}: {msg}", flush=True)
    report(a)


def report(a):
    lines = [json.loads(x) for x in open(os.path.join(a.out, "swapnet2.jsonl"), encoding="utf-8")]
    meta, recs = lines[0]["meta"], lines[1:]
    base = os.path.dirname(os.path.abspath(a.out))
    other = {}
    for f, arms in (("esp_ft/esp_ft.jsonl", ("QK2", "QK3", "PSFR")), ("esp_c32/esp_c32.jsonl", ("C32D",))):
        p = os.path.join(base, f)
        if os.path.exists(p):
            for x in [json.loads(y) for y in open(p, encoding="utf-8")][1:]:
                if x.get("arm") in arms and "error" not in x:
                    other[(x["device"], x["test"], x["arm"])] = x
    lg = lambda r: -math.inf if r.get("esp") is None else r["esp"]  # noqa: E731
    gm = lambda xs: math.exp(sum(math.log(v) for v in xs) / len(xs)) if xs else float("nan")  # noqa: E731
    L = ["# SWAPNET2 (exploratory, nothing predicted)", "",
         f"git head {meta['git_head']}, uncommitted tracked changes: {'none' if not meta['dirty_tracked'] else 'YES'}; "
         f"{meta['tests']} HamLib FakeTorino tests on {', '.join(DEVICES)}; {meta['jobs']} jobs.", ""]
    why = {}
    for r in recs:
        if not r.get("eligible") and "why" in r and r["device"] == DEVICES[0] and r["arm"] == ARMS[0]:
            why[r["why"]] = why.get(r["why"], 0) + 1
    L += ["Not eligible (FakeTorino, SN1): " + ("; ".join(f"{k}: {v}" for k, v in sorted(why.items())) or "none"), ""]
    for d in DEVICES:
        L += [f"## {d}", "", "| arm | eligible | exact | ESP = 0 | q2 / QK3 (gmean, +1) | q2 / QK2 | q2 / C32D | "
              "ESP / QK3 (gmean, both > 0) | ESP / C32D | wins / losses vs QK3 (ESP, 10%) | time (s, summed) | QK3 time |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for x in ARMS:
            rs = [r for r in recs if r["device"] == d and r["arm"] == x and r.get("eligible")]
            def ratio(o, key):
                v = [((r["q2"] + 1) / (other[(d, r["test"], o)]["q2"] + 1)) for r in rs if (d, r["test"], o) in other]
                return gm(v)

            def eratio(o):
                v = [lg(r) - lg(other[(d, r["test"], o)]) for r in rs if (d, r["test"], o) in other]
                v = [u for u in v if math.isfinite(u)]
                return 10 ** (sum(v) / len(v)) if v else float("nan")
            wl = [lg(r) - lg(other[(d, r["test"], "QK3")]) for r in rs if (d, r["test"], "QK3") in other]
            L.append(f"| {x} | {len(rs)} | {sum(1 for r in rs if r['exact'] is True)} | "
                     f"{sum(1 for r in rs if r['esp'] is None)} | {ratio('QK3', 'q2'):.3f} | {ratio('QK2', 'q2'):.3f} | "
                     f"{ratio('C32D', 'q2'):.3f} | {eratio('QK3'):.3f} | {eratio('C32D'):.3f} | "
                     f"{sum(1 for v in wl if v >= math.log10(1.1))} / {sum(1 for v in wl if v <= -math.log10(1.1))} | "
                     f"{sum(r['t'] for r in rs):.1f} | "
                     f"{sum(other[(d, r['test'], 'QK3')]['t'] for r in rs if (d, r['test'], 'QK3') in other):.1f} |")
        L += ["", "| test | n | terms | SNC q2 | QK2 q2 | QK3 q2 | C32D q2 | SNC log10 ESP | QK3 log10 ESP | exact |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for r in sorted((r for r in recs if r["device"] == d and r["arm"] == "SNC" and r.get("eligible")),
                        key=lambda r: r["n"]):
            g = lambda o, k: other.get((d, r["test"], o), {}).get(k, "-")  # noqa: E731
            L.append(f"| {r['test'].split('[')[-1].rstrip(']')[:44]} | {r['n']} | {r['terms']} | {r['q2']} | "
                     f"{g('QK2', 'q2')} | {g('QK3', 'q2')} | {g('C32D', 'q2')} | {r['esp']} | {g('QK3', 'esp')} | "
                     f"{r['exact']} |")
        L += [f"- error ({r['test'][-40:]}, {r['arm']}): {r['error'][:120]}" for r in recs if r["device"] == d and "error" in r]
        L.append("")
    txt = "\n".join(L) + "\n"
    open(os.path.join(a.out, "swapnet2.md"), "w", encoding="utf-8", newline="\n").write(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("run", "one", "report"))
    ap.add_argument("--bp")
    ap.add_argument("--out")
    ap.add_argument("--arm")
    ap.add_argument("--device")
    ap.add_argument("--job")
    ap.add_argument("--par", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    {"run": run, "one": one, "report": report}[a.mode](a)


if __name__ == "__main__":
    main()
