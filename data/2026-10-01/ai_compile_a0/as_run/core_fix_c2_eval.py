"""core_fix_c2_eval.py -- pre-registered evaluation of the PSF-Zero candidates of 2026-10-01 (workplace).

Candidates (both on held-out inputs that were not used while developing them):
  * psf_compile.py 2026-10-01.c2 (base: release 2026-09-28.1)
      (i)   cost-aware consolidation of short 2-qubit blocks (entangling_basis="cx" only)
      (ii)  elide_permutations="auto": ElidePermutations + Split2QUnitaries(split_swap=True) before compile
      (iii) post_routing_resynthesis="auto": routing SWAPs absorbed into neighbouring 2-qubit gates
  * psf_smart_layout.py 2026-10-01.c2 (base: candidate 2026-09-29.c1): exact packing search for disjoint
    2- and 3-qubit paths (budget min(1 s, half the layout budget)).

Parts (all fake-provider snapshots, no IBM access):
  L  layout: tilings of disjoint 3-qubit paths and pairs on FakeAuckland/Torino/Kingston/Nighthawk;
     layout arms m1 (release), c1, c2, all with compile c2.
  C  small dense circuits (3-5 qubits, random seeds 1001-1090 per backend) and textbook circuits
     (W, Dicke, QFT, GHZ, swap networks; gate form and PennyLane-style `unitary` form) on FakeAuckland
     and FakeKingston; compile arms REL (2026-09-28.1) and CAND (c2), both with layout c2; Qiskit L3 as
     reference; canonical-basis identity check; measured-circuit check with Aer.
  R  larger circuits on FakeKingston (random_circuit, dense_pair_blocks, families); REL vs CAND; times.

Usage:
  python core_fix_c2_eval.py run   --rel-compile <psf_compile.py> --cand-compile <psf_compile.py>
                                   --layout-m1 <psf_smart_layout.py> --layout-c1 <...> --layout-c2 <...>
                                   [--families <circuit_family_sweep.py dir>] [--out core_fix_c2_raw.json]
  python core_fix_c2_eval.py score [--out core_fix_c2_raw.json]
Times are wall-clock in this process, median of 3, and are only compared within one run.
"""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
import platform
import random
import statistics
import sys
import time
import warnings

import numpy as np

warnings.simplefilter("ignore")

GT_BUDGET_S = 30.0          # ground-truth packing budget (outside any timed call)
FID_TOL = 1e-9
MAX_COMPONENT = 12

# ---------------------------------------------------------------- module loading


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def norm_sha(path):
    with open(path, "rb") as f:
        txt = f.read().decode("utf-8").replace("\r\n", "\n")
    lines = [ln.rstrip() for ln in txt.split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- helpers


def two_q(c):
    return sum(1 for inst in c.data if len(inst.qubits) == 2)


def digest(c):
    h = hashlib.sha256()
    for inst in c.data:
        h.update(inst.operation.name.encode())
        h.update(str([c.find_bit(q).index for q in inst.qubits]).encode())
        h.update(str([round(float(p), 12) if isinstance(p, (int, float)) else str(p)
                      for p in inst.operation.params]).encode())
    return h.hexdigest()[:16]


def _sub_equiv(qc, out, logical, init, fin, trials=3, seed=1):
    """Fidelity of out (physical) against qc restricted to `logical` qubits. init/fin: logical -> physical.
    Physical qubits touched by out's gates on the component are included as ancillas (start |0>)."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector, random_statevector, partial_trace
    logical = sorted(logical)
    lpos = {q: i for i, q in enumerate(logical)}
    n = len(logical)
    lsub = QuantumCircuit(n)
    for inst in qc.data:
        qs = [qc.find_bit(q).index for q in inst.qubits]
        if inst.operation.name in ("barrier", "measure", "delay"):
            continue
        if all(q in lpos for q in qs):
            lsub.append(inst.operation, [lpos[q] for q in qs])
        elif any(q in lpos for q in qs):
            raise ValueError("logical component is not closed")
    phys_set = {init[q] for q in logical} | {fin[q] for q in logical}
    insts = [(inst.operation, [out.find_bit(q).index for q in inst.qubits]) for inst in out.data
             if inst.operation.name not in ("barrier", "measure", "delay")]
    grow = True
    while grow:
        grow = False
        for _, qs in insts:
            if any(q in phys_set for q in qs) and not all(q in phys_set for q in qs):
                phys_set |= set(qs)
                grow = True
    if len(phys_set) > MAX_COMPONENT:
        return None
    others = sorted(phys_set - {init[q] for q in logical})
    sidx = {init[q]: lpos[q] for q in logical}
    for j, p in enumerate(others):
        sidx[p] = n + j
    m = n + len(others)
    small = QuantumCircuit(m)
    for op, qs in insts:
        if qs[0] in phys_set:
            small.append(op, [sidx[q] for q in qs])
    where = [sidx[fin[q]] for q in logical]
    occupant = {i: None for i in range(m)}
    for v, i in enumerate(where):
        occupant[i] = v
    for v in range(n):
        i = where[v]
        if i != v:
            small.swap(i, v)
            u = occupant[v]
            occupant[v], occupant[i] = v, u
            where[v] = v
            if u is not None:
                where[u] = i
    rng = np.random.default_rng(seed)
    worst = 1.0
    for _ in range(trials):
        psi = random_statevector(2 ** n, seed=int(rng.integers(1 << 30)))
        phi = psi.evolve(lsub).data
        full = Statevector.from_label("0" * (m - n)).tensor(psi) if m > n else psi
        res = full.evolve(small)
        if m > n:
            rho = partial_trace(res, list(range(n, m))).data
        else:
            rho = np.outer(res.data, res.data.conj())
        worst = min(worst, float(np.real(phi.conj() @ rho @ phi)))
    return worst


def component_fidelity(qc, out):
    """Worst fidelity over the logical components of qc (connected by its multi-qubit gates);
    None if some component's physical support exceeds MAX_COMPONENT qubits."""
    n = qc.num_qubits
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for inst in qc.data:
        qs = [qc.find_bit(q).index for q in inst.qubits]
        for a in qs[1:]:
            parent[find(a)] = find(qs[0])
    comps = {}
    for q in range(n):
        comps.setdefault(find(q), []).append(q)
    lay = out.layout
    init = lay.initial_index_layout(filter_ancillas=True)[:n]
    fin = lay.final_index_layout(filter_ancillas=True)[:n]
    # merge logical components that the output entangles physically
    phys_owner = {}
    for inst in out.data:
        if inst.operation.name in ("barrier", "measure", "delay"):
            continue
        qs = [out.find_bit(q).index for q in inst.qubits]
        if len(qs) > 1:
            for p in qs:
                phys_owner.setdefault(p, set()).update(qs)
    # simple closure: logical components whose physical supports overlap are merged
    supports = {}
    for root, qs in comps.items():
        s = {init[q] for q in qs} | {fin[q] for q in qs}
        grow = True
        while grow:
            grow = False
            for p in list(s):
                extra = phys_owner.get(p, set()) - s
                if extra:
                    s |= extra
                    grow = True
            if len(s) > 4 * MAX_COMPONENT:
                break
        supports[root] = s
    roots = list(comps)
    merged = {r: r for r in roots}

    def mfind(r):
        while merged[r] != r:
            r = merged[r]
        return r
    for i, a in enumerate(roots):
        for b in roots[i + 1:]:
            if supports[a] & supports[b]:
                merged[mfind(b)] = mfind(a)
    groups = {}
    for r in roots:
        groups.setdefault(mfind(r), []).extend(comps[r])
    worst = 1.0
    for qs in groups.values():
        if len(qs) > MAX_COMPONENT:
            return None
        f = _sub_equiv(qc, out, qs, init, fin)
        if f is None:
            return None
        worst = min(worst, f)
    return worst


def timed(fn, reps=3):
    ts, out = [], None
    for _ in range(reps):
        t = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t)
    return out, statistics.median(ts)


# ---------------------------------------------------------------- circuit generators


def rand_dense(n, ngates, rng):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate, SwapGate
    qc = QuantumCircuit(n)
    for _ in range(ngates):
        if rng.random() < 0.3:
            q = rng.randrange(n)
            if rng.random() < 0.5:
                getattr(qc, rng.choice(["h", "x", "s", "t"]))(q)
            else:
                qc.ry(rng.uniform(0, 2 * math.pi), q)
        else:
            a, b = rng.sample(range(n), 2)
            g = rng.choice(["cx", "cry", "crz", "cp", "swap", "cz"])
            if g == "swap" and rng.random() < 0.5:
                qc.append(UnitaryGate(SwapGate().to_matrix()), [a, b])
            elif g in ("cx", "swap", "cz"):
                getattr(qc, g)(a, b)
            else:
                getattr(qc, g)(rng.uniform(0, 2 * math.pi), a, b)
    return qc


def to_unitary_form(qc):
    """PennyLane-style: every multi-qubit gate becomes a `unitary` with the same matrix."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    out = QuantumCircuit(qc.num_qubits)
    for inst in qc.data:
        qs = [qc.find_bit(q).index for q in inst.qubits]
        if len(qs) >= 2:
            out.append(UnitaryGate(inst.operation.to_matrix()), qs)
        else:
            out.append(inst.operation, qs)
    return out


def textbook_circuits():
    from qiskit import QuantumCircuit
    cs = []
    for n in (3, 4, 5):  # W state, cry cascade (Diker style)
        qc = QuantumCircuit(n, name=f"W{n}")
        qc.x(0)
        for k in range(n - 1):
            theta = 2 * math.acos(math.sqrt(1.0 / (n - k)))
            qc.cry(theta, k, k + 1)
            qc.cx(k + 1, k)
        cs.append(qc)
    qc = QuantumCircuit(4, name="Dicke42")  # split-and-cyclic-shift style, 2-qubit gates only after decomposition
    qc.x(2); qc.x(3)
    for (a, b, th) in [(2, 3, 2 * math.acos(math.sqrt(1 / 4))), (1, 2, 2 * math.acos(math.sqrt(2 / 4))),
                       (1, 3, 2 * math.acos(math.sqrt(1 / 3))), (0, 1, 2 * math.acos(math.sqrt(2 / 3))),
                       (0, 2, 2 * math.acos(math.sqrt(1 / 2))), (0, 1, 2 * math.acos(math.sqrt(1 / 2)))]:
        qc.cx(a, b); qc.cry(th, b, a); qc.cx(a, b)
    cs.append(qc)
    for n in (3, 4, 5):  # QFT with final swaps
        qc = QuantumCircuit(n, name=f"QFT{n}")
        for j in range(n):
            qc.h(j)
            for k in range(j + 1, n):
                qc.cp(math.pi / 2 ** (k - j), k, j)
        for j in range(n // 2):
            qc.swap(j, n - 1 - j)
        cs.append(qc)
    for n in (4, 5):  # GHZ star
        qc = QuantumCircuit(n, name=f"GHZstar{n}")
        qc.h(0)
        for k in range(1, n):
            qc.cx(0, k)
        cs.append(qc)
    qc = QuantumCircuit(4, name="SwapNet4")  # odd-even swap network with phases
    for layer in range(4):
        for a in range(layer % 2, 3, 2):
            qc.cp(0.3 + 0.1 * a + 0.05 * layer, a, a + 1)
            qc.swap(a, a + 1)
    cs.append(qc)
    qc = QuantumCircuit(5, name="Reverse5")  # qubit reversal by swaps after entangling
    qc.h(0)
    for k in range(4):
        qc.cx(k, k + 1)
    qc.swap(0, 4); qc.swap(1, 3)
    cs.append(qc)
    out = []
    for c in cs:
        out.append((c.name + "/gates", c))
        out.append((c.name + "/unitary", to_unitary_form(c)))
    return out


def tiling_circuit(k3, k2, seed):
    """k3 disjoint 3-qubit paths (a-b-c) and k2 pairs, logical labels shuffled."""
    from qiskit import QuantumCircuit
    rng = random.Random(seed)
    n = 3 * k3 + 2 * k2
    labels = list(range(n))
    rng.shuffle(labels)
    qc = QuantumCircuit(n)
    it = iter(labels)
    for _ in range(k3):
        a, b, c = next(it), next(it), next(it)
        qc.h(a); qc.cx(a, b); qc.ry(rng.uniform(0, 3), b); qc.cx(b, c); qc.rz(rng.uniform(0, 3), c)
    for _ in range(k2):
        a, b = next(it), next(it)
        qc.h(a); qc.cx(a, b); qc.ry(rng.uniform(0, 3), b)
    return qc, 2 * k3 + k2


# Dry-run configuration (harness check only; development cases and other seeds, never the scored inputs).
DRY_LAYOUT_COMBOS = {"FakeAuckland": [(7, 3), (9, 0)], "FakeKingston": [(39, 17)]}
DRY_SEED_OFFSET = 900000

LAYOUT_COMBOS = {
    "FakeAuckland": [(6, 4), (5, 6), (4, 7), (6, 3)],
    "FakeTorino": [(21, 35), (25, 29), (30, 21), (37, 11), (44, 0), (20, 36)],
    "FakeKingston": [(28, 36), (32, 30), (40, 18), (46, 9), (30, 32)],
    "FakeNighthawk": [(10, 45), (20, 30), (40, 0), (26, 21)],
}

# ---------------------------------------------------------------- run


def backend(name):
    from qiskit_ibm_runtime import fake_provider
    tgt = getattr(fake_provider, name)().target
    cm = tgt.build_coupling_map()
    nat = [g for g in ("cx", "cz", "rz", "sx", "x") if g in tgt.operation_names]
    return cm, nat


def cfh(mod, qc, cm, nat, basis="cx"):
    with contextlib.redirect_stdout(io.StringIO()):
        return mod.compile_for_hardware(qc, coupling_map=cm, basis_gates=nat, entangling_basis=basis,
                                        layout_search=True, seed_transpiler=0)


def run(args):
    from qiskit import transpile
    rel = load_module(args.rel_compile, "psfc_rel")
    cand = load_module(args.cand_compile, "psfc_cand")
    lays = {k: load_module(p, "psl_" + k) for k, p in
            (("m1", args.layout_m1), ("c1", args.layout_c1), ("c2", args.layout_c2))}

    def use_layout(k):
        sys.modules["psf_smart_layout"] = lays[k]

    import psf_zero_core
    meta = dict(
        python=platform.python_version(), platform=platform.platform(), processor=platform.processor(),
        cpu_count=os.cpu_count(), qiskit=__import__("qiskit").__version__,
        rel_version=rel.VERSION, cand_version=cand.VERSION,
        layout_versions={k: m.LAYOUT_VERSION for k, m in lays.items()},
        core_version=getattr(psf_zero_core, "CORE_VERSION", None),
        sha={"rel_compile": norm_sha(args.rel_compile), "cand_compile": norm_sha(args.cand_compile),
             "layout_m1": norm_sha(args.layout_m1), "layout_c1": norm_sha(args.layout_c1),
             "layout_c2": norm_sha(args.layout_c2), "script": norm_sha(os.path.abspath(__file__))},
    )
    print("META", json.dumps(meta), flush=True)
    raw = {"meta": meta, "L": [], "C": [], "T": [], "K": [], "M": [], "R": []}

    # ---- Part L
    combos_all = DRY_LAYOUT_COMBOS if args.dry else LAYOUT_COMBOS
    off = DRY_SEED_OFFSET if args.dry else 0
    raw["meta"]["dry"] = bool(args.dry)
    for bname, combos in combos_all.items():
        cm, nat = backend(bname)
        for idx, (k3, k2) in enumerate(combos):
            qc, swapfree = tiling_circuit(k3, k2, seed=off + 5000 + 100 * list(combos_all).index(bname) + idx)
            ipairs = sorted({tuple(sorted((qc.find_bit(i.qubits[0]).index, qc.find_bit(i.qubits[1]).index)))
                             for i in qc.data if len(i.qubits) == 2})
            comps = lays["c2"]._short_path_components(ipairs)
            t0 = time.perf_counter()
            gt = None
            if comps is not None:
                lm = lays["c2"].packing_layout(cm, *comps, time_budget_s=GT_BUDGET_S)
                el = time.perf_counter() - t0
                gt = "feasible" if lm is not None else ("infeasible" if el < 0.95 * GT_BUDGET_S else "unknown")
            row = dict(backend=bname, k3=k3, k2=k2, n=qc.num_qubits, spare=cm.size() - qc.num_qubits,
                       swapfree_2q=swapfree, ground_truth=gt, gt_s=round(time.perf_counter() - t0, 3))
            for k in ("m1", "c1", "c2"):
                use_layout(k)
                out, dt = timed(lambda: cfh(cand, qc, cm, nat))
                f = component_fidelity(qc, out)
                row[k] = dict(two_q=two_q(out), s=round(dt, 4), digest=digest(out), fid=f)
            print("L", json.dumps(row), flush=True)
            raw["L"].append(row)

    use_layout("c2")
    # ---- Part C (random dense) and K (canonical identity, Auckland) and M (measured, Auckland)
    from qiskit_aer import AerSimulator
    from qiskit.quantum_info import Statevector
    sim = AerSimulator()
    for bname in ("FakeAuckland", "FakeKingston"):
        cm, nat = backend(bname)
        for seed in (range(off + 1001, off + 1007) if args.dry else range(1001, 1091)):
            rng = random.Random(seed)
            n = rng.choice([3, 4, 5])
            qc = rand_dense(n, rng.randint(6, 20), rng)
            row = dict(backend=bname, seed=seed, n=n)
            for tag, mod in (("rel", rel), ("cand", cand)):
                out = cfh(mod, qc, cm, nat)
                row[tag] = dict(two_q=two_q(out), fid=component_fidelity(qc, out))
            row["L3"] = two_q(transpile(qc, coupling_map=cm, basis_gates=nat, optimization_level=3,
                                        seed_transpiler=0))
            raw["C"].append(row)
            print("C", json.dumps(row), flush=True)
            if bname == "FakeAuckland":
                a = cfh(rel, qc, cm, nat, basis="canonical")
                b = cfh(cand, qc, cm, nat, basis="canonical")
                raw["K"].append(dict(seed=seed, rel=digest(a), cand=digest(b)))
                if seed < off + 1031:
                    ideal = Statevector(qc).probabilities_dict()
                    qm = qc.copy()
                    qm.measure_all()
                    out = cfh(cand, qm, cm, nat)
                    counts = sim.run(out, shots=20000, seed_simulator=seed).result().get_counts()
                    tvd = 0.5 * sum(abs(counts.get(k, 0) / 20000 - ideal.get(k, 0))
                                    for k in set(counts) | set(ideal))
                    raw["M"].append(dict(seed=seed, tvd=round(tvd, 5)))
        for name, qc in (textbook_circuits()[:4] if args.dry else textbook_circuits()):
            row = dict(backend=bname, name=name, n=qc.num_qubits)
            for tag, mod in (("rel", rel), ("cand", cand)):
                out = cfh(mod, qc, cm, nat)
                row[tag] = dict(two_q=two_q(out), fid=component_fidelity(qc, out))
            row["L3"] = two_q(transpile(qc, coupling_map=cm, basis_gates=nat, optimization_level=3,
                                        seed_transpiler=0))
            raw["T"].append(row)
            print("T", json.dumps(row), flush=True)

    # ---- Part R
    sys.path.insert(0, args.families)
    import circuit_family_sweep as F
    from qiskit.circuit.random import random_circuit
    cm, nat = backend("FakeKingston")
    cases = []
    rc = [(16, 20, off + 2001)] if args.dry else [(16, 20, 2001), (32, 20, 2002), (48, 15, 2003), (80, 10, 2004),
                                                  (120, 8, 2005)]
    for (nq, d, s) in rc:
        cases.append((f"random_circuit {nq}q d{d} s{s}", "other", random_circuit(nq, d, max_operands=2, seed=s)))
    fs = 1 + off
    for nq in ((60,) if args.dry else (60, 120, 156)):
        cases.append((f"dense_pair_blocks {nq}q s{fs}", "unchanged", F.build_dense_pair_blocks_circuit(nq, seed=fs)))
    for fam, kind in ((("ghz_star", "other"),) if args.dry else
                      (("k_chains", "unchanged"), ("linear_chain", "unchanged"),
                       ("random_regular", "other"), ("ghz_star", "other"))):
        cases.append((f"{fam} 80q s{fs}", kind, F.build_circuit_from_family(80, fam, seed=fs)[0]))
    for name, kind, qc in cases:
        row = dict(name=name, kind=kind, n=qc.num_qubits)
        for tag, mod in (("rel", rel), ("cand", cand)):
            out, dt = timed(lambda: cfh(mod, qc, cm, nat))
            row[tag] = dict(two_q=two_q(out), s=round(dt, 4), fid=component_fidelity(qc, out))
        raw["R"].append(row)
        print("R", json.dumps(row), flush=True)

    with open(args.out, "w") as f:
        json.dump(raw, f, indent=1)
    print("wrote", args.out)


# ---------------------------------------------------------------- score


def verdict(ok, bad):
    return "CONFIRMED" if ok else ("REFUTED" if bad else "AMBIGUOUS")


def score(args):
    raw = json.load(open(args.out))
    res = {}
    # C0 harness
    m = raw["meta"]
    c0 = (m["rel_version"] == "2026-09-28.1" and m["cand_version"] == "2026-10-01.c2"
          and m["layout_versions"]["c2"].startswith("2026-10-01.c2")
          and m["layout_versions"]["c1"].startswith("2026-09-29.c1"))
    print("C0 harness versions:", "OK" if c0 else "MISMATCH", m["rel_version"], m["cand_version"], m["layout_versions"])

    def fid_ok(f):
        return f is not None and f > 1 - FID_TOL

    L = raw["L"]
    feas = [r for r in L if r["ground_truth"] == "feasible"]
    infe = [r for r in L if r["ground_truth"] == "infeasible"]
    unk = [r for r in L if r["ground_truth"] == "unknown"]
    l1_hits = [r for r in feas if r["c2"]["two_q"] == r["swapfree_2q"] and r["c2"]["s"] <= 1.0]
    res["L1"] = verdict(len(l1_hits) == len(feas) and feas, len(l1_hits) < len(feas) - 1)
    print(f"L1 feasible tilings placed swap-free within 1 s by c2: {len(l1_hits)}/{len(feas)} "
          f"(infeasible {len(infe)}, unknown {len(unk)}) -> {res['L1']}")
    l2_bad = [r for r in L if r["c2"]["two_q"] > min(r["m1"]["two_q"], r["c1"]["two_q"])]
    l2_dig = [r for r in L if r["c1"]["two_q"] == r["swapfree_2q"] and r["c1"]["digest"] != r["c2"]["digest"]]
    res["L2"] = verdict(not l2_bad and not l2_dig, bool(l2_bad))
    print(f"L2 c2 above min(m1,c1): {len(l2_bad)}; c1-swap-free digests changed by c2: {len(l2_dig)} -> {res['L2']}")
    l3_bad = [r for r in infe + unk if r["c2"]["s"] > r["c1"]["s"] + 1.2]
    res["L3"] = verdict(not l3_bad, len(l3_bad) > 1)
    print(f"L3 infeasible/unknown tilings where c2 time > c1 time + 1.2 s: {len(l3_bad)}/{len(infe) + len(unk)} -> {res['L3']}")
    fl = [(r[k]["fid"]) for r in L for k in ("m1", "c1", "c2")]
    l4_bad = [f for f in fl if f is not None and f <= 1 - FID_TOL]
    res["L4"] = verdict(not l4_bad, bool(l4_bad))
    print(f"L4 layout outputs checked {sum(f is not None for f in fl)}/{len(fl)}, below 1-1e-9: {len(l4_bad)} -> {res['L4']}")

    C = raw["C"]
    allc = C + raw["T"]
    c1_bad = [r for r in allc if r["cand"]["fid"] is not None and not fid_ok(r["cand"]["fid"])]
    c1_unc = [r for r in allc if r["cand"]["fid"] is None]
    c1_rel = [r for r in allc if r["rel"]["fid"] is not None and not fid_ok(r["rel"]["fid"])]
    res["C1"] = verdict(not c1_bad and len(c1_unc) <= 0.05 * len(allc), bool(c1_bad))
    print(f"C1 CAND not equivalent: {len(c1_bad)}, not checkable: {len(c1_unc)} (REL not equivalent: "
          f"{len(c1_rel)}) of {len(allc)} -> {res['C1']}")
    ok2, bad2, ok3, bad3 = True, False, True, False
    for b in ("FakeAuckland", "FakeKingston"):
        rows = [r for r in C if r["backend"] == b]
        sr = sum(r["rel"]["two_q"] for r in rows)
        sc = sum(r["cand"]["two_q"] for r in rows)
        sl = sum(r["L3"] for r in rows)
        worse = sum(r["cand"]["two_q"] > r["rel"]["two_q"] for r in rows) / len(rows)
        above = sum(r["cand"]["two_q"] > r["L3"] for r in rows) / len(rows)
        print(f"   {b}: sum2q REL {sr} CAND {sc} L3 {sl} | CAND>REL {worse:.1%} | CAND/L3 {sc / sl:.3f} CAND>L3 {above:.1%}")
        ok2 &= sc < sr and worse <= 0.05
        bad2 |= sc >= sr or worse > 0.10
        ok3 &= sc <= 1.08 * sl and above <= 0.25
        bad3 |= sc > 1.15 * sl or above > 0.40
    res["C2"] = verdict(ok2, bad2)
    res["C3"] = verdict(ok3, bad3)
    print(f"C2 CAND below REL (sum lower, <=5% circuits worse) -> {res['C2']}")
    print(f"C3 CAND near L3 (sum <=1.08x, <=25% circuits above) -> {res['C3']}")
    T = raw["T"]
    t_bad = [r for r in T if r["cand"]["two_q"] > r["rel"]["two_q"]]
    res["C4"] = verdict(not t_bad, bool(t_bad))
    print(f"C4 textbook circuits where CAND > REL: {len(t_bad)}/{len(T)} -> {res['C4']}")
    for r in T:
        print(f"   {r['backend']:13s} {r['name']:22s} REL {r['rel']['two_q']:3d} CAND {r['cand']['two_q']:3d} L3 {r['L3']:3d}")
    K = raw["K"]
    k_bad = [r for r in K if r["rel"] != r["cand"]]
    res["C5"] = verdict(not k_bad, bool(k_bad))
    print(f"C5 canonical-basis outputs differing REL vs CAND: {len(k_bad)}/{len(K)} -> {res['C5']}")
    M = raw["M"]
    m_bad = [r for r in M if r["tvd"] > 0.03]
    res["C6"] = verdict(not m_bad, bool(m_bad))
    print(f"C6 measured circuits with TVD > 0.03 (20,000 shots): {len(m_bad)}/{len(M)}, max {max(r['tvd'] for r in M):.4f} -> {res['C6']}")

    R = raw["R"]
    r1_bad = [r for r in R if r["cand"]["two_q"] > r["rel"]["two_q"]]
    res["R1"] = verdict(not r1_bad, bool(r1_bad))
    print(f"R1 larger circuits where CAND 2q > REL: {len(r1_bad)}/{len(R)} -> {res['R1']}")
    un = [r for r in R if r["kind"] == "unchanged"]
    ot = [r for r in R if r["kind"] == "other"]
    r2_ok = all(r["cand"]["s"] <= 1.25 * r["rel"]["s"] + 0.010 for r in un)
    r2_bad = any(r["cand"]["s"] > 1.5 * r["rel"]["s"] + 0.020 for r in un)
    r3_ok = all(r["cand"]["s"] <= 2.0 * r["rel"]["s"] + 0.020 for r in ot)
    r3_bad = any(r["cand"]["s"] > 3.0 * r["rel"]["s"] + 0.050 for r in ot)
    res["R2"] = verdict(r2_ok, r2_bad)
    res["R3"] = verdict(r3_ok, r3_bad)
    for r in R:
        print(f"   {r['name']:34s} 2q REL {r['rel']['two_q']:5d} CAND {r['cand']['two_q']:5d} | "
              f"time REL {r['rel']['s'] * 1000:7.1f} ms CAND {r['cand']['s'] * 1000:7.1f} ms | fid checked "
              f"{r['cand']['fid'] is not None}")
    print(f"R2 time on unchanged families (<=1.25x+10 ms) -> {res['R2']}")
    print(f"R3 time on other circuits (<=2x+20 ms) -> {res['R3']}")
    rf = [r[k]["fid"] for r in R for k in ("rel", "cand")]
    r4_bad = [f for f in rf if f is not None and f <= 1 - FID_TOL]
    res["R4"] = verdict(not r4_bad, bool(r4_bad))
    print(f"R4 larger outputs checked {sum(f is not None for f in rf)}/{len(rf)}, below 1-1e-9: {len(r4_bad)} -> {res['R4']}")
    print("SUMMARY", json.dumps(res))
    adopt = c0 and all(v == "CONFIRMED" for v in res.values())
    print("DECISION:", "ADOPT-RECOMMENDED" if adopt else "NOT ALL CONFIRMED (see above)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "score"])
    ap.add_argument("--rel-compile")
    ap.add_argument("--cand-compile")
    ap.add_argument("--layout-m1")
    ap.add_argument("--layout-c1")
    ap.add_argument("--layout-c2")
    ap.add_argument("--families", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--out", default="core_fix_c2_raw.json")
    ap.add_argument("--dry", action="store_true", help="harness check on development cases and other seeds")
    a = ap.parse_args()
    run(a) if a.mode == "run" else score(a)


if __name__ == "__main__":
    main()
