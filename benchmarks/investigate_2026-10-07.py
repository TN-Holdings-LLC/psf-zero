"""investigate_2026-10-07.py -- exploratory, not pre-registered (2026-10-07): three questions left open by BP-MOCK and
BP-MOCK2 (Addenda 390, 392), on candidate c22 (items 46-49; its default call is release 2026-10-06.4's on inputs
without instructions on more than two qubits).

A. Reproducibility. The default call returned different circuits for the same BV input (same counts). Each BV test is
   compiled five times in one process with the stages recorded: PSF-Zero's own compile() (hash of its output), the
   layout search (its layout and what it reports), and the final circuit; then five times with layout_search=False,
   and five times with the first run's layout given as initial_layout. A control circuit (GHZ) is included.
B. BV-like circuits. Qiskit level 2 removes the CX gates of Benchpress's BV-like test (392 -> 0) and uses fewer on BV
   circuits. Each is compiled by QK (Benchpress's call), C22's default call, C22 after Qiskit's CommutativeCancellation
   on the input (CC), and C22 with routing level 2. The same CC pre-pass is then applied to circuits where nothing
   should cancel (SKIP's families, PL family T), to see whether it changes anything there.
C. The recommended call at 100 qubits on FakeTorino used more two-qubit gates than the default call. For each 100-qubit
   test: the default call; whether its output uses an element FakeTorino reports failed (error >= 0.5); the recommended
   call and its counters (PRUNE_STATS, COMPARE_STATS, SKIP_STATS, REFINE_STATS); and the target-aware call with one
   option at a time (pruning only, + placement_refine).

    cd <repo> && python benchmarks/investigate_2026-10-07.py --bp <benchpress clone> [--part A|B|C]
"""
import argparse
import contextlib
import hashlib
import io
import json
import os
import sys
import time
import warnings

import numpy as np

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path[:0] = [HERE, REPO]
RECOMMENDED = dict(placement_refine=True, final_resynthesis="select", compare_level3=True, compare_floor=True,
                   candidate_score="hybrid")


def h(x):
    return hashlib.sha256(json.dumps(x, default=str).encode()).hexdigest()[:10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bp", required=True)
    ap.add_argument("--part", default="FG")
    a = ap.parse_args()
    import bp_mock as B
    import core_fix_c2_eval as H
    sl = H.load_module(os.path.join(HERE, "psf_smart_layout.py"), "psf_smart_layout")
    pc = H.load_module(B.C22_PATH, "psf_compile")
    print(f"c22 {pc.VERSION}, core {pc.CORE_VERSION}")
    spec = {tid: (kind, arg) for _s, tid, kind, arg in
            [(s,) + tuple(t) for s, ts in B.strata(a.bp).items() for t in ts]}

    def build(tid):
        kind, arg = spec[tid]
        return B.build(a.bp, kind, arg)

    def q2(c, backend):
        name = getattr(backend, "two_q_gate_type", None)
        ops = c.count_ops()
        return ops.get(name, 0) if name else sum(ops.get(g, 0) for g in ("cz", "cx", "ecr"))

    def default(qc, backend, **kw):
        basis = [g for g in backend.operation_names if g in ("cx", "cz", "ecr", "rz", "sx", "x", "id")]
        with contextlib.redirect_stdout(io.StringIO()):
            return pc.compile_for_hardware(qc, coupling_map=backend.coupling_map, basis_gates=basis,
                                           entangling_basis="cx", layout_search=kw.pop("layout_search", True),
                                           seed_transpiler=0, **kw)

    # ------------------------------------------------------------------ A
    if "A" in a.part:
        print("\n== A. reproducibility of the default call")
        rec = {}
        orig_compile, orig_layout = pc.compile, sl.smart_vf2_layout

        def rec_compile(*args, **kw):
            out = orig_compile(*args, **kw)
            rec["compressed"] = B.sig_hash(out) if getattr(out, "layout", None) else h(
                [[i.operation.name, [out.find_bit(q).index for q in i.qubits], [repr(p) for p in i.operation.params]]
                 for i in out.data])
            return out

        def rec_layout(*args, **kw):
            t0 = time.perf_counter()
            m, info = orig_layout(*args, **kw)
            rec["layout"] = None if m is None else h(sorted(m.items()))
            rec["layout_map"] = m
            rec["layout_s"] = round(time.perf_counter() - t0, 2)
            rec["info"] = {k: v for k, v in (info or {}).items() if isinstance(v, (str, int, bool))}
            return m, info

        pc.compile, sl.smart_vf2_layout = rec_compile, rec_layout
        for tid in ("test_QASMBench_large[bv_n30-square]", "test_QASMBench_large[bv_n140-linear]",
                    "test_QASMBench_medium[bv_n19-heavy-hex]", "test_BV_100_transpile",
                    "test_QASMBench_large[ghz_n78-square]"):
            qc, backend = build(tid)
            print(f"\n{tid} ({qc.num_qubits} qubits, backend {backend.num_qubits})")
            first_map = None
            for r in range(5):
                rec.clear()
                out = default(qc, backend)
                first_map = first_map or rec.get("layout_map")
                print(f"  layout search  run {r}: compile() {rec.get('compressed')}  layout {rec.get('layout')} "
                      f"({rec.get('layout_s')} s, {rec.get('info')})  final {B.sig_hash(out)[:10]}  q2 {q2(out, backend)}")
            for r in range(5):
                rec.clear()
                out = default(qc, backend, layout_search=False)
                print(f"  no search      run {r}: compile() {rec.get('compressed')}  final {B.sig_hash(out)[:10]}  "
                      f"q2 {q2(out, backend)}")
            if first_map:
                lay = pc._layout_map_to_list(first_map, qc.num_qubits, backend.num_qubits)
                for r in range(5):
                    out = default(qc, backend, layout_search=False, initial_layout=lay)
                    print(f"  fixed layout   run {r}: final {B.sig_hash(out)[:10]}  q2 {q2(out, backend)}")
        pc.compile, sl.smart_vf2_layout = orig_compile, orig_layout

    # ------------------------------------------------------------------ D
    if "D" in a.part:
        from qiskit import transpile
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        print("\n== D. which stage is not reproducible (5 runs each; number of distinct circuits)")
        for tid in ("test_QASMBench_large[bv_n30-square]", "test_QASMBench_large[bv_n140-linear]"):
            qc, backend = build(tid)
            basis = [g for g in backend.operation_names if g in ("cx", "cz", "ecr", "rz", "sx", "x", "id")]
            with contextlib.redirect_stdout(io.StringIO()):
                comp = pc.compile(qc, entangling_basis="cx")
            sigs = set()
            for _ in range(5):
                with contextlib.redirect_stdout(io.StringIO()):
                    c = pc.compile(qc, entangling_basis="cx")
                sigs.add(h([[i.operation.name, [c.find_bit(q).index for q in i.qubits],
                             [repr(p) for p in i.operation.params]] for i in c.data]))
            print(f"\n{tid}: PSF-Zero's compile() alone, 5 runs: {len(sigs)} distinct")
            variants = {
                "default call, no layout search": dict(layout_search=False),
                "  elide_permutations=False": dict(layout_search=False, elide_permutations=False),
                "  post_routing_resynthesis=False": dict(layout_search=False, post_routing_resynthesis=False),
                "  both False": dict(layout_search=False, elide_permutations=False, post_routing_resynthesis=False)}
            for name, kw in variants.items():
                outs = [default(qc, backend, **dict(kw)) for _ in range(5)]
                print(f"  {name:36s} {len({B.sig_hash(o) for o in outs})} distinct, q2 {sorted({q2(o, backend) for o in outs})}")
            outs = [transpile(comp, coupling_map=backend.coupling_map, basis_gates=basis, optimization_level=1,
                              seed_transpiler=0) for _ in range(5)]
            print(f"  {'Qiskit transpile(level 1, seed 0) of compile()':36s} {len({B.sig_hash(o) for o in outs})} "
                  f"distinct, q2 {sorted({q2(o, backend) for o in outs})}")
            outs = [transpile(qc, coupling_map=backend.coupling_map, basis_gates=basis, optimization_level=1,
                              seed_transpiler=0) for _ in range(5)]
            print(f"  {'Qiskit transpile(level 1, seed 0) of the input':36s} {len({B.sig_hash(o) for o in outs})} "
                  f"distinct, q2 {sorted({q2(o, backend) for o in outs})}")

    # ------------------------------------------------------------------ B
    if "B" in a.part:
        from qiskit.transpiler import PassManager
        from qiskit.transpiler.passes import CommutativeCancellation, Unroll3qOrMore
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        def cc(qc):
            return PassManager([Unroll3qOrMore(), CommutativeCancellation()]).run(qc)

        print("\n== B. BV-like circuits: two-qubit gates (QK, C22, C22 after CommutativeCancellation, C22 routing 2)")
        for tid in ("test_BVlike_simplification_transpile", "test_BV_100_transpile",
                    "test_QASMBench_medium[bv_n19-heavy-hex]", "test_QASMBench_large[bv_n30-square]",
                    "test_QASMBench_large[bv_n140-linear]"):
            qc, backend = build(tid)
            qk = generate_preset_pass_manager(2, backend).run(qc)
            o, occ, o2 = default(qc, backend), default(cc(qc), backend), default(qc, backend,
                                                                                routing_optimization_level=2)
            print(f"  {tid[:56]:56s} input 2q {sum(1 for i in qc.data if len(i.qubits) == 2):5d}  after CC "
                  f"{sum(1 for i in cc(qc).data if len(i.qubits) == 2):5d} | QK {q2(qk, backend):5d}  C22 "
                  f"{q2(o, backend):5d}  CC+C22 {q2(occ, backend):5d}  C22 r2 {q2(o2, backend):5d}")
    if "E" in a.part:
        from qiskit.transpiler import PassManager
        from qiskit.transpiler.passes import CommutativeCancellation, Unroll3qOrMore

        def cc(qc):
            return PassManager([Unroll3qOrMore(), CommutativeCancellation()]).run(qc)

        print("\n== E. the same pre-pass where nothing should cancel (C22 default call, two-qubit gates without | with CC):")
        import skip_eval
        from qiskit_ibm_runtime import fake_provider
        t = fake_provider.FakeTorino()
        for fam in ("ring", "brick", "pauli", "qft"):
            for n in (12, 40):
                qc = skip_eval.family_circuit(fam, n, np.random.default_rng(86_000_000 + n))
                o, occ = default(qc, t), default(cc(qc), t)
                print(f"  {fam}{n:<3d} FakeTorino  {q2(o, t):6d} | {q2(occ, t):6d}  same circuit "
                      f"{B.sig_hash(o) == B.sig_hash(occ)}")
        import full_heavyhex_cliff as fh
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import random_unitary
        for dev in ("FakeAuckland", "FakeKingston"):
            bk = getattr(fake_provider, dev)()
            f = fh.graph_facts(bk.target)
            blocks = fh.layout_blocks("T", 0, f["nq"], f["matching"])
            rng = np.random.default_rng(86_100_000)
            qc = QuantumCircuit(sum(len(b) for b in blocks))
            for b in blocks:
                for x, y in ([(b[1], b[0])] if len(b) == 2 else [(b[1], b[0]), (b[2], b[1])]):
                    for _ in range(10):
                        qc.unitary(random_unitary(4, seed=int(rng.integers(2**31))).data, [x, y])
            o, occ = default(qc, bk), default(cc(qc), bk)
            print(f"  T spare 0 {dev:12s} {q2(o, bk):6d} | {q2(occ, bk):6d}  same circuit {B.sig_hash(o) == B.sig_hash(occ)}")

    # ------------------------------------------------------------------ F
    if "F" in a.part:
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        print("\n== F. FakeTorino's failed couplers: does Qiskit avoid them, and what does avoiding them cost Qiskit?")
        for tid in ("test_BV_100_transpile", "test_BVlike_simplification_transpile", "test_QAOA_100_transpile",
                    "test_square_heisenberg_100_transpile", "test_clifford_100_transpile", "test_circSU2_89_transpile"):
            qc, backend = build(tid)
            t = backend.target
            edges, qubits = pc._failed_elements(t, 0.5)
            pruned = pc.prune_coupling_map(backend.coupling_map, t, 0.5)
            basis = [g for g in backend.operation_names if g in ("cx", "cz", "ecr", "rz", "sx", "x", "id")]
            qk = generate_preset_pass_manager(2, backend).run(qc)
            qkp = generate_preset_pass_manager(2, coupling_map=pruned, basis_gates=basis, seed_transpiler=0).run(qc)
            o = default(qc, backend)
            basis_kw = dict(coupling_map=pruned, basis_gates=basis, entangling_basis="cx", layout_search=True,
                            seed_transpiler=0)
            with contextlib.redirect_stdout(io.StringIO()):
                cp = pc.compile_for_hardware(qc, **basis_kw)
                cp2 = pc.compile_for_hardware(qc, routing_optimization_level=2, **basis_kw)
            print(f"  {tid[:40]:40s} QK {q2(qk, backend):6d} (uses failed {pc._uses_failed(qk, edges, qubits)}) | "
                  f"on the pruned map: QK {q2(qkp, backend):6d}, C22 {q2(cp, backend):6d}, C22 routing 2 "
                  f"{q2(cp2, backend):6d} | C22 full map {q2(o, backend):6d}")

    # ------------------------------------------------------------------ G
    if "G" in a.part:
        from qiskit.transpiler import PassManager
        from qiskit.transpiler.passes import CommutativeCancellation, Unroll3qOrMore
        import skip_eval
        from qiskit_ibm_runtime import fake_provider
        print("\n== G. CommutativeCancellation on Hamiltonian inputs: two-qubit gates removed from the input, and the "
              "default call's count without | with it")
        t = fake_provider.FakeTorino()
        for n in (8, 12, 16, 24, 40):
            for k in range(2):
                qc = skip_eval.family_circuit("pauli", n, np.random.default_rng(86_200_000 + 10 * n + k))
                u = PassManager([Unroll3qOrMore()]).run(qc)
                c = PassManager([CommutativeCancellation()]).run(u)
                n2 = lambda x: sum(1 for i in x.data if len(i.qubits) == 2)
                print(f"  pauli{n:<3d} #{k}: input {n2(u):5d} -> {n2(c):5d} two-qubit gates; default call "
                      f"{q2(default(qc, t), t):6d} | {q2(default(c, t), t):6d}")

    # ------------------------------------------------------------------ C
    if "C" in a.part:
        print("\n== C. FakeTorino, 100-qubit tests: default against recommended call (C22)")
        stats = ("PRUNE_STATS", "COMPARE_STATS", "SKIP_STATS", "REFINE_STATS", "RESYNTH_STATS")
        for tid in ("test_BV_100_transpile", "test_BVlike_simplification_transpile", "test_QAOA_100_transpile",
                    "test_square_heisenberg_100_transpile", "test_clifford_100_transpile", "test_circSU2_89_transpile"):
            qc, backend = build(tid)
            t = backend.target
            edges, qubits = pc._failed_elements(t, 0.5)
            o = default(qc, backend)
            print(f"\n{tid}: FakeTorino reports {len(edges)} failed directed couplers and {len(qubits)} failed qubits;"
                  f" the default call ({q2(o, backend)} two-qubit gates) uses a failed element: "
                  f"{pc._uses_failed(o, edges, qubits)}")
            for name, kw in (("prune only (target, nothing else)", {}),
                             ("+ placement_refine", dict(placement_refine=True)),
                             ("recommended call", RECOMMENDED)):
                before = {s: dict(getattr(pc, s)) for s in stats if hasattr(pc, s)}
                t0 = time.perf_counter()
                out = default(qc, backend, target=t, **kw)
                moved = {s: {k: getattr(pc, s)[k] - v for k, v in d.items() if getattr(pc, s)[k] != v}
                         for s, d in before.items()}
                moved = {s: d for s, d in moved.items() if d}
                print(f"   {name:36s} q2 {q2(out, backend):6d}  uses failed {pc._uses_failed(out, edges, qubits)}  "
                      f"{time.perf_counter() - t0:6.1f} s  {moved}")


if __name__ == "__main__":
    main()
