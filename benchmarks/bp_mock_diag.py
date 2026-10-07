"""bp_mock_diag.py -- exploratory, after BP-MOCK's scored run (2026-10-07). Looks into the two REFUTED verdicts.

M1: for every test without a wide instruction whose C22 circuit differed from REL's, compile it again, one call at a
time (no other job running): REL, C22, REL, C22 with the default layout-search time budget (2 s), then REL and C22 with
a budget of 60 s. If REL differs from itself, the difference is not item 48's.

M5: for every test where some arm's output was reported not equivalent, report the input's and the output's qubit
counts and whether the input has a PauliEvolutionGate, then check every arm's output again in two ways that allow for
ancilla qubits and for a product-formula synthesis: (a) c22's item 39 check `_implements` (two random product states,
layout and ancillas handled) against the input expanded through its definitions; (b) the workplace probe's state check
from |0...0> (readout_eval.state_infid) against the same expanded input.

    cd <repo> && python <this file> --bp <benchpress clone> --run data/2026-10-07/bp_mock
"""
import argparse
import contextlib
import io
import json
import os
import sys
import warnings

warnings.simplefilter("ignore")
REPO = os.getcwd()
HERE = os.path.join(REPO, "benchmarks")
WORK = os.path.join(REPO, "data", "2026-10-05", "workplace")
sys.path[:0] = [os.path.join(WORK, "depth1"), HERE, REPO]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bp", required=True)
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    import bp_mock as B
    import core_fix_c2_eval as H
    from qiskit import transpile
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    H.load_module(os.path.join(HERE, "psf_smart_layout.py"), "psf_smart_layout")
    mods = {"REL": H.load_module(B.REL_PATH, "psf_compile_rel_diag"), "C22": H.load_module(B.C22_PATH, "psf_compile")}
    RE = H.load_module(os.path.join(WORK, "readout", "readout_eval.py"), "readout_eval_diag")
    r = json.load(open(os.path.join(a.run, "bp_mock.json")))
    by = {}
    for x in r["rows"]:
        by.setdefault(x["test"], {})[x["arm"]] = x
    spec = {tid: (s, kind, arg) for s, tid, kind, arg in B.sample(a.bp)}

    def build(t):
        s, kind, arg = spec[t]
        return B.build(a.bp, kind, arg)

    def call(name, qc, backend, budget=2.0):
        basis = [g for g in backend.operation_names if g in ("cx", "cz", "ecr", "rz", "sx", "x", "id")]
        with contextlib.redirect_stdout(io.StringIO()):
            return mods[name].compile_for_hardware(qc, coupling_map=backend.coupling_map, basis_gates=basis,
                                                   entangling_basis="cx", layout_search=True, seed_transpiler=0,
                                                   layout_search_time_budget_s=budget)

    print("== M1: inputs without wide instructions whose C22 circuit differed from REL's")
    m1 = [t for t, x in by.items() if x["C22"]["wide"] == 0 and x["C22"]["sig"] != x["REL"]["sig"]]
    for t in m1:
        qc, backend = build(t)
        print(f"\n{t}: in the run REL q2 {by[t]['REL']['q2']} d2 {by[t]['REL']['d2']}, C22 q2 {by[t]['C22']['q2']} "
              f"d2 {by[t]['C22']['d2']}; input {qc.num_qubits} qubits, backend {backend.num_qubits}")
        outs = [(n, call(n, qc, backend)) for n in ("REL", "C22", "REL", "C22")]
        sigs = [B.sig_hash(o) for _, o in outs]
        for (n, o), s in zip(outs, sigs):
            print(f"   budget 2 s  {n}: q2 {o.count_ops().get(backend.two_q_gate_type, 0)}  sig {s[:12]}")
        print(f"   REL = REL {sigs[0] == sigs[2]}, C22 = C22 {sigs[1] == sigs[3]}, REL = C22 {sigs[0] == sigs[1]}, "
              f"run's REL reproduced {by[t]['REL']['sig'] in sigs}, run's C22 reproduced {by[t]['C22']['sig'] in sigs}")
        o60 = [call(n, qc, backend, 60.0) for n in ("REL", "C22")]
        print(f"   budget 60 s: REL = C22 {B.sig_hash(o60[0]) == B.sig_hash(o60[1])}, q2 "
              f"{[o.count_ops().get(backend.two_q_gate_type, 0) for o in o60]}")

    print("\n== M5: tests with an output reported not equivalent")
    m5 = [t for t, x in by.items() if any(x[k].get("equivalent") is False for k in ("QK", "REL", "C22"))]
    for t in m5:
        qc, backend = build(t)
        bare = qc.copy()
        bare.remove_final_measurements(inplace=True)
        ref = transpile(bare, basis_gates=["u", "cx"], optimization_level=0)  # definitions, incl. product formulas
        pe = any(i.operation.name == "PauliEvolution" for i in qc.data)
        print(f"\n{t}: input {qc.num_qubits} qubits, backend {backend.num_qubits}, PauliEvolution {pe}; run: "
              + ", ".join(f"{k} {by[t][k].get('equivalent')}" for k in ("QK", "REL", "C22")))
        outs = {"QK (again, unseeded)": generate_preset_pass_manager(2, backend).run(qc),
                "REL": call("REL", qc, backend), "C22": call("C22", qc, backend)}
        for k, o in outs.items():
            try:
                imp = mods["C22"]._implements(ref, o)
            except Exception as exc:  # noqa: BLE001
                imp = f"n/a {type(exc).__name__}"
            try:
                inf = RE.state_infid(ref, RE.strip_measure(o))
            except Exception as exc:  # noqa: BLE001
                inf = f"n/a {type(exc).__name__}"
            print(f"   {k:22s} output {o.num_qubits} qubits: implements {imp}; state infidelity from |0> "
                  f"{inf if isinstance(inf, str) else f'{inf:.1e}'}")


if __name__ == "__main__":
    main()
