"""c41_exact_sweep.py -- the exactness check of item 73 (prototype 2026-10-11.c41) on real circuits: every Benchpress
QASM file of at most 7 qubits that item 73 reads (not item 72) is compiled by STEINER6's construction on a 7-qubit test
device with differing errors, and the output's distribution over its classical bits is compared with the input's for
6 random product input states each (tolerance 1e-6). A circuit without measurements gets measure_all(). The
distribution, not the unitary, is the test: Qiskit's level 3 removes diagonal gates just before measurements, which
keeps the distribution only; item 73's outputs equal the inputs up to a global phase.

    cd <psf-zero repository>
    python benchmarks/c41_exact_sweep.py --bp <benchpress> [--file patches/psf_compile_c41_2026-10-11/psf_compile.py]
"""
import argparse
import glob
import importlib.util
import os
import random
import sys
import warnings

import numpy as np

REPO = os.path.abspath(os.getcwd())
sys.path[:0] = [os.path.join(REPO, "benchmarks"), REPO]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def device():
    from qiskit.providers.fake_provider import GenericBackendV2
    from qiskit.transpiler import CouplingMap, InstructionProperties
    cm = CouplingMap([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [2, 6]])
    cm.make_symmetric()
    b = GenericBackendV2(7, basis_gates=["cz", "rz", "sx", "x"], coupling_map=cm, seed=72)
    rng = random.Random(72)
    for qa in b.target["cz"]:
        b.target.update_instruction_properties("cz", qa, InstructionProperties(error=rng.uniform(2e-3, 2e-2)))
    return b


def distribution(circ, prep, nclb):
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    body = QuantumCircuit(circ.num_qubits)
    for g, q in prep:
        body.append(g, [q])
    meas = {}
    for i in circ.data:
        if i.operation.name == "measure":
            meas[circ.find_bit(i.clbits[0]).index] = circ.find_bit(i.qubits[0]).index
        elif i.operation.name != "barrier":
            body.append(i.operation, [body.qubits[circ.find_bit(q).index] for q in i.qubits])
    p = Statevector(body).probabilities()
    out = np.zeros(2 ** nclb)
    for idx in np.nonzero(p > 1e-14)[0]:
        c = 0
        for cb, q in meas.items():
            if (idx >> q) & 1:
                c |= 1 << cb
        out[c] += p[idx]
    return out


def same_distribution(out, qc, trials=6, seed=0):
    from qiskit.circuit.library import UGate
    rng = random.Random(seed)
    init = out.layout.initial_index_layout()
    worst = 0.0
    for _ in range(trials):
        gates = [UGate(rng.uniform(0, 3.2), rng.uniform(0, 6.3), rng.uniform(0, 6.3)) for _ in range(qc.num_qubits)]
        a = distribution(qc, [(g, q) for q, g in enumerate(gates)], qc.num_clbits)
        b = distribution(out, [(g, init[q]) for q, g in enumerate(gates)], qc.num_clbits)
        worst = max(worst, float(np.max(np.abs(a - b))))
    return worst


def main():
    warnings.simplefilter("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--bp", required=True)
    ap.add_argument("--file", default=os.path.join("patches", "psf_compile_c41_2026-10-11", "psf_compile.py"))
    a = ap.parse_args()
    from qiskit import QuantumCircuit
    load(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
    m = load(os.path.join(REPO, a.file), "psf_c41_sweep")
    b = device()
    root = os.path.join(a.bp, "benchpress", "qasm")
    same = differ = 0
    for f in sorted(glob.glob(os.path.join(root, "**", "*.qasm"), recursive=True)):
        try:
            qc = QuantumCircuit.from_qasm_file(f)
        except Exception:  # noqa: BLE001 - not loadable
            continue
        if qc.num_qubits > 7 or m._st_parse(qc) is not None or m._st_parse_gates(qc) is None:
            continue
        if not any(i.operation.name == "measure" for i in qc.data):
            qc.measure_all()
        st = m._st_compile(qc, b.target, 0.5, 0, 300_000, 2_500)
        worst = same_distribution(st, qc)
        ok = worst <= 1e-6
        same += ok
        differ += not ok
        print(f"{os.path.relpath(f, root).replace(os.sep, '/')} {qc.num_qubits} {'same' if ok else 'DIFFERENT'} "
              f"(max difference {worst:.1e})", flush=True)
    print(f"same {same}, different {differ}")


if __name__ == "__main__":
    main()
