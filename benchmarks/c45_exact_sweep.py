"""c45_exact_sweep.py -- the exactness check of prototype 2026-10-12.c45's full call (item 80 can return Qiskit level 3's
circuit, which item 39's check cannot verify when the input is measured): every Benchpress QASM file of at most 7
qubits is compiled by `compile_for_hardware(qc, backend=<7-qubit test device>, entangling_basis="cx",
layout_search=True, seed_transpiler=0)`, and the output's distribution over its classical bits is compared with the
input's for 6 random product input states (tolerance 1e-6), as in c41_exact_sweep.py. Reports which outputs are level
3's. A circuit without measurements gets measure_all().

    cd <psf-zero repository>
    python benchmarks/c45_exact_sweep.py --bp <benchpress> [--file patches/psf_compile_c45_2026-10-12/psf_compile.py]
"""
import argparse
import contextlib
import glob
import io
import os
import sys
import warnings

REPO = os.path.abspath(os.getcwd())
sys.path[:0] = [os.path.join(REPO, "benchmarks"), REPO]
from c41_exact_sweep import device, load, same_distribution  # noqa: E402


def main():
    warnings.simplefilter("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--bp", required=True)
    ap.add_argument("--file", default=os.path.join("patches", "psf_compile_c45_2026-10-12", "psf_compile.py"))
    a = ap.parse_args()
    from qiskit import QuantumCircuit
    load(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
    m = load(os.path.join(REPO, a.file), "psf_c45_sweep")
    b = device()
    root = os.path.join(a.bp, "benchpress", "qasm")
    same = differ = from_l3 = 0
    for f in sorted(glob.glob(os.path.join(root, "**", "*.qasm"), recursive=True)):
        try:
            qc = QuantumCircuit.from_qasm_file(f)
        except Exception:  # noqa: BLE001 - not loadable
            continue
        if qc.num_qubits > 7 or "transpiled" in f:
            continue
        if any(i.clbits and i.operation.name != "measure" for i in qc.data):  # classical control: not comparable here
            print(f"{os.path.relpath(f, root).replace(os.sep, '/')} {qc.num_qubits} skipped (classically controlled)")
            continue
        if not any(i.operation.name == "measure" for i in qc.data):
            qc.measure_all()
        before = m.L3C_STATS["kept_level3"]
        with contextlib.redirect_stdout(io.StringIO()):
            out = m.compile_for_hardware(qc, backend=b, entangling_basis="cx", layout_search=True, seed_transpiler=0)
        l3 = m.L3C_STATS["kept_level3"] > before
        worst = same_distribution(out, qc)
        ok = worst <= 1e-6
        same += ok
        differ += not ok
        from_l3 += l3
        print(f"{os.path.relpath(f, root).replace(os.sep, '/')} {qc.num_qubits} {'same' if ok else 'DIFFERENT'} "
              f"(max difference {worst:.1e}){' level3' if l3 else ''}", flush=True)
    print(f"same {same}, different {differ}; level 3's output {from_l3}")


if __name__ == "__main__":
    main()
