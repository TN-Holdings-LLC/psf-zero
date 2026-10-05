"""c14_check.py -- workplace EXPLORATION (TARGET_ONLY=1: target without placement_refine, the C3 call) (not pre-registered): c13 against c14 on devices with one-way failed couplers.
HOLD6's F circuits (benchmarks/hold6_eval.family; seen at home, used here only to compare two candidates), every 5th
circuit of each family; FakeHanoiV2 and FakeGeneva; the base call (target + placement_refine) and the recommended call.
Records recompiles (PRUNE_STATS), compile time, identity of the outputs, failed-direction uses, exactness; and the noisy
state infidelity (restricted noise model, density matrix, final-layout qubits) for circuits whose outputs differ."""
import contextlib, importlib.util, io, json, os, sys, time, warnings
import numpy as np
warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, sys.argv[2])            # repo/benchmarks for hold6_eval
import depth_eval as DE, readout_eval as RE
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace, state_fidelity
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import fake_provider as fp


def load(p, n):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); sys.modules[n] = m
    s.loader.exec_module(m); return m


P13 = load(os.path.join(HERE, "..", "c13", "psf_compile.py"), "k_c13")
P14 = load(os.path.join(HERE, "psf_compile.py"), "k_c14")
import hold6_eval as H  # noqa: E402


def noisy_infid(qc, out, sim):
    fin = list(out.layout.final_index_layout(filter_ancillas=True))
    act = sorted({out.find_bit(b).index for i in out.data for b in i.qubits} | set(fin))
    idx = {p: k for k, p in enumerate(act)}
    red = QuantumCircuit(len(act))
    for i in out.data:
        if i.operation.name in ("barrier", "measure", "delay"):
            continue
        red.append(i.operation, [idx[out.find_bit(b).index] for b in i.qubits])
    red.save_density_matrix(qubits=[idx[p] for p in fin])
    rho = np.asarray(AerSimulator(method="density_matrix", noise_model=sim.reduced_model(act)).run(red).result()
                     .data()["density_matrix"])
    return float(1 - np.real(np.vdot(Statevector(qc).data, rho @ Statevector(qc).data)))


def main(dev):
    be = getattr(fp, dev)(); t = be.target
    basis = [g for g in t.operation_names if g in ("cx", "cz", "rz", "sx", "x")]
    base = dict(coupling_map=t.build_coupling_map(), basis_gates=basis, entangling_basis="cx", layout_search=True,
                seed_transpiler=0, target=t, placement_refine=True)
    full = dict(base, final_resynthesis="select", compare_level3=True, compare_floor=True, candidate_score="hybrid")
    if os.environ.get("TARGET_ONLY"):
        base = dict(base, placement_refine=False); full = base   # the call of Addendum 319's C3 arm
    edges, qubits = P14._failed_elements(t, 0.5)
    sim = DE.Noisy(be)
    rows = []
    for fam in H.FAMILIES:
        for j, (prm, qc) in enumerate(H.family(fam, False)):
            if j % 5:
                continue
            if qc.num_qubits > 8:
                continue
            row = dict(fam=fam, j=j, n=qc.num_qubits)
            for call, kw in (("base", base), ("full", full)):
                res = {}
                for tag, P in (("c13", P13), ("c14", P14)):
                    r0 = P.PRUNE_STATS["recompiled"]
                    t0 = time.perf_counter()
                    with contextlib.redirect_stdout(io.StringIO()):
                        out = P.compile_for_hardware(qc, **kw)
                    res[tag] = dict(out=out, s=time.perf_counter() - t0, rec=P.PRUNE_STATS["recompiled"] - r0)
                same = RE.sig(res["c13"]["out"]) == RE.sig(res["c14"]["out"])
                d = dict(same=same, rec13=res["c13"]["rec"], rec14=res["c14"]["rec"],
                         s13=round(res["c13"]["s"], 4), s14=round(res["c14"]["s"], 4),
                         failed14=P14._uses_failed(res["c14"]["out"], edges, qubits),
                         exact14=RE.state_infid(qc, res["c14"]["out"]))
                if not same:
                    d["inf13"] = noisy_infid(qc, res["c13"]["out"], sim)
                    d["inf14"] = noisy_infid(qc, res["c14"]["out"], sim)
                row[call] = d
            rows.append(row)
    json.dump(dict(device=dev, rows=rows), open(os.path.join(HERE, f"c14check_{dev}{os.environ.get('TARGET_ONLY', '') and '_targetonly'}.json"), "w"))
    for call in ("base", "full"):
        rr = [r[call] for r in rows]
        diff = [r for r in rr if not r["same"]]
        print(dev, call, "circuits", len(rr), "recompiles c13", sum(r["rec13"] for r in rr), "c14", sum(r["rec14"] for r in rr),
              "| outputs differ", len(diff), "| failed-direction uses c14", sum(r["failed14"] for r in rr),
              "| max state infid c14 %.1e" % max(r["exact14"] for r in rr),
              "| time c13 %.2fs c14 %.2fs" % (sum(r["s13"] for r in rr), sum(r["s14"] for r in rr)),
              ("| on differing: mean noisy infid c13 %.4f c14 %.4f, c14 better in %d" %
               (np.mean([r["inf13"] for r in diff]), np.mean([r["inf14"] for r in diff]),
                sum(r["inf14"] < r["inf13"] for r in diff))) if diff else "", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
