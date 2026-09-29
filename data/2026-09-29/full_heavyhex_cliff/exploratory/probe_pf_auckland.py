"""Exploratory (not pre-registered): the release and the patched psf_smart_layout on the
27-qubit FakeAuckland, family T at spare 0 (the harness dry-run device only)."""
import importlib, sys, time, warnings
warnings.simplefilter("ignore")
sys.path[:0] = [sys.argv[1], sys.argv[2]]  # repo, benchmarks
import full_heavyhex_cliff as fh
from qiskit_ibm_runtime.fake_provider import FakeAuckland
t = FakeAuckland().target; cmap = t.build_coupling_map()
f = fh.graph_facts(t)
for spare in (0, 2):
    qc, blocks = fh.build("T", spare, 0, f["nq"], f["matching"])
    pairs = sorted({tuple(sorted((qc.find_bit(i.qubits[0]).index, qc.find_bit(i.qubits[1]).index))) for i in qc.data if len(i.qubits) == 2})
    for label, d in (("release", sys.argv[2]), ("patched", sys.argv[3])):
        sys.path.insert(0, d)
        for m in [k for k in sys.modules if k == "psf_smart_layout"]:
            del sys.modules[m]
        import psf_smart_layout as sl
        lm, info = sl.smart_vf2_layout(cmap, pairs, qc.num_qubits)
        print(spare, label, sl.__file__.split("/")[-2], "found", lm is not None, "phase", info["phase"], info["order_name"],
              "tried", info["orderings_tried"], f"{info['elapsed_s']:.3f}s")
        sys.path.remove(d)
