"""Exploratory (not pre-registered), rerun for the record: the release psf_smart_layout,
the candidate with only the corrected feasibility check (Stage 0b switched off), and the
full candidate, on the 27-qubit FakeAuckland, family T at spare 0 and 2."""
import sys
import warnings
warnings.simplefilter("ignore")
sys.path[:0] = [sys.argv[1], sys.argv[2]]  # repo, benchmarks
import full_heavyhex_cliff as fh
from qiskit_ibm_runtime.fake_provider import FakeAuckland
t = FakeAuckland().target
cmap = t.build_coupling_map()
f = fh.graph_facts(t)
for spare in (0, 2):
    qc, blocks = fh.build("T", spare, 0, f["nq"], f["matching"])
    pairs = sorted({tuple(sorted((qc.find_bit(i.qubits[0]).index, qc.find_bit(i.qubits[1]).index)))
                    for i in qc.data if len(i.qubits) == 2})
    for label, d, shortcut in (("release", sys.argv[2], None), ("check only", sys.argv[3], False),
                               ("candidate", sys.argv[3], True)):
        sys.path.insert(0, d)
        sys.modules.pop("psf_smart_layout", None)
        import psf_smart_layout as sl
        if shortcut is not None:
            sl.USE_PATH_SHORTCUT = shortcut
        lm, info = sl.smart_vf2_layout(cmap, pairs, qc.num_qubits, per_attempt_call_limit=50_000,
                                       time_budget_s=2.0, fallback_call_limit=2_000_000)
        print(f"spare {spare} {label:10s} {sl.LAYOUT_VERSION}: found {lm is not None}, phase {info['phase']}, "
              f"{info['order_name']}, orderings tried {info['orderings_tried']}, {info['elapsed_s']:.3f} s")
        sys.path.remove(d)
