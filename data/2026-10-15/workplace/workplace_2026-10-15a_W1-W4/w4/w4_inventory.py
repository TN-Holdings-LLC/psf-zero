"""W4: which in-scope tests carry symmetric two-qubit instructions, and in which written orientation."""
import sys, os, tempfile, collections, importlib.util
REPO = os.getcwd()
sys.argv = ["c57_val.py", tempfile.mkdtemp() + "/x"]
spec = importlib.util.spec_from_file_location("c57v", os.path.join(REPO, "benchmarks", "c57_val.py"))
V = importlib.util.module_from_spec(spec); sys.modules["c57v"] = V; spec.loader.exec_module(V)
for name in V.IN_SCOPE + V.ODD:
    qc = V.build(name)
    cnt = collections.Counter()
    for ins in qc.data:
        if len(ins.qubits) == 2:
            a, b = (qc.find_bit(q).index for q in ins.qubits)
            cnt[(ins.operation.name, "lo-hi" if a < b else "hi-lo")] += 1
        elif len(ins.qubits) > 2:
            cnt[(ins.operation.name, f"{len(ins.qubits)}q")] += 1
    print(f"{name:16s}", dict(cnt))
