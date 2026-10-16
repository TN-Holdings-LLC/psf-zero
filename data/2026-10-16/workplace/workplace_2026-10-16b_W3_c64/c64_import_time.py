"""Import time of c63a and c64 (reported, not a criterion): each load in a fresh process, 5 times per version,
alternating; median. 'own' = loading the module file after numpy and qiskit are imported; 'total' = the whole import
in a fresh interpreter.   python c64_import_time.py <c63a> <c64>"""
import json
import statistics
import subprocess
import sys

CODE = r'''
import sys, time, importlib.util
t0 = time.perf_counter()
import numpy, qiskit, qiskit.transpiler, qiskit.synthesis, qiskit.quantum_info  # noqa: E401,F401
t1 = time.perf_counter()
spec = importlib.util.spec_from_file_location("m", sys.argv[1])
m = importlib.util.module_from_spec(spec)
sys.modules["m"] = m  # dataclasses look the module up there
spec.loader.exec_module(m)
t2 = time.perf_counter()
print(t2 - t1, t2 - t0)
'''
res = {"c63a": [], "c64": []}
for _ in range(5):
    for name, path in (("c63a", sys.argv[1]), ("c64", sys.argv[2])):
        out = subprocess.run([sys.executable, "-c", CODE, path], capture_output=True, text=True, check=True).stdout
        own, total = map(float, out.split()[-2:])
        res[name].append((own, total))
summary = {k: dict(own_ms=statistics.median(o for o, _ in v) * 1000, total_ms=statistics.median(t for _, t in v) * 1000)
           for k, v in res.items()}
print(json.dumps(summary, indent=1))
