import sys, os, json, time, subprocess, tempfile
sys.path.insert(0, "rust/psfz/harness")
import numpy as np
import twin_check as tc
from qiskit import qasm3
from qiskit_ibm_runtime import fake_provider as fp
pc = tc.load_pc()
rng = np.random.default_rng(20261020)
BIN = sys.argv[1]
with tempfile.TemporaryDirectory() as tmp:
    d = os.path.join(tmp, "F.json"); open(d, "w").write(pc._psfz105_device(fp.FakeFez().target))
    for n, s in [(16, 6), (18, 7), (18, 8), (20, 9)]:
        f = os.path.join(tmp, "in.qasm"); open(f, "w").write(qasm3.dumps(tc.tfim(n, s, rng)))
        rep = os.path.join(tmp, "r.json")
        subprocess.run([BIN, "twin", d, f, rep, "65536"], capture_output=True, timeout=1500)
        j = json.load(open(rep))
        print(n, s, "chi", j.get("max_chi"), "bound 2^%d" % max(j.get("bound_log2") or [0]), "sim %.2f s" % j.get("simulate_s"), "2q", j.get("two_site_gates"), flush=True)
