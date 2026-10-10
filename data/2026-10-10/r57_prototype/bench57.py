import os, sys, time, tempfile, subprocess
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ref57 as R
rng = np.random.default_rng(1)
for k, n in ((10, 2000), (14, 2000)):
    qp = [R.Q(rng.uniform(5e-5, 3e-4), rng.uniform(5e-5, 3e-4)) for _ in range(k)]
    ops = []
    for j in range(n):
        if j % 3 == 2:
            q = [int(x) for x in rng.choice(k, 2, replace=False)]; mat = R.rand_unitary(rng, 4)
        else:
            q = [int(rng.integers(k))]; mat = R.rand_unitary(rng, 2)
        ops.append((mat, q, R.P(1e-3, 6e-8)))
    t0 = time.perf_counter(); ref = R.hybrid_ref(ops, qp, k); tp = time.perf_counter() - t0
    path = os.path.join(tempfile.mkdtemp(), "b"); open(path, "wb").write(R.encode(1, ops, qp, k))
    t0 = time.perf_counter(); out = subprocess.run(["./r57", path], capture_output=True, text=True).stdout; tr = time.perf_counter() - t0
    n2 = sum(len(q) == 2 for _, q, _ in ops)
    print(f"k={k}, {n} ops ({n2} two-qubit): NumPy reference {tp:.2f} s, Rust (process incl.) {tr:.3f} s, "
          f"ratio {tp / tr:.0f}x; values {ref:.15e} / {float(out.split()[1]):.15e}")
