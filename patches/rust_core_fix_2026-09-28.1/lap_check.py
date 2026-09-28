import sys, numpy as np
sys.path[:0] = [sys.argv[1]]
import psf_compile as pc, psf_zero_core as core
d = np.load("blocks_base3000.npz"); U = d["U"]; tags = d["tags"]
print("CORE_VERSION", getattr(core, "CORE_VERSION", None))
worst = []
for i, u in enumerate(U):
    try:
        c, k1, k2, ph = core.geometric_decompose(u.real.tolist(), u.imag.tolist())
        v = pc._reconstruct(c, k1, k2, ph); z = np.vdot(v, u); z /= abs(z)
        e = np.linalg.norm(u - z * v)
    except Exception as ex:
        e = -1.0
    worst.append(e)
w = np.array(worst)
for lap in (2138,):
    sl = slice((lap - 1) * 11, lap * 11)
    print("lap", lap, "block errors:", ["fail" if x < 0 else "%.1e" % x for x in w[sl]])
o = np.argsort(-w)[:5]
print("top 5 over 3,000 laps:", [(int(i // 11) + 1, int(i % 11), "%.1e" % w[i]) for i in o])
