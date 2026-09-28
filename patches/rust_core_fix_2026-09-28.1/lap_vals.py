import sys, numpy as np
sys.path[:0] = [sys.argv[1]]
import psf_compile as pc, psf_zero_core as core
U = np.load("blocks_base3000.npz")["U"]
for lap, b in [(716,4),(790,4),(2480,4),(1274,8),(2922,4),(1609,6),(203,4),(2084,4)]:
    u = U[(lap-1)*11+b]
    try:
        c,k1,k2,ph = core.geometric_decompose(u.real.tolist(), u.imag.tolist()); v = pc._reconstruct(c,k1,k2,ph)
        z = np.vdot(v,u); z/=abs(z); print(lap, b, "%.1e" % np.linalg.norm(u-z*v))
    except Exception as e: print(lap, b, type(e).__name__)
