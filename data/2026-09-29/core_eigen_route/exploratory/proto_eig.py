"""proto_eig.py -- exploratory: for the 15 rejected blocks, (1) the SVD route's
left/right mismatch inside near-tied singular pairs, (2) a numpy prototype of the
eigen route: O2 from the real symmetric pencil of M = u_m^T u_m, O1 from u_m O2^T,
scored by the off-diagonal norm of D = O1^T u_m O2^T."""
import numpy as np
Q = np.array([[1, 1j, 0, 0], [0, 0, 1j, 1], [0, 0, 1j, -1], [1, -1j, 0, 0]]) / np.sqrt(2)
d = np.load("blocks15.npz")

def offd(D):
    return float(np.sqrt(np.sum(np.abs(D - np.diag(np.diag(D))) ** 2)))

def eig_route(um):
    M = um.T @ um
    best = None
    for phi in (0.6, 1.1, 0.25, 1.45, np.pi / 4, np.pi / 2, 0.0, 2.3):
        A = np.cos(phi) * M.real + np.sin(phi) * M.imag
        w, V = np.linalg.eigh(A)
        O2 = V.T                      # M = O2^T D^2 O2
        X = um @ O2.T                 # = O1 D
        O1 = np.empty((4, 4))
        for k in range(4):
            x = X[:, k]
            ph = np.sqrt(x @ x)       # e^{i theta_k} up to sign
            O1[:, k] = (x / ph).real
        D = O1.T @ um @ O2.T
        s = offd(D)
        orth = np.linalg.norm(O1.T @ O1 - np.eye(4))
        if best is None or s < best[0]:
            best = (s, phi, orth, np.min(np.diff(np.sort(w))))
    return best

for u, lap in zip(d["mats"], d["laps"]):
    un = u / np.linalg.det(u) ** 0.25
    um = Q.conj().T @ un @ Q
    U, s, Vt = np.linalg.svd(um.real)
    Dsvd = U.T @ um @ Vt.T
    e = eig_route(um)
    print(f"lap {lap:5d}: numpy-SVD offdiag {offd(Dsvd):.2e} | eigen route offdiag {e[0]:.2e} (phi {e[1]:.2f}, "
          f"O1 orthogonality {e[2]:.1e}, min eigen gap {e[3]:.2e})")
