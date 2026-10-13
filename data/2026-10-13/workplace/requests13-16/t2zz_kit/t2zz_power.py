"""t2zz_power.py -- shot-noise power of the T2*/ZZ measurement's ZZ estimate (E3), for the design choices in v2.

Monte Carlo of the fixed analysis (t2zz_real.zz_fit): Ramsey X and Y counts are drawn binomially for a target whose
Ramsey coherence decays as exp(-tau/T2*) and whose phase rotates by 2 pi zeta tau more when the spectator is in |1>
(and a common offset that cancels);
readout errors shrink the contrast; shots in which the spectator is not read in its prepared state are dropped
(spectator T1 decay during the wait and readout error). Prints the standard error of zeta and the probabilities that
E3 is confirmed (|zeta| >= 1 kHz and >= 3 SE) or refuted on one pair (|zeta| < 0.5 kHz).

    python t2zz_power.py            (about a minute)"""
import math
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t2zz_real import zz_fit  # noqa: E402

P_RO = 0.015      # P(1|0) = P(0|1) per qubit (Heron median readout error is about 1-2%)
T1_SPEC = 150e-6  # spectator T1


def one(rng, taus, shots, t2s, zeta, fit_max):
    rows = []
    for s in (0, 1):
        for tau in taus:
            keep = (1 - P_RO) * (math.exp(-tau / T1_SPEC) if s else 1.0)
            n = rng.binomial(shots, keep)
            c = math.exp(-tau / t2s) * (1 - 2 * P_RO)
            ph = 2 * math.pi * zeta * tau * s + 0.3          # 0.3 rad: a common frequency offset, cancels
            ex, ey = c * math.cos(ph), c * math.sin(ph)
            x = 2 * rng.binomial(n, (1 + ex) / 2) / n - 1
            y = 2 * rng.binomial(n, (1 + ey) / 2) / n - 1
            rows.append(dict(spec=s, tau=tau, x=x, y=y, nx=n, ny=n))
    return zz_fit(rows, fit_max)[:2]


def main():
    rng = np.random.default_rng(20261013)
    T5 = (0, 5e-6, 10e-6, 20e-6, 40e-6)
    T6 = (0, 5e-6, 10e-6, 20e-6, 30e-6, 40e-6)
    designs = [   # all with the v2 estimator (weighted, through the origin); v1 also used fit <= 20 us
        ("v1 grid: 1,000 shots, tau 0/5/10/20/40, fit <= 20 us", T5, 1000, 20e-6),
        ("A: 2,000 shots, v1 grid, fit <= 40 us (rough 84 s)", T5, 2000, 40e-6),
        ("B: 1,000 shots, + tau 30, fit <= 40 us (rough 75 s)", T6, 1000, 40e-6),
        ("v2 = C: 2,000 shots, + tau 30, fit <= 40 us (rough 100 s)", T6, 2000, 40e-6),
    ]
    reps = 1500
    print("| design | T2* (us) | SE of zeta (kHz), zeta = 0 | P(E3 confirmed on a pair), zeta = 1.5 kHz |"
          " P(E3 confirmed), zeta = 1.0 kHz | P(\\|zeta\\| < 0.5 kHz), zeta = 0 |")
    print("|---|---|---|---|---|---|")
    for name, taus, shots, fm in designs:
        for t2s in (15e-6, 30e-6, 60e-6, 120e-6):
            z0 = np.array([one(rng, taus, shots, t2s, 0.0, fm)[0] for _ in range(reps)])
            res15 = [one(rng, taus, shots, t2s, 1.5e3, fm) for _ in range(reps)]
            res10 = [one(rng, taus, shots, t2s, 1.0e3, fm) for _ in range(reps)]
            conf = lambda rs: np.mean([abs(z) >= 1e3 and abs(z) >= 3 * se for z, se in rs])  # noqa: E731
            print(f"| {name} | {t2s*1e6:.0f} | {z0.std()/1e3:.2f} | {conf(res15):.2f} | {conf(res10):.2f} | "
                  f"{np.mean(np.abs(z0) < 500):.2f} |")


if __name__ == "__main__":
    main()
