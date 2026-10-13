"""Offline tests of t2zz_real.py (no IBM connection): python -m pytest -q test_t2zz.py"""
import math
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2zz_real as T  # noqa: E402

warnings.simplefilter("ignore")


def _fez():
    from qiskit_ibm_runtime.fake_provider import FakeFez
    return FakeFez()


def test_pairs_follow_rule_and_are_far_apart():
    b = _fez()
    info, adj = T.device_qubits(b)
    pairs = T.select_pairs(info, adj)
    assert [p["label"] for p in pairs] == ["a", "b", "c"]
    for p in pairs:
        assert p["spectator"] in adj[p["target"]]
        assert p["spectator"] == max(adj[p["target"]], key=lambda v: (info[v]["t1"], -v))
    assert pairs[0]["t2_t"] >= T.MIN_T2_A
    assert pairs[2]["t2_t"] == max(info[q]["t2"] for q in info if info[q]["ok"] and adj[q])
    qs = [(p["target"], p["spectator"]) for p in pairs]
    for i in range(3):
        for j in range(i + 1, 3):
            for u in qs[i]:
                d = T._dist_from(adj, u)
                assert all(d.get(v, 99) >= T.MIN_PAIR_DIST for v in qs[j])


def test_circuits_are_isa_and_counted():
    b = _fez()
    info, adj = T.device_qubits(b)
    pairs = T.select_pairs(info, adj)
    circs, pubs = T.build_all(b, pairs)
    assert len(circs) == len(T.SEQS) * len(T.TAUS_US) * 2 + 2 == 50
    T.check_circuits(circs, b)
    for p in pubs:
        if p["kind"] == "seq":
            assert abs(p["tau_prog_s"] - p["tau_us"] * 1e-6) < 2 * b.target.dt * max(1, b.target.granularity)


def test_noiseless_expectations_and_bit_order():
    """Noiseless: Ramsey X = +1, Ramsey Y = 0, spectator reads its prepared state, for every pair."""
    from qiskit_aer import AerSimulator
    b = _fez()
    info, adj = T.device_qubits(b)
    pairs = T.select_pairs(info, adj)
    circs, pubs = T.build_all(b, pairs)
    sim = AerSimulator()
    man = dict(pairs=pairs, pubs=pubs)
    rows = []
    for i, (c, p) in enumerate(zip(circs, pubs)):
        if p["kind"] == "seq" and p["tau_us"] in (0, 20):
            rows.append(dict(pub=i, counts=sim.run(c, shots=4000, seed_simulator=i).result().get_counts()))
    ex = T.target_expectations(man, dict(rows=rows))
    for (k, seq, tau, s), (val, n, tot) in ex.items():
        assert n == tot                      # spectator always read in its prepared state
        if seq in ("RX", "EX"):
            assert val > 0.99
        else:
            assert abs(val) < 0.06


def test_zz_fit_recovers_exact_rate():
    rows = []
    zeta = 2500.0
    for s in (0, 1):
        for tau in (0, 5e-6, 10e-6, 20e-6, 30e-6, 40e-6):
            c = math.exp(-tau / 50e-6)
            ph = 0.7 + 2 * math.pi * zeta * tau * s
            rows.append(dict(spec=s, tau=tau, x=c * math.cos(ph), y=c * math.sin(ph), nx=2000, ny=2000))
    z, se, ex = T.zz_fit(rows, 40e-6)
    assert abs(z - zeta) < 1e-6 and 0 < se < 300
    zi, sei, _ = T.zz_fit(rows, 40e-6, mode="intercept")
    assert abs(zi - zeta) < 1e-6 and sei > se
