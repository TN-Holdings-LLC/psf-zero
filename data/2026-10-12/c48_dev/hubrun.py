"""c48 development: STEINER6 with hub holders (hub.py) against STEINER6 without, built the same way (the mover's ops,
Qiskit level 3 without routing; no re-placement), with the block reordering on both."""
# Repository form of a development script: workspace paths replaced by repository-relative ones (run from the
# repository root; Benchpress's QASM files under $BENCHPRESS_QASM).

import sys, time, math
import ev48 as E
import hub
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import fake_provider

M, H, S = E.M, E.H, E.S
E.RULE["rule"] = "start"


def construct(qc, target, n_extra_fn):
    ev = M._st_parse(qc)
    ev = ev + ([], None) if ev is not None else M._st_parse_gates(qc)
    prefix, meas, triples, extra_phase, suffix, perm = ev
    perm = perm or list(range(qc.num_qubits))
    triples = M._st_merge(triples)
    tr2, n2, desc = n_extra_fn(triples, qc.num_qubits)
    got = M._st_construct(n2, tr2, target, meas)
    m, pi0 = got
    N = target.num_qubits
    phys = QuantumCircuit(N, qc.num_clbits)
    for op, q in prefix:
        phys.append(op, [pi0[q]])
    for name, q, prm in m.ops:
        if name == "cx":
            phys.cx(*q)
        elif name == "swap":
            phys.swap(*q)
        elif name == "rz":
            phys.rz(prm[0], q[0])
        else:
            getattr(phys, name)(q[0])
    for op, q in suffix:
        phys.append(op, [m.pi[q]])
    for q, cb in meas:
        phys.measure(m.pi[perm[q]], cb)
    out = transpile(phys, target=target, initial_layout=list(range(N)), routing_method="none", optimization_level=3,
                    seed_transpiler=0)
    return out, desc, m


if __name__ == "__main__":
    tests, dev = sys.argv[1].split(","), sys.argv[2]
    confs = [tuple(int(x) for x in c.split("x")) for c in sys.argv[3].split(",")]  # hubs x k, e.g. 1x2,2x2
    b = getattr(fake_provider, dev)()
    t = b.target
    for name in tests:
        qc = H.build_test(name)[0] if name in H.TESTS else E.gens.build(name)
        p0, _ = H.measured_distribution(H._unrolled(qc))
        base = None
        for hubs, k in [(0, 0)] + confs:
            t0 = time.time()
            fn = (lambda tr, n: (tr, n, "none")) if hubs == 0 else (lambda tr, n, hubs=hubs, k=k: hub.transform(tr, n, hubs, k))
            out, desc, m = construct(qc, t, fn)
            met = E.metrics(out, t)
            p1, _ = H.measured_distribution(out) if qc.num_qubits + hubs * k <= 24 else ({}, 0)
            ex = max(abs(p0.get(x, 0) - p1.get(x, 0)) for x in set(p0) | set(p1)) if p1 else float("nan")
            if base is None:
                base = met
            print(f"{name:12s} {dev[4:]:8s} hubs {hubs} k {k}: 2q {met[0]:4d} ESP {met[1]:.3f} dur {met[2]:6.2f} us "
                  f"idle-est {met[3]:.4f} | vs none: dur {met[2]/base[2]:.3f} idle {met[3]/base[3]:.3f} | exact {ex:.1e} "
                  f"| mover makespan {S.asap(m.ops):.0f} | {desc} | {time.time()-t0:.1f}s", flush=True)
