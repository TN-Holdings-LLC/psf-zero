"""R4: a fixed tier rule for item 91's embedding: couplers of the most common length; then couplers at most 1.25 x
that length; then all couplers (first VF2 match in each tier, no score)."""
import sys, os, contextlib, io, tempfile, importlib.util
sys.argv = ["c54_val.py", tempfile.mkdtemp() + "/c54"]
spec = importlib.util.spec_from_file_location("cv", "benchmarks/c54_val.py"); CV = importlib.util.module_from_spec(spec)
sys.modules["cv"] = CV; spec.loader.exec_module(CV)
P = CV.M54
orig = P._chain_lattice
def lattice_tier(target, max_error, factor=1.25):
    K, kadj, fadj, snake = orig(target, max_error)
    d = {}
    for qa, p in target["cz"].items():
        if qa is not None and p is not None: d[tuple(sorted(qa))] = p.duration
    from collections import Counter
    modal = Counter(round(d[(min(a, b), max(a, b))] * 1e12) for a in K for b in kadj[a] if a < b).most_common(1)[0][0] / 1e12
    tadj = {q: {v for v in kadj[q] if d[(min(q, v), max(q, v))] <= factor * modal + 1e-15} for q in K}
    return K, kadj, fadj, snake, tadj
for dev in ("FakeTorino", "FakeFez", "FakeMarrakesh", "FakeKingston"):
    T, F = CV.device(dev)
    for name in ("hhI_n16", "hhI_n20", "ring12_tfim"):
        qc = CV.build(name)
        for tk, tg in (("T", T), ("F", F)):
            K, kadj, fadj, snake, tadj = lattice_tier(tg, 0.5)
            P._chain_lattice = lambda t, e, _r=(K, kadj, tadj, snake): _r  # tier 1 replaced by the <= 1.25 x tier
            got = P._graph_read(qc)
            n = qc.num_qubits
            emb_f = P._graph_place(n, got[-1], K, fadj) is not None
            raw_t = P._graph_brickwork(qc, tg, 0.5, 0)
            P._chain_lattice = orig
            raw = P._graph_brickwork(qc, tg, 0.5, 0)
            with contextlib.redirect_stdout(io.StringIO()):
                own = P._compile_for_hardware_c51(qc, target=tg, entangling_basis="cx", layout_search=True, seed_transpiler=0)
            m = lambda c: CV.metrics(c, tg)["dur_us"] if c is not None else float("nan")
            print(f"{dev:13s} {name:12s} {tk} | embeds on common length: {emb_f} | item 91 now {m(raw):6.3f} | with 1.25x tier {m(raw_t):6.3f} | c53 {m(own):6.3f}", flush=True)
