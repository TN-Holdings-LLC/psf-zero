import sys, os, warnings, time, runpy, tempfile, random
warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ARGS = sys.argv[1:]; _tmp = tempfile.mkdtemp() + "/c55"; sys.argv = ["c55_val", _tmp]
g = runpy.run_path(os.path.join("benchmarks", "c55_val.py"), run_name="imp")
M = g["load"](os.path.join(HERE, "c63", "psf_compile.py"), "c63"); M.WARN_WITHOUT_TARGET = False
t, ft = g["device"]("FakeFez")
for name in ARGS or ["hhtree_heis_n16", "heis_n12_r2", "tfim_n40_r2", "hhI_n20"]:
    qc = g["build"](name); tpl = M.compile_template(qc, t); v = tpl.values_of(qc); gen = tpl._gen
    rng = random.Random(3)
    vals = [[x * rng.uniform(0.5, 1.5) for x in v] for _ in range(40)]
    for x in vals[:3]: tpl.bind_values(x)
    def tm(f):
        ts = []
        for x in vals:
            s = time.perf_counter(); f(x); ts.append(time.perf_counter() - s)
        ts.sort(); return ts[len(ts)//2] * 1e3
    ti = tm(lambda x: list(gen.items(x)))
    tb = tm(tpl.bind_values)
    sl = tpl._slots0
    ta = tm(lambda x: tpl.circuit.assign_parameters(sl))
    nblk = sum(1 for b in gen.prog if b[0] == "block") if not gen.frame else 0
    print(f"{name:16s} T items {ti:.3f} bind {tb:.3f} assign {ta:.3f} slots {len(sl)} blocks {nblk}")
    items = list(gen.items(vals[5])); p = tpl._planned
    r = tpl._batch(items, p)
    print("  batch ok", r is not None, "batch ms", tm(lambda x: tpl._batch(items, p)), "runs", len(p["runs"]), "lmax", p["batch"]["IDX"].shape, M.TEMPLATE100_STATS)
    import gc
    def manual(x):
        T = [time.perf_counter()]
        values = [float(v) for v in x]; T.append(time.perf_counter())
        items = list(gen.items(values)); T.append(time.perf_counter())
        ok = [it[0] for it in items] == p["kinds"]; T.append(time.perf_counter())
        sl = tpl._batch(items, p); T.append(time.perf_counter())
        out = tpl.circuit.assign_parameters(sl); T.append(time.perf_counter())
        return [b - a for a, b in zip(T, T[1:])]
    R = [manual(x) for x in vals]
    import statistics
    print("  manual phases ms", [round(statistics.median(c) * 1e3, 3) for c in zip(*R)], "total", round(statistics.median([sum(r) for r in R]) * 1e3, 3))
    gc.disable(); print("  bind gc off", tm(tpl.bind_values)); gc.enable()
