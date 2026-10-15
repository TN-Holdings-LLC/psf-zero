# Path note (2026-10-15c): the absolute workplace paths of the original run are replaced by the 2026-10-15b
# delivery inside bundle q; run from the repository root. Nothing else is changed.
import sys, os, time, contextlib, io, runpy, tempfile, warnings, statistics
warnings.simplefilter("ignore")
sys.path[:0] = [os.path.join("data", "2026-10-15", "workplace", "workplace_2026-10-15b_W1-W2", "w30"), "benchmarks"]
_tmp = tempfile.mkdtemp() + "/c55"
sys.argv = ["c55_val", _tmp]
g = runpy.run_path(os.path.join("benchmarks", "c55_val.py"), run_name="imp")
M = g["load"](os.path.join(os.path.join("data", "2026-10-15", "workplace", "workplace_2026-10-15b_W1-W2", "w31"), "c61", "psf_compile.py"), "psf_compile"); M.WARN_WITHOUT_TARGET = False
from c29_identity import value_sig
import c60_tests as TT
IN = ("tfim_n12", "tfim_n16", "tfimP_n12", "tfimP_n16", "tfim_n20_r3", "heis_n12_r2", "tfim_n40_r2", "hhI_n16", "hhI_n20", "ring12_tfim", "hhtree_heis_n16", "ghz_n12")
NAT = ("chain_cz_n12", "ring_cz_n12", "hh_cz_n20", "cx_chain_n16", "ra_n12_r4", "su2_n12_r4", "clf_ring12_L4")
t, ft = g["device"]("FakeFez")
for name in IN + NAT:
    qc = g["build"](name) if name in IN else TT.build(name)
    for lab, tg in (("T", t), ("F", ft)):
        with contextlib.redirect_stdout(io.StringIO()):
            s = time.perf_counter(); o = M.compile_for_hardware(qc, target=tg, entangling_basis="cx", layout_search=True, seed_transpiler=0); tc = time.perf_counter() - s
        try:
            s = time.perf_counter(); tpl = M.compile_template(qc, tg); tb = time.perf_counter() - s
        except Exception as e:
            print(name, lab, "TEMPLATE ERROR", type(e).__name__, str(e)[:200]); continue
        vals = tpl.values_of(qc)
        ts = []
        for _ in range(20):
            s = time.perf_counter(); c = tpl.bind_values(vals); ts.append(time.perf_counter() - s)
        s = time.perf_counter(); tpl.bind(qc); tq = time.perf_counter() - s
        mo, mc = g["metrics"](o, tg), g["metrics"](c, tg)
        ex = g["exactness"](qc, c) if mc["active"] <= 20 else None
        same = value_sig(c) == value_sig(o)
        print(f"{name:16s} {lab} {tpl.kind:6s} same {same!s:5s} c60 {mo['dur_us']:6.3f}/{mo['twoq']:4d} tpl {mc['dur_us']:6.3f}/{mc['twoq']:4d} exact {ex} isa {mc['isa']} | compile {tc*1000:6.1f} build {tb*1000:6.1f} bind {statistics.median(ts)*1000:6.3f} bind(qc) {tq*1000:6.2f} ms", flush=True)
print(M.TEMPLATE100_STATS)
