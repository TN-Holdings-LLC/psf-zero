"""a21 item 37 with HANDOFF_CHECK = True: time and verdict of the fixed exactness check on handed-on outputs.
PCFILE selects psf_compile (default c59); CASES lists C55-VAL test names.
    CASES=tfim_n20_r3,hhI_n20 python benchmarks/a21_check_time.py
    CASES=chem_n12_r1,qaoa_n12_p2 PCFILE=psf_compile.py python benchmarks/a21_check_time.py"""
import sys, os, runpy, shutil, time, warnings, contextlib, io
warnings.simplefilter("ignore")
REPO = os.getcwd()  # run from the repository root
sys.path[:0] = [REPO + "/benchmarks", REPO]
import tempfile
S = tempfile.mkdtemp() + "/c55tmp"; sys.argv = ["x", S]
g = runpy.run_path("benchmarks/c55_val.py", run_name="nm"); shutil.rmtree(S, ignore_errors=True)
import core_fix_c2_eval as H
H.load_module("benchmarks/psf_smart_layout.py", "psf_smart_layout")
PC = os.environ.get("PCFILE", "patches/psf_compile_c59_2026-10-15/psf_compile.py")
pc = H.load_module(PC, "psf_compile"); pc.WARN_WITHOUT_TARGET = False
a = H.load_module("patches/psf_ai_compile_a21_2026-10-15/psf_ai_compile.py", "psf_ai_a21"); a.HANDOFF_CHECK = True
t, f = g["device"]("FakeFez")
for nm in os.environ["CASES"].split(","):
    qc = g["build"](nm)
    for tn, tg in (("T", t), ("F", f)):
        s = time.perf_counter()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                out, info = a.compile_for_model_circuit(qc, tg.build_coupling_map(), ["cz", "rz", "sx", "x"], target=tg, return_info=True)
            r = (info["path"], info["chosen"], info.get("exact"))
        except Exception as e:
            r = ("ERR", type(e).__name__, str(e)[:70])
        print(os.path.basename(os.path.dirname(PC)), nm, tn, round(time.perf_counter() - s, 2), "s", r, flush=True)
