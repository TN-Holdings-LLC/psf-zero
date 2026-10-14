"""R1 step 1: is the user's advantage on 3-regular QAOA the layering or the seed? Hand-layered circuit (family_map's
hand_layered) through Qiskit level 3 with seed 0 only; also seeds 0-4 for reference; c53+91 on the circuit as
written and on the hand-layered circuit."""
import sys, os, time, json, contextlib, io, tempfile, importlib.util
sys.argv = ["family_map.py", tempfile.mkdtemp() + "/fm"]
spec = importlib.util.spec_from_file_location("fm", "benchmarks/family_map.py"); FM = importlib.util.module_from_spec(spec)
sys.modules["fm"] = FM; spec.loader.exec_module(FM)
NEW = FM.H.load_module(os.environ.get("PSF_NEW", "patches/psf_compile_c54_2026-10-14/psf_compile.py"), "c53_91_r1"); NEW.WARN_WITHOUT_TARGET = False
from qiskit_ibm_runtime.fake_provider import FakeFez
b = FakeFez(); TG = {"T": b.target, "F": FM.fractional(b.target)}
for name in ("qaoa_n12_p2", "qaoa_n20_p2", "qaoa3reg_n20_r3"):
    qc = FM.build(name); hl = FM.hand_layered(qc)
    for tk, t in TG.items():
        s0 = FM.dur(FM.qk3(hl, t, 0), t)
        s5 = min(FM.dur(FM.qk3(hl, t, s), t) for s in range(5))
        a0 = FM.dur(FM.qk3(qc, t, 0), t)
        with contextlib.redirect_stdout(io.StringIO()):
            p = NEW.compile_for_hardware(qc, target=t, seed_transpiler=0)
            ph = NEW.compile_for_hardware(hl, target=t, seed_transpiler=0)
        print(f"{name:16s} {tk} | HL+QK3 seed0 {s0:6.2f} | HL+QK3 best of 5 {s5:6.2f} | QK3 as written {a0:6.2f} | c53+91 {FM.dur(p, t):6.2f} ({FM.n2(p)}) | c53+91 on HL {FM.dur(ph, t):6.2f} ({FM.n2(ph)})", flush=True)
