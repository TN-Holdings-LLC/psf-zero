"""diag_rel3.py -- REL3-DIAG (2026-10-10; diagnosis, nothing predicted): the release tests of 2026-10-10.3 failed in
test_full_choice_exact_safe_and_not_worse_by_estimate (test_release_2026_10_04_1.py and c10's, c11's): the
recommended call with the full options returned a circuit with a higher hybrid_cost than an older call's. Which of
c35's items causes it? On the tests' 4 circuits x 5 devices, hybrid_cost of the full-option call (the test's FULL) with:
  REL3        c35 as released (ABSORB_SYNTH "qiskit", PRUNE_FIRST True)
  REL3-psf    ABSORB_SYNTH "psf" (item 63 off)
  REL3-nopf   PRUNE_FIRST False (item 61 off)
  REL3-both   both off
  REL2        the kept 2026-10-10.2
and the test's reference: c10 with placement_refine=True, final_resynthesis="select". Also single-qubit gate counts.

    cd <psf-zero repository>; python <this file>
"""
import contextlib
import io
import os
import sys
import warnings

warnings.simplefilter("ignore")
REPO = os.getcwd()
sys.path[:0] = [os.path.join(REPO, "benchmarks"), REPO]
import core_fix_c2_eval as H  # noqa: E402

lay = H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
T = H.load_module(os.path.join(REPO, "benchmarks", "test_release_2026_10_04_1.py"), "t10041_diag")
rel3 = H.load_module(os.path.join(REPO, "psf_compile.py"), "rel3_diag")
rel2 = H.load_module(os.path.join(REPO, "patches", "psf_compile_release_2026-10-10.2", "psf_compile.py"), "rel2_diag")
c10 = H.load_module(os.path.join(REPO, "patches", "psf_compile_c10_2026-10-03", "psf_compile.py"), "c10_diag")
rel3.WARN_WITHOUT_TARGET = False
assert rel3.VERSION == "2026-10-10.3", rel3.VERSION


def q1(c):
    return sum(1 for i in c.data if len(i.qubits) == 1 and i.operation.name not in ("rz", "measure", "barrier"))


def q2(c):
    return sum(1 for i in c.data if len(i.qubits) == 2)


rows, worse = [], {}
for name in ("FakeAuckland", "FakeHanoiV2", "FakeGeneva", "FakeTorino", "FakeKingston"):
    tgt = T.backend(name).target
    kw = T.kw_for(tgt)
    for cname, qc in (("ghz", T.ghz(seed=1)), ("ring6", T.ring(6, seed=2)), ("chain5", T.chain(5, seed=3)),
                      ("xxz", T.xxz_chain(seed=4))):
        res = {}
        with contextlib.redirect_stdout(io.StringIO()):
            for arm, absorb, pf in (("REL3", "qiskit", True), ("REL3-psf", "psf", True), ("REL3-nopf", "qiskit", False),
                                    ("REL3-both", "psf", False)):
                rel3.ABSORB_SYNTH, rel3.PRUNE_FIRST = absorb, pf
                res[arm] = rel3.compile_for_hardware(qc, target=tgt, **T.FULL, **kw)
            rel3.ABSORB_SYNTH, rel3.PRUNE_FIRST = "qiskit", True
            res["REL2"] = rel2.compile_for_hardware(qc, target=tgt, **T.FULL, **kw)
            res["ref"] = c10.compile_for_hardware(qc, target=tgt, placement_refine=True, final_resynthesis="select", **kw)
        hc = {k: rel3.hybrid_cost(v, tgt) for k, v in res.items()}
        for k in hc:
            if k != "ref" and hc[k] > hc["ref"] + 1e-12:
                worse[k] = worse.get(k, 0) + 1
        rows.append((name, cname, hc, {k: (q2(v), q1(v)) for k, v in res.items()}))
        print(f"{name:13s} {cname:6s} " + "  ".join(f"{k} {hc[k]:.5f} ({q2(res[k])},{q1(res[k])})" for k in hc),
              flush=True)
print("\nhybrid_cost above the reference's (of 20 cells):", {k: worse.get(k, 0) for k in
      ("REL3", "REL3-psf", "REL3-nopf", "REL3-both", "REL2")})
print("(two-qubit, one-qubit non-rz gate counts in brackets)")
