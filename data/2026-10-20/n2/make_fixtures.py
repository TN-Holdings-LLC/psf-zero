"""Write N2's device fixtures F1-F4 (W3 §1) into fixtures/, deterministically.

    python data/2026-10-20/n2/make_fixtures.py        (from the repository root; needs qiskit and qiskit-ibm-runtime)

F1 FakeFez and F2 FakeBrisbane in psfz's device format (psf_compile's item 105 writer); F3 = F1 with every coupler's
error set to 0.6 (failed by psfz's rule, error >= 0.5); F4 = F1 plus a synthetic `rzz_native` entry on every coupler
with the interval (0, 0.5] (IR r2 §3). F4 is test-only: no real device has this field.
"""
import contextlib
import importlib.util
import io
import json
import os
import sys

REPO = os.getcwd()
OUT = os.path.join(REPO, "data", "2026-10-20", "n2", "fixtures")


def load_pc():
    sys.path[:0] = [os.path.join(REPO, "benchmarks"), REPO]
    spec = importlib.util.spec_from_file_location("psf_compile", os.path.join(REPO, "psf_compile.py"))
    pc = importlib.util.module_from_spec(spec)
    sys.modules["psf_compile"] = pc
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(pc)
    return pc


def dump(name, obj):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, sort_keys=True, separators=(",", ":"))
        f.write("\n")


def main():
    from qiskit_ibm_runtime import fake_provider as fp
    pc = load_pc()
    os.makedirs(OUT, exist_ok=True)
    f1 = json.loads(pc._psfz105_device(fp.FakeFez().target))
    f1["name"] = "F1 FakeFez"
    f2 = json.loads(pc._psfz105_device(fp.FakeBrisbane().target))
    f2["name"] = "F2 FakeBrisbane"
    f3 = json.loads(json.dumps(f1))
    f3["name"] = "F3 FakeFez, every coupler failed (error 0.6)"
    for c in f3["couplers"]:
        c[2] = 0.6
    f4 = json.loads(json.dumps(f1))
    f4["name"] = "F4 FakeFez plus a synthetic rzz native (test only)"
    f4["rzz_native"] = {"couplers": [[c[0], c[1]] for c in f1["couplers"]], "angle_min_exclusive": 0.0,
                        "angle_max": 0.5}
    for name, obj in (("F1.json", f1), ("F2.json", f2), ("F3.json", f3), ("F4.json", f4)):
        dump(name, obj)
    print("fixtures written:", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
