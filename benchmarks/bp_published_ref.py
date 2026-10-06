"""bp_published_ref.py -- builds published_ref.json for BP-PROBE and BP-MOCK from Benchpress's published results
(branch `previous_results`, commit f87a12a, 2026-07-06): per transpile test, the 2Q gate count, 2Q depth and mean time
of Qiskit 2.5.0rc1 and Tket 2.18.0 (both run by the Benchpress authors on an AMD Ryzen 9 7900).

    python bp_published_ref.py <dir with qiskit_2.5.0rc1.json and tket-*.json> published_ref.json

Source files on that branch: benchpress_v1/qiskit/qiskit_2.5.0rc1.json and benchpress_v1/tket/2.18.0/tket-*.json.
"""
import json
import os
import sys


def entries(path):
    """{test name: {q2, d2, t}} for every transpile benchmark ("Transpile - Abstract" or "- Device")."""
    out = {}
    for b in json.load(open(path))["benchmarks"]:
        x = b.get("extra_info", {})
        if not b.get("group", "").startswith("Transpile") or "output_gate_count_2q" not in x or not b.get("stats"):
            continue
        out[b["name"]] = dict(d2=x["output_depth_2q"], q2=x["output_gate_count_2q"], t=b["stats"]["mean"])
    return out


def main(src, dst):
    ref = {}
    for name, e in entries(os.path.join(src, "qiskit_2.5.0rc1.json")).items():
        ref.setdefault(name, {})["qiskit_2.5.0rc1"] = e
    for f in sorted(os.listdir(src)):
        if f.startswith("tket-") and f.endswith(".json"):
            for name, e in entries(os.path.join(src, f)).items():
                if name in ref:
                    ref[name]["tket_2.18.0"] = e
    json.dump(ref, open(dst, "w"), indent=1, sort_keys=True)
    print(len(ref), "tests;", sum("tket_2.18.0" in v for v in ref.values()), "with Tket")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
