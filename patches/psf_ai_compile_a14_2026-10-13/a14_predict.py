"""Write a14's idle-aware estimate (item 19, form fixed in docs/findings/a14-design-2026-10-13.md) for compiled circuits
BEFORE they are run, so that it can be scored as a prediction afterwards. No IBM connection: the Target is read from a
pickle the owner saved (for example the live Target of the device chosen for the run).

  python patches/psf_ai_compile_a14_2026-10-13/a14_predict.py CIRCUITS.qpy TARGET.pkl OUT.json [--labels LABELS.json]

LABELS.json (optional): a list with one name per circuit, e.g. "ba_n14|PSF". Circuits with more than MAX_ACTIVE active
qubits are skipped (the estimate keeps the ideal state of the active qubits).
"""
import hashlib
import json
import os
import pickle
import sys
import time
import warnings

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [os.path.join(REPO, "benchmarks"), REPO]
MAX_ACTIVE = 22


def main():
    qpy_path, tgt_path, out_path = sys.argv[1:4]
    labels = None
    if "--labels" in sys.argv:
        labels = json.load(open(sys.argv[sys.argv.index("--labels") + 1], encoding="utf-8"))
    import core_fix_c2_eval as H
    H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
    H.load_module(os.path.join(REPO, "psf_compile.py"), "psf_compile")
    a14 = H.load_module(os.path.join(HERE, "psf_ai_compile.py"), "psf_ai_compile_a14")
    from qiskit import qpy
    target = pickle.load(open(tgt_path, "rb"))
    circs = qpy.load(open(qpy_path, "rb"))
    rows = []
    for i, c in enumerate(circs):
        pos = {q: k for k, q in enumerate(c.qubits)}
        used = sorted({pos[q] for ins in c.data for q in ins.qubits})
        name = labels[i] if labels else str(i)
        if len(used) > MAX_ACTIVE:
            rows.append(dict(index=i, name=name, active=len(used), skipped=True))
            continue
        ident = {q: q for q in used}
        gate = a14._score_weights(a14._state_weights(c), target, ident)
        iw, T = a14.idle_weights(c, target)
        idle = a14.idle_cost(iw, target, ident)
        rows.append(dict(index=i, name=name, active=len(used), gate_readout=gate, idle=idle, total=gate + idle,
                         duration_us=T * 1e6))
        print(f"{name:24s} gate+readout {gate:.4f}  idle {idle:.4f}  total {gate + idle:.4f}  {T * 1e6:.2f} us", flush=True)
    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
    json.dump(dict(written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), a14=a14.AI_COMPILE_VERSION,
                   a14_sha256=sha(os.path.join(HERE, "psf_ai_compile.py")), circuits_sha256=sha(qpy_path),
                   target_sha256=sha(tgt_path), rows=rows), open(out_path, "w", encoding="utf-8"), indent=1)
    print("written", out_path, sha(out_path))


if __name__ == "__main__":
    main()
