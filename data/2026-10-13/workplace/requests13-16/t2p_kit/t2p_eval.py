"""t2p_eval.py -- evaluation harness for prototype c49 (item 84, "T2P": T2-aware placement score), written by the
workplace (the evaluator) before c49 exists; c49 itself is written by home (the implementer). Pre-registration:
t2-placement-preregistration-draft-2026-10-13.md (predictions V0-V7); this harness fixes how each is measured.

    python -u <kit>/t2p_eval.py circuits --out <dir>                 # list and hash the circuits (no compile)
    python -u <kit>/t2p_eval.py run --c49 <path to c49 psf_compile.py> --out <dir> [--devices FakeFez,...]
                                    [--tests a,b,...] [--arms C48,T2P,...]
    python -u <kit>/t2p_eval.py score --out <dir>
    python -u <kit>/t2p_eval.py rehearse --out <dir>                 # mock c49 (= c48, switch ignored), small set

Run from the psf-zero repository root (it loads patches/psf_compile_c48_2026-10-12, benchmarks/psf_smart_layout.py
and the HW2 kit's circuit builders). `run` appends to <dir>/t2p_eval_rows.jsonl and skips rows already there, so it
can be resumed.

**Contract with c49 (the only thing the harness assumes about it).** c49's `compile_for_hardware` accepts
`placement_score` with values "esp" (c48's behaviour, bit for bit), "t2p" and "t2p_half"; everything else as c48.
If home names the switch differently, edit ARMS below and record the edit (the harness's hash changes).

Arms (each called as `compile_for_hardware(qc, backend=<fake device>, seed_transpiler=0, **extra)`, i.e. as a user
calls the release, with the device's Target):
  C48       c48, no extra argument
  T2P       c49, placement_score="t2p"
  T2P_half  c49, placement_score="t2p_half"
  C49_esp   c49, placement_score="esp"       (harness check H0: must reproduce C48's output on every circuit)
  QK3       Qiskit level 3 with the device (generate_preset_pass_manager(3, backend, seed_transpiler=0))

Every metric is computed here, from the compiled circuit and the fake device's Target, by code that does not come
from c48 or c49 (ESP, exposure, idle term, durations, failed elements, low-T2 qubits, determinism, exactness).
Exactness: total variation distance between the compiled circuit's and the logical circuit's exact outcome
distributions (Aer statevector of the active qubits, at most 24; the logical circuit unrolled by the HW2 kit's
`_unrolled`, so that a Trotterised evolution is compared with itself)."""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import importlib.util
import io
import json
import math
import os
import platform
import statistics
import sys
import time
import warnings

import numpy as np

REPO = os.path.abspath(os.getcwd())
for p in (os.path.join(REPO, "benchmarks"), REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

DEVICES = ("FakeFez", "FakeMarrakesh", "FakeKingston", "FakeTorino")
HW2_TESTS = ("qaoa_n12_p2", "qaoa_n20_p2", "ba_n14", "ba_n16", "ba_n18", "ba_n20", "tfim_n12", "tfim_n16", "ghz_n12",
             "tfimP_n12", "tfimP_n16")      # HW2's part R tests (nine) and the two tfimP tests
NEW_BA_N = (14, 16, 18, 20)
NEW_BA_SEEDS = 5
NEW_BA_SEED_BASE = 4_840_000                # fixed now: seed = base + 1000 n + i, i = 0..4 (not used anywhere before)
NEW_BA_M = 2                                # Barabasi-Albert attachment, as Benchpress's BA files (24 edges at n = 14)
NEW_BA_P = 3                                # QAOA layers
QAOA_GAMMA, QAOA_BETA = 0.55, 0.45          # HW2's linear ramp
ARMS = {
    "C48": ("c48", {}),
    "T2P": ("c49", {"placement_score": "t2p"}),
    "T2P_half": ("c49", {"placement_score": "t2p_half"}),
    "C49_esp": ("c49", {"placement_score": "esp"}),
    "QK3": ("qiskit", {}),
}
TWICE = ("T2P", "T2P_half")                 # compiled a second time for determinism (V0)
C48_FILE = os.path.join(REPO, "patches", "psf_compile_c48_2026-10-12", "psf_compile.py")
LAYOUT_FILE = os.path.join(REPO, "benchmarks", "psf_smart_layout.py")
HW2_FILE = os.path.join(REPO, "data", "2026-10-13", "hw2_kit", "hw2_real.py")
FAILED = 0.5
MISSING_T_S = 1e-6
EXACT_TOL = 1e-6
EXACT_MAX_ACTIVE = 24
LOW_T2_PCT = 10
KIT_CIRCUITS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "circuits", "t2p_circuits.json")


# ------------------------------------------------------------------ helpers

def stamp():
    u = dt.datetime.now(dt.timezone.utc)
    try:
        from zoneinfo import ZoneInfo
        return f"{u.isoformat(timespec='seconds')} ({u.astimezone(ZoneInfo('Europe/Zurich')).strftime('%Y-%m-%d %H:%M:%S %Z')})"
    except Exception:  # noqa: BLE001
        return u.isoformat(timespec="seconds")


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def gmean(v):
    v = [x for x in v if x is not None and x > 0 and math.isfinite(x)]
    return math.exp(sum(math.log(x) for x in v) / len(v)) if v else float("nan")


# ------------------------------------------------------------------ circuits

def new_ba(n, i):
    """BA QAOA, NEW_BA_P layers, HW2's angle ramp, on networkx's Barabasi-Albert graph (m = 2, seed fixed above)."""
    import networkx as nx
    from qiskit.circuit.library import QAOAAnsatz
    from qiskit.quantum_info import SparsePauliOp
    seed = NEW_BA_SEED_BASE + 1000 * n + i
    g = nx.barabasi_albert_graph(n, NEW_BA_M, seed=seed)
    edges = sorted(tuple(sorted(e)) for e in g.edges())

    def lab(a, b):
        s = ["I"] * n
        s[n - 1 - a] = s[n - 1 - b] = "Z"
        return "".join(s)
    qa = QAOAAnsatz(SparsePauliOp.from_list([(lab(a, b), 1.0) for a, b in edges]), reps=NEW_BA_P)
    vals = {}
    for prm in qa.parameters:
        k = int(prm.name.split("[")[1].rstrip("]"))
        vals[prm] = QAOA_GAMMA * (k + 1) / NEW_BA_P if prm.name.startswith("γ") else QAOA_BETA * (1 - k / NEW_BA_P)
    qc = qa.assign_parameters(vals)
    qc.measure_all()
    qc.name = f"nba_n{n}_s{i}"
    return qc, seed, edges


def all_circuits(H, only=None):
    """[(name, family, logical circuit, info)] -- HW2's eleven, then the twenty new BA circuits."""
    out = []
    for t in HW2_TESTS:
        if only and t not in only:
            continue
        qc, _ = H.build_test(t)
        out.append((t, "BA" if t.startswith("ba_") else "HW2", qc, {}))
    for n in NEW_BA_N:
        for i in range(NEW_BA_SEEDS):
            name = f"nba_n{n}_s{i}"
            if only and name not in only:
                continue
            qc, seed, edges = new_ba(n, i)
            out.append((name, "BA", qc, dict(seed=seed, edges=len(edges))))
    return out


def logical_sig(qc):
    from qiskit import qasm2
    return hashlib.sha256(qasm2.dumps(qc.decompose(reps=3)).encode()).hexdigest()[:16]


# ------------------------------------------------------------------ metrics (independent of c48/c49)

def _props(target, nm, qs):
    try:
        return target[nm][qs]
    except KeyError:
        return None


def out_sig(out):
    """deterministic signature of a compiled circuit: every instruction's name, qubits, clbits and rounded params."""
    h = hashlib.sha256()
    for ins in out.data:
        h.update(ins.operation.name.encode())
        h.update(repr(tuple(out.find_bit(q).index for q in ins.qubits)).encode())
        h.update(repr(tuple(out.find_bit(c).index for c in ins.clbits)).encode())
        h.update(repr(tuple(round(float(p), 10) if not hasattr(p, "parameters") else str(p)
                            for p in ins.operation.params)).encode())
    return h.hexdigest()[:16]


def device_stats(target):
    t2 = []
    for q in range(target.num_qubits):
        p = target.qubit_properties[q] if target.qubit_properties else None
        if getattr(p, "t2", None):
            t2.append(p.t2)
    return dict(t2_p10=float(np.percentile(t2, LOW_T2_PCT)), t2_median=float(np.median(t2)))


def metrics(out, target, stats):
    """ESP (product of 1 - error over gates and measurements), exposure (sum over used qubits of (end - first gate) /
    T2, as soon as possible, measurements left out), idle term (sum over used qubits of I_q (1/(6 T1) + 1/(3 T2)),
    I_q = window - own gate time; T2 capped at 2 T1; missing T1/T2 = 1 us) and its T2/2 form, combined cost
    (-ln ESP + idle term), two-qubit count, duration, ISA, failed elements, used low-T2 qubits."""
    pos = {q: i for i, q in enumerate(out.qubits)}
    ln_esp, isa, failed, twoq = 0.0, True, 0, 0
    clock, first, busy = {}, {}, {}
    for ins in out.data:
        nm = ins.operation.name
        if nm == "barrier":
            continue
        qs = tuple(pos[q] for q in ins.qubits)
        pr = _props(target, nm, qs)
        if nm == "delay":
            continue
        if nm not in target.operation_names or qs not in target[nm]:
            isa = False
        e = getattr(pr, "error", None) or 0.0
        if e >= FAILED:
            failed += 1
        ln_esp += -math.log(max(1.0 - e, 1e-300))
        if nm == "measure":
            continue
        if len(qs) == 2:
            twoq += 1
        d = (getattr(pr, "duration", None) or 0.0)
        s = max(clock.get(q, 0.0) for q in qs)
        for q in qs:
            first.setdefault(q, s)
            clock[q] = s + d
            busy[q] = busy.get(q, 0.0) + d
    T = max(clock.values()) if clock else 0.0
    expo = idle = idle_half = 0.0
    low = 0
    for q in clock:
        p = target.qubit_properties[q] if target.qubit_properties else None
        t1 = getattr(p, "t1", None) or MISSING_T_S
        t2raw = getattr(p, "t2", None) or MISSING_T_S
        t2 = min(t2raw, 2 * t1)
        w = T - first[q]
        iq = max(w - busy[q], 0.0)
        expo += w / t2raw
        idle += iq * (1 / (6 * t1) + 1 / (3 * t2))
        idle_half += iq * (1 / (6 * t1) + 2 / (3 * t2))
        if t2raw < stats["t2_p10"]:
            low += 1
    return dict(esp=math.exp(-ln_esp), ln_esp=ln_esp, exposure=expo, idle=idle, idle_half=idle_half,
                cost=ln_esp + idle, cost_half=ln_esp + idle_half, twoq=twoq, dur_us=T * 1e6, isa=isa,
                failed=failed, low_t2=low, used=len(clock))


def measured_probs(circ):
    """Exact probabilities over the classical bits of a circuit whose measurements are all at the end ({bitstring:
    p}, clbit 0 rightmost), by Aer's statevector on the active qubits; (None, active) above EXACT_MAX_ACTIVE."""
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    meas, body = {}, []
    for ins in circ.data:
        nm = ins.operation.name
        if nm == "measure":
            meas[circ.find_bit(ins.clbits[0]).index] = circ.find_bit(ins.qubits[0]).index
        elif nm in ("barrier", "delay"):
            continue
        else:
            body.append((ins.operation, [circ.find_bit(q).index for q in ins.qubits]))
    active = sorted({q for _, qs in body for q in qs} | set(meas.values()))
    if len(active) > EXACT_MAX_ACTIVE:
        return None, len(active)
    pos = {q: i for i, q in enumerate(active)}
    c = QuantumCircuit(len(active))
    for op, qs in body:
        c.append(op, [pos[q] for q in qs])
    ncl = circ.num_clbits
    order = [pos[meas[k]] for k in range(ncl)]          # clbit k <- this local qubit
    c.save_probabilities(order)
    p = AerSimulator(method="statevector", precision="double").run(c, shots=None).result().data()["probabilities"]
    out = {}
    for i in np.nonzero(np.asarray(p) > 1e-15)[0]:
        out[format(int(i), f"0{ncl}b")] = float(p[i])
    return out, len(active)


def exactness(ideal, out):
    d, act = measured_probs(out)
    if d is None:
        return None, act
    keys = set(d) | set(ideal)
    return 0.5 * sum(abs(d.get(k, 0.0) - ideal.get(k, 0.0)) for k in keys), act


# ------------------------------------------------------------------ compiling

def compilers(c49_path):
    load(LAYOUT_FILE, "psf_smart_layout")
    mods = {"c48": load(C48_FILE, "t2p_c48")}
    if c49_path:
        mods["c49"] = load(c49_path, "t2p_c49")
    for m in mods.values():
        if hasattr(m, "WARN_WITHOUT_TARGET"):
            m.WARN_WITHOUT_TARGET = False
    return mods


def compile_arm(arm, qc, backend, mods):
    kind, extra = ARMS[arm]
    t0 = time.perf_counter()
    if kind == "qiskit":
        from qiskit.transpiler import generate_preset_pass_manager
        out = generate_preset_pass_manager(3, backend, seed_transpiler=0).run(qc)
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            out = mods[kind].compile_for_hardware(qc, backend=backend, seed_transpiler=0, **extra)
    return out, time.perf_counter() - t0


def header(mods, c49_path):
    import qiskit
    print(f"{stamp()} | platform {platform.platform()} | python {platform.python_version()} | qiskit "
          f"{qiskit.__version__}")
    for m in ("qiskit_aer", "qiskit_ibm_runtime", "psf_zero_core", "networkx"):
        try:
            mod = __import__(m)
            print(f"  {m} {getattr(mod, '__version__', getattr(mod, 'CORE_VERSION', '?'))}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {m} MISSING ({type(exc).__name__})")
    print("SCRIPT", os.path.basename(__file__), sha256_file(os.path.abspath(__file__)))
    for f in (C48_FILE, LAYOUT_FILE, HW2_FILE) + ((c49_path,) if c49_path else ()):
        print("FILE", os.path.relpath(f, REPO) if os.path.exists(f) else f, sha256_file(f) if os.path.exists(f)
              else "MISSING")


# ------------------------------------------------------------------ steps

def cmd_circuits(a):
    H = load(HW2_FILE, "t2p_hw2")
    rows = []
    for name, fam, qc, info in all_circuits(H):
        rows.append(dict(name=name, family=fam, qubits=qc.num_qubits, sig=logical_sig(qc), **info))
        print(json.dumps(rows[-1]))
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "t2p_circuits.json"), "w", encoding="utf-8") as f:
        import networkx
        import qiskit
        json.dump(dict(written=stamp(), seed_base=NEW_BA_SEED_BASE, networkx=networkx.__version__,
                       qiskit=qiskit.__version__, circuits=rows), f, indent=1)
    print(f"{len(rows)} circuits")


def cmd_run(a, mock=False):
    warnings.simplefilter("ignore")
    os.makedirs(a.out, exist_ok=True)
    c49 = a.c49
    if mock:
        c49 = os.path.join(a.out, "mock_c49.py")
        with open(c49, "w", encoding="utf-8") as f:
            f.write(MOCK_C49.replace("@C48@", C48_FILE))
    if not c49 and any(ARMS[x][0] == "c49" for x in a.arms):
        raise SystemExit("--c49 <path> is needed for the T2P arms")
    mods = compilers(c49)
    header(mods, c49)
    H = load(HW2_FILE, "t2p_hw2")
    path = os.path.join(a.out, "t2p_eval_rows.jsonl")
    done = set()
    if os.path.exists(path):
        for ln in open(path, encoding="utf-8"):
            r = json.loads(ln)
            done.add((r["device"], r["circuit"], r["arm"]))
    from qiskit_ibm_runtime import fake_provider
    circs = all_circuits(H, set(a.tests) if a.tests else None)
    if os.path.exists(KIT_CIRCUITS):         # the twenty new BA circuits must be the ones fixed with the harness
        fixed = {c["name"]: c["sig"] for c in json.load(open(KIT_CIRCUITS, encoding="utf-8"))["circuits"]}
        bad = [n for n, _, qc, _ in circs if n.startswith("nba_") and fixed.get(n) != logical_sig(qc)]
        if bad:
            raise SystemExit(f"new BA circuits differ from the fixed list (networkx/qiskit version?): {bad}")
        print(f"new BA circuits match the fixed list ({sum(n.startswith('nba_') for n, *_ in circs)} checked)")
    dists = {}
    with open(path, "a", encoding="utf-8") as fh:
        for dev in a.devices:
            backend = getattr(fake_provider, dev)()
            tgt = backend.target
            stats = device_stats(tgt)
            for name, fam, qc, info in circs:
                for arm in a.arms:
                    if (dev, name, arm) in done:
                        continue
                    out, secs = compile_arm(arm, qc, backend, mods)
                    m = metrics(out, tgt, stats)
                    r = dict(device=dev, circuit=name, family=fam, arm=arm, secs=round(secs, 3), sig=out_sig(out),
                             **m, **{k: v for k, v in info.items() if k != "edges"})
                    if arm in TWICE:
                        out2, _ = compile_arm(arm, qc, backend, mods)
                        r["same_twice"] = out_sig(out2) == r["sig"]
                    if arm != "QK3" and a.exact:
                        if name not in dists:
                            dists[name] = measured_probs(H._unrolled(qc))[0]
                        tvd, act = exactness(dists[name], out)
                        r["tvd"], r["active"] = tvd, act
                    fh.write(json.dumps(r) + "\n")
                    fh.flush()
                    print(f"{dev} {name} {arm}: ESP {m['esp']:.4f} expo {m['exposure']:.3f} cost {m['cost']:.4f} "
                          f"2q {m['twoq']} low {m['low_t2']} {secs:.1f}s tvd {r.get('tvd')}", flush=True)
    cmd_score(a)


def _verdict(conf, refu):
    return "CONFIRMED" if conf else "REFUTED" if refu else "AMBIGUOUS"


def cmd_score(a):
    rows = [json.loads(ln) for ln in open(os.path.join(a.out, "t2p_eval_rows.jsonl"), encoding="utf-8")]
    R = {(r["device"], r["circuit"], r["arm"]): r for r in rows}
    devs = [d for d in DEVICES if any(r["device"] == d for r in rows)]
    circs = sorted({r["circuit"] for r in rows})
    L = [f"# c49 (T2P) evaluation, scored {stamp()}", "", f"devices: {', '.join(devs)}; circuits: {len(circs)}", ""]
    V, out = {}, dict(per_device={})

    def pair(d, c, arm, base="C48"):
        x, y = R.get((d, c, arm)), R.get((d, c, base))
        return (x, y) if x and y else None
    # H0: C49_esp reproduces C48
    h0 = [(d, c) for d in devs for c in circs if pair(d, c, "C49_esp")]
    h0_bad = [(d, c) for d, c in h0 if R[(d, c, "C49_esp")]["sig"] != R[(d, c, "C48")]["sig"]]
    L.append(f"**H0 (harness check):** C49_esp equals C48 on {len(h0) - len(h0_bad)} of {len(h0)} compiles"
             + (f"; differs on {h0_bad[:5]}" if h0_bad else ""))
    # V0
    v0_rows = [R[k] for k in R if k[2] in TWICE]
    v0_fail = []
    for r in v0_rows:
        why = []
        if not r["isa"]:
            why.append("not ISA")
        if r["failed"]:
            why.append("failed element")
        if r.get("same_twice") is False:
            why.append("differs on second call")
        if r.get("tvd") is not None and r["tvd"] > EXACT_TOL:
            why.append(f"tvd {r['tvd']:.2e}")
        if why:
            v0_fail.append((r["device"], r["circuit"], r["arm"], why))
    n_exact = sum(1 for r in v0_rows if r.get("tvd") is not None)
    V["V0"] = ("NOT TESTABLE (no T2P rows)" if not v0_rows else _verdict(not v0_fail, bool(v0_fail))
               + f" ({len(v0_rows)} outputs, exactness checked on {n_exact}; failures: {v0_fail[:5]})")
    # per device numbers
    L += ["", "| device | V1 share cost T2P <= C48 | V2 gmean exposure T2P/C48 (BA) | V3 gmean ESP T2P/C48 | "
          "V3 min ESP ratio | V4 low-T2 qubits T2P / C48 | V5 median 2q ratio | V5 median duration ratio | "
          "V6 median time ratio |", "|---|---|---|---|---|---|---|---|---|"]
    t_all, q_all, d_all = [], [], []
    for d in devs:
        P = [pair(d, c, "T2P") for c in circs]
        P = [p for p in P if p]
        if not P:
            continue
        share = sum(x["cost"] <= y["cost"] + 1e-12 for x, y in P) / len(P)
        ba = [x["exposure"] / y["exposure"] for x, y in P if x["family"] == "BA" and y["exposure"] > 0]
        esp = [x["esp"] / y["esp"] for x, y in P]
        lowx, lowy = sum(x["low_t2"] for x, _ in P), sum(y["low_t2"] for _, y in P)
        q = [x["twoq"] / y["twoq"] for x, y in P if y["twoq"]]
        du = [x["dur_us"] / y["dur_us"] for x, y in P if y["dur_us"]]
        tt = [x["secs"] / y["secs"] for x, y in P if y["secs"] > 0]
        t_all += tt
        q_all += q
        d_all += du
        e = dict(n=len(P), v1_share=share, v2_gmean_expo_ba=gmean(ba), n_ba=len(ba), v3_gmean_esp=gmean(esp),
                 v3_min_esp=min(esp), v4_low_t2p=lowx, v4_low_c48=lowy, v5_q=statistics.median(q) if q else None,
                 v5_dur=statistics.median(du) if du else None, v6_time=statistics.median(tt) if tt else None)
        out["per_device"][d] = e
        L.append(f"| {d} | {share:.3f} ({len(P)}) | {e['v2_gmean_expo_ba']:.3f} ({len(ba)}) | {e['v3_gmean_esp']:.4f} | "
                 f"{e['v3_min_esp']:.4f} | {lowx} / {lowy} | {e['v5_q']:.3f} | {e['v5_dur']:.3f} | {e['v6_time']:.2f} |")
    pdv = out["per_device"]
    if pdv:
        V["V1"] = _verdict(all(e["v1_share"] >= 0.95 for e in pdv.values()), any(e["v1_share"] < 0.85 for e in pdv.values()))
        V["V2"] = _verdict(all(e["v2_gmean_expo_ba"] <= 0.90 for e in pdv.values()),
                           any(e["v2_gmean_expo_ba"] > 0.97 for e in pdv.values()))
        V["V3"] = _verdict(all(e["v3_gmean_esp"] >= 0.97 for e in pdv.values()) and all(e["v3_min_esp"] >= 0.90 for e in pdv.values()),
                           any(e["v3_gmean_esp"] < 0.95 for e in pdv.values()))
        el = {d: e for d, e in pdv.items() if e["v4_low_c48"] > 0}
        V["V4"] = ("NOT TESTABLE (C48 uses no low-T2 qubit on any device)" if not el else
                   _verdict(all(e["v4_low_t2p"] <= 0.5 * e["v4_low_c48"] for e in el.values()),
                            any(e["v4_low_t2p"] >= e["v4_low_c48"] for e in el.values())))
        mq, md = statistics.median(q_all), statistics.median(d_all)
        V["V5"] = _verdict(0.95 <= mq <= 1.05 and 0.95 <= md <= 1.05, not (0.90 <= mq <= 1.10) or not (0.90 <= md <= 1.10)) \
            + f" (pooled medians: 2q {mq:.3f}, duration {md:.3f})"
        mt = statistics.median(t_all)
        V["V6"] = _verdict(mt <= 1.2, mt > 1.5) + f" (pooled median {mt:.2f})"
        out["pooled"] = dict(v5_q=mq, v5_dur=md, v6_time=mt)
    # V7 (reported)
    L += ["", "**V7 (reported).**", "", "| device | T2P_half/T2P exposure (gmean) | T2P_half/T2P ESP (gmean) | "
          "T2P/QK3 ESP | T2P/QK3 exposure | C48/QK3 ESP | C48/QK3 exposure | cost T2P - QK3 (mean) | cost C48 - QK3 (mean) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for d in devs:
        hp = [pair(d, c, "T2P_half", "T2P") for c in circs]
        hp = [p for p in hp if p]
        tq = [pair(d, c, "T2P", "QK3") for c in circs]
        tq = [p for p in tq if p]
        cq = [pair(d, c, "C48", "QK3") for c in circs]
        cq = [p for p in cq if p]
        f = lambda P, k: gmean([x[k] / y[k] for x, y in P if y[k] > 0])  # noqa: E731
        g = lambda P: (sum(x["cost"] - y["cost"] for x, y in P) / len(P)) if P else float("nan")  # noqa: E731
        L.append(f"| {d} | {f(hp, 'exposure'):.3f} | {f(hp, 'esp'):.4f} | {f(tq, 'esp'):.4f} | {f(tq, 'exposure'):.3f} | "
                 f"{f(cq, 'esp'):.4f} | {f(cq, 'exposure'):.3f} | {g(tq):+.4f} | {g(cq):+.4f} |")
    L += ["", "| ID | verdict |", "|---|---|"] + [f"| {k} | {v} |" for k, v in V.items()]
    out["verdict"], out["h0"] = V, dict(n=len(h0), differ=h0_bad)
    txt = "\n".join(L) + "\n"
    with open(os.path.join(a.out, "t2p_eval_score.md"), "w", encoding="utf-8") as f:
        f.write(txt)
    with open(os.path.join(a.out, "t2p_eval_score.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=str)
    print(txt)


MOCK_C49 = '''"""REHEARSAL ONLY: a stand-in for c49 that accepts placement_score and ignores it (= c48). Not T2P."""
import importlib.util, sys
_s = importlib.util.spec_from_file_location("mock_c49_inner", r"@C48@")
_m = importlib.util.module_from_spec(_s)
sys.modules["mock_c49_inner"] = _m
_s.loader.exec_module(_m)
WARN_WITHOUT_TARGET = False
_m.WARN_WITHOUT_TARGET = False


def compile_for_hardware(*args, placement_score="esp", **kwargs):
    assert placement_score in ("esp", "t2p", "t2p_half")
    return _m.compile_for_hardware(*args, **kwargs)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("circuits", "run", "score", "rehearse"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--c49")
    ap.add_argument("--devices", default=",".join(DEVICES))
    ap.add_argument("--tests", default="")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--no-exact", dest="exact", action="store_false")
    a = ap.parse_args()
    a.devices = [x for x in a.devices.split(",") if x]
    a.tests = [x for x in a.tests.split(",") if x]
    a.arms = [x for x in a.arms.split(",") if x]
    if a.step == "circuits":
        cmd_circuits(a)
    elif a.step == "run":
        cmd_run(a)
    elif a.step == "score":
        cmd_score(a)
    else:
        if not a.tests:
            a.tests = ["ba_n14", "tfimP_n12", "ghz_n12", "nba_n14_s0", "nba_n16_s1"]
        if a.devices == list(DEVICES):
            a.devices = ["FakeFez", "FakeTorino"]
        cmd_run(a, mock=True)
        print("REHEARSAL DONE (mock c49 = c48: H0 must pass; T2P must equal C48, so V1 confirmed, V2-V4 refuted or "
              "ambiguous, V5 = 1.0 -- the verdicts mean nothing about T2P)")


if __name__ == "__main__":
    main()
