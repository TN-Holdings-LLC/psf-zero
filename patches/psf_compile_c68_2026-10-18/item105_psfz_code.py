

# ---------------------------------------------------------------------------------------------------------------------
# Item 105 (prototype c68, 2026-10-18): chains and rings by psfz r4. psfz is PSF-Zero's compiler in Rust
# (rust/psfz/, Addenda 518-522): circuits whose two-qubit interaction graph is one chain or one even ring, placed by the
# heavy-hex lattice's coordinates and written by closed forms, CZ (Heron) and ECR (Eagle). Validated against release
# 2026-10-17.1 in PSFZ-VAL2 (Addendum 522): 3,238 of 3,238 inputs exact, native, deterministic and never longer.
# The default call (target= or backend=, no other option than seed_transpiler and the default prune_max_error) first
# hands the circuit to psfz's program (an executable, run as a separate process; found by $PSFZ_BIN, else
# rust/psfz/target/release/psfz in this file's folder or
# one or two folders up). psfz's output is used as it is when psfz compiles the circuit and
# reports version r4; in every other case (no program, another version, psfz refuses the circuit, an option is given,
# the circuit cannot be written as OpenQASM 3 in psfz's gates) the call is the release's path, unchanged. psfz's output
# is equal to the input up to a global phase (not carried over). Counts in PSFZ105_STATS.
# ---------------------------------------------------------------------------------------------------------------------

import json as _json105  # noqa: E402  (item 105)
import os as _os105  # noqa: E402  (item 105)
import subprocess as _sp105  # noqa: E402  (item 105)
import tempfile as _tf105  # noqa: E402  (item 105)

PSFZ105 = True  # item 105: False never calls psfz
PSFZ105_VERSION = "psfz r4 (2026-10-18)"  # item 105: the only psfz version whose output is used
PSFZ105_STATS = {"calls": 0, "no_program": 0, "not_device": 0, "options": 0, "not_written": 0, "refused": 0,
                 "other_version": 0, "failed": 0, "used": 0}
_PSFZ105_BASIS = ["rx", "ry", "rz", "h", "x", "y", "z", "s", "sdg", "sx", "p", "cx", "cz", "rzz", "rxx", "ryy"]
_PSFZ105_KEYS = {"target", "backend", "seed_transpiler", "prune_max_error"}
_PSFZ105_DEVICES = {}  # id(target) -> (target, device file text)


def _psfz105_program():
    """Item 105: psfz's executable, or None."""
    p = _os105.environ.get("PSFZ_BIN")
    if p:
        return p if _os105.path.isfile(p) else None
    here = _os105.path.dirname(_os105.path.abspath(__file__))
    for up in (here, _os105.path.dirname(here), _os105.path.dirname(_os105.path.dirname(here))):
        for name in ("psfz", "psfz.exe"):
            c = _os105.path.join(up, "rust", "psfz", "target", "release", name)
            if _os105.path.isfile(c):
                return c
    return None


def _psfz105_device(target):
    """Item 105: psfz's device file (JSON) for `target`: per qubit sx's error and duration and measure's error; per
    coupler of the two-qubit gate (cz, else ecr) its direction, error and duration. As PSFZ-VAL2's `device_json`."""
    hit = _PSFZ105_DEVICES.get(id(target))
    if hit is not None and hit[0] is target:
        return hit[1]
    g2 = "cz" if "cz" in target.operation_names else "ecr"
    qs = []
    for q in range(target.num_qubits):
        sx, m = target["sx"].get((q,)), target["measure"].get((q,))
        qs.append(dict(sx_error=None if sx is None else sx.error, sx_duration=None if sx is None else sx.duration,
                       meas_error=None if m is None else m.error))
    cps = [[a, b, None if p is None else p.error, None if p is None else p.duration] for (a, b), p in target[g2].items()]
    text = _json105.dumps(dict(name="target", num_qubits=target.num_qubits, two_qubit_gate=g2, qubits=qs, couplers=cps))
    if len(_PSFZ105_DEVICES) >= 16:
        _PSFZ105_DEVICES.clear()
    _PSFZ105_DEVICES[id(target)] = (target, text)
    return text


def _psfz105_compile(qc, target, kwargs):
    """Item 105: psfz's output for `qc` on `target` as a circuit on the device's qubits, or None (see the comment)."""
    if not PSFZ105:
        return None
    if set(kwargs) - _PSFZ105_KEYS or kwargs.get("prune_max_error", 0.5) != 0.5:
        PSFZ105_STATS["options"] += 1
        return None
    ops = set(target.operation_names)
    if not ({"rz", "sx", "x", "measure"} <= ops and ("cz" in ops or "ecr" in ops)):
        PSFZ105_STATS["not_device"] += 1
        return None
    prog = _psfz105_program()
    if prog is None:
        PSFZ105_STATS["no_program"] += 1
        return None
    if [b for r in qc.cregs for b in r] != list(qc.clbits):
        PSFZ105_STATS["not_written"] += 1  # classical bits outside registers, or registers sharing bits
        return None
    try:
        from qiskit import qasm3
        from qiskit import transpile as _transpile105
        src = qasm3.dumps(_transpile105(qc, basis_gates=_PSFZ105_BASIS, optimization_level=0))
    except Exception:  # noqa: BLE001 - any failure to write it leaves the call to the release's path
        PSFZ105_STATS["not_written"] += 1
        return None
    with _tf105.TemporaryDirectory() as d:
        dev, inp = _os105.path.join(d, "device.json"), _os105.path.join(d, "in.qasm")
        out_q, rep_f = _os105.path.join(d, "out.qasm"), _os105.path.join(d, "report.json")
        with open(dev, "w", encoding="utf-8") as f:
            f.write(_psfz105_device(target))
        with open(inp, "w", encoding="utf-8") as f:
            f.write(src)
        try:
            _sp105.run([prog, "compile", dev, inp, out_q, rep_f], capture_output=True, timeout=120)
            with open(rep_f, encoding="utf-8") as f:
                rep = _json105.load(f)
        except (OSError, ValueError, _sp105.SubprocessError):
            PSFZ105_STATS["failed"] += 1
            return None
    if rep.get("version") != PSFZ105_VERSION:
        PSFZ105_STATS["other_version"] += 1
        return None
    if not rep.get("ok"):
        PSFZ105_STATS["refused"] += 1
        return None
    from qiskit.circuit.library import ECRGate
    out = QuantumCircuit(target.num_qubits, name=qc.name)
    for r in qc.cregs:
        out.add_register(r)
    for name, qs, ps in rep["ops"]:
        qs = [int(q) for q in qs]
        if name == "ecr":
            out.append(ECRGate(), qs)
        else:
            getattr(out, name)(*ps, *qs)
    for q, cb in rep["measure"]:
        out.measure(int(q), out.clbits[int(cb)])
    out._layout = _n99_layout(qc, out, [int(p) for p in rep["layout"]])
    PSFZ105_STATS["used"] += 1
    return out


_i105_compile = compile_for_hardware


def compile_for_hardware(qc, *args, **kwargs):  # noqa: F811  (item 105 wraps items 97-104)
    """Item 105 on items 97-104: a chain or ring circuit compiled by psfz r4 when psfz takes it (see the item's
    comment); every other call is items 97-104's path unchanged."""
    target = kwargs.get("target") or getattr(kwargs.get("backend"), "target", None)
    if target is not None and not args:
        PSFZ105_STATS["calls"] += 1
        out = _psfz105_compile(qc, target, kwargs)
        if out is not None:
            return out
    return _i105_compile(qc, *args, **kwargs)
