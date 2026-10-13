"""ns_snapshot.py -- saves the calibration of HW2's device (read from HW2's manifest) for the NS2 simulation.

Read-only: it opens the saved IBM account, reads the backend's Target and properties, and writes them to JSON. It
submits nothing and uses no QPU time. It never reads or writes the API key. Run from the repository root, after
HW2's `prepare` (VPN off, as for the run):
    python ns_snapshot.py data/2026-10-13/hw2            (real device; refuses if the snapshot exists)
    python ns_snapshot.py <dir> --fake FakeFez           (rehearsal)"""
import argparse
import datetime as dt
import json
import os


def snapshot(backend):
    t = backend.target
    props = None
    try:
        props = backend.properties()
    except Exception:  # noqa: BLE001
        props = None

    def pval(q, name):
        if props is None:
            return None
        try:
            for p in props.qubits[q]:
                if p.name == name:
                    return float(p.value)
        except Exception:  # noqa: BLE001
            return None
        return None

    qubits = {}
    for q in range(t.num_qubits):
        qp = t.qubit_properties[q] if t.qubit_properties is not None else None
        sx = t["sx"].get((q,)) if "sx" in t.operation_names else None
        me = t["measure"].get((q,)) if "measure" in t.operation_names else None
        qubits[q] = dict(t1_s=getattr(qp, "t1", None), t2_s=getattr(qp, "t2", None),
                         sx_error=getattr(sx, "error", None), sx_length_s=getattr(sx, "duration", None),
                         readout_error=getattr(me, "error", None), readout_length_s=getattr(me, "duration", None),
                         prob_meas1_prep0=pval(q, "prob_meas1_prep0"), prob_meas0_prep1=pval(q, "prob_meas0_prep1"),
                         init_error=None)
    g2 = next((g for g in ("cz", "ecr", "cx") if g in t.operation_names), None)
    couplers = []
    if g2:
        for qa, pr in t[g2].items():
            if qa is not None:
                couplers.append([qa[0], qa[1], getattr(pr, "error", None), getattr(pr, "duration", None)])
    last = None
    try:
        last = str(props.last_update_date) if props is not None else None
    except Exception:  # noqa: BLE001
        last = None
    return dict(device=backend.name, two_qubit_gate=g2, calibration_time=last,
                written=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), qubits=qubits,
                couplers=couplers), (props.to_dict() if props is not None else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--fake", default=None)
    a = ap.parse_args()
    if a.fake:
        from qiskit_ibm_runtime import fake_provider
        backend = getattr(fake_provider, a.fake)()
    else:
        man = json.load(open(os.path.join(a.out, "hw2_manifest.json"), encoding="utf-8"))
        from qiskit_ibm_runtime import QiskitRuntimeService
        backend = QiskitRuntimeService().backend(man["device"])
    path = os.path.join(a.out, f"ns_snapshot_{backend.name}.json")
    if os.path.exists(path):
        raise SystemExit(f"{path} exists: not written again")
    snap, raw = snapshot(backend)
    os.makedirs(a.out, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snap, f, indent=1, default=str)
    if raw is not None:
        with open(os.path.join(a.out, f"ns_properties_{backend.name}.json"), "w", encoding="utf-8") as f:
            json.dump(raw, f, default=str)
    n_t1 = sum(1 for v in snap["qubits"].values() if v["t1_s"])
    print(f"SNAPSHOT WRITTEN {path}: {len(snap['qubits'])} qubits ({n_t1} with T1), {len(snap['couplers'])} "
          f"couplers, calibration {snap['calibration_time']}")


if __name__ == "__main__":
    main()
