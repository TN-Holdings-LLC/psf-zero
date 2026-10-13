"""snapshot_target.py -- a qiskit Target built from a calibration snapshot written by ns_snapshot.py (HW2's device),
so that estimates that read a Target (a14's) can be computed offline, after the run, from the calibration of the day.

The Target has the snapshot's qubits; sx, x and rz on every qubit; measure on every qubit; the two-qubit gate on
every coupler the snapshot lists (both directions, as listed); delay on every qubit. Errors and durations are the
snapshot's (rz: error 0, duration 0). T1 and T2 go into the qubit properties."""
import json


def target_from_snapshot(path, dt=None):
    from qiskit.circuit import Delay, Measure, Parameter
    from qiskit.circuit.library import CZGate, ECRGate, CXGate, RZGate, SXGate, XGate
    from qiskit.transpiler import InstructionProperties, QubitProperties, Target
    d = json.load(open(path, encoding="utf-8"))
    qs = sorted(int(q) for q in d["qubits"])
    n = max(qs) + 1
    qp = [QubitProperties(t1=d["qubits"][str(q)].get("t1_s"), t2=d["qubits"][str(q)].get("t2_s")) if str(q) in d["qubits"]
          else QubitProperties() for q in range(n)]
    t = Target(num_qubits=n, dt=dt, qubit_properties=qp)

    def one(key_err, key_len):
        return {(q,): InstructionProperties(error=d["qubits"][str(q)].get(key_err),
                                            duration=d["qubits"][str(q)].get(key_len)) for q in qs}
    t.add_instruction(SXGate(), one("sx_error", "sx_length_s"))
    t.add_instruction(XGate(), one("sx_error", "sx_length_s"))
    t.add_instruction(RZGate(Parameter("theta")), {(q,): InstructionProperties(error=0.0, duration=0.0) for q in qs})
    t.add_instruction(Measure(), one("readout_error", "readout_length_s"))
    t.add_instruction(Delay(Parameter("t")), {(q,): None for q in qs})
    gate = {"cz": CZGate(), "ecr": ECRGate(), "cx": CXGate()}[d.get("two_qubit_gate") or "cz"]
    t.add_instruction(gate, {(int(a), int(b)): InstructionProperties(error=e, duration=ln) for a, b, e, ln in d["couplers"]})
    return t
