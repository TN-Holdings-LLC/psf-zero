"""measured_compile.py -- compile a circuit WITH the measurements it will be sampled with, then read it back.

Why (workplace READOUT test, 2026-10-05): PSF-Zero's re-placement (release item 33) and Qiskit's VF2PostLayout score
`measure` errors only if the circuit contains measurements; the candidate estimates (c13 item 40, a10 item 15) do the
same. Circuits compiled without their measurements are placed readout-blind (FakeTorino: the classifier's output
qubit had readout error 0.048 instead of 0.008).

    out = compile_measured(lambda c: compile_for_hardware(c, ...), qc, measured=[0])
    z_qubits = measured_physical_qubits(out)          # physical qubit of each measured logical qubit, in clbit order
    sim_circ = without_final_measurements(out)        # for density-matrix / statevector reading, layout kept
"""
from qiskit import QuantumCircuit


def with_measurements(qc, measured=None):
    """A copy of `qc` with one classical bit per measured logical qubit (default: all), measured at the end in order.
    A circuit that already measures is returned unchanged."""
    if any(i.operation.name == "measure" for i in qc.data):
        return qc
    measured = list(range(qc.num_qubits)) if measured is None else list(measured)
    m = QuantumCircuit(qc.num_qubits, len(measured), name=qc.name)
    m.global_phase = qc.global_phase
    m.compose(qc, inplace=True)
    for c, q in enumerate(measured):
        m.measure(q, c)
    return m


def compile_measured(compile_fn, qc, measured=None):
    """compile_fn(circuit) -> compiled circuit; called on `qc` with its measurements."""
    return compile_fn(with_measurements(qc, measured))


def measured_physical_qubits(out):
    """Physical qubit measured into each classical bit, in clbit order."""
    pairs = sorted((out.find_bit(i.clbits[0]).index, out.find_bit(i.qubits[0]).index)
                   for i in out.data if i.operation.name == "measure")
    return [q for _, q in pairs]


def without_final_measurements(out):
    """A copy without measurements, keeping the layout (for exact density-matrix reading)."""
    c = out.copy_empty_like()
    for ins in out.data:
        if ins.operation.name != "measure":
            c.append(ins.operation, ins.qubits, ins.clbits)
    c._layout = out._layout
    return c
