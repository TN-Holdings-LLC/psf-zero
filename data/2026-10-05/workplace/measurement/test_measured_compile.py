"""Tests for measured_compile.py (PYTHONPATH with the Rust core; psf_compile c13 next to this file)."""
import contextlib, io, os, sys, warnings
import numpy as np
warnings.simplefilter("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_ibm_runtime import fake_provider as fp
import measured_compile as M
import psf_compile as pc

BE = fp.FakeTorino(); T = BE.target


def psf(c):
    basis = [g for g in T.operation_names if g in ("cz", "rz", "sx", "x")]
    with contextlib.redirect_stdout(io.StringIO()):
        return pc.compile_for_hardware(c, coupling_map=T.build_coupling_map(), basis_gates=basis, entangling_basis="cx",
                                       layout_search=True, seed_transpiler=0, target=T, placement_refine=True,
                                       final_resynthesis="select", compare_level3=True, compare_floor=True,
                                       candidate_score="hybrid")


def classifier(n=6, L=12, seed=3):
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    for _ in range(L):
        for q in range(n):
            qc.ry(float(rng.uniform(-1, 1)), q); qc.rz(float(rng.uniform(-1, 1)), q)
        for q in range(n):
            qc.cz(q, (q + 1) % n)
    return qc


def test_with_measurements_idempotent_and_ordered():
    qc = classifier(4, 1)
    m = M.with_measurements(qc, [2, 0])
    assert m.num_clbits == 2 and [m.find_bit(i.qubits[0]).index for i in m.data if i.operation.name == "measure"] == [2, 0]
    assert M.with_measurements(m) is m


def test_measured_qubit_is_final_position_and_state_exact():
    qc = classifier()
    out = M.compile_measured(psf, qc, [0])
    fin = list(out.layout.final_index_layout(filter_ancillas=True))
    assert M.measured_physical_qubits(out) == [fin[0]]
    s = M.without_final_measurements(out)
    assert not any(i.operation.name == "measure" for i in s.data) and s.layout is not None


def test_readout_aware_on_torino():
    """The FakeTorino case of the READOUT test: n = 6, L = 12, output qubit readout without vs with measurement."""
    qc = classifier()
    un = psf(qc)
    me = M.compile_measured(psf, qc, [0])
    e_un = T["measure"][(un.layout.final_index_layout(filter_ancillas=True)[0],)].error
    e_me = T["measure"][(M.measured_physical_qubits(me)[0],)].error
    assert e_me <= e_un
