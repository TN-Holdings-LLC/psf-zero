"""Tests for candidate psf_compile 2026-10-05.c13 (item 40). Run from this folder:
    python -m pytest test_c13_readout.py -q        (PYTHONPATH must contain the Rust core)
C12 is the candidate file of exact_stage_v1 (path via C12_PATH, default ../qml2/cand/psf_compile.py)."""
import contextlib, importlib.util, io, os, sys, warnings
import numpy as np
import pytest
warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from qiskit import QuantumCircuit, transpile  # noqa: E402
from qiskit_ibm_runtime import fake_provider as fp  # noqa: E402
import depth_eval as DE  # noqa: E402
import readout_eval as RE  # noqa: E402

C12_PATH = os.environ.get("C12_PATH", os.path.join(HERE, "..", "qml2", "cand", "psf_compile.py"))
P13 = RE.load(os.path.join(HERE, "psf_compile.py"), "t_psf_c13")
P12 = RE.load(C12_PATH, "t_psf_c12")
DEV = {d: getattr(fp, d)() for d in ("FakeTorino", "FakeKingston", "FakeAuckland")}


def call(P, qc, be):
    t = be.target
    basis = [g for g in t.operation_names if g in ("cx", "cz", "rz", "sx", "x")]
    with contextlib.redirect_stdout(io.StringIO()):
        return P.compile_for_hardware(qc, coupling_map=t.build_coupling_map(), basis_gates=basis, entangling_basis="cx",
                                      layout_search=True, seed_transpiler=0, target=t, placement_refine=True,
                                      final_resynthesis="select", compare_level3=True, compare_floor=True,
                                      candidate_score="hybrid")


def classifier(n=6, L=4, seed=5):
    rng = np.random.default_rng(seed)
    th = rng.normal(0, 1, DE.n_params(n, L))
    return DE.circuit(rng.uniform(-1, 1, n), th, n, L)


def test_versions():
    assert P13.VERSION == "2026-10-05.c13" and P12.VERSION == "2026-10-05.c12"


def test_readout_cost_sums_measured_qubits_once():
    t = DEV["FakeTorino"].target
    c = QuantumCircuit(t.num_qubits, 3)
    c.x(5); c.measure(5, 0); c.measure(5, 1); c.measure(7, 2)
    exp = t["measure"][(5,)].error + t["measure"][(7,)].error
    assert abs(P13.readout_cost(c, t) - exp) < 1e-15
    c2 = QuantumCircuit(t.num_qubits); c2.x(5)
    assert P13.readout_cost(c2, t) == 0.0


@pytest.mark.parametrize("dev", ["FakeTorino", "FakeAuckland"])
def test_hybrid_cost_adds_only_readout(dev):
    be = DEV[dev]
    out = call(P12, RE.measured(classifier()), be)
    h12, h13 = P12.hybrid_cost(out, be.target), P13.hybrid_cost(out, be.target)
    assert abs(h13 - h12 - P13.readout_cost(out, be.target)) < 1e-12
    un = RE.strip_measure(out)
    assert P13.hybrid_cost(un, be.target) == P12.hybrid_cost(un, be.target)


@pytest.mark.parametrize("dev", ["FakeTorino", "FakeKingston", "FakeAuckland"])
def test_unmeasured_identical_to_c12(dev):
    for seed in (1, 2, 3):
        qc = classifier(n=6, L=2, seed=seed)
        assert RE.sig(call(P13, qc, DEV[dev])) == RE.sig(call(P12, qc, DEV[dev]))


@pytest.mark.parametrize("dev", ["FakeTorino", "FakeKingston", "FakeAuckland"])
def test_measured_output_exact_and_measured_where_logical0_ends(dev):
    qc = classifier(n=6, L=4, seed=7)
    out = call(P13, RE.measured(qc), DEV[dev])
    fin0 = out.layout.final_index_layout(filter_ancillas=True)[0]
    mq = [out.find_bit(g.qubits[0]).index for g in out.data if g.operation.name == "measure"]
    assert mq == [fin0]
    assert RE.state_infid(qc, RE.strip_measure(out)) <= 1e-6


def test_readout_term_can_change_the_choice():
    """On FakeKingston, 6-qubit rings: c13 never puts the output on a worse-readout qubit than c12 does, and does
    better on at least one of these circuits (the dev case of the workplace exploration had 0.0149 -> 0.0079)."""
    be = DEV["FakeKingston"]; t = be.target
    better = 0
    for seed in range(6):
        m = RE.measured(classifier(n=6, L=4, seed=100 + seed))
        e = []
        for P in (P12, P13):
            out = call(P, m, be)
            e.append(t["measure"][(out.layout.final_index_layout(filter_ancillas=True)[0],)].error)
        assert e[1] <= e[0] + 1e-15
        better += e[1] < e[0]
    assert better >= 1
