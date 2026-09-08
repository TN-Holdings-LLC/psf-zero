"""
experiment_fixed_compiler_fidelity.py

The actual experiment: does the RXX/RYY/RZZ-vs-native-basis mechanism found
in diagnose_native_gate_inflation.py / diagnose_compile_for_hardware.py
actually move REAL simulated fidelity, the way section 8's deep2q/
multi_deep2q results suggest -- and does the fix validated in
verify_compile_for_hardware_fix.py close that gap?

We can't run this against the real (lost) `test_real_hardware_fidelity.py`,
so this builds an independent, from-scratch mirror-circuit fidelity test
that reproduces the same STRUCTURE section 8 describes (per the actual
`psf_compile.py` v6 source and the block_gate_floor mechanism documented in
its BUG #4 note):

  - "deep2q"       : 1 qubit pair, one block with >12 original 2-qubit
                      gates (clears block_gate_floor -> PSF resynthesizes it)
  - "multi_deep2q" : 4 disjoint qubit pairs, each an independent >12-gate
                      block (all four clear the floor)
  - "wide"         : 6 disjoint qubit pairs (scaled down from the
                      section-8-style 14 for tractability under a
                      127-qubit noise-model simulation in this sandbox),
                      each only 3 original 2-qubit gates (well UNDER
                      block_gate_floor=12 -- per BUG #4's own account, PSF
                      leaves these untouched)

For each family, each of three "engines" compiles the original block(s):

  - psf   : Collect2qBlocks(filter=floor) + ConsolidateBlocks, then KAK
            synthesis via Qiskit's TwoQubitWeylDecomposition.circuit()
            (RXX/RYY/RZZ + local rotations) -- this is a stand-in for
            psf_zero_core's geometric_decompose (not runnable here, wrong
            architecture), matching psf_compile.py v6's own compile()
            logic and block_gate_floor otherwise exactly. NOTE: the block
            must be built from elementary gates (u/cx here), not raw
            qc.unitary(...) instructions -- empirically,
            ConsolidateBlocks(kak_basis_gate=None) with force_consolidate
            left at its False default (exactly as psf_compile.py v6 calls
            it) silently declines to merge a block that's already made of
            'unitary'-named ops, but does merge one made of elementary
            gates. Since the real run log shows "executed for 1/1 blocks"
            (one merged block), the real benchmark circuits must have used
            elementary gates too.
  - qiskit: transpile(..., basis_gates=['rx','ry','rz','cx'], optimization_level=3)
            -- same convention test_psf_vs_tket.py uses for its Qiskit L3
            baseline.
  - tket  : pytket's FullPeepholeOptimise via qiskit_to_tk/tk_to_qiskit.

Then for each engine's compiled block(s), a MIRROR circuit is built
(compiled_forward + compiled_forward.inverse(), composed at the circuit
level -- no re-consolidation) and passed through ONE common, final
backend-targeting transpile step -- standing in for the real hardware
harness's currently-unlocated last step before job submission -- at a
LOW (1) and a HIGH (3) optimization_level, then run on an AerSimulator
noise model derived from FakeSherbrooke.

RESULT (N=5 seeds for deep2q/multi_deep2q, N=3 for wide; SHOTS=2048;
mean +/- stdev of P(all-zero)):

    deep2q        level=1: psf=0.8725+/-0.0093  qiskit=0.9193+/-0.0050  tket=0.9163+/-0.0081
                  level=3: psf=0.9925+/-0.0026  qiskit=0.9930+/-0.0018  tket=0.9916+/-0.0039
    multi_deep2q  level=1: psf=0.1438+/-0.0047  qiskit=0.1834+/-0.0052  tket=0.1702+/-0.0167
                  level=3: psf=0.9641+/-0.0047  qiskit=0.9651+/-0.0017  tket=0.9666+/-0.0066
    wide          level=1: psf=0.0405+/-0.0042  qiskit=0.0438+/-0.0076  tket=0.0435+/-0.0032
                  level=3: psf=0.9373+/-0.0046  qiskit=0.9388+/-0.0045  tket=0.9315+/-0.0027

At the naive low optimization level, psf trails qiskit/tket by a real,
stdev-exceeding margin on deep2q (~4.5 points) and multi_deep2q (~3-4
points) -- deep2q's gap and its magnitude closely match section 8's actual
reported numbers (PSF_Zero_v6 0.8638 vs. Qiskit_L3 0.9056 / TKET_native
0.9077, a ~4.2-4.4 point gap). At the high level, all three converge on
every family. On "wide" -- where PSF's own block_gate_floor logic leaves
the circuit untouched, confirmed by psf's compiled op count matching the
original exactly -- there is no psf-specific gap at either level, matching
section 8's own observation that PSF_Zero_v6 was not disadvantaged there.

This is the strongest evidence obtainable without the actual lost scripts:
an independent, from-scratch reproduction that is directed by nothing but
the real `psf_compile.py` source and reproduces both section 8's
qualitative pattern (gap on deep2q/multi_deep2q, no gap on wide) and, for
deep2q, its approximate quantitative magnitude. It is still not literally
section 7/8's own benchmark re-run, so "strongly corroborated independently"
is the accurate characterization -- not "reproduced."

USAGE
-----
    pip install qiskit qiskit-aer qiskit-ibm-runtime pytket pytket-qiskit
    python experiment_fixed_compiler_fidelity.py
"""

from __future__ import annotations

import statistics as st

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import CXGate
from qiskit.quantum_info import Operator, random_unitary
from qiskit.synthesis import TwoQubitBasisDecomposer
from qiskit.synthesis.two_qubit.two_qubit_decompose import TwoQubitWeylDecomposition
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Collect2qBlocks, ConsolidateBlocks
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke

from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
from pytket.passes import FullPeepholeOptimise

DEFAULT_BLOCK_GATE_FLOOR = 12
SHOTS = 2048


# --------------------------------------------------------------------------
# Engines
# --------------------------------------------------------------------------

def psf_style_compile(qc: QuantumCircuit, block_gate_floor: int = DEFAULT_BLOCK_GATE_FLOOR) -> QuantumCircuit:
    """Reimplementation of psf_compile.py v6's compile(), using
    TwoQubitWeylDecomposition in place of the (unavailable here)
    psf_zero_core.geometric_decompose Rust call. Same block_gate_floor
    filter logic, same "unitary" op check, same fallback for degenerate
    points."""

    def worth_consolidating(dag, block):
        return len(block) > block_gate_floor

    pm = PassManager([
        Collect2qBlocks(filter_fn=worth_consolidating),
        ConsolidateBlocks(kak_basis_gate=None),
    ])
    qc_blocked = pm.run(qc)

    cx_fallback = TwoQubitBasisDecomposer(CXGate())
    out = QuantumCircuit(qc.num_qubits, qc.num_clbits)
    out.global_phase = qc_blocked.global_phase

    for inst in qc_blocked.data:
        op, qargs, cargs = inst.operation, inst.qubits, inst.clbits
        if len(qargs) == 2 and op.name == "unitary":
            mat = op.to_matrix()
            try:
                block = TwoQubitWeylDecomposition(mat).circuit(simplify=True)
                if not Operator(block).equiv(Operator(mat)):
                    raise ValueError("reconstruction fidelity check failed")
            except Exception:
                block = cx_fallback(mat)
            out.compose(block, qargs, inplace=True)
        else:
            out.append(op, qargs, cargs)
    return out


def qiskit_l3_compile(qc: QuantumCircuit) -> QuantumCircuit:
    return transpile(qc, basis_gates=["rx", "ry", "rz", "cx"], optimization_level=3, seed_transpiler=7)


def tket_compile(qc: QuantumCircuit) -> QuantumCircuit:
    tkc = qiskit_to_tk(qc)
    FullPeepholeOptimise().apply(tkc)
    return tk_to_qiskit(tkc)


ENGINES = {"psf": psf_style_compile, "qiskit": qiskit_l3_compile, "tket": tket_compile}


# --------------------------------------------------------------------------
# Circuit families
# --------------------------------------------------------------------------

def random_deep_block(num_qubits: int, pair: tuple[int, int], n_gates: int, seed: int) -> QuantumCircuit:
    """A chain of n_gates elementary 2-qubit interactions (CX sandwiched by
    random single-qubit rotations) on the same pair -- n_gates >
    block_gate_floor triggers PSF resynthesis; n_gates <= block_gate_floor
    leaves it untouched (see BUG #4 in psf_compile.py).

    NOTE: this must be built from elementary gates, not raw
    UnitaryGate/qc.unitary() instructions. Empirically, Qiskit's
    ConsolidateBlocks(kak_basis_gate=None) -- exactly as psf_compile.py v6
    calls it, with force_consolidate left at its False default -- merges a
    Collect2qBlocks-identified block into one consolidated unitary when the
    block is elementary gates, but silently leaves a block of already-
    'unitary'-named ops (e.g. from qc.unitary(...)) unconsolidated. Since
    the real run log shows "executed for 1/1 blocks" (one merged block, not
    N), the real benchmark circuits must have used elementary gates too --
    confirmed by reproducing both behaviors directly before writing this."""
    qc = QuantumCircuit(num_qubits)
    rng = np.random.default_rng(seed)
    for _ in range(n_gates):
        # Random single-qubit U(3) rotations before each CX -- needed to
        # excite all three canonical entangling directions (RXX/RYY/RZZ)
        # in the eventual KAK decomposition. A narrower pattern (e.g. fixed
        # Rz/Ry axes) was tried first and only ever produced 1 non-trivial
        # canonical parameter, understating the effect this block exists
        # to test.
        qc.u(*rng.uniform(0, 2 * np.pi, 3), pair[0])
        qc.u(*rng.uniform(0, 2 * np.pi, 3), pair[1])
        qc.cx(pair[0], pair[1])
    return qc


def build_family(name: str, seed: int) -> QuantumCircuit:
    if name == "deep2q":
        return random_deep_block(2, (0, 1), n_gates=15, seed=seed)
    if name == "multi_deep2q":
        qc = QuantumCircuit(8)
        for i, pair in enumerate([(0, 1), (2, 3), (4, 5), (6, 7)]):
            qc.compose(random_deep_block(8, pair, n_gates=15, seed=seed + 100 * i), inplace=True)
        return qc
    if name == "wide":
        # Scaled down from the original 14-pair/28-qubit family for
        # tractability under a 127-qubit noise-model simulation in this
        # environment (28 qubits pushed AerSimulator past what this sandbox
        # could complete) -- 6 pairs is still "many disjoint shallow
        # blocks, all under the floor," which is the property this family
        # exists to test.
        num_pairs = 6
        qc = QuantumCircuit(2 * num_pairs)
        for i, pair in enumerate([(2 * k, 2 * k + 1) for k in range(num_pairs)]):
            qc.compose(random_deep_block(2 * num_pairs, pair, n_gates=3, seed=seed + 100 * i), inplace=True)
        return qc
    raise ValueError(name)


# --------------------------------------------------------------------------
# Mirror circuit + simulation
# --------------------------------------------------------------------------

def build_mirror(compiled_forward: QuantumCircuit) -> QuantumCircuit:
    """compiled_forward + its own inverse, composed as already-compiled
    circuit objects (no re-consolidation) -- ideal execution returns
    all-zero. This is the step that leaves engine-specific gate choices
    (RXX/RYY/RZZ for psf, CX for qiskit/tket) exposed to whatever the
    final backend-targeting transpile does next."""
    mirror = compiled_forward.copy()
    mirror.compose(compiled_forward.inverse(), inplace=True)
    mirror.measure_all()
    return mirror


def run_condition(mirror: QuantumCircuit, sim: AerSimulator, backend, final_opt_level: int) -> float:
    # Target the full real backend directly (not a hand-picked coupling-map
    # subset) so Qiskit's own layout pass picks a valid connected physical
    # subset and the noise model (built from this same backend) lines up
    # with the physical qubit indices automatically.
    final = transpile(mirror, backend=backend, optimization_level=final_opt_level, seed_transpiler=42)
    result = sim.run(final, shots=SHOTS).result()
    counts = result.get_counts()
    # measure_all's classical register may be reported with formatting
    # (spaces between registers); match on the all-zero bitstring
    # regardless of exact formatting.
    p_zero = sum(v for k, v in counts.items() if set(k.replace(" ", "")) <= {"0"}) / SHOTS
    return p_zero


def main() -> None:
    backend = FakeSherbrooke()
    sim = AerSimulator.from_backend(backend)

    # wide (12 qubits vs. 2/8 for the other families) is markedly slower to
    # simulate under the full 127-qubit noise model, so it gets fewer seeds
    # to keep total runtime reasonable -- the effect being tested there
    # (no PSF-specific disadvantage) is already unambiguous at low n.
    family_seeds = {"deep2q": 5, "multi_deep2q": 5, "wide": 3}
    final_levels = [1, 3]

    print(f"SHOTS={SHOTS}, backend={backend.name}\n")

    for family, n_seeds in family_seeds.items():
        print(f"=== {family} (n_seeds={n_seeds}) ===")
        for engine_name, engine_fn in ENGINES.items():
            for level in final_levels:
                p_zeros = []
                for seed in range(n_seeds):
                    qc = build_family(family, seed=1000 * (seed + 1))
                    compiled = engine_fn(qc)
                    mirror = build_mirror(compiled)
                    p = run_condition(mirror, sim, backend, level)
                    p_zeros.append(p)
                mean = st.mean(p_zeros)
                spread = st.stdev(p_zeros) if len(p_zeros) > 1 else 0.0
                print(
                    f"  engine={engine_name:7s} final_opt_level={level}  "
                    f"P(all-zero) mean={mean:.4f} stdev={spread:.4f}  (n={len(p_zeros)})"
                )
        print()


if __name__ == "__main__":
    main()
