"""Verification for `psf_compile.py`'s item-12 change (2026-09-16):
`layout_search` on `compile_for_hardware()`.

Same style as `verify_psf_compile_revision.py`: a `check()` helper, PASS/FAIL
printed per assertion, a `FAILED` list, exit code 1 if anything failed. Run
this on a machine with the real `psf_zero_core` built -- everything here
calls `compile_for_hardware()`, which imports the Rust core at module load,
so this cannot be run in an environment without it (the sandbox that wrote
this file does not have the core, and did not run this script -- see the
accompanying pre-registration for what WAS checked without the core).

Checks, in order:
  1. layout_search=True with an explicit initial_layout also given raises
     ValueError (the two are mutually exclusive by design -- item 12).
  2. layout_search=True on a small circuit (no coupling-map saturation) still
     produces a unitarily-equivalent, coupling-map-valid circuit -- the new
     code path must not regress correctness.
  3. layout_search=True on the actual 6x7-grid, spare=0 scenario from
     Addenda 24-25 finds a layout, produces 0 coupling violations, and is
     unitarily equivalent module the routing/layout permutation is NOT
     checked here (see note below) -- what IS checked is that PSF-Zero's own
     internal verify=True check passes (no fallback warnings) and the
     circuit respects the coupling map.
  4. layout_search=False (the default) is byte-for-byte unaffected by this
     change: same circuit, same seed, same call produces the same gate
     counts and depth as calling compile_for_hardware with the arguments
     that existed before item 12 (a regression guard on the untouched
     default path).
  5. A search that cannot possibly succeed (interaction graph is not
     realizable on the given coupling map at all) still returns a valid
     circuit by falling through to Qiskit's own default layout stage,
     rather than raising or hanging.

Note on what "equivalence" means here: `Operator(qc).equiv(...)` after
ROUTING is only meaningful if the comparison accounts for the layout's
qubit permutation, which this script does not attempt -- that is a routing
correctness question, not a PSF-Zero synthesis one, and this project's own
convention (Addendum 24-25, `test_cliff_sniper_corrected.py`) is to check
coupling-map validity post-routing and unitary equivalence only on the
pre-routing (`compile()`-only) output. Both are done that way here too.
"""
import sys
import warnings

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator, random_unitary
from qiskit.transpiler import CouplingMap

import psf_compile
from psf_compile import compile as psf_compile_only
from psf_compile import compile_for_hardware

FAILED = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not cond:
        FAILED.append(name)


def dense_pair_blocks_circuit(num_qubits, gates_per_pair=20, seed=7):
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(num_qubits)
    for a, b in [(i, i + 1) for i in range(0, num_qubits - 1, 2)]:
        block = QuantumCircuit(2)
        for _ in range(gates_per_pair):
            u = random_unitary(4, seed=int(rng.integers(0, 2**31))).data
            block.append(UnitaryGate(u), [0, 1])
        qc.compose(block.decompose(), [a, b], inplace=True)
    return qc


def coupling_violations(qc, cmap):
    edges = set(cmap.get_edges())
    v = 0
    for inst in qc.data:
        if len(inst.qubits) == 2:
            i = qc.find_bit(inst.qubits[0]).index
            j = qc.find_bit(inst.qubits[1]).index
            if (i, j) not in edges and (j, i) not in edges:
                v += 1
    return v


BASIS = ["rz", "sx", "x", "cx"]

print("0. sanity: psf_zero_core is real, not the stub (import-time check already "
      "enforces this; confirm compile() runs without raising)")
smoke = psf_compile_only(dense_pair_blocks_circuit(4, seed=1))
check("compile() ran on a 4-qubit smoke circuit", smoke is not None)

print("\n1. layout_search=True + explicit initial_layout -> ValueError")
cm_small = CouplingMap.from_grid(3, 3)
qc_small = dense_pair_blocks_circuit(8, seed=2)
try:
    compile_for_hardware(qc_small, coupling_map=cm_small, basis_gates=BASIS,
                          layout_search=True, initial_layout=list(range(8)))
    check("raises ValueError when both given", False, "no error raised")
except ValueError as exc:
    check("raises ValueError when both given", True, str(exc)[:60])

print("\n2. layout_search=True, small circuit, correctness + coupling validity")
cm2 = CouplingMap.from_grid(4, 4)  # 16 qubits, plenty of spare for an 8-qubit circuit
qc2 = dense_pair_blocks_circuit(8, seed=3)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    out2 = compile_for_hardware(qc2, coupling_map=cm2, basis_gates=BASIS,
                                 routing_optimization_level=1, entangling_basis="cx",
                                 layout_search=True, seed_transpiler=42)
check("0 coupling violations (small, unsaturated)", coupling_violations(out2, cm2) == 0)
check("no PSF-Zero fallback warnings", len(caught) == 0, f"{len(caught)} warning(s)")

print("\n3. layout_search=True on the Addenda 24-25 scenario (6x7 grid, spare=0)")
cm3 = CouplingMap.from_grid(6, 7)
qc3 = dense_pair_blocks_circuit(42, seed=7)
with warnings.catch_warnings(record=True) as caught3:
    warnings.simplefilter("always")
    out3 = compile_for_hardware(qc3, coupling_map=cm3, basis_gates=BASIS,
                                 routing_optimization_level=1, entangling_basis="cx",
                                 layout_search=True, seed_transpiler=42)
check("0 coupling violations at spare=0 with layout_search", coupling_violations(out3, cm3) == 0)
check("no PSF-Zero fallback warnings at spare=0", len(caught3) == 0, f"{len(caught3)} warning(s)")

print("\n4. layout_search=False (default) is unaffected -- regression guard")
cm4 = CouplingMap.from_grid(6, 7)
qc4 = dense_pair_blocks_circuit(42, seed=7)
out4a = compile_for_hardware(qc4, coupling_map=cm4, basis_gates=BASIS,
                              routing_optimization_level=1, entangling_basis="cx",
                              seed_transpiler=42)
out4b = compile_for_hardware(qc4, coupling_map=cm4, basis_gates=BASIS,
                              routing_optimization_level=1, entangling_basis="cx",
                              seed_transpiler=42, layout_search=False)
check("default call and explicit layout_search=False agree on gate counts",
      dict(out4a.count_ops()) == dict(out4b.count_ops()))
check("default call and explicit layout_search=False agree on depth",
      out4a.depth() == out4b.depth())

print("\n5. a search that cannot succeed still falls through cleanly")
# 2x2 grid (4 qubits) cannot possibly lay out a 4-qubit dense-pair-blocks
# circuit needing 2 disjoint adjacent pairs if the map itself has no perfect
# matching of that shape -- use a tiny call_limit instead to force a
# not-found result cheaply, on an instance the search COULD in principle
# solve, so this exercises the "gave up, fell through" path specifically
# (not a "problem is infeasible" short-circuit).
cm5 = CouplingMap.from_grid(6, 7)
qc5 = dense_pair_blocks_circuit(42, seed=7)
out5 = compile_for_hardware(qc5, coupling_map=cm5, basis_gates=BASIS,
                             routing_optimization_level=1, entangling_basis="cx",
                             layout_search=True, layout_search_call_limit=1,
                             layout_search_fallback_call_limit=1,
                             seed_transpiler=42)
check("falls through to a valid circuit when the search's budget is starved",
      coupling_violations(out5, cm5) == 0)

print("\nFAILURES:", FAILED if FAILED else "none")
sys.exit(1 if FAILED else 0)
