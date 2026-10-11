"""Builds candidate 2026-10-11.c38 from release 2026-10-11.1 by exact substitutions (changelog item 68: a Pauli
evolution whose terms all commute is compiled by STEINER6's construction, Addendum 444; every other input is compiled
as by the release).
    python make_c38.py <release psf_compile.py> <st_core.py> <st_driver.py> <output psf_compile.py>"""
import sys

src, core, driver, dst = sys.argv[1:5]
s = open(src, encoding="utf-8").read()
block = ("# ---------------------------------------------------------------- item 68: STEINER6 (Addendum 444)\n"
         "import heapq  # noqa: E402\nimport math  # noqa: E402\nfrom collections import deque  # noqa: E402\n\n_ST_FAILED = 0.5\n\n\n"
         + open(core, encoding="utf-8").read().rstrip("\n") + "\n" + open(driver, encoding="utf-8").read().rstrip("\n")
         + "\n\n\n")
for old, new in [
    ("VERSION: 2026-10-11.1 -- release, adopted on 2026-10-11 from candidate 2026-10-10.c36, accepted in C36-VAL "
     "(Addenda 440-441) (previous release: 2026-10-10.3)",
     "VERSION: 2026-10-11.c38 -- candidate (from release 2026-10-11.1): item 68"),
    ('VERSION = "2026-10-11.1"  # release (from candidate 2026-10-10.c36, C36-VAL): 2026-10-10.3 + item 64 '
     '(routing at level 3 when the device is given)',
     'VERSION = "2026-10-11.c38"  # candidate: release 2026-10-11.1 + item 68 (commuting Pauli evolutions by '
     'STEINER6)'),
    ('''    a Target nothing changes. An explicit integer keeps its meaning.
"""''', '''    a Target nothing changes. An explicit integer keeps its meaning.
68. **ACCURACY: a Pauli evolution whose terms all commute is built on the device by STEINER6 (candidate
    2026-10-11.c38).** With the device's Target (`steiner="auto"`) and no `coupling_map` or `initial_layout` of the
    caller's, an input that is one PauliEvolutionGate on all its qubits (Qiskit's default first-order product
    formula, real coefficients, optional final measurements) whose terms all commute is compiled by construction,
    without routing: terms are run in an order of the construction's choice (exact, as they commute), qubits are moved
    by SWAPs planned from the terms still to come (a SWAP joining a ZZ term's block costs one CNOT), placements and
    SWAPs are chosen by the device's errors, then Qiskit level 3 on the fixed layout and item 33's re-placement. The
    output carries its layout (initial placement and the final permutation). Every other input, and an output that
    would use a failed element, goes by the release's path unchanged. The rule was fixed from STEINER6's development
    data only (Addendum 444: on all-commuting tests it beat the release on 12 of 13 runnable FakeTorino/FakeKingston
    cases; on the others it was even). Counts in `STEINER_STATS`.
"""'''),
    ("    placement_max_trials: int = 2_500,\n",
     "    placement_max_trials: int = 2_500,\n"
     '    steiner: Union[bool, str] = "auto",\n'),
    ('''    `placement_refine` (candidate 2026-10-10.c32, item 60): "auto" (default) is True when a Target is given and False
    otherwise.
    """
    if backend is not None:  # item 58''', '''    `placement_refine` (candidate 2026-10-10.c32, item 60): "auto" (default) is True when a Target is given and False
    otherwise.
    `steiner` (candidate 2026-10-11.c38, item 68): "auto" (default) is True when the device is given (`backend=` or
    `target=`) and neither `coupling_map` nor `initial_layout` is; a Pauli evolution whose terms all commute is then
    compiled by STEINER6's construction. False compiles every input as the release does.
    """
    if steiner == "auto":  # item 68: decided before the device fills in the coupling map
        steiner = (backend is not None or target is not None) and coupling_map is None and initial_layout is None
    if backend is not None:  # item 58'''),
    ('''    if placement_refine and target is None:
        raise ValueError("placement_refine=True needs the device `target` (changelog item 33)")
''', '''    if placement_refine and target is None:
        raise ValueError("placement_refine=True needs the device `target` (changelog item 33)")
    if steiner and target is None:
        raise ValueError("steiner=True needs the device `target` (changelog item 68)")
'''),
    ('''    if target is not None:
        args = dict(qc=qc, coupling_map=coupling_map, basis_gates=basis_gates, block_gate_floor=block_gate_floor,''',
     '''    if steiner and target is not None and not (final_resynthesis or compare_level3 or compare_floor):  # item 68
        st = _st_compile(qc, target, prune_max_error, seed_transpiler, placement_call_limit, placement_max_trials)
        if st is not None:
            return st
    if target is not None:
        args = dict(qc=qc, coupling_map=coupling_map, basis_gates=basis_gates, block_gate_floor=block_gate_floor,'''),
    ("def _failed_elements(target, max_error: float,", block + "def _failed_elements(target, max_error: float,"),
]:
    assert s.count(old) == 1, old[:80]
    s = s.replace(old, new)
open(dst, "w", encoding="utf-8", newline="\n").write(s)
print("written", dst)
