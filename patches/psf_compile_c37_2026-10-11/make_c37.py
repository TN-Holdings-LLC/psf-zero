"""Builds candidate 2026-10-11.c37 from release 2026-10-11.1 by exact substitutions (changelog items 65 and 66: a last
relabelling scored on every operation's error, readout included, that also works when several qubits carry no
two-qubit gate; the Pauli parts of single-qubit runs carried through two-qubit Clifford gates) and 67 (several routing
tries, kept by ESP).
    python make_c37.py <release psf_compile.py> <c37_block.py> <output psf_compile.py>"""
import sys

src, block, dst = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(src, encoding="utf-8").read()
code = open(block, encoding="utf-8").read()
for old, new in [
    ("VERSION: 2026-10-11.1 -- release, adopted on 2026-10-11 from candidate 2026-10-10.c36, accepted in C36-VAL "
     "(Addenda 440-441) (previous release: 2026-10-10.3)",
     "VERSION: 2026-10-11.c37 -- candidate (from release 2026-10-11.1): items 65-67"),
    ('VERSION = "2026-10-11.1"  # release (from candidate 2026-10-10.c36, C36-VAL): 2026-10-10.3 + item 64 '
     '(routing at level 3 when the device is given)',
     'VERSION = "2026-10-11.c37"  # candidate: release 2026-10-11.1 + items 65-67 (last relabelling by every error; '
     'Paulis carried through two-qubit Clifford gates; routing tries kept by ESP)'),
    ('''    a Target nothing changes. An explicit integer keeps its meaning.
"""''', '''    a Target nothing changes. An explicit integer keeps its meaning.
65. **ACCURACY: a last relabelling by every error, readout included (candidate 2026-10-11.c37).** With the device's
    Target (`final_relabel="auto"`), after item 33 the physical qubits are relabelled once more, scored on the summed
    -ln(1 - error) of all operations: two-qubit gates, single-qubit gates and measurements. Item 33 is Qiskit's exact
    VF2PostLayout, which gives up when two or more qubits carry no two-qubit gate (a circuit of single-qubit gates
    and measurements, or measured qubits left out of the entangling part): GAP-DIAG (2026-10-11) found inverseqft1
    and inverseqft2 measured on qubits 0-3 of FakeTorino, at 12 times level 3's readout cost. The qubits with
    two-qubit gates are matched by VF2PostLayout without the others, which are then assigned to the remaining
    physical qubits by an exact assignment. Gates are unchanged. Kept only if the cost falls and no failed element
    is used. Counts in `FINAL_STATS`.
66. **ACCURACY: Paulis carried through two-qubit Clifford gates (candidate 2026-10-11.c37).** With the device's
    Target (`final_push="auto"`), the Pauli part of a single-qubit run is carried forward through a CZ (X on one
    qubit becomes X on it and Z on the other; a Z joins an rz at no cost) until a run absorbs it at no extra cost.
    GAP-DIAG found the release with 3-7% more non-rz single-qubit gates than level 3 at the same two-qubit count,
    many of them lone X gates between CZs (ham_JW-8 on FakeTorino: 426 x against level 3's 67). Exact, global phase
    included; kept only if the single-qubit gates' summed -ln(1 - error) falls. Two-qubit gates and measurements
    are unchanged. Counts in `FINAL_STATS`.
67. **ACCURACY: several routing tries, kept by ESP (candidate 2026-10-11.c37).** With the device's Target
    (`route_tries="auto"`, 4), the circuit is compiled with seeds seed, ..., seed + 3, each on the map without the
    failed elements (item 61) and on the full map; outputs that use a failed element are discarded and the one of
    the highest ESP (lowest summed -ln(1 - error) over all operations) is returned. Routing's two-qubit count moves
    widely with the seed (tof_10 on FakeMarrakesh: 145 to 201 over six seeds), and routing on the pruned map costs
    2-5% more two-qubit gates than on the full map, whose outputs are often clean. Up to 2 x route_tries compiles;
    one when the first output has more than ROUTE_TRIES_MAX_2Q (5000) two-qubit gates. Without a Target, or with
    route_tries=1, one compile as before. Counts in `TRIES_STATS`.
"""'''),
    ("    placement_max_trials: int = 2_500,\n",
     "    placement_max_trials: int = 2_500,\n"
     '    final_relabel: Union[bool, str] = "auto",\n'
     '    final_push: Union[bool, str] = "auto",\n'
     '    route_tries: Union[int, str] = "auto",\n'),
    ("    _refine_target=None,\n    _cancel_done: bool = False,\n",
     "    _refine_target=None,\n    _final=None,\n    _cancel_done: bool = False,\n"),
    ('''    if placement_refine and target is None:
        raise ValueError("placement_refine=True needs the device `target` (changelog item 33)")
''', '''    if placement_refine and target is None:
        raise ValueError("placement_refine=True needs the device `target` (changelog item 33)")
    if final_relabel == "auto":  # item 65: on with a Target, off without one
        final_relabel = target is not None
    if final_push == "auto":  # item 66: on with a Target, off without one
        final_push = target is not None
    if (final_relabel or final_push) and target is None:
        raise ValueError("final_relabel=True and final_push=True need the device `target` (changelog items 65-66)")
    if route_tries == "auto":  # item 67: 4 with a Target, 1 without one
        route_tries = 4 if target is not None else 1
    if not isinstance(route_tries, int) or isinstance(route_tries, bool) or route_tries < 1:
        raise ValueError('route_tries must be "auto" or an integer >= 1 (changelog item 67)')
    if route_tries > 1 and target is None:
        raise ValueError("route_tries > 1 needs the device `target` (changelog item 67)")
'''),
    ('''                out = _recompile_pruned(args, out, edges, qubits, on_failed_elements)  # item 58
        out = _resynthesise(out, final_resynthesis, target, prune_max_error)
''', '''                out = _recompile_pruned(args, out, edges, qubits, on_failed_elements)  # item 58
        if route_tries > 1:  # item 67
            out = _route_tries(args, out, target, coupling_map, edges, qubits, prune_max_error, route_tries)
        out = _resynthesise(out, final_resynthesis, target, prune_max_error)
'''),
    ("                    _refine_target=target if placement_refine else None, _inner=True)\n",
     "                    _refine_target=target if placement_refine else None, _inner=True,\n"
     "                    _final=((target, prune_max_error, bool(final_push), bool(final_relabel))\n"
     "                            if final_relabel or final_push else None))\n"),
    ("    if permutation is None and not post_routing_resynthesis and _refine_target is None:\n",
     "    if permutation is None and not post_routing_resynthesis and _refine_target is None and _final is None:\n"),
    ('''        if pm.optimization is None:
            pm.optimization = PassManager(refine)
        else:
            pm.optimization.append(refine)
    return pm.run(qc_compressed, callback=callback)''', '''        if pm.optimization is None:
            pm.optimization = PassManager(refine)
        else:
            pm.optimization.append(refine)
    if _final is not None:  # items 65-66, after item 33
        f_target, f_max, f_push, f_relabel = _final
        tail = [_PushPaulis(f_target, f_max)] if f_push else []
        if f_relabel:
            tail += [_FinalRelabel(f_target, f_max, placement_call_limit, placement_max_trials),
                     ConditionalController(ApplyLayout(), condition=_relabel_found)]
        if pm.optimization is None:
            pm.optimization = PassManager(tail)
        else:
            pm.optimization.append(tail)
    return pm.run(qc_compressed, callback=callback)'''),
    ("def _failed_elements(target, max_error: float,", code + "def _failed_elements(target, max_error: float,"),
]:
    assert s.count(old) == 1, old[:80]
    s = s.replace(old, new)
open(dst, "w", encoding="utf-8", newline="\n").write(s)
print("written", dst)
