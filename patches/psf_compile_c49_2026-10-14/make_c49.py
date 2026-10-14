"""Builds prototype 2026-10-14.c49 from prototype 2026-10-12.c48 by exact substitutions:
  - item 84: every choice between two compiled circuits (items 69, 74, 80, 82 and STEINER6's choice among its
    constructions) by the as-soon-as-possible duration on the device's reported gate durations, then the two-qubit
    count; no error value of the device decides any more (a failed element still excludes);
  - item 85: with the device given, its failed qubits are cut out of the device before compiling (the circuit is
    compiled on the remaining qubits and written back on the device's own indices).
    python make_c49.py <c48 psf_compile.py> <c49_code.py> <output psf_compile.py>"""
import sys

src, code, dst = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(src, encoding="utf-8").read()
c49 = open(code, encoding="utf-8").read()


def sub(old, new, count=1, text=None):
    global s
    t = s if text is None else text
    n = t.count(old)
    if n != count:
        raise SystemExit(f"expected {count} occurrence(s), found {n}: {old[:80]!r}")
    t = t.replace(old, new)
    if text is None:
        s = t
    return t


sub("VERSION: 2026-10-12.c48 -- prototype (from c47; c43 = release 2026-10-11.4): items 75 revised, 77-80, 82, 83",
    "VERSION: 2026-10-14.c49 -- prototype (from c48; c43 = release 2026-10-11.4): items 75 revised, 77-80, 82-85")
sub('VERSION = "2026-10-12.c48"  # prototype: item 83 (time-aware block order)',
    'VERSION = "2026-10-14.c49"  # prototype: items 84 (choices by duration) and 85 (failed qubits cut first) + item 83 (time-aware block order)')

sub('''    single-qubit gates 0.3), is shorter. `ST_SCHEDULE = False` turns it off. Counts in `STEINER_STATS` ("reordered",
    "reorder_kept_order").
"""''', '''    single-qubit gates 0.3), is shorter. `ST_SCHEDULE = False` turns it off. Counts in `STEINER_STATS` ("reordered",
    "reorder_kept_order").
84. **DURATION: choices between compiled circuits by duration, not by the device's error table (prototype c49).**
    On two IBM devices (Addenda 455, 461) the summed -ln(1 - error) of a circuit's operations (ESP) did not pick the
    circuit the device ran better, and the same estimate ranked differently from one device and day to the next; the
    as-soon-as-possible duration picked the winner on 7 of 7 of HW2's pairs (after the fact). Every place where c48
    kept one of two compiled circuits by ESP now keeps the shorter one (`_better`): item 69 (STEINER6 against the
    usual path), item 74 (the near-failed alternative), item 80 (Qiskit level 3 as a candidate), item 82 (the
    mirrored layers) and STEINER6's choice among its constructions (by its own schedule in CX units after item 83's
    order, then its CX count, in place of its error-weighted estimate). Duration is the makespan on the gate
    durations the Target reports (rz 0 when not reported); when a duration is missing, both circuits are measured in
    item 83's CX units. Equal durations (within 1e-12 s): fewer two-qubit gates; a full tie keeps the circuit c48
    would have kept on a tie. A circuit with an operation the device does not offer, or reported at error >= 1, still
    loses. Error values still shape the candidates themselves (placement by errors, items 60 and 75; STEINER6's
    error-distance construction; Qiskit level 3's own passes); they no longer decide between them. `SELECT_BY = "esp"`
    restores c48. Counts in `SELECT_STATS`.
85. **SAFETY: the device's failed qubits are cut out before compiling (prototype c49).** Item 61 compiled on a
    coupling map without the failed couplers, but kept every qubit, failed ones included, as isolated points: a
    qubit of the circuit with no two-qubit gate could then be placed on a failed qubit, and item 58 refused the
    output (SCALE1, Addendum 475: 144-qubit circuits on ibm_marrakesh). With the device given (`backend=` or
    `target=`) and no `coupling_map`, `initial_layout` or layout error tables of the caller's, a Target without the
    failed qubits (sx or readout error >= `prune_max_error`, item 58's rule) is built (`_working_target`), the
    circuit is compiled on it, and the output is written back on the device's own qubit indices with its layout
    (`_embed_working`); the cut qubits carry ancillas that nothing touches. When fewer working qubits remain than the
    circuit has, "raise" raises `FailedElementsError` and "keep" compiles as c48 did. A device without failed qubits
    is compiled exactly as by c48. `CUT_FAILED_QUBITS = False` restores c48. Counts in `CUT_STATS`.
"""''')

# the new functions, after item 69's cost (they use _st_cost, _count_2q, _ST_DUR and math)
sub('''    return total


def _st_evolution(qc):''', '''    return total
''' + c49 + '''

def _st_evolution(qc):''')

# item 84 in item 80
sub("    if not _st_cost(l3, target) < _st_cost(out, target):\n        L3C_STATS[\"kept_psf\"] += 1",
    "    if not _better(l3, out, target):  # item 84\n        L3C_STATS[\"kept_psf\"] += 1")

# item 84 in item 74 (c43's revision only; c42's function stays as recorded)
head, sep, tail = s.partition("def _near_alternative_c42(")
if not sep:
    raise SystemExit("c42 record function not found")
head = sub('''        NEAR_STATS["no_alternative"] += 1
        return out
    if _st_cost(alt, target) < _st_cost(out, target):''', '''        NEAR_STATS["no_alternative"] += 1
        return out
    if _better(alt, out, target):  # item 84''', text=head)
s = head + sep + tail

# item 84 in item 82
sub("(final is None or _st_cost(outs[1], target) < _st_cost(final, target)):",
    "(final is None or _better(outs[1], final, target)):  # item 84")

# item 84 in item 69
sub("            if usual is None or _st_cost(st, target) < _st_cost(usual, target):",
    "            if usual is None or _better(st, usual, target):  # item 84")

# item 84 in STEINER6's choice among its constructions
sub('''            est = 0.0
            for name, q, _ in m.ops:
                if len(q) == 2:
                    est += wedge.get((min(q), max(q)), w_med) * (1 if name == "cx" else 3)
            est -= 2 * w_med * m.stats["merged_swaps"]
            est += sum(readout.get(m.pi[q], 0.0) for q, _ in meas)
            cands.append((est, mode, label, m, pi0))''', '''            if SELECT_BY == "esp":
                est = 0.0
                for name, q, _ in m.ops:
                    if len(q) == 2:
                        est += wedge.get((min(q), max(q)), w_med) * (1 if name == "cx" else 3)
                est -= 2 * w_med * m.stats["merged_swaps"]
                est += sum(readout.get(m.pi[q], 0.0) for q, _ in meas)
            else:  # item 84: the construction's own schedule (item 83's order when shorter), then its CX count
                ops = m.ops
                if ST_SCHEDULE:
                    new = _st_reorder(m.ops, getattr(m, "st_blocks", None))
                    if new is not None and _st_asap(new) < _st_asap(ops) - 1e-9:
                        ops = new
                ncx = sum(1 if name == "cx" else 3 for name, q, _ in m.ops if len(q) == 2)
                est = (_st_asap(ops), ncx - 2 * m.stats["merged_swaps"])
            cands.append((est, mode, label, m, pi0))''')
sub("distances, two starts each) and STEINER5's embeddings, each run by _StMover2; the lowest estimated -ln ESP kept.",
    "distances, two starts each) and STEINER5's embeddings, each run by _StMover2; the shortest schedule kept (item 84;\n"
    "    the lowest estimated -ln ESP with SELECT_BY = \"esp\").")

# item 85 at the entry of compile_for_hardware
sub('''    if backend is not None:  # item 58: the device, from which the Target, map and basis follow unless given
        if target is None:
            target = backend.target
    if basis_gates is None and target is not None:  # item 79: also when the Target is given alone''',
    '''    if backend is not None:  # item 58: the device, from which the Target, map and basis follow unless given
        if target is None:
            target = backend.target
    if (CUT_FAILED_QUBITS and not _inner and target is not None and coupling_map is None and initial_layout is None
            and _refine_target is None and layout_edge_errors is None and layout_qubit_errors is None):  # item 85
        out85 = _cut_failed_qubits(dict(locals()))
        if out85 is not None:
            return out85
        del out85  # item 50 copies this call's locals as its arguments
    if basis_gates is None and target is not None:  # item 79: also when the Target is given alone''')

open(dst, "w", encoding="utf-8").write(s)
print("written", dst)
