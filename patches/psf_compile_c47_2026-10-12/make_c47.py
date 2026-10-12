"""Builds prototype 2026-10-12.c47 from prototype 2026-10-12.c45 by exact substitutions:
  - item 82: repeated layers of the same commuting set (QAOA's cost layers, Trotter steps of Hamiltonians whose
    multi-qubit terms commute) are compiled by scheduling the first layer once and replaying it forwards and backwards
    (`_StMirror`); the result is compared with STEINER6's own on the device's errors and the better kept.
    python make_c47.py <c45 psf_compile.py> <mirror.py> <output psf_compile.py>"""
import sys

src, mir, dst = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(src, encoding="utf-8").read()
mirror_code = open(mir, encoding="utf-8").read()


def sub(old, new, count=1):
    global s
    n = s.count(old)
    if n != count:
        raise SystemExit(f"expected {count} occurrence(s), found {n}: {old[:80]!r}")
    s = s.replace(old, new)


sub('''STEINER_STATS = {"not_applicable": 0, "not_commuting": 0, "used": 0, "fallback": 0, "too_large": 0, "chosen": 0,
                 "kept_usual": 0, "gate_level": 0, "permuted": 0}  # changelog items 68-76''',
    '''STEINER_STATS = {"not_applicable": 0, "not_commuting": 0, "used": 0, "fallback": 0, "too_large": 0, "chosen": 0,
                 "kept_usual": 0, "gate_level": 0, "permuted": 0, "mirror_tried": 0,
                 "mirror_kept": 0}  # changelog items 68-76, 82''')

# _st_compile: the construction's remainder becomes finish(m, pi0); item 82's construction is a second candidate
a = s.index("def _st_compile(qc, target, max_error, seed, call_limit, max_trials):")
b = s.index("def _failed_elements(target, max_error: float")
body = s[a:b]
head, tail = body.split("    m, pi0 = got\n", 1)
end_old = '''    STEINER_STATS["used"] += 1
    return final
'''
assert tail.count(end_old) == 1
tail = tail.replace(end_old, "    return final\n")
fail_old = '''    if _uses_failed(final, edges, bad):
        STEINER_STATS["fallback"] += 1
        return None
'''
assert tail.count(fail_old) == 1
tail = tail.replace(fail_old, '''    if _uses_failed(final, edges, bad):
        return None
''')
inner = "".join(("    " + ln if ln.strip() else ln) for ln in tail.splitlines(True))
new_body = head + '''    def finish(m, pi0):  # item 82: the circuit of one construction, or None if it uses a failed element
''' + inner.rstrip("\n") + '''

    outs = [finish(*got)]
    blocks = _st_mirror_blocks(triples)  # item 82
    if blocks is not None:
        STEINER_STATS["mirror_tried"] += 1
        mir = _st_mirror_construct(n, triples, target, blocks)
        if mir is not None:
            outs.append(finish(*mir))
    if outs[0] is None and (len(outs) == 1 or outs[1] is None):
        STEINER_STATS["fallback"] += 1
        return None
    final = outs[0]
    if len(outs) == 2 and outs[1] is not None and (final is None or _st_cost(outs[1], target) < _st_cost(final, target)):
        final = outs[1]
        STEINER_STATS["mirror_kept"] += 1
    STEINER_STATS["used"] += 1
    return final


'''
s = s[:a] + mirror_code.rstrip("\n") + '''


def _st_mirror_construct(n, triples, target, blocks):
    """Item 82: (mover-like, initial placement) of the mirrored layers, or None."""
    c1 = next(idx for kind, idx in blocks if kind == "C")
    got = _st_construct(n, [triples[j] for j in c1], target, [])
    if got is None:
        return None
    first, pi0 = got
    adj, _ = _st_working_graph(target)
    dist, par = _st_bfs_all(adj)
    err = None
    if first.err is not None:
        err = _st_error_distances(adj, target)
    return _StMirror(n, triples, adj, dist, par, pi0, err, blocks).build(first, c1), pi0


''' + new_body + s[b:]

sub('VERSION: 2026-10-12.c45 -- prototype (from c44; c43 = release 2026-10-11.4): items 75 revised, 77-80',
    'VERSION: 2026-10-12.c47 -- prototype (from c45; c43 = release 2026-10-11.4): items 75 revised, 77-80, 82')
sub('VERSION = "2026-10-12.c45"  # prototype:', 'VERSION = "2026-10-12.c47"  # prototype: item 82 (mirrored layers) +')
OLD_TAIL = '''    diagnosis). Item 36's opt-in `compare_level3` (by `excitation_cost`, small circuits) is unchanged.
"""'''
NEW_TAIL = '''    diagnosis). Item 36's opt-in `compare_level3` (by `excitation_cost`, small circuits) is unchanged.
82. **ACCURACY: repeated layers of commuting terms, mirrored (prototype c47).** QAOA's cost layers, and the Trotter steps
    of a Hamiltonian whose multi-qubit terms commute (an Ising model in a transverse field, for example), repeat one
    set of commuting terms p times, with single-qubit terms between. STEINER6 scheduled all the terms at once, and its
    later layers cost more than the first: on qaoa_barabasi_albert_N40 (three layers, FakeFez) 128, 134 and 193
    SWAPs, as the placement drifted. When the terms are p >= 2 such layers -- the k-th occurrence of every label in
    layer k, every label p times, the first layer's multi-qubit terms commuting, and the order C_1 L_1 C_2 L_2 ...
    keeping every anticommuting pair in its original order (so the reordering is exact) -- `_StMirror` schedules C_1
    once with STEINER6's mover, from its own initial placement, and replays its events (terms and SWAPs) forwards in
    odd layers and backwards in even ones: every layer costs what the first does, and every second one ends where the
    first began. The single-qubit terms run where their qubits are. Both circuits are built and the one with the lower
    summed -ln(1 - error) on the device is kept (STEINER6's own on a tie). Counts in `STEINER_STATS` ("mirror_tried",
    "mirror_kept").
"""'''
sub(OLD_TAIL, NEW_TAIL)
open(dst, "w", encoding="utf-8").write(s)
print("wrote", dst)
