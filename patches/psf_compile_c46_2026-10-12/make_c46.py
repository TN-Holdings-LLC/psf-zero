"""Builds prototype 2026-10-12.c46 from prototype 2026-10-12.c45 by exact substitutions:
  - item 81: item 73 also reads measured circuits whose final Clifford acts on several qubits at once, when every
    measured qubit's Z, pulled back through it, is a product of Z's (a parity of the wires): that Clifford is then
    replaced by a CNOT network, synthesised on the device's coupling graph along error-weighted Steiner trees (RowCol),
    that puts each measured parity on its own wire; the measured distribution is the input's.
    python make_c46.py <c45 psf_compile.py> <lin_synth.py> <output psf_compile.py>"""
import sys

src, lin, dst = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(src, encoding="utf-8").read()
lin_code = open(lin, encoding="utf-8").read()


def sub(old, new, count=1):
    global s
    n = s.count(old)
    if n != count:
        raise SystemExit(f"expected {count} occurrence(s), found {n}: {old[:80]!r}")
    s = s.replace(old, new)


sub('''STEINER_STATS = {"not_applicable": 0, "not_commuting": 0, "used": 0, "fallback": 0, "too_large": 0, "chosen": 0,
                 "kept_usual": 0, "gate_level": 0, "permuted": 0}  # changelog items 68-76''',
    '''STEINER_STATS = {"not_applicable": 0, "not_commuting": 0, "used": 0, "fallback": 0, "too_large": 0, "chosen": 0,
                 "kept_usual": 0, "gate_level": 0, "permuted": 0, "linear": 0,
                 "linear_unconnected": 0}  # changelog items 68-76, 81''')

# the parser: a final Clifford on several qubits at once -> item 81's linear suffix when the measured Z's allow it
sub('''                return None  # the Clifford left at the end acts on more than one qubit here''',
    '''                return _st_linear_suffix(prefix, meas, terms, phase, frame, n)  # item 81''')
sub('''            elif both != 1 << wire:
                return None''', '''            elif both != 1 << wire:
                return _st_linear_suffix(prefix, meas, terms, phase, frame, n)  # item 81''')

sub('''def _st_compile(qc, target, max_error, seed, call_limit, max_trials):''', lin_code + '''

def _st_linear_suffix(prefix, meas, terms, phase, frame, n):
    """Item 81: the parse result of a measured circuit whose final Clifford C acts on several qubits at once, when for
    every measured qubit q, C^dagger Z_q C is +-(a product of Z's): the outcome of q is then the parity of those wires'
    values (with the sign as a flip), and C itself need not be run. Suffix ("linear", [(q, clbit, wire bitmask,
    flip)]); None otherwise (the circuit is not read)."""
    if not meas:
        return None
    spec = []
    for q, cb in meas:
        x, z, k = frame[q][1]
        if x or not z or k % 2:
            return None
        spec.append((q, cb, z, (k // 2) % 2))
    STEINER_STATS["gate_level"] += 1
    STEINER_STATS["linear"] += 1
    return prefix, meas, terms, phase, ("linear", spec), None


def _st_compile(qc, target, max_error, seed, call_limit, max_trials):''')

sub('''    for op, q in suffix:  # item 73: the Clifford gates left at the end, at the final places
        phys.append(op, [m.pi[q]])
    for q, cb in meas:
        phys.measure(m.pi[perm[q]], cb)  # item 76: logical q's state is on wire perm[q]'s place
''', '''    if isinstance(suffix, tuple):  # item 81: a CNOT network puts each measured parity on its own wire
        masks = []
        for _, _, mask, flip in suffix[1]:
            if (mask, flip) not in masks:
                masks.append((mask, flip))
        got = _lin_targets(n, masks)
        adj, _ = _st_working_graph(target)
        wedge = {}
        g2 = next(g for g in ("cz", "ecr", "cx") if g in target.operation_names)
        for qa, pr in target[g2].items():
            if qa is not None and pr is not None and pr.error is not None:
                k = (min(qa), max(qa))
                wedge[k] = max(wedge.get(k, 0.0), -math.log(1.0 - min(pr.error, 0.999)) + 1e-9)
        cnots = None if got is None else _lin_synth(got[0], [m.pi[j] for j in range(n)], adj, wedge)
        if cnots is None:
            STEINER_STATS["linear_unconnected"] += 1
            STEINER_STATS["fallback"] += 1
            return None
        for c, t in cnots:
            phys.cx(c, t)
        place = got[1]
        for k, (mask, flip) in enumerate(masks):
            if flip:
                phys.x(m.pi[place[k]])
        for q, cb, mask, flip in suffix[1]:
            phys.measure(m.pi[place[masks.index((mask, flip))]], cb)
    else:
        for op, q in suffix:  # item 73: the Clifford gates left at the end, at the final places
            phys.append(op, [m.pi[q]])
        for q, cb in meas:
            phys.measure(m.pi[perm[q]], cb)  # item 76: logical q's state is on wire perm[q]'s place
''')

sub('VERSION: 2026-10-12.c45 -- prototype (from c44; c43 = release 2026-10-11.4): items 75 revised, 77-80',
    'VERSION: 2026-10-12.c46 -- prototype (from c45; c43 = release 2026-10-11.4): items 75 revised, 77-81')
sub('VERSION = "2026-10-12.c45"  # prototype:', 'VERSION = "2026-10-12.c46"  # prototype: item 81 (final linear part) +')

OLD_TAIL = '''    diagnosis). Item 36's opt-in `compare_level3` (by `excitation_cost`, small circuits) is unchanged.
"""'''
NEW_TAIL = '''    diagnosis). Item 36's opt-in `compare_level3` (by `excitation_cost`, small circuits) is unchanged.
81. **ACCURACY: the final linear part of measured circuits (prototype c46).** Item 73 carries Clifford gates to the end
    of the circuit and required what is left there to act on each qubit alone (item 76: or to permute them). Of
    Benchpress's 519 QASM files, 99 were not read only because of that, among them the arithmetic circuits (adders,
    GF(2^m) multipliers, Feynman's reversible benchmarks) and W states. When the circuit ends with measurements and,
    for every measured qubit q, the final Clifford C maps Z_q back to +-(a product of Z's) (44 of the 99), the
    outcome of q is a parity of the wires' values before C. C is then not run: a CNOT network that puts each measured
    parity on its own wire (an X where the sign is -1) is synthesised on the device's coupling graph by RowCol
    elimination (Wu et al. 2023; Steiner-Gauss, Kissinger and Meijer-van de Griend 2019) along Steiner trees weighted
    by the couplers' -ln(1 - error), only on the qubits that hold the circuit (None if they are not connected), with
    the free choice of which wire carries which parity made by a bipartite matching within each parity's support. The
    output gives the input's measured distribution; it does not implement the input's unitary (the unmeasured part of
    the final state differs), as Qiskit's level 3 also only keeps the distribution when it removes gates before
    measurements. Item 69's choice by ESP applies as before. Counts in `STEINER_STATS` ("linear").
"""'''
sub(OLD_TAIL, NEW_TAIL)
open(dst, "w", encoding="utf-8").write(s)
print("wrote", dst)
