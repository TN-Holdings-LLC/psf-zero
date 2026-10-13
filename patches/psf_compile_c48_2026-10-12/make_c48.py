"""Builds prototype 2026-10-12.c48 from prototype 2026-10-12.c47 by exact substitutions:
  - item 83: STEINER6's output is re-emitted block by block (a gadget per term, a SWAP) by list scheduling, where
    blocks that commute as unitaries may change places; kept when the schedule in CX units is shorter. The gates are
    the same; only their order changes.
    python make_c48.py <c47 psf_compile.py> <sched_code.py> <output psf_compile.py>"""
import sys

src, code, dst = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(src, encoding="utf-8").read()
sched = open(code, encoding="utf-8").read()


def sub(old, new, count=1):
    global s
    n = s.count(old)
    if n != count:
        raise SystemExit(f"expected {count} occurrence(s), found {n}: {old[:80]!r}")
    s = s.replace(old, new)


sub("VERSION: 2026-10-12.c47 -- prototype (from c45; c43 = release 2026-10-11.4): items 75 revised, 77-80, 82",
    "VERSION: 2026-10-12.c48 -- prototype (from c47; c43 = release 2026-10-11.4): items 75 revised, 77-80, 82, 83")
sub('VERSION = "2026-10-12.c47"  # prototype: item 82 (mirrored layers)',
    'VERSION = "2026-10-12.c48"  # prototype: item 83 (time-aware block order) + item 82 (mirrored layers)')

sub('''    summed -ln(1 - error) on the device is kept (STEINER6's own on a tie). Counts in `STEINER_STATS` ("mirror_tried",
    "mirror_kept").
"""''', '''    summed -ln(1 - error) on the device is kept (STEINER6's own on a tie). Counts in `STEINER_STATS` ("mirror_tried",
    "mirror_kept").
83. **DURATION: STEINER6's blocks in a time-aware order (prototype c48).** STEINER6's mover emits one block per term (a
    Steiner-tree gadget, exactly exp(-i t c P) on its terminals and the identity on its Steiner qubits) and one per
    SWAP, and its emission order fixed the schedule: terms that commute but share qubits ran one after another in the
    order given (in a transverse-field Ising Trotter circuit, a staircase over the chain). On the real device
    (Addendum 455) the measure that ranked PSF-Zero against Qiskit was the duration, not the gate count. The mover now
    records its blocks (a SWAP that joins the last gadget on its pair stays fused to it) and `_st_reorder` re-emits
    them by list scheduling: a block is ready once every earlier block that shares a qubit with it and does not commute
    with it as a unitary has been emitted, and the ready block that can start earliest goes next (ties: the longer
    chain of dependent work after it). Every non-commuting pair keeps its order, so the action is unchanged, and so are
    the gates. The new order is kept when its as-soon-as-possible schedule, in CX units (CX 1, SWAP 3, rz 0, other
    single-qubit gates 0.3), is shorter. `ST_SCHEDULE = False` turns it off. Counts in `STEINER_STATS` ("reordered",
    "reorder_kept_order").
"""''')

sub('''                 "kept_usual": 0, "gate_level": 0, "permuted": 0, "mirror_tried": 0,
                 "mirror_kept": 0}  # changelog items 68-76, 82''',
    '''                 "kept_usual": 0, "gate_level": 0, "permuted": 0, "mirror_tried": 0,
                 "mirror_kept": 0, "reordered": 0, "reorder_kept_order": 0}  # changelog items 68-76, 82, 83''')

# block recording in the movers
sub('''        self.order = []
        self.last2q = {}  # physical qubit -> the pair of the last two-qubit gate on it (single-qubit gates join blocks)
''', '''        self.order = []
        self.last2q = {}  # physical qubit -> the pair of the last two-qubit gate on it (single-qubit gates join blocks)
        self.st_blocks, self._lastb = [], {}  # item 83: the emitted blocks; physical qubit -> its last block
''')
sub('''    def run_term(self, j):
        lb, c, t = self.triples[j]
        ex = self.excess(j)
''', '''    def run_term(self, j):
        lb, c, t = self.triples[j]
        first_op = len(self.ops)  # item 83
        pauli = {self.pi[q]: ch for q, ch in enumerate(reversed(lb)) if ch != "I"}
        ex = self.excess(j)
''')
sub('''        if ex > 0:
            self.stats["bridged_terms"] += 1
            self.stats["bridge_excess"] += ex
        self.stats["terms"] += 1
        self.order.append(j)
''', '''        if ex > 0:
            self.stats["bridged_terms"] += 1
            self.stats["bridge_excess"] += ex
        self.stats["terms"] += 1
        self.order.append(j)
        _st_record_term(self, first_op, pauli)  # item 83
''')
sub('''    def swap(self, p, q):
        if self.mergeable(p, q):
            self.stats["merged_swaps"] += 1
        self.ops.append(("swap", (p, q), ()))
''', '''    def swap(self, p, q):
        if self.mergeable(p, q):
            self.stats["merged_swaps"] += 1
        self.ops.append(("swap", (p, q), ()))
        _st_record_swap(self, len(self.ops) - 1, p, q)  # item 83
''')
# _StMirror
sub('''        self.ops, self.gphase = [], 0.0
        self.stats = dict(swaps=0, merged_swaps=0, bridged_terms=0, bridge_excess=0, terms=0)

    def _term(self, j):
        lb, c, t = self.triples[j]
''', '''        self.ops, self.gphase = [], 0.0
        self.stats = dict(swaps=0, merged_swaps=0, bridged_terms=0, bridge_excess=0, terms=0)
        self.st_blocks, self._lastb = [], {}  # item 83

    def _term(self, j):
        lb, c, t = self.triples[j]
        first_op = len(self.ops)  # item 83
        pauli = {self.pi[q]: ch for q, ch in enumerate(reversed(lb)) if ch != "I"}
''')
sub('''        self.stats["terms"] += 1
        if ex > 0:
            self.stats["bridged_terms"] += 1

    def _swap(self, p, q):
        self.ops.append(("swap", (p, q), ()))
''', '''        self.stats["terms"] += 1
        if ex > 0:
            self.stats["bridged_terms"] += 1
        _st_record_term(self, first_op, pauli)  # item 83

    def _swap(self, p, q):
        self.ops.append(("swap", (p, q), ()))
        _st_record_swap(self, len(self.ops) - 1, p, q)  # item 83
''')
# the scheduling code, after the movers
sub('''_ST_FAILED = 0.5  # set from prune_max_error by _st_compile
''', sched.lstrip("\n") + '''

_ST_FAILED = 0.5  # set from prune_max_error by _st_compile
''')
# finish(): the reordered ops
sub('''        at = list(range(N))  # at[p]: the physical qubit whose state is at p (SWAPs move idle qubits' states too)
        for name, q, prm in m.ops:
''', '''        at = list(range(N))  # at[p]: the physical qubit whose state is at p (SWAPs move idle qubits' states too)
        ops = m.ops
        if ST_SCHEDULE:  # item 83
            new = _st_reorder(m.ops, getattr(m, "st_blocks", None))
            if new is not None and _st_asap(new) < _st_asap(m.ops) - 1e-9:
                ops = new
                STEINER_STATS["reordered"] += 1
            else:
                STEINER_STATS["reorder_kept_order"] += 1
        for name, q, prm in ops:
''')
open(dst, "w", encoding="utf-8").write(s)
print("wrote", dst)
