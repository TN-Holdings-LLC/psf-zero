"""Builds prototype 2026-10-14.c52 from prototype 2026-10-14.c51 by exact substitutions:
  - item 89: in item 73's reader, when every qubit is measured at the end, a diagonal Clifford left before the
    measurements is dropped (it cannot change the outcome distribution);
  - item 90: the estimate-based parts of the default call that the workplace's ablation (N5) found without effect are
    off by default (items 33/60 re-placement by errors, item 64 routing at level 3, item 78 re-placement at Qiskit's
    routing stage), and item 75's deterministic re-placement is scored by the device's gate durations instead of its
    errors.
    python make_c52.py <c51 psf_compile.py> <output psf_compile.py>"""
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src, encoding="utf-8").read()


def sub(old, new, count=1):
    global s
    n = s.count(old)
    if n != count:
        raise SystemExit(f"expected {count} occurrence(s), found {n}: {old[:80]!r}")
    s = s.replace(old, new)


sub("VERSION: 2026-10-14.c51 -- prototype (from c49 as c50, item 86 revised; c43 = release 2026-10-11.4): items 75 revised, 77-80, 82-86",
    "VERSION: 2026-10-14.c52 -- prototype (from c51; c43 = release 2026-10-11.4): items 75 revised, 77-80, 82-86, 89, 90")
sub('VERSION = "2026-10-14.c51"  # prototype: item 86 revised',
    'VERSION = "2026-10-14.c52"  # prototype: items 89 (diagonal Clifford before measurement dropped) and 90 (estimate-based defaults off, item 75 scored by duration) + item 86 revised')
sub('''    rzz does not run.
"""''', '''    rzz does not run.
89. **DURATION: a diagonal Clifford left before the measurements is dropped (prototype c52).** Item 73 reads a
    circuit written as gates by carrying its Clifford gates to the end; it applied only when the Clifford left at the
    end acts on each qubit alone (or permutes them, item 76). A layered classifier with an odd number of CZ rings
    leaves one ring there and was not read (workplace N3, Addendum 482: 32 of 32 odd cells), so it went by the usual
    path. When every qubit is measured at the end, only the images of the Z operators matter: if each C^dagger Z_q C
    is a single-qubit Pauli, the Clifford is a single-qubit Clifford L times a diagonal Clifford D (C L^dagger
    commutes with every Z), and D before measurements leaves every outcome probability unchanged. The reader then
    keeps L and drops D. The output reproduces the input's outcome distribution, not its unitary; it applies only
    when every qubit is measured at the end. `DROP_DIAGONAL_BEFORE_MEASURE = False` restores c51. Counts in
    `STEINER_STATS` ("diag_dropped").
90. **NO ESTIMATE AS JUDGE: estimate-based defaults off; item 75 scored by duration (prototype c52).** The
    workplace's ablation (N5, Addendum 482) removed each remaining estimate-based part of c49's default call on
    C48-VAL's 16 tests (FakeFez, FakeTorino): items 33/60 (re-placement by the Target's errors) changed 3 of 32
    outputs and never lengthened one; item 64 (routing at level 3) 2 of 32, equal CZ and duration; item 78
    (re-placement at Qiskit's routing stage) 2 of 32, at most +0.3%. With `MAP_FREE_DEFAULTS = True` (default):
    `placement_refine="auto"` is off, `routing_optimization_level="auto"` is 1, and item 80's level-3 candidate runs
    without item 78's routing-stage re-placement. Item 75's re-placement (in Qiskit's passes built here, item 33 when
    asked for, item 68's relabel) scores placements by the summed duration of the placed operations on the Target
    (`PLACE_SCORE = "duration"`); on a device whose gates all have one length per type every placement ties and the
    current one is kept, and where lengths differ (FakeTorino) the shorter gates are taken. A failed element still
    excludes. `MAP_FREE_DEFAULTS = False` and `PLACE_SCORE = "esp"` restore c51.
"""''')

sub('''STEINER_STATS = {"not_applicable": 0, "not_commuting": 0, "used": 0, "fallback": 0, "too_large": 0, "chosen": 0,
                 "kept_usual": 0, "gate_level": 0, "permuted": 0, "mirror_tried": 0,
                 "mirror_kept": 0, "reordered": 0, "reorder_kept_order": 0}  # changelog items 68-76, 82, 83''',
    '''STEINER_STATS = {"not_applicable": 0, "not_commuting": 0, "used": 0, "fallback": 0, "too_large": 0, "chosen": 0,
                 "kept_usual": 0, "gate_level": 0, "permuted": 0, "mirror_tried": 0,
                 "mirror_kept": 0, "reordered": 0, "reorder_kept_order": 0,
                 "diag_dropped": 0}  # changelog items 68-76, 82, 83, 89
DROP_DIAGONAL_BEFORE_MEASURE = True  # item 89
MAP_FREE_DEFAULTS = True  # item 90: items 33/60, 64 and 78 off by default
PLACE_SCORE = "duration"  # item 90: item 75 scored by the placed operations' durations ("esp": c51)''')

# item 89 in item 73's reader
sub('''    suffix = []
    table = None
    perm = list(range(n))  # item 76: perm[q] is the wire whose state logical qubit q holds at the end''',
    '''    if DROP_DIAGONAL_BEFORE_MEASURE and meas and {q for q, _ in meas} == set(range(n)):  # item 89
        dropped = False
        for q in range(n):
            xi, zi = frame[q]
            bz, bx = zi[0] | zi[1], xi[0] | xi[1]
            if not bz or bz & (bz - 1):
                continue  # Z's image is not a single-qubit Pauli: no single-qubit L exists
            w = bz.bit_length() - 1
            if bx and not bx & (bx - 1) and bx == bz:
                continue  # already a single-qubit Clifford on that wire
            # an X image on wire w that anticommutes with Z's image: Z_w if Z's image has an X part, else X_w
            frame[q] = [(0, 1 << w, 0) if zi[0] & bz else (1 << w, 0, 0), zi]
            dropped = True
        if dropped:
            STEINER_STATS["diag_dropped"] += 1
    suffix = []
    table = None
    perm = list(range(n))  # item 76: perm[q] is the wire whose state logical qubit q holds at the end''')

# item 90: defaults
sub('''    if routing_optimization_level == "auto":  # item 64: level 3 with the device's Target, level 1 without
        routing_optimization_level = 3 if target is not None else 1''',
    '''    if routing_optimization_level == "auto":  # item 64: level 3 with the device's Target, level 1 without
        routing_optimization_level = 3 if target is not None and not MAP_FREE_DEFAULTS else 1  # item 90''')
sub('''    if placement_refine == "auto":  # item 60: on with a Target, off without one
        placement_refine = target is not None''',
    '''    if placement_refine == "auto":  # item 60: on with a Target, off without one
        placement_refine = target is not None and not MAP_FREE_DEFAULTS  # item 90''')
sub('''        with _DeterministicVF2(target, routing_stage=True):  # item 78''',
    '''        with _DeterministicVF2(target, routing_stage=not MAP_FREE_DEFAULTS):  # item 78 (off by item 90)''')
# item 90: item 75 scored by duration
sub('''                table[None if qargs is None else tuple(qargs)] = math.inf if failed else -math.log1p(-max(e, 0.0))''',
    '''                if PLACE_SCORE == "duration":  # item 90: the placed operation's duration, not its error
                    d = props.duration if props is not None and props.duration is not None else 0.0
                    table[None if qargs is None else tuple(qargs)] = math.inf if failed else d
                else:
                    table[None if qargs is None else tuple(qargs)] = math.inf if failed else -math.log1p(-max(e, 0.0))''')
sub('''                    e = props.error
                    v = -math.log1p(-e) if e < 1.0 else 1e3
                    own[qargs[0]] = own.get(qargs[0], 0.0) + v''',
    '''                    e = props.error
                    if PLACE_SCORE == "duration":  # item 90: search order by the qubit's own durations
                        v = (props.duration or 0.0) if e < 1.0 else 1e3
                    else:
                        v = -math.log1p(-e) if e < 1.0 else 1e3
                    own[qargs[0]] = own.get(qargs[0], 0.0) + v''')
open(dst, "w", encoding="utf-8").write(s)
print("written", dst)
