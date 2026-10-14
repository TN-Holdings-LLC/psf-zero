"""psf_ai_compile.py -- PSF-Zero front end for circuits written by a language model, prototype 2026-10-14.a15.

a15 changes what happens at or below SMALL_MAX_QUBITS when the device's Target is given. Without a Target, and above
SMALL_MAX_QUBITS, it is a14 (patches/psf_ai_compile_a14_2026-10-13), which is a12 there.

Why. a2-a14 chose among many compiles (several routing seeds per starting point, layouts borrowed from Qiskit level 3,
re-placements scored over up to 5,000 placements) by an estimate built from the device's reported errors and T1/T2
(a2's ESP, a4's state-aware estimate, a14's idle term). On two IBM devices those estimates did not pick the circuit
the device ran better, and the same estimate ranked differently from one device and day to the next (Addenda 455,
461, 476). The compiler's own constructions won where they made the circuit shorter (STEINER6, item 83).

a15 (with a Target):
  23. Each deterministic starting point of a0-a1 (the input; its commutative cancellation; gates on 3+ qubits split;
      2-qubit `unitary` gates expanded to CX form) is compiled ONCE by psf_compile's default call with the device
      (`target=`, first seed): its constructions (STEINER6 and item 83's time-aware order where the input reads as
      Pauli rotations), its choice by duration (item 84) and its exclusion of failed qubits (item 85) apply.
  24. a0's polish (commutative cancellation and re-synthesis of 2-qubit blocks, exact) is applied to each, and both
      the compiled and the polished circuit are candidates.
  25. The candidate returned is chosen by psf_compile's `_better` (item 84): the shorter as-soon-as-possible
      schedule on the device's reported gate durations, then fewer two-qubit gates; on a full tie the earlier
      candidate (the input's own compile first). No routing seeds are searched, no placement is re-scored, and no
      error value or T1/T2 decides.
  26. A candidate that uses an element the device reports failed is never returned (a14's item 22 kept); if every
      candidate does, psf_compile's default call on the input is returned (it raises FailedElementsError when nothing
      avoids them). a11's direction backstop is kept.
  27. Search kept as a recorded fallback, not as the method: a14's own output (its seeds, borrowed layouts and
      re-placements, chosen by its estimate) is added as the LAST candidate and judged by the same rule (item 25), so
      it is returned only where it is strictly shorter than every construction (or, at equal duration, has fewer
      two-qubit gates). `A15_STATS["chosen_search"]` counts where the constructions lost: those are the inputs a
      formula still has to cover. SEARCH_FALLBACK = False turns it off (constructions only).
Needs psf_compile 2026-10-14.c49 or later (`_better`); with an older psf_compile, a15 falls back to a14.
"""
import contextlib
import importlib.util
import io
import os
import sys
import warnings

from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import CommutativeCancellation

import psf_compile as pc

_A14_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "psf_ai_compile_a14_2026-10-13",
                         "psf_ai_compile.py")


def _load_a14():
    name = "psf_ai_compile_a14_for_a15"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _A14_FILE)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


a14 = _load_a14()

AI_COMPILE_VERSION = "2026-10-14.a15"  # prototype: a14 + items 23-27 (constructed candidates chosen by duration; a14's search as the last candidate)
SMALL_MAX_QUBITS = a14.SMALL_MAX_QUBITS
DEFAULT_SEEDS = a14.DEFAULT_SEEDS
CONSTRUCT_BY_DURATION = True  # items 23-26; False gives a14
SEARCH_FALLBACK = True  # item 27: a14's searched output as the last candidate
A15_STATS = {"calls": 0, "candidates": 0, "chosen_polished": 0, "chosen_start": {}, "failed_skipped": 0,
             "release_fallback": 0, "compile_failed": 0, "chosen_search": 0, "search_failed": 0}


def _starts(qc):
    """a0-a1's deterministic starting points, in a14's order: (name, circuit)."""
    starts = [("input", qc)]
    cc = PassManager([CommutativeCancellation()]).run(qc)
    if a14._two_q(cc) < a14._two_q(qc) or [i.operation.name for i in cc.data] != [i.operation.name for i in qc.data]:
        starts.append(("commuted", cc))
    split = a14._split_multi_qubit(qc)
    if split is not qc:
        starts.append(("split", split))
        starts.append(("split+commuted", PassManager([CommutativeCancellation()]).run(split)))
        qc = split
    if a14.EXPANDED_START and any(len(i.qubits) == 2 and i.operation.name == "unitary" for i in qc.data):
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ex = pc.compile(qc, entangling_basis="cx")
        starts.append(("expanded", PassManager([CommutativeCancellation()]).run(ex)))
    return starts


def compile_for_model_circuit(qc, coupling_map, basis_gates, entangling_basis="cx", seeds=DEFAULT_SEEDS,
                              commutation_start=True, do_polish=True, layout_search=True, return_info=False,
                              target=None, **kwargs):
    """a15: with the device's Target and at most SMALL_MAX_QUBITS qubits, the shortest of psf_compile's constructions
    of a few deterministic starting points (items 23-26); otherwise a14. `coupling_map` is not passed to psf_compile
    when the Target is given (the device's own map is used, so that items 68, 80 and 85 apply)."""
    if (not CONSTRUCT_BY_DURATION or target is None or basis_gates is None or qc.num_qubits > SMALL_MAX_QUBITS
            or not hasattr(pc, "_better")):
        return a14.compile_for_model_circuit(qc, coupling_map, basis_gates, entangling_basis=entangling_basis,
                                             seeds=seeds, commutation_start=commutation_start, do_polish=do_polish,
                                             layout_search=layout_search, return_info=return_info, target=target,
                                             **kwargs)
    A15_STATS["calls"] += 1
    seed = seeds[0] if seeds else 0

    def device_call(circ):
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return pc.compile_for_hardware(circ, target=target, basis_gates=basis_gates,
                                           entangling_basis=entangling_basis, layout_search=layout_search,
                                           seed_transpiler=seed, **kwargs)

    starts = _starts(qc) if commutation_start else [("input", qc)]
    cands = []  # (label, circuit), in a fixed order: earlier wins a full tie
    for sname, circ in starts:
        try:
            out = device_call(circ)
        except pc.FailedElementsError:
            raise  # nothing avoids the failed elements: as psf_compile's default call
        except Exception:  # noqa: BLE001 - a starting point psf_compile cannot compile is skipped, counted
            A15_STATS["compile_failed"] += 1
            continue
        cands.append((sname, out))
        if do_polish:
            cands.append((sname + "+polish", a14.polish(out, basis_gates)))
    if SEARCH_FALLBACK:  # item 27: last, so it loses every tie
        try:
            with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cands.append(("search", a14.compile_for_model_circuit(
                    qc, coupling_map, basis_gates, entangling_basis=entangling_basis, seeds=seeds,
                    commutation_start=commutation_start, do_polish=do_polish, layout_search=layout_search,
                    target=target, **kwargs)))
        except Exception:  # noqa: BLE001 - counted; the constructions stand
            A15_STATS["search_failed"] += 1
    A15_STATS["candidates"] += len(cands)
    edges, bad = pc._failed_elements(target, a14.FAILED_MAX_ERROR)
    ok = [(lb, c) for lb, c in cands if not pc._uses_failed(c, edges, bad)]  # item 26
    if len(ok) < len(cands):
        A15_STATS["failed_skipped"] += len(cands) - len(ok)
    if not ok:
        A15_STATS["release_fallback"] += 1
        best, label = device_call(qc), "REL"
    else:
        label, best = ok[0]
        for lb, c in ok[1:]:
            if pc._better(c, best, target):  # item 25
                label, best = lb, c
    A15_STATS["chosen_start"][label] = A15_STATS["chosen_start"].get(label, 0) + 1
    if label == "search":
        A15_STATS["chosen_search"] += 1
    if label.endswith("+polish"):
        A15_STATS["chosen_polished"] += 1

    def _release():
        return device_call(qc)

    kept = a14._keep_direction(best, target, _release)  # a11 (item 16)
    info = {"path": "small-constructed", "chosen": label, "candidates": [lb for lb, _ in cands],
            "version": AI_COMPILE_VERSION, "direction_fix": kept is not best}
    return (kept, info) if return_info else kept
