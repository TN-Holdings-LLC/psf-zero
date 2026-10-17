# psf_compile changelog (items 1-104)

Release 2026-10-17.1 is prototype c67 (release 2026-10-16.2 = c66, plus item 104; option A's unreachable code
removed). Its module docstring is short; the
changelog lives here.

- **Items 1-101:** c63a's module docstring, moved unchanged by the workplace's c64 (below).
- **c63a (W3 option A, 2026-10-15):** in a template, item 88's frame is fixed at build (`TEMPLATE_FRAME_FIXED`), so a
  sign change alters angles only, never the shape. Kept by the owner's decision (Addendum 508): no rebuilds, flat
  memory and fast binds in training loops; on fractional targets, same-sign inputs cost up to 1.21-1.35 x c62's
  duration (Addendum 507).
- **c64 (workplace, 2026-10-16):** c63a without the code its default calls cannot reach: 9,240 lines to 3,045, every
  switch fixed at c63a's default, the same output on all 49,590 C63-VAL inputs (Addendum 505).
- **Item 102 (c65, 2026-10-16):** item 96's embedding test decides by formula before VF2: an odd cycle cannot embed
  in a bipartite device, nor a cycle shorter than the device's shortest; a path on the snake's longest working piece
  embeds; otherwise VF2. The same output, bit for bit; TOQB's ring circuits 11-12% faster (Addenda 504-505).
- **Item 103 (c66, 2026-10-16; release 2026-10-16.2):** in a template, item 88's frame by segments. The non-diagonal
  single-qubit runs ("anchors") cut each qubit's timeline into segments; conjugating an inner segment by X merges into
  the anchors' runs and flips the sign of every rzz in it on that qubit. Which segments to conjugate so that every rzz
  angle is positive is a GF(2) system (one equation per rzz), solved by a spanning forest found at build. An rzz on
  basis states, or after the last anchors in a measured circuit, needs no sign; an rzz closing a cycle, or joining
  two fixed segments, keeps option A's pinned form; folds are a Z frame carried to the next anchor. The template's
  shape is fixed as in option A, without its pulses: on fractional targets 0.81-1.00 x c62's median duration
  (C66-VAL, Addendum 513). Default calls and CZ targets unchanged, bit for bit.
- **Item 104 (c67, 2026-10-17; release 2026-10-17.1):** devices whose two-qubit gate is ECR (IBM Eagle). CZ(a, b) is
  ECR(a, b) with fixed single-qubit gates around it (checked at import). The default call runs the CZ path on a CZ view
  of the device (CZ on every ECR coupler, with its error and duration), writes every CZ as the coupler's ECR in its
  direction, and merges the single-qubit runs into rz, sx and x. Templates on ECR devices are refused. C67-VAL
  (Addendum 516): every criterion confirmed; 0.31 x Qiskit level 3's duration (geometric mean), at most 1.072 x.

---

# Changelog of psf_compile.py up to 2026-10-15.c63a (items 1-101), moved unchanged from its module docstring

```text
PSF-Zero -- the compiler. **This file is the latest version of it.**

VERSION: 2026-10-15.c63 -- workplace proposal (from c62: item 100 added, planned binding; release 2026-10-14.1 = c57)

Where to look for what
----------------------
This module is the compiler. It defines exactly two entry points --
`compile()` and `compile_for_hardware()` -- and a file that does not define
those two is not this module, whatever it happens to be named. The
layout-search prototype (`smart_vf2_layout()`) is a separate thing that lives
in `psf_smart_layout.py`. The contents of those two files have been swapped by
accident more than once, so the `VERSION` line above and this paragraph are
here to answer "is this the current compiler?" the moment the file is opened.

Versioning is by content, not by filename: when this file changes, bump
`VERSION` (and `__version__`, which mirrors it). Nothing has to be renamed to
mark a newer copy, and no second file has to exist alongside this one.

Changes in the 2026-09-15 revision
----------------------------------
1. FIXED: the Rust core import. The previous revision of this file imported
   `psf_zero_core_stub` -- the Qiskit-based Python stand-in written for a
   sandbox that could not load the real `.so` -- instead of `psf_zero_core`.
   With that import in place nothing in the Rust core ran at all: the
   degeneracy handling was never exercised, and every "PSF-Zero vs Qiskit"
   measurement was really Qiskit's own TwoQubitWeylDecomposition being
   compared against Qiskit, while the debug line still announced "PSF-Zero
   Rust Core executed for N blocks". The import is now the real core, and a
   missing core raises immediately with an explanation rather than silently
   substituting something else.

2. Verification is no longer the dominant cost, so it no longer has to be
   switched off to be competitive. The old `verify=True` path rebuilt the
   synthesized circuit with `Operator(qc)` and compared -- measured at
   ~1.35 ms per block against ~0.22 ms for everything else put together,
   i.e. ~87% of the total, which is the entire reason this project's
   measured speed advantage was only available with `verify=False`. The
   core now returns the reconstruction infidelity itself
   (`geometric_decompose_checked`), computed from a handful of 4x4 products
   on values it already has. Measured on 400 random SU(4) blocks:

       verify via Operator(qc)                 1.352 ms/block
       verify via numpy reconstruction         0.111 ms/block
       verify via the core's own check        ~0.000 ms/block (in the FFI call)

   `verify=True` therefore stays the default and is now essentially free.
   `verify="strict"` keeps the old `Operator(qc)` behavior for anyone who
   wants the circuit object itself checked rather than the decomposition
   (see `_verify_block` for exactly what each one does and does not cover).

3. `logging` instead of `print`. A library writing to stdout on every call
   forced this project's own benchmark scripts to wrap it in
   `contextlib.redirect_stdout`, and would do the same to anything else that
   embeds it. The per-block debug line is now a single `logger.debug`.

4. Fallbacks are counted and reported once at the end instead of raising a
   `warnings.warn` per block, and they are now classified: a legitimately
   degenerate input and an unexpected core failure are different events, and
   the core's new exception types let them be told apart.

5. `on_unsupported` is exposed on `compile()` (it was hardcoded to "keep"),
   and `compile_for_hardware()` accepts `seed_transpiler` so its internal
   routing call can be pinned -- the asymmetry this project's own benchmarks
   ran into, where an unpinned `optimization_level>=2` transpile returns a
   different answer and a different runtime on every call.

6. The entangling core is appended directly rather than built as a separate
   QuantumCircuit and composed, and the CX-basis form of a given canonical
   triple is cached -- Trotter and QAOA layers repeat the same (a, b, c)
   across many blocks, and each miss otherwise costs an `Operator()` build
   plus a full Qiskit KAK.

Changes in this (2026-09-16) revision
-------------------------------------
None of the following changes any published number. Items 7 and 10 change
behaviour only in cases that previously either crashed or were misreported;
item 9 swaps one computation of a quantity for a numerically equivalent,
cheaper one. **None of them has been benchmarked** -- the claims below about
cost are structural (fewer calls, fewer allocations), not measured, and should
be treated as such until someone times them.

7. The output circuit is built with `qc_blocked.copy_empty_like()` instead of
   `QuantumCircuit(qc.num_qubits, qc.num_clbits)`. The old construction made a
   circuit with fresh registers, which meant the source circuit's registers,
   loose bits, name and metadata were all dropped, and the per-instruction
   `find_bit` lookups in the main loop existed only to work around that. Two
   consequences, both silent:
     - a circuit carrying a classical condition on one of its own registers
       could not be rebuilt, because the target register no longer existed;
     - any named or split register came back as a single anonymous "q".
   `copy_empty_like()` keeps the registers, the bits, the name, the metadata
   and the global phase, so instructions can be re-appended with the original
   bit objects and the two `find_bit` calls per instruction disappear from the
   hot loop.

8. `verify`, `entangling_basis` and `on_unsupported` are validated at entry
   instead of being interpreted loosely. Previously `verify="Strict"` (capital
   S) or `verify=1` silently selected the cheap check rather than the strict
   one, and `entangling_basis="CX"` silently emitted the canonical basis --
   three ways to think you had asked for something you had not. They now raise
   `ValueError` naming the accepted values. This is the same reasoning as this
   project's "no silent fallback" rule, applied to its own arguments.

9. The core's own infidelity is now used for `entangling_basis="cx"` as well.
   Both `verify=True` paths check the same quantity -- the decomposition
   (`cartan`, `k1`, `k2`, phase) against the target unitary -- and the core
   already computed it during the FFI call, so recomputing the identical
   number in numpy on the cx path (~0.111 ms/block) bought nothing. What the
   cx path does *not* check is the CX substitution itself, and that was
   already true before this change: the substitution is Qiskit's own exact
   decomposer applied to an exact matrix, and `verify="strict"` remains the
   only mode that validates the emitted circuit object.

10. The `try` around the core call no longer also covers circuit
    construction. A failure inside `_build_circuit` -- i.e. a bug in this
    file, not in the input -- was being reported as "Decomposition failed or
    degenerate" and, when the core's typed exceptions are unavailable,
    counted as an *expected* degeneracy. The two are now separate blocks with
    separate messages, and a construction failure is always counted as
    unexpected.

11. Housekeeping: an unused eigenvalue array is no longer bound, `__all__`
    names the public surface, and `compile()` has the docstring it was
    missing.

Additional change, same day (2026-09-16), item 12
--------------------------------------------------
12. **NEW, opt-in: `layout_search` on `compile_for_hardware()`.** Addenda
    24-25 (spare-qubit-cliff series) measured that at the spare-qubit cliff,
    `compile_for_hardware()`'s advantage over plain Qiskit L3 comes almost
    entirely from running its internal routing call at a lower
    `routing_optimization_level`, not from avoiding the cliff mechanism
    itself (`VF2Layout` failing, falling back to `SabreLayout`) -- raising
    the level toward 3 made the cliff, and PSF-Zero's own advantage, both
    disappear together. This item addresses the cliff at its actual cause
    instead: when `layout_search=True`, `compile_for_hardware()` now runs
    this project's own `smart_vf2_layout()` (previously a `prototypes/`-only
    tool a caller had to invoke and wire in by hand, per Addendum 17) against
    the *compressed* circuit's own interaction graph before calling
    `transpile()`, and threads a found layout through `initial_layout`
    itself. On a topology/interaction-graph combination the search can
    solve near-instantly (grids and lines with a matching-type interaction
    pattern, per Addendum 13), this is intended to let `routing_optimization_level`
    stay low without inheriting a slow `VF2Layout` failure to get there.

    Deliberately conservative about failure modes, per this project's
    "no silent fallback" rule:
      - `layout_search=True` together with an explicit `initial_layout` is a
        contradiction (which one should win?) and raises `ValueError` rather
        than silently picking one.
      - `psf_smart_layout` not being importable raises `ImportError` with an
        explicit message, only when `layout_search=True` is actually
        requested -- normal `compile_for_hardware()` calls that never ask for
        this are completely unaffected, and nothing about this failure is
        swallowed into "fell back to default behaviour" the way a bare
        `except Exception` would.
      - If the search does not find a layout within its budget (this can and
        does happen -- see `psf_smart_layout.py`'s own docstring on `brick`
        and other hard topologies), `compile_for_hardware()` falls through to
        Qiskit's own default layout stage, exactly as if `layout_search` had
        been left `False` -- but the time already spent searching is real and
        is not hidden from the caller's own wall-clock measurement of this
        call, matching this project's Addendum 14 principle that a failed
        search's cost must be counted, not absorbed silently.

    **Not yet benchmarked in this file's own revision history** -- this
    changelog entry describes what the code now does, not a measured result.
    The pre-registered prediction and the benchmark built to test it live
    separately, in this project's spare-qubit-cliff addenda (see Addendum 26
    once it exists), not in this docstring, per this project's own
    convention of keeping code changelog entries and measured findings in
    separate documents.

Additional change, same day (2026-09-16), item 13
--------------------------------------------------
13. **NEW, opt-in: `callback` on `compile_for_hardware()`.** Addendum 27
    found that PSF-Zero's own default (`layout_search=False`) path still has
    rare, large, unexplained slowdowns exactly at the spare-qubit cliff's
    peak (spare=0) -- 3 of 30 runs across 5 rounds, up to 8x the median, not
    tied to one seed. This project's own established technique for seeing
    *where inside a transpile call* time actually goes is Qiskit's own
    `transpile(callback=...)` (already used, from outside this file, in
    Addendum 10 -- see the `initial_layout` docstring above for that
    finding). Until now there was no way to use it from *inside* a
    `compile_for_hardware()` call, since this function builds its own
    internal `transpile()` call and gave the caller no hook into it.

    `callback`, when given, is forwarded verbatim to that internal
    `transpile()` call -- one parameter, one line of forwarding, the same
    minimal-and-additive shape as `initial_layout` (Addendum 17) and
    `layout_search` (item 12 above). Default `None`, so every existing call
    is unaffected. It is Qiskit's own `callback(pass_, dag, time, property_set,
    count, running_time=...)`-style callable (see Qiskit's `transpile()`
    documentation for the exact signature Qiskit invokes it with); this file
    does nothing with it beyond passing it through, so anyone already
    familiar with the plain-`transpile()` version needs to learn nothing new
    to use it here.

    **Not yet used to explain anything** -- this changelog entry describes
    the hook, not a finding. The investigation it exists to support (the
    Addendum 27 spare=0 outlier hunt) lives separately, per this project's
    changelog/findings separation convention.

Changes in the 2026-09-21 revision
----------------------------------
14. The CX-basis decomposer is configured for the {rz, sx} basis:
    `TwoQubitBasisDecomposer(CXGate(), euler_basis="ZSX")` instead of
    `TwoQubitBasisDecomposer(CXGate())`. Constructed without an Euler basis,
    it emitted a general rotation between each pair of its three CXs, which
    became two `sx` pulses per qubit per gap after basis translation; Qiskit's
    own transpilation places at most one. On a repeated-pair variational
    ansatz this left PSF-Zero with 1.6x Qiskit's `sx` count at the same CX
    count (spare-qubit-cliff Addendum 114). With this setting, `sx` count, sx
    depth and total depth equal Qiskit's own re-compile exactly (128 -> 80
    `sx`, depth 23 -> 16 at 16 qubits; 336 -> 210, 23 -> 16 at 42); CX count
    unchanged; 300/300 random blocks still exact (Addendum 116).
    `pulse_optimize=True` changed nothing measurable and is left out.

    Scope and caveats: the benefit is for devices whose single-qubit basis is
    {rz, sx}; other bases are still translated by `transpile()`, untested
    here. This decomposer also serves the degenerate-block fallback, which
    Addendum 116 did not exercise. What changes and what does not:
    `compile()` and `compile_for_hardware()` default to
    `entangling_basis="canonical"`, which uses this decomposer only on that
    fallback, so default-basis output -- including the 10,000-iteration
    gate-synthesis speed benchmark (`test_cumulative_compile_scale.py`) --
    is essentially unaffected. Output with `entangling_basis="cx"` changes:
    same two-qubit counts, fewer `sx`, lower total depth than the 2026-09-16
    revision (e.g. the README's coupling-map cliff table, measured with
    "cx", reports depth 23 for `layout_search=True`; expected 16 now). A
    per-circuit speed change of order 10-20% on the "cx" path could not be
    ruled in or out (Addendum 116, P4).

Changes in the 2026-09-26.3 revision (spare-qubit-cliff Addenda 191-194)
------------------------------------------------------------------------
15. **CX-basis core in closed form.** With `entangling_basis="cx"`, each
    block's canonical core exp(i(a XX + b YY + c ZZ)) used to be built as a
    circuit, turned into an `Operator`, and decomposed again by Qiskit's
    `TwoQubitBasisDecomposer` -- a second KAK decomposition of a matrix whose
    decomposition was already known. On random blocks the cache never hits,
    and this was 23% of `compile_for_hardware`'s time on the Nighthawk cliff
    (Addendum 191). The core is now emitted directly as three CXs
    (Vatan-Williams form) with the rotations in the two middle gaps moved
    through the CXs so that each gap is exactly one `sx` after translation
    (an `rx(pi/2)` on the target side of the middle CX; see
    `_append_cx_core_closed_form`). Checked in isolation against
    `expm(i(aXX+bYY+cZZ))` on 20,000 random triples over [-pi, pi]^3:
    worst Frobenius distance 2.7e-15 including global phase; both middle
    gaps have |off-diagonal| = 1/sqrt(2) to 2.2e-16 (one `sx` each).
    Degenerate triples -- any coordinate within 1e-6 of a multiple of pi/2,
    where fewer than three CXs suffice -- keep the previous path, so the CX
    count can only stay equal. `USE_CX_CLOSED_FORM = False` restores the
    previous behavior everywhere (used for A/B validation).

16. **NEW, opt-in: `layout_edge_errors` on `compile_for_hardware()`.** A
    `{(p, q): error}` map of the native 2-qubit gate's error per physical
    edge (`edge_errors_from_target(backend.target)` builds one). When given
    with `layout_search=True`, and the interaction graph is a set of disjoint
    pairs (the matching shortcut of `psf_smart_layout`, LAYOUT_VERSION
    2026-09-26.m1, Addenda 192-193), the layout is the maximum-weight
    matching with weight log(1 - error) per edge, i.e. it maximizes the
    product of the edge fidelities the pairs land on. Ignored for any other
    interaction graph (VF2 is unweighted). Single-qubit and readout errors
    are not considered. Like any `initial_layout`, it skips VF2PostLayout.

Changes in the 2026-09-26.4 revision (spare-qubit-cliff Addenda 195-196)
------------------------------------------------------------------------
17. **FIX (correctness): every block left to Qiskit's CX decomposer is now
    checked.** Addendum 195 found that Qiskit's
    `TwoQubitBasisDecomposer(CXGate(), euler_basis="ZSX")` (this file's
    `_CX_DECOMPOSER`, Qiskit 2.5.2) returns a wrong circuit -- average gate
    infidelity 6.99e-2 -- for unitaries whose smallest canonical coordinate
    is about 3e-8 to 3e-7, and that plain `transpile()` to basis
    [cx, rz, sx, x] fails the same way. Every earlier revision emitted such
    blocks unchanged on the `entangling_basis="cx"` path (the degenerate
    core via `_cx_core_cached`, and whole blocks via the fallback), and
    `verify=True` did not catch it because it checks the decomposition, not
    the emitted circuit. Now `_guarded_cx_synthesis` computes the emitted
    circuit's unitary and accepts it only if its average gate infidelity is
    <= 1e-8 (ten times Qiskit's own default requested fidelity, so its
    documented approximations still pass), correcting its global phase to
    match exactly. On rejection it retries
    with the default Euler basis (exact on every input tested), and for a
    canonical core finally with the closed form of item 15 regardless of
    degeneracy (three CXs, exact). A whole block that fails both raises
    `RuntimeError` rather than being emitted. The check costs one 4x4
    `Operator` per decomposer call, which after item 15 happens only on
    degenerate blocks. `USE_CX_GUARD = False` restores the unchecked
    behavior (for A/B validation only). Counts in `GUARD_STATS`.

18. **Closed-form core with native middle gaps.** The 2026-09-26.3 form
    left `ry`/`rx` in the two middle gaps for Qiskit to translate; drift
    under repeated recompilation rose about 2.6x (Addendum 195, C5). The
    gaps are now emitted directly in {rz, sx}, using
    rx(pi/2) ry(t) = rz(t) rx(pi/2), ry(t) rx(-pi/2) = rx(-pi/2) rz(t),
    rx(pi/2) = e^{-i pi/4} sx and rx(-pi/2) = -e^{-i pi/4} rz(pi) sx rz(pi):
    gap 1 is `sx, rz(2a - pi/2)`, gap 2 is `rz(3pi/2 - 2b), sx, rz(pi)`,
    global phase 3pi/4 in total. Checked in isolation on 20,000 random
    triples: worst 3.6e-15 including global phase. `USE_NATIVE_GAPS =
    False` restores the 2026-09-26.3 gaps.

19. **Single-qubit errors in the weighted layout.** `layout_qubit_errors`
    (`{q: error}` of the single-qubit `sx`; `qubit_errors_from_target`
    builds it) adds each endpoint's single-qubit log-fidelity to the edge
    weight. This is exact for a matching, where every matched edge covers
    exactly its two qubits. Gate counts per pair are estimated from the
    circuit: n2 = mean 2-qubit gates per interacting pair, n1 = 2 n2 + 1
    `sx` per qubit (one per middle gap plus the outer layers; matches the
    measured 7 per qubit for 3-CX blocks). Addendum 195 (Q3): without it,
    the weighted layout placed pairs on a qubit whose `sx` has error 1.0.

20. **Polish batched over all blocks of a circuit.** `compile()` now
    decomposes every block first, then runs the polish of Addendum 186 on
    all of them at once (`_refine_batch`): one vectorized residual check,
    and Gauss-Newton steps on the blocks above the threshold with a batched
    Jacobian and a batched SVD least-squares solve (same cutoff as
    `numpy.linalg.lstsq(rcond=None)`), with the same per-block stopping
    rules as `_refine_decomposition`. What is computed is unchanged; only
    numpy's per-call overhead is removed. Addendum 206: the per-block
    polish cost 12.3 ms per 120-qubit compile (80 us per check, 0.55 ms per
    step) and cannot be loosened without a 16-22x rise in drift under
    repeated recompilation. `SU4GeodesicPSFSynthesizer.synthesize()` (one
    block) is unchanged. `USE_BATCHED_POLISH = False` restores the
    per-block polish inside `compile()` (for A/B validation).

21. **Default `block_gate_floor` 12 -> 8.** At 12, the second layer of a
    brick-layer ansatz lost its leading single-qubit rotations to the
    neighbouring blocks during collection, kept runs of 9 gates, and passed
    through unconsolidated: 48 two-qubit gates instead of 33 on the 12-qubit
    training circuit (Addendum 206). Addendum 210 checked 20 instances of 8
    families (brick, Heisenberg Trotter, QAOA, HEA, dense pairs, random,
    QFT, 120-qubit pair blocks): at 8, no two-qubit count rose, depth rose
    by at most 2%, every compile was exact, and the brick circuits fell from
    48 to 33 (depth 35 -> 31) for +16% compile time. Lower floors are not
    safe as a default: at 6 and 4 the HEA depth grew from 33 to 41 and 91
    with no change in two-qubit count. Pass `block_gate_floor=12` for the
    previous behaviour.

22. **Default `block_gate_floor` back to 12 (item 21 withdrawn as a
    default).** 100,000 compiles (Addendum 214) found that at 8, 13% of the
    brick-layer training blocks fall back from the Rust core
    (`SU2ExtractionSingular`) to Qiskit's CX synthesis, and 4 of 3,000
    checked compiles lost accuracy (loss error 1.7e-9 to 1.3e-7, against
    <= 1e-14 everywhere else). At 12 those blocks are not synthesized and the
    result is exact. The likely mechanism -- Qiskit's Weyl decomposition
    snapping near-special inputs within about 1e-9 infidelity, which the
    guard's 1e-8 tolerance accepts -- is under investigation.
    `block_gate_floor=8` remains available and gives 33 instead of 48
    two-qubit gates on that circuit, with this accuracy caveat.

23. **FIX (correctness): every circuit taken from Qiskit's CX decomposer is
    now checked by phase-aligned operator distance, and rebuilt exactly when
    it is not exact.** Addendum 216: the losses of item 22 came only from
    blocks that fell back to Qiskit's synthesis; the guard of item 17
    measures average gate infidelity (tolerance 1e-8), which is quadratic in
    the operator error and admitted operator errors up to about 1e-4 (one
    block with infidelity 6.7e-16 still cost a loss error of 4.6e-9). A
    decomposer result is now accepted only if, additionally, its
    phase-aligned Frobenius distance to the target is at most
    `_EXACT_TOL` = 1e-13; otherwise the block is rebuilt from Qiskit's Weyl
    decomposition without specialization (`fidelity=None`), converted to
    PSF-Zero's own parameters (ZYZ angles of each local factor, Weyl
    coordinates, global phase), polished by `_refine_decomposition` (the
    Gauss-Newton step of Addendum 186, which works in the same Frobenius
    distance), and emitted as PSF-Zero emits its own blocks, with the
    closed-form core (`force=True`, three CXs). Exact results from Qiskit,
    including 2-CX ones, are kept as before. Rebuilt blocks are counted in
    `GUARD_STATS["exact_rebuilt"]`. If even the rebuild misses 1e-13, the
    most accurate candidate is used when its distance is at most 1e-10
    (`GUARD_STATS["best_effort"]`, worst distance in
    `GUARD_STATS["best_effort_worst"]`); otherwise the block is reported as
    before. `USE_EXACT_FALLBACK = False` restores the 2026-09-27.3 behavior.
    Revision 2026-09-27.4 (withdrawn before release) rebuilt the local
    factors with an Euler decomposition and no polish; on one fallback block
    its rebuild missed 1e-13 and the compile stopped (Addendum 217,
    amendment).

24. **FIX (correctness): PSF-Zero's own blocks are held to the same
    operator-distance standard.** With item 23 in place (revision .5), one
    of the 3,003 floor-8 training compiles still lost 4.6e-9 in loss
    (Addendum 218): not a fallback block, but a block PSF-Zero synthesized
    itself, off by 5.9e-7 in operator distance after the polish. It was
    accepted because the block check compares infidelity (the core's own,
    against `tol` = 1e-5), in which 5.9e-7 is about 1e-13. A block whose
    polished residual (`_refine_decomposition`'s Frobenius distance) is
    above `_EXACT_TOL` is now synthesized instead through the checked path
    of item 23 (`_guarded_cx_synthesis`: Qiskit's decomposer if exact,
    otherwise the exact rebuild). Counted in `GUARD_STATS["psf_rerouted"]`.
    Addendum 206 measured polished residuals of at most 9.8e-14 on cliff and
    floor-12 training circuits, so those are not expected to change.

25. **Default `block_gate_floor` 12 -> 8 again (item 21 restored).** The
    accuracy caveat of item 22 is removed by items 23 and 24. Addendum 219
    pre-registered the rule: 8 becomes the default again only if the
    100,000-compile run of Addendum 213 clears all six flags at floor 8 with
    revision .6. It did (Addendum 222): no exception, no memory growth, no
    slow-down, linear drift, every checked compile exact (worst loss error
    1.9e-15 over 3,000 training checks, against 1.3e-7 in Addendum 214;
    worst per-pair check 1.0e-15 over 50 cliff checks). In the training
    part, 43,829 of about 330,000 blocks still fall back from the Rust core
    (`SU2ExtractionSingular`, unchanged since Addendum 214), and 150 of
    them and 1 PSF-Zero block were rebuilt exactly; none needed the
    best-effort path. Gains, as in item 21: 33 instead of 48 two-qubit
    gates on the 12-qubit brick-layer training circuit, no change in
    two-qubit count on seven other families (Addendum 210). The only change
    from 2026-09-27.6 is this constant and its docstrings; pass
    `block_gate_floor=12` for the previous behaviour.

26. **`REFINE_THRESHOLD` 1e-13 -> 1e-14** (= `_REFINE_TARGET`, the polish's
    own stopping point). With the Rust core fix of 2026-09-28 (core changelog
    item 11, `CORE_VERSION` 2026-09-28.1), the core's raw residual on some
    blocks is a smooth function of the block that lies just below 1e-13 (one
    block at 8.3e-14 in the compounding chain of Addendum 219, part C). Such a
    block was never polished, and in a chain that compiles its own output
    again and again the same error was added in the same direction on every
    lap: the chain's drift doubled against the pre-fix core (1.66e-9 at lap
    20,000; Addenda 235-237). At 1e-14 about twice as many blocks are polished
    and the drift falls to 1.22e-10, seven times below the pre-fix core's,
    for 5-8% compile time in the workplace sandbox (Addenda 238-239); every
    output stays exact, and the whole-circuit GPU check at 20-26 qubits gives
    a worst error of 1.1e-14 with unchanged CX counts (Addenda 240-241). The
    threshold was not measured with the pre-fix core. Every change of this kind alters output bits, not structure. Setting
    `REFINE_THRESHOLD` (and the defaults of `_refine_batch` and
    `_refine_decomposition`) back to 1e-13 restores the previous behaviour.

27. **`CORE_VERSION`** is exported: the loaded Rust core's own version string
    (`psf_zero_core.CORE_VERSION`), or None for cores built before
    2026-09-28, so that logs can show which core ran.

2026-10-01.1 (release; candidate 2026-10-01.c2, adopted on 2026-10-01 after its pre-registered evaluation):

28. **Cost-aware consolidation of short blocks** (`CONSOLIDATE_IF_CHEAPER`). A same-pair block at or below
    `block_gate_floor` is still consolidated when it holds at least two 2-qubit gates and its optimal CX
    count (Weyl) is below its CX cost as written (`_CX_COST`; SWAP 3, controlled rotations 2, CX/CZ 1).
    Only with `entangling_basis="cx"`: with "canonical" the re-expanded RXX/RYY/RZZ cost more CX than the
    gates as written (40-qubit random circuit 641 -> 648 CX with the rule on). The block's 4x4 unitary is
    built with numpy (`_block_unitary_4x4`), about 10x cheaper than `Operator(QuantumCircuit)`.
29. **`compile_for_hardware(elide_permutations="auto")`**: before compiling, ElidePermutations and
    Split2QUnitaries(split_swap=True) remove SWAPs (also a SWAP written as a 2-qubit `unitary`, as PennyLane
    tapes arrive) by relabelling later gates; the permutation is handed to the preset pipeline
    (`_CarryPermutation`), so it appears in `out.layout.final_layout` exactly as at optimization levels
    2 and 3. Read the output qubits through `final_index_layout()` (the e2e and IBM pipelines already do).
30. **`compile_for_hardware(post_routing_resynthesis="auto")`**: in the preset pipeline's post_routing
    stage, every 2-qubit block that holds a routing SWAP and costs more CX than its optimum is consolidated
    and re-synthesised by the PSF-Zero core (`_AbsorbRoutingSwaps`). Blocks without a SWAP are untouched.
    "auto" means on for `entangling_basis="cx"`, off for "canonical" (items 29 and 30), and the canonical
    path is unchanged.

2026-10-02.1 (release; candidate 2026-10-02.c3, adopted on 2026-10-02 after its pre-registered evaluation, Addenda 303-304):

31. **NEW, opt-in: `compile_for_hardware(target=...)` avoids failed couplers and qubits.** Addendum 302
    found the release placing 7-14 CZ of a 6-qubit ring on FakeTorino's coupler (15, 19), whose reported
    error is 1.0: the layout search and the routing see only `coupling_map`, which still lists edges the
    device reports as failed. With `target` given, the circuit is first compiled exactly as without it; if
    the result uses no failed element -- no 2-qubit gate on a directed edge whose native 2-qubit gate error
    is >= `prune_max_error` (default 0.5), no gate on a qubit whose `sx` error is >= `prune_max_error` --
    that result is returned unchanged. Otherwise the circuit is compiled again on `prune_coupling_map(...)`,
    the coupling map without those edges and without every edge touching such a qubit. Recompiling only when
    needed keeps every unaffected output identical to the release: the layout search is error-blind, so
    pruning alone would move unaffected layouts too (Addendum 303, development). Qubit indices and
    `coupling_map.size()` are unchanged. Without `target` nothing changes. Counts in `PRUNE_STATS`. If no
    placement on the pruned map exists, the first output is kept, with a warning (item 43). Gate and
    readout errors below the threshold are still ignored by the layout (a separate, later step).

2026-10-02.2 (release; candidate 2026-10-02.c5, adopted on 2026-10-02 after its pre-registered evaluation, Addenda 309-310):

33. **NEW, opt-in: `compile_for_hardware(target=..., placement_refine=True)`.** (Item 32, candidate c4, was
    not adopted: Addendum 307.) Addendum 308 found why handing placement to Qiskit's level-1 layout stage
    failed: it ranks placements by an averaged per-qubit error that mixes in readout, while Qiskit level 3
    ends with a re-placement scored on the exact per-instruction errors of the gates actually placed. With
    `placement_refine=True` and a `target`, the circuit is compressed, laid out and routed exactly as by the
    release, and the routing pass manager then ends with that same exact re-placement:
    `VF2PostLayout(target, strict_direction=True, seed=-1)` with Qiskit level 3's limits for it
    (`placement_call_limit=300_000`, `placement_max_trials=2_500`), and `ApplyLayout` when it finds a
    placement that scores better. It relabels physical qubits only: gates, their count and the routing are
    unchanged. Item 31 stays the backstop. Requires `target`; default False, identical to the release.
    Counts in `REFINE_STATS`.

2026-10-03.1 (release; candidate 2026-10-03.c8, adopted on 2026-10-03 after its pre-registered evaluation, Addenda
323-324). Item 34 is the held candidate c6 (floor-aware re-placement score, Addendum 321), which this release does not
contain:

35. **NEW, opt-in: `final_resynthesis` re-synthesises every two-qubit block with Qiskit at the end, always (True) or
    when an excitation-aware estimate says it helps ("select").** On cx
    devices the release trailed Qiskit level 3 on chains by 9-22% with the same qubits and the same cx gates
    (Addendum 322). Under depolarizing noise alone the two were level; the whole gap was thermal relaxation:
    the local frames PSF-Zero's two-qubit synthesis chooses around each cx (about 11 x gates per 60-cx chain,
    against about 2 for Qiskit) leave qubits excited for longer during the long cx gates. Re-synthesising every
    block with Qiskit removed 72-87% of that gap on open chains and changed nothing on a cz device. With
    `final_resynthesis=True` the finished circuit (after item 31's backstop) is passed through
    `ConsolidateBlocks(force_consolidate=True)`, `UnitarySynthesis` and `Optimize1qGatesDecomposition`, all with
    the target and exact (approximation_degree=1.0). The layout is carried over unchanged: these passes move no
    qubit. If the result has an instruction the target does not provide, or a two-qubit gate in a direction the
    target reports failed (error >= prune_max_error), the circuit before re-synthesis is returned instead.
    "select" builds both circuits and keeps the one with the lower excitation-aware estimate
    (`excitation_cost`): the summed -log(1 - reported error) of the gates as placed, plus, for every gate, its
    duration / T1 times P(1) on each of its qubits, with P(1) from the noiseless state just before the gate (the
    population that amplitude damping acts on). The estimate needs a statevector of the touched qubits; above
    `RESYNTH_MAX_QUBITS` (16) touched qubits, or with a non-unitary instruction, "select" keeps the release's
    circuit. Why "select": an unconditional re-synthesis (candidate c7, never locked) won 3-15% on the F3 chains
    in its smoke run but lost 1-13% on F1, F2, F5 and F6 on all nine devices, cz devices included, with the same
    two-qubit count and a much deeper circuit (Addendum 323, section 5).
    Requires `target`; default False, identical to the release. Counts in `RESYNTH_STATS`.

2026-10-03.2 (release; candidate 2026-10-03.c9, adopted on 2026-10-03 after its pre-registered evaluation, Addenda 327-328):

36. **NEW, opt-in: `compare_level3=True` also compiles the input with Qiskit's level-3 preset on the target and keeps
    whichever circuit has the lower `excitation_cost`.** After release 2026-10-03.1 the remaining gaps to Qiskit
    level 3 with the Target were periodic chains on cx devices (12-40%), F1-type rings on cz devices (6-8%) and
    QFT circuits (about 5%) (Addendum 324). A diagnosis on those circuits (Addendum 326) found three causes:
    on periodic chains on cx devices the same qubits and the same cx count, with PSF-Zero's routed circuit carrying
    about 50% more sx gates even after item 35's re-synthesis; on the cz rings a placement that needs more SWAPs
    (59 against 54 two-qubit gates); on QFT one or two more two-qubit gates after routing. In all three, choosing
    per circuit between the release's circuit and level 3's by `excitation_cost` matched the better of the two
    (measured) closely: 1.000 of level 3 on the periodic chains, 0.984-0.998 on the rings, 0.996-1.003 on QFT.
    With `compare_level3=True`, `transpile(qc, target=target, optimization_level=3, seed_transpiler=...,
    approximation_degree=1.0)` is run on the input and kept if (a) it has no instruction the target does not
    provide, no failed qubit and no two-qubit gate in a direction the target reports failed, and (b) its
    `excitation_cost` is lower than that of the circuit the release would return. If either estimate cannot be made
    (above `RESYNTH_MAX_QUBITS` touched qubits), the release's circuit is kept. Requires `target`; default False,
    identical to release 2026-10-03.1. Counts in `COMPARE_STATS`.

2026-10-03.3 (release; candidate 2026-10-03.c10, adopted on 2026-10-03 after its pre-registered evaluation, Addenda 331-332;
recommended on cx devices only):

37. **NEW, opt-in: `compare_floor=True` adds a third candidate, and `candidate_score="pauli"` chooses among the
    candidates with a state-aware Pauli estimate that includes dephasing.** After release 2026-10-03.2 the AI front
    end a7 was still ahead, most on GHZ-type chains (F5) on the cx devices, with the same gates and depth: a
    placement effect (Addendum 330). The held candidate c6 (item 34; its functions `decoherence_floor` and
    `floor_aware_target` are carried over unchanged) re-places on a Target whose errors are max(reported error,
    T1/T2 floor), and that placement alone matched a7 on those chains. `excitation_cost` sees amplitude damping
    only, so it cannot tell the two placements of a GHZ chain apart (it ranked the best of three candidates in
    60% of circuits on FakeAuckland); `pauli_cost` (the first-order estimate of the AI front end a4, simplified)
    ranked 87%. In that diagnosis the choice among the three by `pauli_cost` came within 0.1-0.5% of the measured
    best on every device.
    - `compare_floor=True`: the release's pipeline is run a second time with the re-placement of item 33 scored on
      `floor_aware_target(target)` (the backstop of item 31 and item 35's `final_resynthesis` applied as in the
      first run), and the result is a candidate if `_acceptable`.
    - `candidate_score`: "excitation" (default) or "pauli". The candidates (the release's circuit, the floor
      candidate if `compare_floor`, level 3's circuit if `compare_level3` and `_acceptable`) are scored and the
      lowest kept; ties and any estimate that cannot be made keep the release's circuit. Item 35's "select" is
      unchanged (it still uses `excitation_cost`).
    - `pauli_cost(circ, target)`: per gate, on the noiseless state right after the gate, for each of the gate's
      qubits the Pauli-twirled thermal relaxation for the gate's duration (p_X = p_Y = (1 - exp(-t/T1)) / 4,
      p_Z = (1 - exp(-t/T2)) / 2 - p_X, T2 capped at 2 T1) costs p_P (1 - <P>^2), plus the reported error above
      the thermal floor times (d + 1) / d (state-independent). None above RESYNTH_MAX_QUBITS touched qubits.
    With `compare_floor=False` and `candidate_score="excitation"` (the defaults) nothing changes. Counts in
    `COMPARE_STATS`.

2026-10-04.1 (release; candidate 2026-10-04.c11, adopted on 2026-10-04 after its pre-registered evaluation, Addenda 336-337;
recommended on every device):

38. **NEW, opt-in: `candidate_score="hybrid"` chooses among the candidates of items 36-37 with an estimate that keeps
    both amplitude damping and pure dephasing.** HOLD5 (Addendum 332) found `pauli_cost` choosing well where
    dephasing decides (GHZ chains) and losing up to 1.8% on XXZ chains, where amplitude damping decides and
    `excitation_cost` had ranked correctly: twirling relaxation into symmetric Pauli errors loses its non-unital
    part. In the exploratory diagnosis of Addendum 335 (in-sample, HOLD5's circuits, nine devices) the choice by
    `hybrid_cost` among the release's candidates was better than the choice by `pauli_cost` on every device
    (0.06-0.51%), better than release 2026-10-03.2's choice by `excitation_cost` on every device (0.02-2.2%), and
    within 0.2% of the measured best.
    - `hybrid_cost(circ, target)`: per gate with a reported duration, for each of the gate's qubits,
      duration / T1 x P(1) on the noiseless state just before the gate (amplitude damping, as `excitation_cost`),
      plus p_phi (1 - <Z>^2) on the noiseless state just after it (pure dephasing), with
      p_phi = (1 - exp(-t / T_phi)) / 2, 1 / T_phi = 1 / T2 - 1 / (2 T1), T2 capped at 2 T1; plus, per gate, the
      reported error above the thermal floor times (d + 1) / d (as `pauli_cost`). None above RESYNTH_MAX_QUBITS
      touched qubits or when an instruction has no matrix.
    - `candidate_score="hybrid"` uses it in `_choose`; everything else is as in item 37 (ties and any estimate that
      cannot be made keep the release's circuit; item 35's "select" still uses `excitation_cost`).
    Intended call on every device: `compare_level3=True, compare_floor=True, candidate_score="hybrid"` with
    item 35's `final_resynthesis="select"` and item 33's `placement_refine=True`. Other values change nothing.

2026-10-05.1 (release; candidate 2026-10-05.c12, adopted on 2026-10-05 after its pre-registered evaluation, Addenda 342-344;
a correctness fix for releases 2026-10-03.1 to 2026-10-04.1):

39. **FIX (correctness): circuits made by Qiskit are now checked for equivalence before they can be returned.**
    Item 17's guard checks every block this file leaves to Qiskit's CX decomposer, but items 35-37 brought in
    circuits that Qiskit synthesised as a whole and that nothing checked: item 35's re-synthesis
    (`UnitarySynthesis`) and items 36-37's level-3 output. On cx devices both are exposed to Qiskit issue #17057
    (Addendum 294; near-boundary two-qubit unitaries). A workplace probe (2026-10-05, Addendum 340) found the
    recommended call of release 2026-10-04.1 returning a non-equivalent circuit for 6 of 10 explicit near-boundary
    unitary circuits on FakeAuckland (infidelity up to 0.19), and item 35's re-synthesis wrong on 20 of 20 near-boundary
    Trotter circuits; 2026-10-02.2's call was exact on all of them.
    - `_same_action(ref, new)`: two circuits on the same physical qubits and layout (item 35) act alike on two seeded
      random product states of every touched qubit (state infidelity <= `EXACT_TOL`, global phase ignored).
    - `_implements(qc, out)`: a compiled circuit maps two seeded random product states of `qc`'s qubits, placed by
      its initial layout with every other touched qubit in |0>, to `qc`'s output states at its final layout, with
      the other touched qubits back in |0>.
    - Item 35's re-synthesis is kept only if `_same_action`; the floor and level-3 candidates of items 36-37 are
      used only if `_implements`. A check that cannot be made (more than RESYNTH_MAX_QUBITS touched qubits, an
      instruction without a matrix such as a reset or a conditional; from item 44 an instruction on more than
      EXACT_MAX_GATE_QUBITS qubits is expanded through its definition) refuses the candidate: the release's own,
      guarded circuit is kept. Counts in `EXACT_STATS`.
    The checks are statevector simulations of at most RESYNTH_MAX_QUBITS qubits, like the estimates of items 35-38.
    With every Qiskit-made circuit exact (all HOLD tests), the result is the same as 2026-10-04.1.

2026-10-06.1 (release; candidates 2026-10-05.c14 (items 40-41, written at the workplace) and 2026-10-06.c15 (item 42),
adopted on 2026-10-06 after their pre-registered evaluations RECR and KRAUS, Addenda 358-362):

40. **Readout of measured qubits in item 38's estimate.** `hybrid_cost` skipped `measure`, so no candidate choice
    (items 36-38) ever saw readout error, and circuits compiled without measurements gave the placement step
    (item 33) nothing to see either. On FakeTorino the qubit that carried a classifier's output had a mean readout
    error of 0.034 under the guarded call and 0.025 under the recommended call, against 0.011 for Qiskit level 3
    (workplace DEPTH stage 1, 2026-10-05; Addendum 297 found the same at home).
    - `readout_cost(circ, target)`: the sum, over the circuit's `measure` instructions, of the Target's measure error
      on that qubit (each measured qubit counted once; 0 if the Target reports none).
    - `hybrid_cost` adds it. A circuit without measurements gets exactly c12's estimate, so its choice is unchanged.
    Callers who want readout taken into account compile the circuit with its final measurements (item 33's
    re-placement then scores them too, as Qiskit's VF2PostLayout does).
    RECR (Addenda 358, 360): compiled with their measurements, sampled circuits on six fake devices had 0.96-1.00 x
    the classical infidelity of 2026-10-05.1's call with measurements, and lower summed readout error on every
    device; without measurements the output was 2026-10-05.1's on every circuit.

41. **Item 31's check is direction-aware, like its pruning.** `_uses_failed` flagged a two-qubit gate whenever either
    direction of its coupler was reported failed, while `prune_coupling_map` and item 36's `_acceptable` remove or
    refuse only the failed direction. On a coupler that has failed one way only (FakeHanoiV2's cx(5, 8) and
    cx(19, 20), FakeGeneva's (16, 14) and (20, 19); Addendum 319, section 4), a circuit using only the healthy direction
    was recompiled on the pruned map for nothing. Now a gate is flagged if its own direction is reported failed, or,
    for a symmetric gate (cz, swap, rzz, ...), either direction; a gate on a failed qubit is flagged as before.

42. **NEW, opt-in: `candidate_score="kraus"` chooses among the candidates of items 36-37 by `kraus_cost`, an estimate
    that is exact to first order for a noise model of thermal relaxation after each gate plus a depolarizing remainder.**
    The exploratory diagnosis of Addendum 339 found two approximations in `hybrid_cost` (item 38) that add up against
    one placement (HOLD6's H4: FakeAlgiers 4-qubit GHZ chains, 6.7% worse than the floor-placed candidate):
    amplitude damping counted on the state before each gate, and the remainder counted as if the gate's qubits were
    maximally mixed. Rescoring HOLD5's circuits in-sample, the choice by this estimate equalled the measured best
    on all nine devices, to four decimals.
    - `kraus_cost(circ, target)`: on the noiseless state just after each gate with a reported duration, for each of
      its qubits with T1 (T2 capped at 2 T1), 1 - sum_k |Tr(rho K_k)|^2 over the Kraus operators of amplitude
      damping (gamma = 1 - exp(-t / T1)) followed by pure dephasing (p_phi = (1 - exp(-t / T_phi)) / 2,
      1 / T_phi = 1 / T2 - 1 / (2 T1)); plus, per gate, the reported error above the thermal floor as a depolarizing
      channel at the gate qubits' reduced purity: lambda (1 - Tr rho_S^2 / d), lambda = excess d / (d - 1). The same
      terms as `kraus_pur` in Addendum 339's `h4_diag.py`. None above RESYNTH_MAX_QUBITS touched qubits or when an
      instruction has no matrix.
    - `candidate_score="kraus"` uses it in `_choose`; everything else is as in items 37-39. Other values change
      nothing.
    It does not include item 40's readout term, so the recommended call stays `candidate_score="hybrid"`.
    KRAUS (Addenda 359, 361): on 13,554 fresh circuits on nine fake devices it chose at least as well as `hybrid_cost`
    on every device and repaired HOLD6's H4 case (6.3% lower infidelity), for 3-5% more compile time.
    Its limit: it reproduces the simulator's own noise model (Aer), so a noisy-simulation test favours it by
    construction; on hardware relaxation acts during the gate and there are errors that model lacks.

43. **Item 31's recompile no longer raises when no placement avoids the failed elements.** With a `target`, an output
    that uses a failed element is compiled again on the pruned coupling map. When the circuit needs more of the
    device than the pruned map offers -- a circuit on every qubit of FakeKingston, as in the PennyLane loop of
    Addendum 254 -- the layout search finds nothing, Qiskit's SabreLayout raises "A connected component of the
    DAGCircuit is too large for any of the connected components in the coupling map", and 2026-10-06.1 passed
    that exception to the caller, so the recommended call and the AI front end failed where the default call
    succeeds (PL-REDO smoke, 2026-10-06). Now that recompile (and compare_floor's) catches `TranspilerError`, keeps
    the first output, counts it in `PRUNE_STATS["unavoidable"]` and warns. Nothing else changes: where the
    recompile succeeds, or is not needed, the output is 2026-10-06.1's.

2026-10-06.c18 (candidate, Addendum 377):

44. **FIX (crash): item 39's checks no longer turn a wide instruction into a matrix.** `_ops_of` called
    `to_matrix()` on every instruction of the logical circuit, and `_implements` did so before its 16-qubit limit
    was applied. Benchpress's HamLib tests hand the compiler one `PauliEvolutionGate` on all qubits. With the
    recommended call (`compare_floor` or `compare_level3`), and so through the AI front end, 2026-10-06.2:
    - aborted the Python process on a 48-qubit Hamiltonian on FakeTorino: Qiskit asked for (2^48 + 1) x 8 bytes,
      the row index of a 2^48-row sparse matrix, and Rust's allocator aborted (no exception reaches the caller);
    - ran past 600 s on a 14-qubit one (a dense 2^14 x 2^14 matrix), where the default call took 0.3 s
      (BP-PROBE, 2026-10-06). The default call never runs these checks and was not affected.
    Now the qubit limits are applied first (the logical circuit's width in `_implements`, the touched qubits in
    `_same_action`), and an instruction on more than `EXACT_MAX_GATE_QUBITS` (6) qubits is expanded through its
    `definition`, recursively, instead of being made a matrix. No definition, or an expansion longer than
    `EXACT_MAX_OPS` instructions, makes the check one that cannot be made, which refuses the candidate as before.
    For a `PauliEvolutionGate` the definition is its synthesis (a product formula), the circuit every compiler
    builds from it, whereas `to_matrix()` is the exact exponential; the two agree when the terms commute.
    Nothing else changes: on circuits whose instructions act on at most 6 qubits every matrix, every check and
    every output is 2026-10-06.2's.

2026-10-06.c17 (candidate, Addendum 379):

45. **SPEED: alternatives that item 39 cannot check are no longer built.** A check that cannot be made refuses
    the candidate (item 39), and `_implements` cannot be made for a logical circuit of more than
    `RESYNTH_MAX_QUBITS` (16) qubits, nor `_same_action` for a circuit touching more than 16. Yet the floor
    candidate (item 37: a second full compile) and Qiskit level 3 (item 36) were still built for such circuits,
    and item 35's re-synthesis for such outputs, only to be refused. On FakeAuckland at full occupancy (27 qubits)
    the recommended call took 6.4 s and returned the default call's circuit on 30 of 30 laps (Addendum 376, G9);
    on a 48-qubit Hamiltonian it built level 3 and the floor and refused both (Addendum 377). Now:
    - with more than 16 logical qubits, neither the floor candidate nor level 3 is built;
    - item 35's re-synthesis is not built for a circuit touching more than 16 qubits.
    Each skip is counted in `SKIP_STATS`. The output is the one 2026-10-06.3 returns, because what is skipped is
    exactly what it refused; only counters (`COMPARE_STATS`, `EXACT_STATS`, `RESYNTH_STATS`, `PRUNE_STATS` of the
    floor's compile) and a `callback`'s calls during the floor's compile differ.

2026-10-07.c19 (candidate, Addendum 383):

46. **SPEED: the state-vector estimates and checks apply fewer, larger matrices.** Up to 16 qubits the recommended
    call simulates its circuits gate by gate on a numpy state vector: item 39's checks (`_same_action`,
    `_implements`) and the estimates of items 35 and 38 (`excitation_cost`, `hybrid_cost`). At 16 qubits that took
    20-60 s per compile for Hamiltonian and QFT circuits (SKIP, Addendum 380), almost all of it in these
    simulations (profile, Addendum 383). Now:
    - **checks:** before simulating, every single-qubit gate is multiplied into the next gate on its qubit that acts
      on more qubits, and consecutive gates on the same qubits are multiplied into one (`_fuse_ops`); single-qubit
      gates left at the end are kept as they are. The action is the same; only the order of floating-point
      operations changes.
    - **estimates:** a single-qubit gate whose matrix is diagonal and which has no reported error and no duration
      (an `rz` on IBM devices) is multiplied into the next gate on its qubit instead of being applied on its own. It
      adds nothing to either estimate, and a diagonal gate changes neither the populations the estimates read
      before a gate nor the state after the gate it is multiplied into, so every term is the same up to rounding.
    - **`hybrid_cost` and `excitation_cost`** read a qubit's populations with one pass over the state
      (`_populations`); `hybrid_cost` no longer builds the qubit's reduced density matrix, of which it used only
      the diagonal.
    - **ties:** the choice among candidates (`_choose`), item 35's "select" and item 36's comparison now treat two
      estimates within a relative 1e-12 (`ESTIMATE_TIE_TOL`) as equal, and a tie keeps the earlier candidate, as an
      exact tie always did. Without this the new rounding broke exact ties: FUSE's smoke run (Addendum 383) found a
      16-qubit ring on FakeKingston whose own circuit and level 3's differed only by two `rz` gates, which add
      nothing to the estimate; 2026-10-06.4 scored them equal to the bit and kept its own, while the first version
      of this candidate scored level 3's lower by 2e-17 and took it.
    `excitation_cost` now builds each gate's matrix before its cost terms (it returned None for a gate without a
    matrix either way). `pauli_cost` and `kraus_cost` are unchanged. The decisions taken from these numbers can
    differ from 2026-10-06.4's only where 2026-10-06.4 saw two estimates differ by less than 1e-12 (relative)
    without being equal, or a check land within rounding of `EXACT_TOL`.

2026-10-07.c20 (candidate, based on c19):

47. **SPEED: item 39's checks are made only where their result can change the output.** Item 39 checks, by
    simulation, that a circuit the recommended call could return does what the input does: `_same_action` for item
    35's re-synthesis, `_implements` for the floor candidate (item 37) and for Qiskit level 3 (item 36). Up to
    2026-10-07.c19 every check was made before the estimates, so a candidate was simulated two more times (two
    seeded product states) even when its estimate then lost and it was not returned. Now the estimates come first:
    - **item 35's "select"** (`_select_resynthesis`): the re-synthesised circuit is built and checked against the
      target as before (`_resynthesis_candidate`), then both estimates are made, and `_same_action` is made only if
      the re-synthesised circuit's estimate is lower, i.e. only when it would be returned. If the check fails, the
      original is returned, as before. `final_resynthesis=True` (always) is unchanged (`_final_resynthesis`).
    - **item 36** (`_compare_level3`): `_implements` is made on level 3's circuit only if its estimate is lower.
    - **item 37** (the candidates of `compare_floor`/`compare_level3` with any `candidate_score`): the estimates of
      the release's circuit and of every acceptable candidate are made first; `_implements` is made only on a
      candidate whose estimate is None or not higher than the release's by more than `ESTIMATE_TIE_TOL`. A candidate
      whose estimate is higher than the release's by more than that can be neither the lowest nor within the tie band
      of the lowest, and leaving it out does not change the lowest, so whether it implements the input cannot change
      the choice (`_choose_lazy`). If the release's own estimate is None, the release's circuit is returned without
      any check, as 2026-10-07.c19 returned it either way.
    The output is 2026-10-07.c19's in every case. What differs:
    - fewer checks: `EXACT_STATS["checked"]` and the `refused_*` counts count only the checks made; a candidate that
      was not checked is counted in `RESYNTH_STATS["not_checked"]` or `COMPARE_STATS["not_checked"]`;
    - `RESYNTH_STATS["applied"]` under "select" counts re-synthesised circuits that passed the check (it counted those
      that passed it before the estimates), and `selected_original` now also counts those not checked;
    - estimates are made for candidates that 2026-10-07.c19 refused before estimating them (one simulation each,
      against the two of the check), and `COMPARE_STATS["not_estimable"]` counts a None estimate whenever an
      acceptable candidate exists;
    - in item 37's comparison a candidate refused by `_implements` is counted as before (`floor_refused`,
      `level3_refused`).

2026-10-07.c21 (candidate, based on c20):

48. **QUALITY: instructions on three or more qubits are expanded before PSF-Zero's own pipeline.** The default
    call handed instructions on more than two qubits (`ccx`, a `PauliEvolutionGate`, `qft`, `EfficientSU2`, a
    Quantum Volume layer, a QASMBench `add4`) to its pipeline as they were. On Benchpress and SKIP circuits that
    contain them it returned up to 6.3 times Qiskit level 2's two-qubit gates (EfficientSU2 on 100 qubits: 1,887
    against 300; DISPATCH-PROBE, 2026-10-07). Expanding only those instructions first, with Qiskit's
    `Unroll3qOrMore` (each through its `definition`, recursively, until no instruction acts on more than two
    qubits; two-qubit gates are left as they are), lowered the two-qubit count on every such circuit of that probe
    (to 1.00-1.38 of level 2's) and changed nothing elsewhere (DISPATCH-PROBE 2). Expanding the whole circuit to the
    basis instead also broke two-qubit structure the pipeline uses (a flat 100-qubit QFT: 17,544 against 13,456).
    - Where: at the start of the pipeline without a `target` (`_unroll_wide`), so in the default call, in the
      first compile of the recommended call and in its floor candidate. Qiskit level 3's candidate (item 36) and
      item 39's checks still take the input as given.
    - Only when the input has such an instruction (barriers aside): every other circuit reaches the pipeline as the
      same object, so its output is 2026-10-07.c20's.
    - An instruction Qiskit cannot expand (no definition) leaves the input as it was, with a warning; counts in
      `UNROLL_STATS`.
    Exploratory so far: DISPATCH-PROBE's circuits chose this change, so they cannot test it.

2026-10-07.c22 (candidate, based on c21):

49. **SPEED: the estimates follow single-qubit gates on each qubit's 2x2 reduced state.** After items 46-47 the
    recommended call still spent most of its time at 16 qubits in `excitation_cost` and `hybrid_cost` (profile,
    2026-10-07: 74 of 87 s for a 16-qubit Hamiltonian on FakeTorino, two thirds of it reading populations and
    applying single-qubit gates to the whole 2^16 state). Both estimates read, per gate, only the populations of the
    gate's own qubits before it (and `hybrid_cost` after it). A single-qubit gate changes the reduced state of its own
    qubit only, as rho -> U rho U^dagger, and changes no other qubit's reduced state. Now:
    - each qubit's 2x2 reduced density matrix is kept (`_rho1`); it is |0><0| at the start, follows every
      single-qubit gate on the qubit by U rho U^dagger, and is read again from the state after every gate on more
      qubits that acts on it;
    - a single-qubit gate is not applied to the state; it is multiplied into the next gate on more qubits that acts
      on its qubit (as item 46 did for free diagonal gates only), and the populations it is charged for come from
      the 2x2 matrix;
    - every cost term is the one item 46 computed, up to rounding.
    `pauli_cost` and `kraus_cost` are unchanged. Decisions can differ from 2026-10-07.c21's only where two estimates
    lie within rounding of each other's tie band (`ESTIMATE_TIE_TOL`).

2026-10-07.c23 (candidate, based on c22):

50. **QUALITY: two-qubit gates that cancel through commutation are removed when that helps.** Benchpress's BV-like
    test (100 qubits, two CX from every qubit to one target) reduces to no two-qubit gate: CX gates that share a
    target commute, and each pair cancels. Qiskit level 2 removes them (`CommutativeCancellation`); PSF-Zero's
    pipeline, which routes at level 1, did not, and returned 392 two-qubit gates (BP-MOCK, Addendum 390). Applying
    the pass to every input is not neutral: on Hamiltonian inputs it removed no two-qubit gate yet changed the default
    call's count both ways (-6% to +5%), because it also merges and moves single-qubit gates (Addendum 393). Now, at
    the start of the pipeline without a `target` (after item 48's expansion):
    - Qiskit's `CommutativeCancellation` is run on the input (`_cancel_candidate`); if it removes no two-qubit gate,
      nothing else happens and the input goes on as before;
    - if it removes at least one, the pipeline is run on both the input and the cancelled input, and the result with
      fewer two-qubit gates is returned (the input's on a tie). Counts in `CANCEL_STATS`.
    The pass is exact (it removes or merges only gates whose product is the identity or a single rotation), so both
    results implement the input. Where it removes no two-qubit gate the output is 2026-10-07.c22's; where it removes
    some, the output has no more two-qubit gates than 2026-10-07.c22's would have, for twice the pipeline's time.
53. **SPEED: gate matrices are kept, and one-qubit gates are embedded without np.kron (candidate 2026-10-09.c26).**
    (Items 51 and 52 are candidates 2026-10-07.c24 and 2026-10-09.c25, not adopted; Addenda 406, 410-412.) The
    recommended call's estimates and exactness checks build a matrix for every instruction they read. In the profile
    of 2026-10-09 (rec_profile.py, 30 FakeTorino development tests), `_embed_1q` took 8.9% of the time and `_ops_of`
    7.6%. Now:
    - `_gate_matrix` keeps the matrix of a standard gate with numeric parameters, keyed by (name, parameters), for the
      last 4,096 keys, read-only; any other instruction is built as before;
    - `_embed_1q` returns the matrix itself on one qubit, and on two qubits forms the Kronecker product's products as
      one broadcast multiplication.
    The values are the same (the same products of the same numbers; at most the sign of a zero can differ, which
    changes no sum or product that is read), so every estimate and check returns what it returned. Nothing else
    changes.
56. **SPEED: no estimate for a candidate that cannot be checked (candidate 2026-10-10.c29).** A candidate circuit
    (re-synthesis, the floor-aware compile, Qiskit level 3's) replaces the release's circuit only if item 39's
    exactness check passes, and that check cannot be made -- it returns False -- when `_ops_of` gives up on a circuit
    (more than EXACT_MAX_OPS matrices, item 44) or more than RESYNTH_MAX_QUBITS qubits are involved (items 44-45).
    On hwb10 the release spent 819 s on estimates whose candidates could never pass that check (Addendum 420). Now,
    before a candidate is built or estimated, counts made without any matrix decide whether its check certainly
    cannot be made, and if so it is neither built nor estimated and the circuit in hand is kept:
    - `_n_narrow`: the instructions `_ops_of` turns into one matrix each (not barrier, measure or delay, and on at
      most EXACT_MAX_GATE_QUBITS qubits). Each of them adds one entry to its list or makes it give up, so more than
      EXACT_MAX_OPS of them means it gives up;
    - `_narrow_touched`: their qubits, which are all among the qubits of `_ops_of`'s list;
    - `_cannot_implement(qc, out)`: True only where `_implements(qc, out)` would return False;
    - used in `_choose_lazy` (a candidate that can never pass is dropped before the estimates), `_compare_level3`
      (level 3's circuit is not estimated), `_checkable_logical` (the floor-aware and level-3 compiles are not made
      when the input itself cannot be checked) and `_resynthesise` (the re-synthesis is not built when the circuit
      cannot be checked).
    Wherever the release returns a circuit, this returns the same circuit: every path the release takes from those
    points returns the circuit in hand. Where the skipped work would have raised, this returns the circuit in hand
    instead. Counters differ (FEASIBILITY_STATS counts the skips). Exploratory candidate 2026-10-09.c28 (Addendum 421)
    counted every instruction and qubit, including those of instructions `_ops_of` expands through their
    definitions, which can count more than `_ops_of` sees; these counts cannot.
57. **SPEED: the estimates' and checks' per-gate loops in Rust, when psf_zero_core57 is installed (part b;
    candidate 2026-10-10.c30).** (Part a, the reduced states in one NumPy pass, was tried in exploratory candidate
    c28 and saved little: Addendum 421.) `excitation_cost`, `hybrid_cost` and item 39's `_apply_ops` spend their
    time in a loop over the gates: Python's work per gate and NumPy's copies per amplitude. Now Python still reads
    the circuit -- the instructions, their qubits, `_gate_matrix`, the Target's error and duration, T1 and T2 -- and
    decides the cases that return None, then passes one byte buffer to `psf_zero_core57` (a module of its own, built
    from patches/psf_zero_core57_2026-10-10, so that the release's psf_zero_core is not changed). The core walks the
    gates exactly as the Python code does: single-qubit gates waiting with their reduced states (item 49), each wider
    gate applied to the state, a two-qubit gate and both of its qubits' reduced states in one pass, and every cost
    term in the same order. `hybrid_cost` adds `readout_cost` in Python, as before. The values agree with the Python
    code's to rounding (sums are formed in another order), not bit for bit, so this is not identity by
    construction; it is tested. Without psf_zero_core57, or for a call it cannot take (an instruction on more than
    six qubits), the Python code runs. CORE57_STATS counts the calls made each way.
58. **SAFETY: a circuit that uses an element the device reports as failed is never returned silently (candidate
    2026-10-10.c31).** An output with a gate on a failed coupler or qubit gives no usable result on the device, and
    nothing in the output says so. Item 31 already avoided failed elements, but only when the caller passed `target`
    (the recommended call), and item 43 then returned the failing circuit with a warning when no placement avoided
    them. Now:
    - `backend=` (new) supplies the device: its Target, and from it the coupling map and the basis, unless given.
      `coupling_map` may be left out when `target` or `backend` is given. With a device, the default call avoids
      failed elements exactly as item 31 does; with the same arguments as before, every output is unchanged.
    - Without a Target, the first call in a process warns once that failed elements cannot be avoided
      (`WARN_WITHOUT_TARGET` switches this off).
    - A qubit whose measurement error is >= `prune_max_error` is a failed qubit too (`_failed_elements`,
      `prune_coupling_map`): a result read from it is noise.
    - After item 31's recompile on the pruned map the output is checked again. If no placement avoids the failed
      elements, or the output still uses one, `on_failed_elements="raise"` (the default) raises `FailedElementsError`
      instead of returning it; `"keep"` returns it with item 43's warning, as before.
59. **EXACTNESS (part a: measurement only; candidate 2026-10-10.c31).** Item 39 accepts a circuit when its state
    infidelity against the reference is <= `EXACT_TOL` (1e-6). Dropping a Z rotation of angle theta, as Qiskit
    level 3's CommutativeCancellation does below |theta| = 1.2566e-4 (Addendum 247), changes the infidelity by at
    most theta^2 / 4 (4e-9), far inside that tolerance. Before the tolerance is tightened, the infidelities that
    exact paths actually produce are recorded: `EXACT_SEEN` keeps, per check, the largest infidelity found and
    where (`_same_action`, `_implements`), up to `EXACT_SEEN_MAX` entries. No decision changes.
60. **QUALITY ON THE DEVICE: with a Target, the default call places by the device's errors (candidate
    2026-10-10.c32).** With item 58 the default call avoids failed elements, but it chose among the working qubits
    without looking at their errors: on BP-FINAL's 106 FakeTorino tests its estimated success probability was 0.66
    (FakeTorino) and 0.84 (FakeKingston) of Qiskit level 2's (ESP-C31). `placement_refine` (item 33, Qiskit level 3's
    exact re-placement by the Target's errors) is now "auto" by default: on whenever a Target is given (by `target`
    or `backend`), off otherwise. `placement_refine=False` gives c31's default call with a Target; calls without a
    Target, and calls that pass `placement_refine` explicitly, are unchanged.
61. **SPEED: with a Target, one compile on the map without the failed elements (candidate 2026-10-10.c33).** Item 31
    compiled on the full map first and, when the result used a failed element, compiled again on the pruned map: on
    FakeTorino 63 of BP-FINAL's 106 tests were compiled twice (ESP-C31). Now, when the Target reports failed elements,
    the circuit is compiled once, on `prune_coupling_map(...)`, and checked as item 58 checks a recompile. When no
    placement exists there, "raise" raises FailedElementsError; "keep" compiles on the full map, with item 43's
    warning. `PRUNE_FIRST = False` restores c32's order. The output can differ from c32's where c32's first compile
    used no failed element (the router then saw the whole map).
62. **SPEED, and switches to measure the compiler's own steps (candidate 2026-10-10.c34).** Timers around the stages
    of c33's default call (STAGETIME, 2026-10-10) found the time in PSF-Zero's own steps, not in Qiskit's: the SWAP
    absorption after routing (item 30) took 30% of it, the compression 16-22%, and on small circuits the first call's
    layout search 23% (networkx's import). Here:
    - `_uses_failed` reads each qubit's index from a table instead of `find_bit` (the same answer, faster);
    - `ABSORB_SYNTH = "psf"` (default) synthesises the absorbed blocks as before; `"qiskit"` uses Qiskit's
      TwoQubitBasisDecomposer on CX (Rust; exact; the same CX count for a two-qubit unitary);
    - `COMPRESS = True` (default) compresses as before; False routes the input uncompressed.
    The defaults change no output. psf_smart_layout's two matching-size checks use rustworkx instead of networkx
    (the size of a maximum matching does not depend on the algorithm), in patches/psf_compile_c34_2026-10-10.
63. **SPEED: the SWAP absorption's blocks synthesised by Qiskit's decomposer (candidate 2026-10-10.c35).**
    `ABSORB_SYNTH` is "qiskit" by default. On BP-FINAL's 106 FakeTorino tests (ABLATE-C34, arm C34Q) every output had
    c34's two-qubit count, ESP was 1.005 of c34's, and the summed time 0.89 of it; the outputs differ in their
    single-qubit gates only. Both of PSF-Zero's own steps paid for themselves there: without the absorption 10% more
    two-qubit gates (ESP 0.918), without the compression 11% more (ESP 0.956). "psf" restores c34's synthesis.
64. **ACCURACY: routing at level 3 when the device is given (candidate 2026-10-10.c36).** `routing_optimization_level`
    is "auto": 3 when a Target is given (`backend=` or `target=`), 1 otherwise. In ROUTE-X (Addendum 439; release
    2026-10-10.3's plain call with backend= on BP-FINAL's 106 FakeTorino tests) level 3 raised ESP by 5% (FakeTorino)
    and 10% (FakeKingston) over level 1, to 0.975 and 0.985 of Qiskit level 3's, at 1.15-1.2 times the time. Without
    a Target nothing changes. An explicit integer keeps its meaning.
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
69. **ACCURACY: STEINER6's output is kept only where it is better (candidate 2026-10-11.c39).** When item 68 applies,
    the input is also compiled by the usual path (`steiner=False`), and the output with the lower summed
    -ln(1 - error) over all its operations on the device (the higher ESP) is returned; on a tie, the usual path's.
    C38-VAL (Addendum 445) found item 68 worse than the usual path on the dense two-local diagonal type (HamLib tsp
    in unary encoding: 0.87 of its ESP), which no structural rule from the data separated from types it wins on (a
    dense two-local gnp instance won by 2-3 times). Two fixed constructions compared by the device's own error table,
    not a search over seeds. Counts in `STEINER_STATS` ("chosen", "kept_usual").
70. **SPEED: a size limit for item 68 (candidate 2026-10-11.c39).** An evolution of more than 2000 terms takes the
    usual path. The largest commuting test runnable on a device so far had 449 terms (Addenda 444-445); C38-VAL's
    slowest call was a 5335-term, 144-qubit evolution whose ESP is near 0 either way (129 s against the usual path's
    10 s). Counts in `STEINER_STATS` ("too_large").
71. **ACCURACY: STEINER6 also for terms that do not all commute (prototype c40).** Item 68 required all terms to
    commute; with item 69 choosing by ESP, the construction is tried on every input it can read and kept only where it
    is better. Terms never pass an anticommuting earlier term, so the result is exact.
72. **ACCURACY: more inputs read as Pauli rotations (prototype c40).** Product formulas of any order and repetitions
    (LieTrotter, SuzukiTrotter, from their own `expand()`), sequences of PauliEvolutionGates and rotation gates (rz, rx,
    ry, rzz, rxx, ryy), composite instructions such as QAOAAnsatz (read through their definitions), a leading layer of
    single-qubit gates (as QAOA's Hadamards) and final measurements. Repeated terms are merged where exact (no
    anticommuting term in between): a commuting Hamiltonian over r Trotter steps becomes one step. The size limit of
    item 70 applies after merging.
73. **ACCURACY: circuits written as gates are read too (prototype c41).** A circuit that item 72 cannot read is read
    gate by gate: single-qubit gates before a qubit's first multi-qubit gate stay in front; Clifford gates after that
    (cx, cz, swap, ecr, h, s, sx, rotations by multiples of pi/2: any gate on one or two qubits that maps Paulis to
    Paulis, from its matrix) are carried to the end, turning each later rotation exp(-i a P) into exp(-i a C^dagger P C);
    rotations, Pauli evolutions, p, t and any other single-qubit gate (as rz.ry.rz) give the terms; other gates are read
    through their definitions. It applies when the Clifford gates left at the end act on each qubit alone; they are
    then placed after the terms at the qubits' final places. The output equals the input up to a global phase (the
    Clifford gates' phases are not followed). QASM files of QAOA and Trotter circuits (cx-rz-cx) are of this kind.
74. **ACCURACY: near-failed elements (prototype c42).** Item 31 avoids elements with error >= `prune_max_error`
    (0.5). An element just below it -- a coupler with 15% error, a qubit whose readout is wrong one time in five -- was
    used like any other, as often as routing chose (C41-VAL: on FakeBoston one Heisenberg circuit crossed a 15% coupler
    hundreds of times, ESP 2e-6 where an error-aware placement got 0.04). With the device's Target, when the output uses
    an element with error in [`near_failed_error`, `prune_max_error`) -- a two-qubit gate on such a coupler, any gate
    on such a qubit (sx error), or a measurement of a qubit with such a readout error -- the input is compiled again
    with every element of error >= `near_failed_error` avoided, and the output with the lower summed -ln(1 - error)
    (item 69's measure) is returned; on a tie, or when the second compile cannot avoid them, the first. Default 0.1
    (fixed before any measurement of it: an order of magnitude above these devices' median two-qubit errors of
    0.001-0.004, a fifth of the failed threshold); None turns it off. Counts in `NEAR_STATS`.
    **Revised in prototype c43:** the second compile avoids only the near-failed elements the first output used (it
    sees a copy of the Target with those marked failed). In C42-VAL, avoiding every element of error >= 0.1 at once left
    no placement in 100 of 198 attempts (FakeTorino has 19 qubits with readout error above 0.1).
75. **REPRODUCIBILITY: a deterministic exact re-placement (prototype c43).** Qiskit's VF2PostLayout, with a fixed seed,
    returns different layouts from call to call (C42-VAL, Addendum 450): with the routing call's own Target, which has
    no errors, it chooses among equal layouts by an order that changes between calls. In its place -- in Qiskit's preset
    pass managers built here, in item 33's re-placement and in item 68's relabel -- `_DeterministicPostLayout` scores
    subgraph matchings found by rustworkx's VF2 in a fixed order (`id_order=True`) by the summed -ln(1 - error) of
    every operation on the device's own Target, places qubits without two-qubit gates one by one on their cheapest free
    qubit, and applies the lowest score if it is below the current placement's. Without a device Target nothing is
    re-placed (there are no errors to place by). The same input then gives the same output. Counts in `DET_STATS`.
76. **ACCURACY: permutations left at the end (prototype c43).** Item 73 required the Clifford gates left at the end to
    act on each qubit alone. They may now also permute the qubits (as QFT's final swaps do): the permutation costs no
    gate, it is carried into the measurements and the output's final layout. Counts in `STEINER_STATS` ("permuted").
    **Item 75 revised in prototype c44:** the search runs three times, each up to the trial limit: in c43's index order;
    with the device's qubits ordered by their own error (sx and readout plus the mean of their couplers', best first)
    and the circuit's by two-qubit gate count (busiest first); and in rustworkx's own VF2 order (`id_order=False`). The
    lowest score of all is applied if below the current placement's. Each order finds placements the others miss. C43-VAL: with the index order alone, 2500
    trials missed the placements Qiskit's pass found on a 48-qubit random circuit (0.48-0.90 of the release's ESP).
77. **REPRODUCIBILITY: no time budget in the layout search (prototype c44).** `layout_search_time_budget_s` now defaults
    to None: the layout search stops by its call limits only (`layout_search_call_limit` per ordering,
    `layout_search_fallback_call_limit` for VF2's heuristic), so its result no longer depends on the machine's speed. A
    number restores the time budget. Measured before the change: of 154 compiles, one (dtc_100 on FakeTorino) reached
    the 2 s budget, without a layout either way (5.4 s without the budget). The packing search for paired and 3-qubit
    path interactions (psf_smart_layout) keeps its own short time limit; it did not bind in those 154 compiles.
78. **ACCURACY: item 75's re-placement at Qiskit's routing stage, for item 80 (prototype c45).** Qiskit's preset pass
    managers re-place the routed circuit right after routing, before it is translated to the device's gates
    (`strict_direction=False`). Item 75 gave up there whenever the circuit had a gate the device does not run natively
    (swap, cx on a cz device, h, ...), that is, on almost every circuit. In item 80's level-3 compile
    (`_DeterministicVF2(target, routing_stage=True)`) such gates are now scored by what they translate to: every
    two-qubit gate as 1-3 native two-qubit gates on its coupler (swap 3, gates locally equivalent to cx 1, others 2:
    `_NATIVE2_COUNT`), every single-qubit gate but measure as 0-2 sx gates on its qubit (`_SX_COUNT`); an element of
    error >= `DET_FAILED_ERROR` (0.5) is never placed on, nor is any qubit with an operation placed on a qubit whose
    sx, x or readout error is (a qubit with only rz and measure there gets sx gates later; found on FakeBoston, qubit
    146). Qiskit's preset asks for up to 250,000 trials (30 million VF2 calls); scored in Python that took 38 s on a
    100-qubit Bernstein-Vazirani circuit, so each search order is capped at `DET_TRIALS_CAP` = 25,000 trials and
    `DET_CALL_CAP` = 3,000,000 calls (on five circuits: the same ESP as 250,000 trials, 4.8 s against 38 s on that
    circuit; 2,500 trials lost 22% there).
    Diagnosis: on gen_random_n48 and FakeBoston, Qiskit level 3's ESP is 0.029 with its own pass, 5.5e-8 without it,
    and 0.029 with this one. **PSF-Zero's own pipeline keeps c44's behaviour** (the routing-stage call is skipped as
    before): tried there, it did not help (ESP 0.996 times c44's on 118 runnable tests of C44-VAL, development data)
    and lost 45% on one (Bernstein-Vazirani n = 100, FakeTorino): the proxy's best placement before translation
    narrowed item 33's later search.
79. **FIX: the basis gates from a Target given alone (prototype c45).** `basis_gates` was taken from the Target only
    when `backend=` was given; with `target=` alone the output was routed but not translated (swap, h, cp, ... left in
    it: not runnable on the device). It is now taken from the Target in both cases.
80. **ACCURACY: Qiskit level 3 as a candidate, by ESP (prototype c45).** With the device (`backend=` or `target=`, and
    no `coupling_map` or `initial_layout` of the caller's: `level3_candidate="auto"`), the input is also compiled by
    Qiskit's level-3 preset on the device's Target, under item 75's deterministic re-placement, and that circuit is
    returned when its summed -ln(1 - error) on the device is lower than PSF-Zero's (a tie keeps PSF-Zero's). It is
    refused if it uses a failed element, or, where item 39's check can be made (at most `RESYNTH_MAX_QUBITS` qubits, no
    measurement), if it does not implement the input; above that size Qiskit's circuit is not checked (counted as
    "unverified" in `L3C_STATS`). `level3_candidate=False` turns it off. This is a stopgap, not a method: where it is
    chosen, PSF-Zero's own pipeline lost to Qiskit's on this device, mostly by an error-blind layout (item 78's
    diagnosis). Item 36's opt-in `compare_level3` (by `excitation_cost`, small circuits) is unchanged.
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
86. **DURATION: the device's native fractional gates (prototype c50).** IBM's Heron devices offer rzz(theta), a ZZ
    rotation in one two-qubit pulse of a CZ's length (68 ns on ibm_marrakesh, 2026-10-13), and rx(theta) in one
    single-qubit pulse (36 ns), when the backend is loaded with `use_fractional_gates=True`. A ZZ rotation built from
    CZ gates takes two CZ and the single-qubit gates between them. When the Target offers rzz (`FRACTIONAL = "auto"`),
    the finished circuit is re-synthesised on its own physical qubits and layout: two-qubit blocks consolidated and
    re-synthesised (Qiskit's ConsolidateBlocks and UnitarySynthesis on a copy of the Target with every error removed,
    so no reported error chooses), single-qubit runs merged, rzz angles folded into the executable range
    0 < theta <= pi/2 (qiskit-ibm-runtime's FoldRzzAngle). The result is kept when it is shorter (item 84) and, where
    item 39's check can be made (at most RESYNTH_MAX_QUBITS touched qubits), acts as the circuit before it; above that
    size it is kept unverified (counted), as item 80 keeps Qiskit's circuit. The re-synthesis runs once, on the
    outermost call (`compile_for_hardware` wraps c49's function, now `_compile_for_hardware_c49`). Without rzz in the
    Target, the output is c49's. Counts in `FRAC_STATS`.
    **Revised in prototype c51:** C50-VAL (Addendum 480) found rzz angles outside 0 < theta <= pi/2 in 69 of 90 of
    c50's outputs: Qiskit's own stages inside the compiler (routing, item 80's level-3 candidate, translation) emit
    rzz with any angle when the Target offers it; only a qiskit-ibm-runtime backend object folds them by itself.
    Every circuit returned on a Target with rzz is now folded (`_fold`) and checked; the re-synthesis is compared with
    the folded circuit. `FRACTIONAL = False` turns off the re-synthesis only; the folding stays, since an unfolded
    rzz does not run.
87. **DURATION: a chain Hamiltonian built by formula as a brickwork (workplace proposal N1, on c51).** W2
    (2026-10-14) found that Qiskit level 3 given a hand-written brickwork circuit equals PSF-Zero on TFIM chains up to
    n = 50, and beats it at n = 100 (item 70's size limit sent 10 Trotter steps to the usual path: 82 times longer; on
    FakeMarrakesh STEINER6's placement around the failed couplers was not a clean path: 2.9-4.1 times longer). With the
    device given (`steiner` "auto" true) and its two-qubit gate cz, an input whose terms (read by items 72-73, merged
    where exact) act on one or two qubits, the two-qubit ones forming one path through all the circuit's qubits, is
    compiled without STEINER6 and without item 70's limit: W5's cut (failed qubits and couplers removed, largest
    connected component kept); the chain on the longest surviving piece of Heron's snake, centred, or else on the
    first path rustworkx's VF2 finds with nodes in index order (no score) -- both tried first on the couplers of the
    device's most common two-qubit gate length (gate lengths are fixed per device), then on all couplers; every term
    at level 1 + the largest level of an earlier term that shares a qubit and does not commute with it, emitted by
    level, even edges of the chain, then odd edges, then single-qubit terms (exact: every non-commuting pair keeps its
    order); Qiskit's level-3 preset translates on that fixed layout with no routing, given the coupling map and basis
    only. On W2's 48 cells (c49 + N1): CZ at or below the hand-written brickwork's in 48, duration in 46 (0.80-1.02 of
    it); the two exceptions are FakeMarrakesh n = 100 (8 and 10 steps, +0.2% and +1.6%), where every 100-qubit path
    uses 80 ns couplers (accepted by home: the path is not chosen by duration). The brickwork is compared with c51's
    own output for the same call (c49, then item 86) by item 84's rule and returned unless c51's is shorter (a tie
    keeps the brickwork). The comparison is made in the outermost wrapper (`compile_for_hardware`, which wraps c51's
    `_compile_for_hardware_c51`); nested calls go straight to c51. Any other input, or a chain with no path of n
    qubits, goes on as by c51. `CHAIN_BRICKWORK = False` restores c51. Counts in `CHAIN_STATS`.
88. **DURATION: item 87's terms as the device's rzz / rx, the signs in a Pauli frame (workplace proposal, on
    c51).** With rzz and rx in the Target (`CHAIN_FRAME`), item 87 emits each two-qubit term as one rzz between
    single-qubit basis changes and each X term as rx, drops diagonal gates directly before measurements, and makes
    every rzz angle executable (0 < theta <= pi/2) without a second rzz: angles reduced to (-pi, pi], those of size
    above pi/2 moved by pi with Z x Z (two virtual rz(pi)), and a negative angle made positive by one x before the rzz
    (merged into the single-qubit run before it) while the x after it is kept in a Pauli X frame (single-qubit gates
    conjugated, later rzz signs flipped). The frame is undone inside the circuit, by one x before the measurement or
    at the end of an unmeasured qubit, merged into the last single-qubit gate where there is one: the circuit outputs
    the input's distribution, and acts as its unitary when not measured; no measured bit is flipped afterwards. Every
    single-qubit run is merged into rz rx rz (one pulse). An angle left outside the range (none seen) goes through
    item 86's `_fold`. Without rzz or rx in the Target, item 87's output is the CZ brickwork (N1). `CHAIN_FRAME =
    False` gives the CZ brickwork on every Target.
    **In prototype c53 (home's merge onto c52, Addendum 483):** "c51's own output" in items 87-88 is c52's
    (`_compile_for_hardware_c51` wraps c52's body, items 89-90 included); the code is the workplace's, unchanged.
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
91. **DURATION: an interaction graph that embeds in the device, built by edge colouring (workplace proposal Q1, on
    c53).** Item 87 generalised from a path to any graph heavy-hex contains. With item 87's conditions (device given,
    cz, none of the caller's coupling_map / initial_layout / final_resynthesis / compare_level3 / compare_floor) and an
    input whose terms (items 72-73's reader, merged where exact) act on one or two qubits but are not one chain: the
    interaction graph (max degree 3, bipartite) is placed by the first embedding rustworkx's VF2 finds in the cut
    device (nodes in index order, no score), first on the couplers of the device's most common gate length, then on
    all of them. Its edges are coloured with max-degree colours (Koenig; deterministic alternating-path colouring,
    at most 3 on heavy-hex), each colour a matching. Every term keeps item 87's level (1 + the largest level of an
    earlier term that shares a qubit and does not commute with it: exact); within a level the two-qubit terms go by
    colour, one layer per colour, then the single-qubit terms. Emitted as item 87 (Qiskit level 3 on the fixed layout,
    CZ) or, with rzz and rx in the Target, item 88 (rzz, rx, Pauli frame undone inside the circuit), and compared with
    c53's own output by item 87's rule. Heavy-hex never has two degree-3 qubits side by side and has no cycle shorter
    than 12, so a graph with adjacent degree-3 vertices (most random trees), a cycle of length 4-11, or a vertex of
    degree 4 (a square lattice, C52-VAL's heis2d) does not embed: then nothing changes. `GRAPH_BRICKWORK = False`
    turns it off. Counts in `GRAPH_STATS`.
93. **CORRECTNESS: every output checked exact where it can be, with fallbacks (prototype c54).** The workplace's Q3
    (Addendum 485) traced a total variation of 1.4e-5 in one AI-front-end output to Qiskit's two-qubit synthesis: its
    Weyl decomposition snaps a nearly degenerate block when the snap costs at most 1e-9 in gate fidelity, even at
    approximation_degree 1.0, here inside STEINER6's final level-3 transpile. Nothing checked whole outputs before.
    On the outermost call, the output is now checked (`_exact93`): a measured circuit by its exact outcome
    distribution against the input's own synthesis (largest probability difference at most EXACT_CHECK_TOL = 1e-6,
    the tolerance every validation since C48-VAL has used), an unmeasured one by item 39's check at the same
    tolerance against the input's own synthesis (a Pauli evolution's product formula, not its exact exponential); only with at most RESYNTH_MAX_QUBITS active qubits and final measurements, otherwise not checked
    (counted). An output that fails is replaced by the usual path's (`steiner=False, level3_candidate=False`) or else
    Qiskit's level-1 preset on the device (refused if it uses a failed element), whichever is first exact; if none is,
    TranspilerError is raised. At 1e-9 the check refused ordinary outputs: the same synthesis leaves differences of
    about 1e-7 (6.1e-7 on c48's 12-qubit TFIM ring on FakeTorino) in many outputs, which only a synthesis without the
    snap would remove; the largest difference accepted is recorded (`EXACT93_STATS["largest_passed"]`).
    `EXACT_CHECK = False` restores c53 + item 91. Counts in `EXACT93_STATS`.
94. **DURATION: runs of commuting ZZ terms re-layered by edge colouring (workplace proposal R1, on c54; off by default since c56, dropped: Addendum 488).** The family
    map (Addendum 484) found the user's best ahead on 3-regular QAOA, always from the hand-layered circuit. The
    workplace's R1 separated the two effects with one fixed seed: the hand-layered circuit through Qiskit level 3 at
    seed 0 alone beat c54 on qaoa_n12_p2 (5.53 against 5.98 us, F 4.07 against 4.64) but not on qaoa_n20_p2 or
    qaoa3reg_n20_r3 (6.48 / 13.38 against 6.31 / 11.08), where the user's margin comes from trying seeds (placement).
    No fixed placement tried by formula came near (a first Hamiltonian path on item 87's line, reverse Cuthill-McKee
    on the line or on a compact patch: 1.5-3 times c54). On the outermost call, the input's maximal runs of pairwise
    commuting ZZ terms (items 72-73's reader, in their order) are regrouped into layers of disjoint pairs by a
    first-fit edge colouring in that order, written as rzz (other terms unchanged, as their own PauliEvolutionGate):
    exact by construction (only commuting terms move). c54 compiles the rewrite too, and its output is returned when
    item 84's rule prefers it and item 93's check against the input does not refuse it. No seed, placement or score
    is searched: two fixed inputs, chosen by duration. `ZZ_RELAYER = False` restores c54. Counts in `ZZ94_STATS`.
95. **CORRECTNESS: item 93's distribution check up to 20 active qubits (prototype c55).** The workplace's R2
    (Addendum 486) scanned C52-VAL's 240 cells on c54: every output is within 5.1e-14 of its input's distribution
    except tfim_n20_r3's six (1.5e-8 on the Target without fractional gates on every device, 3.2e-7 with them on
    FakeTorino), left by Qiskit's two-qubit synthesis and CommutativeCancellation inside STEINER6's final level-3
    transpile. They have 20 active qubits, so item 93, stopping at RESYNTH_MAX_QUBITS = 16, never saw them. A
    measured circuit's check now goes up to EXACT_CHECK_MAX_QUBITS = 20 (a statevector of 16 MB); an unmeasured one
    stays at RESYNTH_MAX_QUBITS (it compares unitaries). The tolerance stays 1e-6: at 1e-9 these six outputs would
    fall back to the usual path, 17-21 times longer (R2), and the exact synthesis that removes them costs about 10%
    and is not built into the compiler yet. `EXACT_CHECK_MAX_QUBITS = RESYNTH_MAX_QUBITS` restores c54's reach.
96. **SCOPE: a circuit PSF-Zero does not build by formula is handed to Qiskit's level 3 unchanged (prototype c56).**
    On real devices (HW1, HW2, TIME1, FRAC1; 54 pairs pooled after the fact, Addendum 488) PSF-Zero's construction
    won 19 of 23 pairs on Ising chains (F 1.21 x) but its compiler choices alone won 9 of 25 on QAOA and
    Barabasi-Albert circuits (F 0.78 x); TIME1 (Addendum 487) traced the losses to placement where routing is needed,
    which needs search or calibration, which PSF-Zero does not use by design. The owner declared the scope the same
    day: PSF-Zero competes only where it builds by formula. With the device's Target, the input's two-qubit
    interaction graph -- the terms' supports when items 72-73's reader reads it as Pauli terms on one or two qubits
    (item 91's `_graph_read`), else every two-qubit operation as written; an operation on more than two qubits is out
    of scope -- is tested for an embedding in the device's working coupling graph (failed couplers and qubits
    removed; rustworkx's first VF2 match, nodes in index order, no score; not found within 2,000,000 states counts as
    none). Embeds, or no two-qubit operation: the call is c55's (item 94 off). Otherwise the input goes to Qiskit's
    `generate_preset_pass_manager(3, target, seed_transpiler)` unchanged (seed 0 when none is given, so the output is
    reproducible); if that output uses a failed element, Qiskit's level 3 on the device without its failed qubits
    (item 85's working Target), written back on the device's indices; only if that fails too, c55's own path
    (counted). The handed-on output is checked by item 93's test where it can be and returned with a warning if
    inexact. Arguments such as `steiner=` or `level3_candidate=` do not apply to a handed-on circuit. Without a
    Target nothing changes. `SCOPE_GATE = False` restores c55 with item 94 off. Counts in `SCOPE96_STATS`.
    Revised in c57: C56-VAL (Addendum 490) refuted its V6. On a Target with IBM's fractional gates, Qiskit's level 3
    emitted rzz angles outside the executable range 0 < theta <= pi/2 in 79 of the 80 handed-on outputs, because
    only a qiskit-ibm-runtime backend object folds them by itself, not a Target. With rzz in the Target the handed-on
    output is now folded by item 86's `_fold` (qiskit-ibm-runtime's FoldRzzAngle, then single-qubit runs merged), as
    IBM's backend would do; if folding fails, c55's own path is used (counted as "fold_failed_fallback"). On a Target
    without rzz nothing changes.
97. **SPEED: formula only inside the scope (prototype c58).** Measured on 2026-10-15 (Addendum 493): inside the scope
    the release spent 0.6-37 s per call, most of it on item 93/95's state-vector checks (17-30 s at 20 qubits) and on
    candidates that lose, while item 87's or item 91's formula alone took 27-73 ms. With the device's Target and an
    input inside item 96's scope, the output of `_chain_brickwork` or else `_graph_brickwork` is returned as it is:
    nothing else is built or compared, and nothing is simulated (`FORMULA_CHECK = True` adds item 93's check). An
    in-scope input the formulas do not build is handed to Qiskit's level 3 as item 96 does. `FORMULA_ONLY = False`
    restores 2026-10-14.1. Counts in `FORMULA97_STATS`.
98. **SCOPE: the gate closed (prototype c59).** The owner's decision of 2026-10-15 (Addendum 494): PSF-Zero stops
    handing circuits to Qiskit. With `GATE_CLOSED = True` (default), `compile_for_hardware` builds an in-scope input by
    formula (item 97) and refuses every other call with `OutOfScopeError` (a TranspilerError) saying why: no device
    Target; an operation on more than two qubits; an interaction graph that does not embed in the device's working
    couplers; or an in-scope input neither formula reads. The caller compiles those with Qiskit (or another compiler)
    themselves. `GATE_CLOSED = False` restores c58. Counts in `GATE98_STATS`.
99. **SCOPE: native gate circuits on an embeddable graph, by formula (workplace proposal c60).** CZ / CX chains and
    lattices (hardware-efficient ansaetze, `real_amplitudes`, `efficient_su2` expanded, CZ-ring classifiers) are
    inside item 96's scope but neither item 87 nor item 91 reads them, so c59 refused them. `_native_brickwork`
    builds them as written in one fixed pass: item 91's placement (first VF2 embedding), commuting diagonal gates
    ordered by item 91's edge colouring, each two-qubit gate in a fixed native form (cz; rzz native and folded, or
    cz rz cz; others through RZZ or the closed-form CZ decomposition, each multiplied out and checked), single-qubit
    runs merged in closed form (ZYZ). No pass manager, no comparison, no simulation. Outputs of items 87 and 91 are
    unchanged. `NATIVE_BRICKWORK = False` restores c59. Counts in `NATIVE99_STATS`.
101. **SCOPE: items 87/91's CZ path emitted by PSF-Zero (workplace proposal c62).** Without item 88's frame (no rzz
    and rx in the Target), items 87 and 91 handed their logical circuit to Qiskit's level-3 preset on the fixed layout:
    the last outside stage inside the scope. `_cz101_emit` emits it instead, by block kind: one two-qubit term (a
    single interaction) as a local Clifford around RZZ in closed form; several terms on one pair (with the
    single-qubit operations between them) by the Weyl decomposition with closed-form circuits; diagonal operations
    directly before the measurements left out; runs merged in closed form (ZYZ, theta = pi as one x pulse). Placement
    and order are items 87/91's. Measured in C62-VAL (0.76-0.83 x c60 on Ising models in C61-VAL). `CZ_OWN_EMIT = False`
    restores c60. Counts in `CZ101_STATS`.
100. **SPEED: the compile template (workplace proposal c61; planned binding c63).** `compile_template(qc, target)`
    builds an in-scope circuit once, by the formula path (item 87, 91 or 99), into a parametrised circuit on the
    device; `CompileTemplate.bind_values` / `.bind(qc)` give the compiled circuit for new angles in closed form only
    (no pass manager, no simulation). A shape signature covers every value-dependent branch; a new shape is built and
    counted, a shape seen before is taken from the template's cache. c63: each shape keeps its emission plan (every
    run's factors, class and slots), so a bind multiplies and merges the runs without re-emitting -- the same
    arithmetic, the same output. Item 99's outputs are c62's value for value; items 87/91's frame path equals c62's in
    duration and two-qubit count; their CZ path uses c61's fast closed form for XX + YY + ZZ blocks (not item 101's
    Weyl form); a run with a factor of a general Weyl block is pinned to the full form, so the decomposition's
    discrete branches do not change the shape. c62's default call is unchanged. Counts in `TEMPLATE100_STATS`.

```
