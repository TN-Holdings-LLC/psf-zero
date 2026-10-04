"""PSF-Zero -- the compiler. **This file is the latest version of it.**

VERSION: 2026-10-04.1 -- release, adopted on 2026-10-04 from candidate 2026-10-04.c11 (previous release: 2026-10-03.3)

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
    `coupling_map.size()` are unchanged. Without `target` nothing changes. Counts in `PRUNE_STATS`. Gate and
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
"""
from __future__ import annotations

import logging
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from typing import Union

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import CXGate
from qiskit.quantum_info import Operator
from qiskit.synthesis import TwoQubitBasisDecomposer, TwoQubitWeylDecomposition
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
from qiskit.transpiler import CouplingMap, PassManager, generate_preset_pass_manager
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler.basepasses import AnalysisPass, TransformationPass
from qiskit.transpiler.passes import (BasisTranslator, Collect2qBlocks, ConsolidateBlocks, ElidePermutations,
                                      Optimize1qGatesDecomposition, Split2QUnitaries, UnitarySynthesis)
from qiskit.transpiler import ConditionalController
from qiskit.transpiler.passes import ApplyLayout, VF2PostLayout
from qiskit.transpiler.passes.layout.vf2_post_layout import VF2PostLayoutStopReason

try:
    import psf_zero_core
    from psf_zero_core import geometric_decompose
except ImportError as exc:  # pragma: no cover - environment problem, not logic
    raise ImportError(
        "psf_zero_core (the compiled Rust core) could not be imported. Build it "
        "with `maturin develop --release`. Do NOT substitute psf_zero_core_stub "
        "here: it is a Qiskit-based stand-in for environments that cannot load "
        "the real extension, and silently swapping it in makes every benchmark "
        "in this project measure Qiskit against Qiskit."
    ) from exc

VERSION = "2026-10-04.1"  # release (from candidate 2026-10-04.c11): 2026-10-03.3 + choice by an estimate with amplitude damping and pure dephasing (item 38)
__version__ = VERSION

# Changelog item 27: the version string of the loaded Rust core, for logs.
# Cores built before 2026-09-28 do not define it.
CORE_VERSION = getattr(psf_zero_core, "CORE_VERSION", None)

__all__ = [
    "VERSION",
    "CORE_VERSION",
    "compile",
    "compile_for_hardware",
    "GeodesicPSFHyper",
    "SU4GeodesicPSFSynthesizer",
    "unitary_fidelity",
    "edge_errors_from_target",
    "qubit_errors_from_target",
    "GUARD_STATS",
]

# Optional, newer core entry points. Older builds of the core have neither;
# the code below degrades to an equivalent (slower) path rather than failing.
_CORE_CHECKED = getattr(psf_zero_core, "geometric_decompose_checked", None)
_PSF_DEGENERATE_ERRORS = tuple(
    err
    for err in (
        getattr(psf_zero_core, "PsfDegenerateError", None),
        getattr(psf_zero_core, "PsfSU2SingularError", None),
        getattr(psf_zero_core, "PsfNumericError", None),
    )
    if err is not None
)

logger = logging.getLogger(__name__)

# Changelog items 21, 22 and 25: 8 in 2026-09-27.2, 12 in 2026-09-27.3 to .6,
# 8 again from 2026-09-27.7.
DEFAULT_BLOCK_GATE_FLOOR = 8

# Candidate 2026-10-01.c1: a block at or below the floor is still consolidated when it holds at least two
# 2-qubit gates and its optimal CX count (from the Weyl decomposition) is smaller than the CX cost of the
# gates as written. Measured motivation (Addendum 271, section 3): on 4-qubit W and Dicke circuits written by
# a language model, short (cry, cx) runs on one pair were left alone by the floor and cost 9 CX on the device
# where 6 suffice. A block is only re-synthesised when that saves 2-qubit gates.
# Applies only when entangling_basis="cx" (see worth_consolidating in compile()).
CONSOLIDATE_IF_CHEAPER = True

# CX cost of a 2-qubit gate as written (what the router's basis translation will emit for it).
_CX_COST = {"cx": 1, "cz": 1, "cy": 1, "ecr": 1, "swap": 3, "iswap": 2, "dcx": 2,
            "crx": 2, "cry": 2, "crz": 2, "cp": 2, "cu1": 2, "ch": 2, "cs": 2, "csdg": 2, "csx": 2,
            "rxx": 2, "ryy": 2, "rzz": 2, "rzx": 2, "xx_plus_yy": 2, "xx_minus_yy": 2}
_CX_DECOMPOSER = None


def _cx_cost_as_written(block):
    total = 0
    for node in block:
        if len(node.qargs) == 2:
            total += _CX_COST.get(node.op.name, 3)
    return total


_SWAP_4 = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
_EYE_2 = np.eye(2, dtype=complex)


def _block_unitary_4x4(block, qubits):
    """4x4 unitary of a 2-qubit block in Qiskit's little-endian order (qubits[0] is the low bit).
    Built with numpy directly (`Operator(QuantumCircuit)` costs ~10x more per call), with runs of
    single-qubit gates multiplied as 2x2 matrices first: routed blocks are mostly single-qubit gates."""
    u = np.eye(4, dtype=complex)
    pend = [None, None]  # pending 2x2 product on qubits[0] (low) and qubits[1] (high)

    def flush():
        nonlocal u
        lo, hi = pend
        if lo is None and hi is None:
            return
        t = u.reshape(2, 2, 4)  # [high, low, column]
        if lo is not None:
            t = np.einsum("ab,hbc->hac", lo, t)
        if hi is not None:
            t = np.einsum("ab,blc->alc", hi, t)
        u = t.reshape(4, 4)
        pend[0] = pend[1] = None

    for node in block:
        m = node.op.to_matrix()
        if len(node.qargs) == 1:
            k = 0 if node.qargs[0] == qubits[0] else 1
            pend[k] = m if pend[k] is None else m @ pend[k]
            continue
        flush()
        if node.qargs[0] != qubits[0]:
            # op's own qubit 0 is qubits[1]: conjugate by SWAP to express it in the block's order
            m = _SWAP_4 @ m @ _SWAP_4
        u = m @ u
    flush()
    return u


def _block_saves_cx(block):
    """True when the block's optimal CX count is below its CX cost as written (candidate 2026-10-01.c1)."""
    global _CX_DECOMPOSER
    cost = 0
    n_two = 0
    qubits = []
    for node in block:
        if getattr(node.op, "condition", None) is not None or not hasattr(node.op, "to_matrix"):
            return False
        if len(node.qargs) == 2:
            n_two += 1
            cost += _CX_COST.get(node.op.name, 3)
        elif len(node.qargs) != 1:
            return False
        for q in node.qargs:
            if q not in qubits:
                qubits.append(q)
    if n_two < 2 or len(qubits) != 2:
        return False
    if cost > 3:
        return True  # no 2-qubit unitary needs more than 3 CX
    try:
        u = _block_unitary_4x4(block, qubits)
    except Exception:
        return False
    if _CX_DECOMPOSER is None:
        _CX_DECOMPOSER = TwoQubitBasisDecomposer(CXGate())
    return _CX_DECOMPOSER.num_basis_gates(u) < cost

_VALID_ENTANGLING_BASES = ("canonical", "cx")
_VALID_ON_UNSUPPORTED = ("keep", "raise")

# Reuses Qiskit's own exact CX-optimal decomposer, configured for the
# {rz, sx} basis so the single-qubit layers between its CXs are minimal
# (changelog item 14).
_CX_DECOMPOSER = TwoQubitBasisDecomposer(CXGate(), euler_basis="ZSX")
# Changelog item 17: the retry decomposer, and the guard's switch and counts.
_CX_DECOMPOSER_DEFAULT_EULER = TwoQubitBasisDecomposer(CXGate())
USE_CX_GUARD = True
# Average gate infidelity accepted from Qiskit's decomposer: ten times its own
# default requested fidelity (1 - 1e-9), so its documented approximations
# (dropping a tiny interaction to save a CX) pass and the 7e-2 failures do not.
_GUARD_TOL = 1e-8
GUARD_STATS = {"checked": 0, "zsx_rejected": 0, "default_rejected": 0, "closed_form_forced": 0,
               "inexact": 0, "exact_rebuilt": 0, "best_effort": 0, "best_effort_worst": 0.0,
               "psf_rerouted": 0, "psf_rerouted_worst_residual": 0.0}
# Changelog item 23. Module-level so a validation run can switch it off.
USE_EXACT_FALLBACK = True
_EXACT_TOL = 1e-13
_BEST_EFFORT_TOL = 1e-10


def _aligned_errors(u: np.ndarray, circ: QuantumCircuit):
    """(average gate infidelity, phase angle p, phase-aligned Frobenius
    distance ||u - exp(i p) V||) between `u` and the circuit's unitary V,
    with u ~= exp(i p) * V. Infidelity is quadratic in the operator error;
    the Frobenius distance is linear in it (changelog item 23)."""
    v = Operator(circ).data
    t = np.trace(v.conj().T @ u)
    ph = t / abs(t) if abs(t) > 0 else 1.0
    f = abs(t) / 4.0
    return (float(1.0 - (4.0 * f * f + 1.0) / 5.0), float(np.angle(ph)),
            float(np.linalg.norm(u - ph * v)))


def _aligned_distance(u: np.ndarray, circ: QuantumCircuit):
    """Average gate infidelity and phase angle (kept for existing callers)."""
    infid, phase, _ = _aligned_errors(u, circ)
    return infid, phase


def _weyl_exact(u: np.ndarray):
    """Qiskit's Weyl decomposition without specialization (no snapping of
    near-special inputs onto the special case)."""
    try:
        return TwoQubitWeylDecomposition(u, fidelity=None)
    except TypeError:
        from qiskit._accelerate.two_qubit_decompose import Specialization
        return TwoQubitWeylDecomposition(u, _specialization=Specialization.General)


def _su2_zyz(k: np.ndarray):
    """(phi, theta, lam) with _zyz_matrix(phi, theta, lam) = exp(-i alpha) k,
    and alpha (half the phase of det k). Exact for any 2x2 unitary."""
    alpha = float(np.angle(np.linalg.det(k)) / 2.0)
    s = k * np.exp(-1j * alpha)
    a, b = s[0, 0], s[1, 0]
    theta = 2.0 * float(np.arctan2(abs(b), abs(a)))
    ssum = -2.0 * float(np.angle(a)) if abs(a) > 1e-12 else 0.0
    sdiff = 2.0 * float(np.angle(b)) if abs(b) > 1e-12 else 0.0
    return ((ssum + sdiff) / 2.0, theta, (ssum - sdiff) / 2.0), alpha


def _exact_rebuild(u: np.ndarray) -> QuantumCircuit:
    """Exact CX-basis circuit for `u` (changelog item 23): Qiskit's
    unspecialized Weyl decomposition, converted to PSF-Zero's parameters,
    polished, and emitted like a PSF-Zero block with the closed-form core."""
    d = _weyl_exact(u)
    t1l, a1 = _su2_zyz(np.asarray(d.K1l))
    t1r, a2 = _su2_zyz(np.asarray(d.K1r))
    t2l, a3 = _su2_zyz(np.asarray(d.K2l))
    t2r, a4 = _su2_zyz(np.asarray(d.K2r))
    cartan = (float(d.a), float(d.b), float(d.c))
    phase = float(d.global_phase) + a1 + a2 + a3 + a4
    (cartan, k1, k2, phase), _, _ = _refine_decomposition(u, cartan, (t1l, t1r), (t2l, t2r), phase)
    qc = QuantumCircuit(2, global_phase=float(phase))

    def local(triple, qubit):
        phi, theta, lam = (float(x) for x in triple)
        qc.rz(lam, qubit)
        qc.ry(theta, qubit)
        qc.rz(phi, qubit)

    local(k2[0], 1)
    local(k2[1], 0)
    _append_cx_core_closed_form(qc, *(float(x) for x in cartan), force=True)
    local(k1[0], 1)
    local(k1[1], 0)
    return qc


def _accept(u, circ, stat_key):
    """Returns the phase-corrected circuit if it passes both checks, else
    None. `stat_key` is counted when the infidelity check (item 17) fails."""
    infid, phase, frob = _aligned_errors(u, circ)
    if infid > _GUARD_TOL:
        GUARD_STATS[stat_key] += 1
        return None
    if USE_EXACT_FALLBACK and frob > _EXACT_TOL:
        GUARD_STATS["inexact"] += 1
        return None
    circ.global_phase += phase
    return circ


def _guarded_cx_synthesis(u: np.ndarray):
    """Qiskit's CX-basis synthesis of `u`, checked (changelog item 17).

    Returns (circuit, ok). An accepted circuit has its global phase corrected
    so that its unitary matches `u`, not only up to phase. ok is False only
    when both the ZSX and the default-Euler decomposers exceed `_GUARD_TOL`
    in average gate infidelity; the caller decides what to do then.
    """
    circ = _CX_DECOMPOSER(u)
    if not USE_CX_GUARD:
        return circ, True
    GUARD_STATS["checked"] += 1
    ok = _accept(u, circ, "zsx_rejected")
    if ok is not None:
        return ok, True
    logger.debug("PSF-Zero guard: ZSX decomposer result rejected; retrying")
    circ = _CX_DECOMPOSER_DEFAULT_EULER(u)
    ok = _accept(u, circ, "default_rejected")
    if ok is not None:
        return ok, True
    if USE_EXACT_FALLBACK:
        # Changelog item 23: neither decomposer returned an exact circuit.
        candidates = []
        for c in (_CX_DECOMPOSER(u), circ):
            infid, phase, frob = _aligned_errors(u, c)
            if infid <= _GUARD_TOL:
                candidates.append((frob, phase, c))
        try:
            rebuilt = _exact_rebuild(u)
        except Exception as exc:  # reported; the best remaining candidate applies
            logger.debug("PSF-Zero guard: exact rebuild failed: %s", exc)
        else:
            infid, phase, frob = _aligned_errors(u, rebuilt)
            if frob <= _EXACT_TOL:
                rebuilt.global_phase += phase
                GUARD_STATS["exact_rebuilt"] += 1
                return rebuilt, True
            candidates.append((frob, phase, rebuilt))
        if candidates:
            frob, phase, best = min(candidates, key=lambda x: x[0])
            if frob <= _BEST_EFFORT_TOL:
                best.global_phase += phase
                GUARD_STATS["best_effort"] += 1
                GUARD_STATS["best_effort_worst"] = max(GUARD_STATS["best_effort_worst"], frob)
                return best, True
    return circ, False

# XX, YY and ZZ commute and share one eigenbasis, so the canonical core's
# matrix exponential is a diagonal scaling in a basis that can be computed
# once at import instead of per block. The 1/2/4 coefficients are there to
# make the combined spectrum non-degenerate, so `eigh` returns a basis that
# simultaneously diagonalises all three rather than an arbitrary rotation
# inside a degenerate eigenspace. Only used by the numpy verification
# fallback, for cores too old to have `geometric_decompose_checked`.
_XX = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]], dtype=complex)
_YY = np.array([[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]], dtype=complex)
_ZZ = np.diag([1, -1, -1, 1]).astype(complex)
_, _CORE_BASIS = np.linalg.eigh(_XX + 2.0 * _YY + 4.0 * _ZZ)
_CORE_BASIS_H = _CORE_BASIS.conj().T
_WX = np.real(np.diag(_CORE_BASIS_H @ _XX @ _CORE_BASIS))
_WY = np.real(np.diag(_CORE_BASIS_H @ _YY @ _CORE_BASIS))
_WZ = np.real(np.diag(_CORE_BASIS_H @ _ZZ @ _CORE_BASIS))


def _validate_verify(verify: Union[bool, str]) -> Union[bool, str]:
    """Accept exactly True, False or "strict".

    Deliberately strict about types: `verify=1` is not `verify=True` here, and
    `"Strict"` is not `"strict"`. Both used to fall through to the cheap check
    without saying so, which is the failure mode this project's own rules are
    written against.
    """
    if verify is True or verify is False or verify == "strict":
        return verify
    raise ValueError(
        f"verify must be True, False or 'strict' (got {verify!r}). "
        "True = check the decomposition using the core's own reconstruction; "
        "'strict' = rebuild the emitted circuit with Operator(qc) and check "
        "that; False = no check."
    )


@dataclass
class GeodesicPSFHyper:
    tol: float = 1e-5
    phase_fix: bool = True
    on_unsupported: str = "keep"
    entangling_basis: str = "canonical"  # "canonical" (default) | "cx" (native CX output directly)

    def __post_init__(self) -> None:
        if self.entangling_basis not in _VALID_ENTANGLING_BASES:
            raise ValueError(
                f"entangling_basis must be one of {_VALID_ENTANGLING_BASES} "
                f"(got {self.entangling_basis!r})."
            )
        if self.on_unsupported not in _VALID_ON_UNSUPPORTED:
            raise ValueError(
                f"on_unsupported must be one of {_VALID_ON_UNSUPPORTED} "
                f"(got {self.on_unsupported!r})."
            )
        if not self.tol > 0.0:
            raise ValueError(f"tol must be positive (got {self.tol!r}).")


def unitary_fidelity(U_target: np.ndarray, qc: QuantumCircuit) -> float:
    """Average gate fidelity between a target unitary and a circuit.

    Unchanged, and still the most independent check available -- it builds the
    circuit's operator with Qiskit rather than trusting anything this package
    computed. It is also the most expensive one by more than an order of
    magnitude, which is why it is no longer on the default path; see
    `_verify_block`.
    """
    U_out = Operator(qc).data
    tr = np.trace(U_target.conj().T @ U_out)
    d = 4.0
    return float((np.abs(tr) ** 2 + d) / (d * (d + 1)))


def _zyz_matrix(triple) -> np.ndarray:
    phi, theta, lam = triple
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    ep, em = np.exp(-0.5j * phi), np.exp(0.5j * phi)
    lp, lm = np.exp(-0.5j * lam), np.exp(0.5j * lam)
    return np.array([[ep * c * lp, -ep * s * lm], [em * s * lp, em * c * lm]], dtype=complex)


def _reconstruct(cartan, k1, k2, phase: float) -> np.ndarray:
    """Rebuild the 4x4 from the values the core returned, following
    `synthesize()`'s own recipe (k2 locals, canonical core, k1 locals, global
    phase) so that what gets checked is the gate about to be emitted."""
    a, b, c = cartan
    core = (_CORE_BASIS * np.exp(1j * (a * _WX + b * _WY + c * _WZ))) @ _CORE_BASIS_H
    left = np.kron(_zyz_matrix(k1[0]), _zyz_matrix(k1[1]))
    right = np.kron(_zyz_matrix(k2[0]), _zyz_matrix(k2[1]))
    return np.exp(1j * phase) * (left @ core @ right)


def _infidelity(U_target: np.ndarray, U_out: np.ndarray) -> float:
    tr = np.trace(U_target.conj().T @ U_out)
    d = 4.0
    return float(1.0 - (np.abs(tr) ** 2 + d) / (d * (d + 1)))


# Changelog item 26: 1e-13 until 2026-09-27.7.
REFINE_THRESHOLD = 1e-14
_REFINE_TARGET = 1e-14
_HALF_Z = np.diag([-0.5j, 0.5j])


def _pack(cartan, k1, k2, phase):
    return np.array([*cartan, *k1[0], *k1[1], *k2[0], *k2[1], phase], dtype=float)


def _unpack(p):
    return ((p[0], p[1], p[2]),
            ((p[3], p[4], p[5]), (p[6], p[7], p[8])),
            ((p[9], p[10], p[11]), (p[12], p[13], p[14])),
            p[15])


def _zyz_and_derivatives(triple):
    """_zyz_matrix(triple) and its derivatives in phi, theta and lam."""
    phi, theta, lam = triple
    m = _zyz_matrix(triple)
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    ep, em = np.exp(-0.5j * phi), np.exp(0.5j * phi)
    lp, lm = np.exp(-0.5j * lam), np.exp(0.5j * lam)
    d_theta = 0.5 * np.array([[-ep * s * lp, -ep * c * lm], [em * c * lp, -em * s * lm]], dtype=complex)
    return m, (_HALF_Z @ m, d_theta, m @ _HALF_Z)


def _reconstruct_with_jacobian(p):
    """_reconstruct(*_unpack(p)) and its derivative with respect to each of
    the 16 parameters, computed in closed form (Addendum 189)."""
    (a, b, c), k1, k2, phase = _unpack(p)
    diag = np.exp(1j * (a * _WX + b * _WY + c * _WZ))
    core = (_CORE_BASIS * diag) @ _CORE_BASIS_H
    l0, dl0 = _zyz_and_derivatives(k1[0])
    l1, dl1 = _zyz_and_derivatives(k1[1])
    r0, dr0 = _zyz_and_derivatives(k2[0])
    r1, dr1 = _zyz_and_derivatives(k2[1])
    left, right = np.kron(l0, l1), np.kron(r0, r1)
    g = np.exp(1j * phase)
    cr = core @ right
    lc = left @ core
    u = g * (left @ cr)
    jac = []
    for w in (_WX, _WY, _WZ):
        jac.append(g * (left @ ((_CORE_BASIS * (1j * w * diag)) @ _CORE_BASIS_H) @ right))
    for d in dl0:
        jac.append(g * (np.kron(d, l1) @ cr))
    for d in dl1:
        jac.append(g * (np.kron(l0, d) @ cr))
    for d in dr0:
        jac.append(g * (lc @ np.kron(d, r1)))
    for d in dr1:
        jac.append(g * (lc @ np.kron(r0, d)))
    jac.append(1j * u)
    return u, jac


def _refine_decomposition(U_target, cartan, k1, k2, phase, threshold=REFINE_THRESHOLD, max_iter=3):
    """Polish the core's decomposition in parameter space (Gauss-Newton on
    the 16 real parameters, residual = _reconstruct(...) - U_target).

    Addendum 185 traced PSF-Zero's per-block error to the core's returned
    parameters: up to ~1.8e-11 (Frobenius) on inputs near the Weyl-chamber
    face c = 0, where Qiskit's decomposer stays near 1e-14. Each Newton step
    roughly squares the error, so one step reaches machine precision.
    Addendum 189: the residual check uses the plain reconstruction; the
    closed-form Jacobian is computed only when a step is actually taken, and
    iteration stops once the residual is <= 1e-14 or a step fails to halve
    it. Returns the (possibly unchanged) decomposition and the residual norms
    before and after.
    """
    p = _pack(cartan, k1, k2, phase)
    d = (_reconstruct(cartan, k1, k2, phase) - U_target).ravel()
    before = float(np.linalg.norm(d))
    if before <= threshold:
        return (cartan, k1, k2, phase), before, before
    norm_r = before
    for _ in range(max_iter):
        _, jac = _reconstruct_with_jacobian(p)
        jm = np.stack([j.ravel() for j in jac], axis=1)
        jr = np.concatenate([jm.real, jm.imag], axis=0)
        r = np.concatenate([d.real, d.imag])
        q = p + np.linalg.lstsq(jr, -r, rcond=None)[0]
        d_q = (_reconstruct(*_unpack(q)) - U_target).ravel()
        norm_q = float(np.linalg.norm(d_q))
        if norm_q >= norm_r:
            break
        progress = norm_q < 0.5 * norm_r
        p, d, norm_r = q, d_q, norm_q
        if norm_r <= _REFINE_TARGET or not progress:
            break
    return _unpack(p), before, norm_r


# Changelog item 20. Module-level so a validation run can switch it off.
USE_BATCHED_POLISH = True


def _zyz_batch(t: np.ndarray) -> np.ndarray:
    """_zyz_matrix for each row (phi, theta, lam) of t, shape (N, 3) -> (N, 2, 2)."""
    phi, theta, lam = t[:, 0], t[:, 1], t[:, 2]
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    ep, em = np.exp(-0.5j * phi), np.exp(0.5j * phi)
    lp, lm = np.exp(-0.5j * lam), np.exp(0.5j * lam)
    out = np.empty((t.shape[0], 2, 2), dtype=complex)
    out[:, 0, 0] = ep * c * lp
    out[:, 0, 1] = -ep * s * lm
    out[:, 1, 0] = em * s * lp
    out[:, 1, 1] = em * c * lm
    return out


def _zyz_derivatives_batch(t: np.ndarray, m: np.ndarray):
    """Derivatives of _zyz_batch(t) in phi, theta and lam (each (N, 2, 2)),
    as in _zyz_and_derivatives; m is _zyz_batch(t)."""
    phi, theta, lam = t[:, 0], t[:, 1], t[:, 2]
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    ep, em = np.exp(-0.5j * phi), np.exp(0.5j * phi)
    lp, lm = np.exp(-0.5j * lam), np.exp(0.5j * lam)
    d_theta = np.empty_like(m)
    d_theta[:, 0, 0] = 0.5 * (-ep * s * lp)
    d_theta[:, 0, 1] = 0.5 * (-ep * c * lm)
    d_theta[:, 1, 0] = 0.5 * (em * c * lp)
    d_theta[:, 1, 1] = 0.5 * (-em * s * lm)
    return _HALF_Z @ m, d_theta, m @ _HALF_Z


def _kron_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """np.kron of each pair of 2x2 matrices, (N, 2, 2) x (N, 2, 2) -> (N, 4, 4)."""
    return np.einsum("nij,nkl->nikjl", a, b).reshape(a.shape[0], 4, 4)


def _reconstruct_batch(p: np.ndarray, with_jacobian: bool = False):
    """_reconstruct for each packed parameter row of p (N, 16) -> (N, 4, 4);
    with_jacobian also returns the derivative with respect to each of the 16
    parameters, (N, 16, 4, 4), in the order of _reconstruct_with_jacobian."""
    diag = np.exp(1j * (p[:, 0:1] * _WX + p[:, 1:2] * _WY + p[:, 2:3] * _WZ))
    core = (_CORE_BASIS[None, :, :] * diag[:, None, :]) @ _CORE_BASIS_H
    l0, l1 = _zyz_batch(p[:, 3:6]), _zyz_batch(p[:, 6:9])
    r0, r1 = _zyz_batch(p[:, 9:12]), _zyz_batch(p[:, 12:15])
    left, right = _kron_batch(l0, l1), _kron_batch(r0, r1)
    g = np.exp(1j * p[:, 15])[:, None, None]
    cr = core @ right
    u = g * (left @ cr)
    if not with_jacobian:
        return u
    lc = left @ core
    jac = []
    for w in (_WX, _WY, _WZ):
        dcore = (_CORE_BASIS[None, :, :] * (1j * w * diag)[:, None, :]) @ _CORE_BASIS_H
        jac.append(g * (left @ dcore @ right))
    for d in _zyz_derivatives_batch(p[:, 3:6], l0):
        jac.append(g * (_kron_batch(d, l1) @ cr))
    for d in _zyz_derivatives_batch(p[:, 6:9], l1):
        jac.append(g * (_kron_batch(l0, d) @ cr))
    for d in _zyz_derivatives_batch(p[:, 9:12], r0):
        jac.append(g * (lc @ _kron_batch(d, r1)))
    for d in _zyz_derivatives_batch(p[:, 12:15], r1):
        jac.append(g * (lc @ _kron_batch(r0, d)))
    jac.append(1j * u)
    return u, np.stack(jac, axis=1)


def _lstsq_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimum-norm least-squares solution of a[n] x = b[n] for each n,
    (N, M, K) and (N, M) -> (N, K), by SVD with the cutoff that
    numpy.linalg.lstsq uses for rcond=None (eps * max(M, K) * s_max)."""
    u, s, vh = np.linalg.svd(a, full_matrices=False)
    cutoff = np.finfo(float).eps * max(a.shape[1], a.shape[2]) * s[:, :1]
    keep = s > cutoff
    s_inv = np.where(keep, 1.0 / np.where(keep, s, 1.0), 0.0)
    ub = np.einsum("nmk,nm->nk", u, b)
    return np.einsum("nkj,nk->nj", vh, s_inv * ub)


def _refine_batch(U_targets: np.ndarray, p: np.ndarray, threshold=REFINE_THRESHOLD, max_iter=3):
    """_refine_decomposition for N blocks at once (changelog item 20).

    U_targets (N, 4, 4), p (N, 16) packed parameters. Returns the polished
    parameters (N, 16), and the residual norms before and after (N,). The
    per-block rules are those of _refine_decomposition: blocks at or below
    `threshold` are left untouched; otherwise up to `max_iter` Gauss-Newton
    steps, a step is accepted only if it lowers the residual, and a block
    stops once its residual is <= _REFINE_TARGET or a step fails to halve it.
    """
    p = p.copy()
    n = p.shape[0]
    d = (_reconstruct_batch(p) - U_targets).reshape(n, 16)
    before = np.linalg.norm(d, axis=1)
    norm_r = before.copy()
    active = before > threshold
    for _ in range(max_iter):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        _, jac = _reconstruct_batch(p[idx], with_jacobian=True)
        jm = jac.reshape(idx.size, 16, 16).transpose(0, 2, 1)
        jr = np.concatenate([jm.real, jm.imag], axis=1)
        r = np.concatenate([d[idx].real, d[idx].imag], axis=1)
        q = p[idx] + _lstsq_batch(jr, -r)
        d_q = (_reconstruct_batch(q) - U_targets[idx]).reshape(idx.size, 16)
        norm_q = np.linalg.norm(d_q, axis=1)
        improved = norm_q < norm_r[idx]
        progress = norm_q < 0.5 * norm_r[idx]
        acc = idx[improved]
        p[acc], d[acc], norm_r[acc] = q[improved], d_q[improved], norm_q[improved]
        active[idx] = improved & progress & (norm_q > _REFINE_TARGET)
    return p, before, norm_r


class SU4GeodesicPSFSynthesizer:
    """Synthesize a 2-qubit unitary via the Rust core's Cartan decomposition.

    `verify` selects what, if anything, is checked before a synthesized block
    is accepted:

      True     -- check the decomposition, using the core's own reconstruction
                  when the core provides one and a numpy reconstruction
                  otherwise. Catches a bad decomposition. Does not independently
                  re-derive the circuit object, so it would not catch a bug in
                  this file's circuit construction (the core's reconstruction
                  mirrors that construction deliberately, which is what makes
                  the check meaningful, and also what makes it not independent
                  of it). On `entangling_basis="cx"` it checks the same
                  decomposition; the CX substitution applied afterwards is
                  Qiskit's own exact decomposer and is not re-checked here.
      "strict" -- build `Operator(qc)` from the emitted circuit and compare.
                  Independent of both the core and this file's own recipe, and
                  the only mode that validates the actual circuit object. ~12x
                  the cost of the default; worth it in CI, rarely in production.
      False    -- no check. The core still reports genuine failures as
                  exceptions, which are still caught and fall back.
    """

    def __init__(self, hyper: GeodesicPSFHyper, verify: Union[bool, str] = True):
        self.hyper = hyper
        self.verify = _validate_verify(verify)
        self.fallback_count = 0
        # Blocks whose core decomposition was polished by _refine_decomposition
        # (Addendum 186); the maximum residual seen before and after polishing.
        self.refine_count = 0
        self.refine_max_before = 0.0
        self.refine_max_after = 0.0
        self.degenerate_count = 0
        self.unexpected_count = 0
        self._last_reasons: list[str] = []

    def _fallback(self, U_target: np.ndarray, msg: str, expected: bool) -> QuantumCircuit:
        self.fallback_count += 1
        if expected:
            self.degenerate_count += 1
        else:
            self.unexpected_count += 1
        if self.hyper.on_unsupported == "raise":
            raise RuntimeError(msg)
        # Collected rather than warned per block: a large circuit can produce
        # thousands of these, and one summary at the end is both cheaper and
        # more useful than a flood of identical lines.
        if len(self._last_reasons) < 5:
            self._last_reasons.append(msg)
        logger.debug("PSF-Zero fallback: %s", msg)
        circ, ok = _guarded_cx_synthesis(U_target)
        if not ok:
            # Never observed; emitting a block known to be wrong is worse
            # than stopping (changelog item 17).
            raise RuntimeError(
                "CX-basis synthesis of a fallback block failed verification "
                "with both Euler bases"
            )
        return circ

    def fallback_summary(self) -> str:
        if not self.fallback_count:
            return ""
        parts = [
            f"{self.fallback_count} block(s) fell back to CX-basis synthesis "
            f"({self.degenerate_count} degenerate/numeric, "
            f"{self.unexpected_count} unexpected)"
        ]
        parts.extend(f"  e.g. {r}" for r in self._last_reasons)
        return "\n".join(parts)

    def _entangling_core(self, qc: QuantumCircuit, a: float, b: float, c: float) -> None:
        """Append the canonical entangling core directly to `qc`.

        The previous version built a second QuantumCircuit and composed it in;
        appending straight to the target skips an object allocation and a
        compose per block for an identical result.
        """
        if self.hyper.entangling_basis == "cx":
            if USE_CX_CLOSED_FORM and _append_cx_core_closed_form(qc, a, b, c):
                return
            sub = _cx_core_cached(a, b, c)
            if sub is not None:
                qc.compose(sub, [0, 1], inplace=True)
            return
        if abs(a) > 1e-10:
            qc.rxx(-2 * a, 0, 1)
        if abs(b) > 1e-10:
            qc.ryy(-2 * b, 0, 1)
        if abs(c) > 1e-10:
            qc.rzz(-2 * c, 0, 1)

    def _build_circuit(self, cartan, k1, k2, global_phase: float) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.global_phase = global_phase

        def local(triple, qubit):
            phi, theta, lam = triple
            qc.rz(lam, qubit)
            qc.ry(theta, qubit)
            qc.rz(phi, qubit)

        local(k2[0], 1)
        local(k2[1], 0)
        self._entangling_core(qc, *cartan)
        local(k1[0], 1)
        local(k1[1], 0)
        return qc

    def _verify_block(self, U_target, qc, cartan, k1, k2, phase, core_infid):
        """Return the infidelity to test against `tol`, or None to skip."""
        if self.verify is False:
            return None
        if self.verify == "strict":
            return 1.0 - unitary_fidelity(U_target, qc)
        if core_infid is not None:
            # The core already reconstructed this decomposition and told us how
            # far off it was; nothing further to compute. This is the same
            # quantity the numpy path below produces, for either entangling
            # basis -- what is being validated is the decomposition, not the
            # emitted circuit (only "strict" does that).
            return core_infid
        # Older core with no self-check: recompute the same reconstruction here.
        return _infidelity(U_target, _reconstruct(cartan, k1, k2, phase))

    def synthesize(self, U_target: np.ndarray) -> QuantumCircuit:
        dec = self._decompose(U_target)
        if isinstance(dec, QuantumCircuit):
            return dec
        cartan, k1, k2, global_phase, core_infid = dec
        # Polish the core's parameters before building the circuit (Addendum
        # 186). Near the Weyl-chamber face c = 0 the core's output can be off
        # by ~1e-11; one Gauss-Newton step on the 16 parameters brings it to
        # machine precision. A no-op when the residual is already <= 1e-13.
        (cartan, k1, k2, global_phase), res_before, res_after = _refine_decomposition(
            U_target, cartan, k1, k2, global_phase
        )
        return self._finish(U_target, cartan, k1, k2, global_phase, core_infid, res_before, res_after)

    def synthesize_many(self, U_targets: list) -> list:
        """Synthesize several blocks; the polish runs on all of them at once
        (changelog item 20). Returns [(circuit, fell_back)] in input order.
        Results match calling synthesize() on each block, up to rounding in
        the polished parameters."""
        decs = []
        out: list = [None] * len(U_targets)
        for i, u in enumerate(U_targets):
            before = self.fallback_count
            dec = self._decompose(u)
            if isinstance(dec, QuantumCircuit):
                out[i] = (dec, self.fallback_count != before)
            else:
                decs.append((i, dec))
        if not decs:
            return out
        if USE_BATCHED_POLISH:
            us = np.stack([U_targets[i] for i, _ in decs])
            p0 = np.stack([_pack(*dec[:4]) for _, dec in decs])
            p1, before_all, after_all = _refine_batch(us, p0)
        for j, (i, (cartan, k1, k2, global_phase, core_infid)) in enumerate(decs):
            u = U_targets[i]
            if USE_BATCHED_POLISH:
                res_before, res_after = float(before_all[j]), float(after_all[j])
                if res_before > REFINE_THRESHOLD:
                    cartan, k1, k2, global_phase = _unpack(p1[j])
            else:
                (cartan, k1, k2, global_phase), res_before, res_after = _refine_decomposition(
                    u, cartan, k1, k2, global_phase
                )
            before = self.fallback_count
            circ = self._finish(u, cartan, k1, k2, global_phase, core_infid, res_before, res_after)
            out[i] = (circ, self.fallback_count != before)
        return out

    def _decompose(self, U_target: np.ndarray):
        """Rust core decomposition of one block. Returns (cartan, k1, k2,
        global_phase, core_infid), or the fallback circuit if the core failed."""
        if U_target.shape != (4, 4):
            raise ValueError("Input must be a 4x4 unitary matrix.")

        u_r = U_target.real.tolist()
        u_i = U_target.imag.tolist()

        core_infid = None
        try:
            # "strict" rebuilds the circuit with Operator(qc) and ignores
            # core_infid entirely, so asking the core for it is pure waste.
            want_core_check = self.verify is True
            if _CORE_CHECKED is not None and want_core_check:
                cartan, k1, k2, global_phase, core_infid = _CORE_CHECKED(u_r, u_i)
            else:
                cartan, k1, k2, global_phase = geometric_decompose(u_r, u_i)
        except Exception as exc:
            # A degenerate or numerically singular input is an expected event
            # that the CX path handles correctly; anything else means the core
            # itself misbehaved, and the two are worth counting separately.
            expected = isinstance(exc, _PSF_DEGENERATE_ERRORS) if _PSF_DEGENERATE_ERRORS else True
            return self._fallback(U_target, f"Decomposition failed or degenerate: {exc}", expected)
        return cartan, k1, k2, global_phase, core_infid

    def _finish(self, U_target, cartan, k1, k2, global_phase, core_infid, res_before, res_after):
        """Bookkeeping of the polish, circuit construction and verification
        of one decomposed (and polished) block."""
        if USE_EXACT_FALLBACK and res_after > _EXACT_TOL:
            # Changelog item 24: the polished decomposition is not exact.
            circ, ok = _guarded_cx_synthesis(np.asarray(U_target))
            if ok:
                GUARD_STATS["psf_rerouted"] += 1
                GUARD_STATS["psf_rerouted_worst_residual"] = max(
                    GUARD_STATS["psf_rerouted_worst_residual"], float(res_after))
                return circ
        if res_before > REFINE_THRESHOLD:
            self.refine_count += 1
            self.refine_max_before = max(self.refine_max_before, res_before)
            self.refine_max_after = max(self.refine_max_after, res_after)
            if core_infid is not None:
                # The core's self-check described the unpolished parameters;
                # re-derive it for the parameters actually emitted.
                core_infid = _infidelity(U_target, _reconstruct(cartan, k1, k2, global_phase))

        try:
            qc = self._build_circuit(cartan, k1, k2, global_phase)
        except Exception as exc:
            # The core returned a decomposition and this file failed to turn it
            # into a circuit. That is a bug here, not a property of the input,
            # and is never an expected degeneracy.
            return self._fallback(
                U_target,
                f"Circuit construction failed after a successful decomposition: {exc}",
                expected=False,
            )

        infid = self._verify_block(U_target, qc, cartan, k1, k2, global_phase, core_infid)
        if infid is not None and infid > self.hyper.tol:
            return self._fallback(
                U_target, f"Fidelity loss exceeded tolerance: {infid:.2e}", expected=False
            )
        return qc


def _cx_core_cached(a: float, b: float, c: float):
    """CX-basis form of the canonical core for one (a, b, c).

    Cached on the exact triple: Trotter steps and QAOA layers apply the same
    canonical angles to many pairs, and every miss costs an `Operator()` build
    plus a full Qiskit KAK on a matrix whose decomposition we already know.
    Keyed on the exact floats, so a cache hit is bit-identical -- no rounding,
    no accuracy traded for the speedup.

    LRU rather than "stop caching once full": the previous version kept the
    first 4096 triples forever and silently degraded to no caching at all
    after that, which is the wrong way round for a long-lived process whose
    working set moves.

    Not synchronised. Two threads racing here can each build the same entry,
    which wastes a decomposition but cannot produce a wrong one; the cache is
    keyed on the inputs and the value is a function of them alone.
    """
    key = (a, b, c)
    if key in _CX_CORE_CACHE:
        _CX_CORE_CACHE.move_to_end(key)
        return _CX_CORE_CACHE[key]
    core = QuantumCircuit(2)
    if abs(a) > 1e-10:
        core.rxx(-2 * a, 0, 1)
    if abs(b) > 1e-10:
        core.ryy(-2 * b, 0, 1)
    if abs(c) > 1e-10:
        core.rzz(-2 * c, 0, 1)
    if len(core.data) == 0:
        result = None
    else:
        result, ok = _guarded_cx_synthesis(Operator(core).data)
        if not ok:
            # Both decomposers missed; the closed form is exact for any
            # triple, at the price of three CXs (changelog item 17).
            GUARD_STATS["closed_form_forced"] += 1
            result = QuantumCircuit(2)
            _append_cx_core_closed_form(result, a, b, c, force=True)
    _CX_CORE_CACHE[key] = result
    if len(_CX_CORE_CACHE) > _CX_CORE_CACHE_MAX:
        _CX_CORE_CACHE.popitem(last=False)
    return result


_CX_CORE_CACHE: "OrderedDict[tuple, object]" = OrderedDict()
_CX_CORE_CACHE_MAX = 4096

# Changelog item 15. Module-level so a validation run can switch it off.
USE_CX_CLOSED_FORM = True
# Changelog item 18: middle gaps emitted in {rz, sx}.
USE_NATIVE_GAPS = True
# A coordinate this close to a multiple of pi/2 makes the core reducible to
# fewer than three CXs; those triples keep the decomposer path.
_CLOSED_FORM_DEGENERATE = 1e-6
_HALF_PI = float(np.pi / 2)


def _append_cx_core_closed_form(qc: QuantumCircuit, a: float, b: float, c: float,
                                force: bool = False) -> bool:
    """Append exp(i(a XX + b YY + c ZZ)) to qubits (0, 1) of `qc` as three
    CXs, exactly (including global phase). Returns False, appending nothing,
    when the triple is degenerate (see `_CLOSED_FORM_DEGENERATE`).

    Vatan-Williams form, adapted: q1 is the control of the outer CXs and the
    target of the middle one, so an `rx` on q1 commutes with the middle CX.
    Inserting rx(pi/2) rx(-pi/2) around it turns each middle-gap rotation
    into rx(pi/2) ry(t) or ry(t) rx(-pi/2), whose off-diagonal magnitude is
    1/sqrt(2) for every t -- one `sx` after translation, instead of two.
    With USE_NATIVE_GAPS (changelog item 18) the two gaps are written in
    {rz, sx} directly. `force=True` skips the degeneracy check (used when
    Qiskit's decomposer failed on a degenerate triple; the result is still
    exact, with three CXs).
    """
    if not force:
        for x in (a, b, c):
            if abs(x - _HALF_PI * round(x / _HALF_PI)) < _CLOSED_FORM_DEGENERATE:
                return False
    half_pi = _HALF_PI
    if USE_NATIVE_GAPS:
        qc.rz(-half_pi, 1)
        qc.cx(1, 0)
        qc.rz(half_pi - 2.0 * c, 0)
        qc.sx(1)
        qc.rz(2.0 * a - half_pi, 1)
        qc.cx(0, 1)
        qc.rz(3.0 * half_pi - 2.0 * b, 1)
        qc.sx(1)
        qc.rz(np.pi, 1)
        qc.cx(1, 0)
        qc.rz(half_pi, 0)
        qc.global_phase += 3.0 * np.pi / 4
        return True
    qc.rz(-half_pi, 1)
    qc.cx(1, 0)
    qc.rz(half_pi - 2.0 * c, 0)
    qc.ry(2.0 * a - half_pi, 1)
    qc.rx(half_pi, 1)
    qc.cx(0, 1)
    qc.rx(-half_pi, 1)
    qc.ry(half_pi - 2.0 * b, 1)
    qc.cx(1, 0)
    qc.rz(half_pi, 0)
    qc.global_phase += np.pi / 4
    return True


def compile(
    qc: QuantumCircuit,
    block_gate_floor: int = DEFAULT_BLOCK_GATE_FLOOR,
    verify: Union[bool, str] = True,
    entangling_basis: str = "canonical",
    on_unsupported: str = "keep",
    tol: float = 1e-5,
) -> QuantumCircuit:
    """Collect 2-qubit blocks, consolidate them, and re-synthesize each one
    through the Rust core's Cartan (KAK) decomposition.

    Only runs of more than `block_gate_floor` gates on the same qubit pair are
    collected, so a wide-and-shallow circuit (`random_circuit()`, say) has few
    or no blocks and comes back largely untouched by design rather than by
    accident (at the default of 8: 0-6 blocks on 8-qubit, depth-12 random
    circuits, with the same two-qubit count as at 12; Addendum 210).
    Everything that is not a 2-qubit `unitary` block is copied through as-is,
    keeping the input's registers, bits, name and metadata.

    Args:
        qc: the circuit to compile.
        block_gate_floor: minimum run length before a same-pair block is worth
            consolidating and re-synthesizing.
        verify: True (default, checks the decomposition via the core's own
            reconstruction -- effectively free), "strict" (rebuilds the emitted
            circuit with `Operator(qc)` and checks that, ~12x the cost), or
            False (no check). See `SU4GeodesicPSFSynthesizer`.
        entangling_basis: "canonical" emits RXX/RYY/RZZ directly; "cx"
            re-expresses the entangling core through Qiskit's exact CX-basis
            decomposer, which costs 2x fewer native gates on hardware whose
            native 2-qubit gate is CX-like (see docs/findings/entangling-basis.md).
        on_unsupported: "keep" (default) falls back to CX-basis synthesis for a
            block the core cannot decompose; "raise" turns it into an error.
        tol: infidelity above which a synthesized block is rejected and falls
            back instead of being emitted.

    Returns:
        A new circuit with the same structure and qubit/clbit layout as `qc`.
    """
    verify = _validate_verify(verify)

    def worth_consolidating(dag, block):
        if len(block) > block_gate_floor:
            return True
        # Only with entangling_basis="cx": the saving is counted in CX, and the canonical RXX/RYY/RZZ output
        # re-expands to more CX than the gates as written (measured, sandbox, 2026-10-01: 40-qubit random
        # circuit 641 -> 648 CX with the rule on under "canonical").
        return CONSOLIDATE_IF_CHEAPER and entangling_basis == "cx" and _block_saves_cx(block)

    pm_consolidate = PassManager([
        Collect2qBlocks(filter_fn=worth_consolidating),
        ConsolidateBlocks(kak_basis_gate=None, force_consolidate=True),
    ])
    qc_blocked = pm_consolidate.run(qc)

    hyper = GeodesicPSFHyper(
        tol=tol, on_unsupported=on_unsupported, entangling_basis=entangling_basis
    )
    synth = SU4GeodesicPSFSynthesizer(hyper, verify=verify)

    # `copy_empty_like()` keeps the registers, loose bits, name, metadata and
    # global phase of the input. The previous version built
    # `QuantumCircuit(qc.num_qubits, qc.num_clbits)` and then mapped every bit
    # back through `find_bit`, which dropped register structure (so a
    # classically-conditioned instruction had nowhere to point) and paid two
    # lookups per instruction to do it.
    qc_psf = qc_blocked.copy_empty_like()
    qc_psf.global_phase = qc_blocked.global_phase

    blocks_processed = 0
    blocks_seen = 0

    # All blocks are decomposed and polished together (changelog item 20),
    # then emitted in their original order.
    block_mats = {}
    for i, inst in enumerate(qc_blocked.data):
        op = inst.operation
        if len(inst.qubits) == 2 and op.name == "unitary":
            mat = op.to_matrix()
            if mat is not None and mat.shape == (4, 4):
                block_mats[i] = mat
    order = list(block_mats)
    synthesized = dict(zip(order, synth.synthesize_many([block_mats[i] for i in order])))

    for i, inst in enumerate(qc_blocked.data):
        if i in synthesized:
            synthesized_block, fell_back = synthesized[i]
            blocks_seen += 1
            if not fell_back:
                blocks_processed += 1
            qc_psf.compose(synthesized_block, inst.qubits, inplace=True)
            continue
        qc_psf.append(inst.operation, inst.qubits, inst.clbits)

    logger.debug(
        "PSF-Zero Rust Core executed for %d/%d blocks (%d fell back); "
        "block_gate_floor=%d; verify=%s; entangling_basis=%s.",
        blocks_processed,
        blocks_seen,
        synth.fallback_count,
        block_gate_floor,
        verify,
        entangling_basis,
    )
    if synth.fallback_count:
        # One summary, once, instead of a warning per block.
        warnings.warn(synth.fallback_summary(), UserWarning, stacklevel=2)
    return qc_psf


def edge_errors_from_target(target, gate_names=("cz", "ecr", "cx")) -> dict:
    """`{(p, q): error}` for the first of `gate_names` the target supports,
    undirected (p < q), the smaller error when both directions are listed.
    Edges without an error value are left out. For `layout_edge_errors`
    (changelog item 16)."""
    name = next((g for g in gate_names if g in target.operation_names), None)
    if name is None:
        raise ValueError(f"target supports none of {gate_names}")
    out = {}
    for qargs, props in target[name].items():
        if qargs is None or props is None or props.error is None:
            continue
        key = (min(qargs), max(qargs))
        out[key] = min(out.get(key, props.error), props.error)
    return out


def qubit_errors_from_target(target, gate_name: str = "sx") -> dict:
    """`{q: error}` of the single-qubit `gate_name` (default `sx`, the only
    non-virtual single-qubit gate on IBM devices). Qubits without an error
    value are left out. For `layout_qubit_errors` (changelog item 19)."""
    out = {}
    if gate_name not in target.operation_names:
        return out
    for qargs, props in target[gate_name].items():
        if qargs is None or props is None or props.error is None:
            continue
        out[qargs[0]] = props.error
    return out


PRUNE_STATS = {"calls": 0, "edges_removed": 0, "qubits_isolated": 0, "checked": 0, "recompiled": 0}
REFINE_STATS = {"calls": 0, "applied": 0}  # changelog item 33


class _RecordRefine(AnalysisPass):
    """Counts how often the exact re-placement (changelog item 33) found a better placement."""

    def run(self, dag):
        REFINE_STATS["calls"] += 1
        if self.property_set["VF2PostLayout_stop_reason"] is VF2PostLayoutStopReason.SOLUTION_FOUND:
            REFINE_STATS["applied"] += 1


RESYNTH_STATS = {"applied": 0, "kept_original": 0, "selected_resynthesised": 0, "selected_original": 0,
                 "not_estimable": 0}
RESYNTH_MAX_QUBITS = 16


def excitation_cost(circ, target):
    """Item 35's estimate for "select": sum over gates of -log(1 - reported error), plus duration / T1 x P(1) on each
    of the gate's qubits, P(1) from the noiseless state (from |0...0>) just before the gate. None if more than
    RESYNTH_MAX_QUBITS qubits are touched or an instruction has no matrix (other than barrier, measure, delay).
    The state is a plain numpy tensor (axis j = j-th touched qubit); gate matrices are Qiskit's (little-endian)."""
    import math
    ops = [(ins.operation, tuple(circ.find_bit(b).index for b in ins.qubits)) for ins in circ.data
           if ins.operation.name not in ("barrier", "measure", "delay")]
    active = sorted({i for _, q in ops for i in q})
    if len(active) > RESYNTH_MAX_QUBITS:
        return None
    k = max(len(active), 1)
    pos = {p: j for j, p in enumerate(active)}
    qp = getattr(target, "qubit_properties", None) or []
    psi = np.zeros((2,) * k, dtype=complex)
    psi[(0,) * k] = 1.0
    cost = 0.0
    for op, q in ops:
        props = target[op.name].get(q, None) if op.name in target.operation_names else None
        if props is not None and props.error is not None:
            cost += -math.log(max(1.0 - props.error, 1e-300))
        dur = props.duration if props is not None and props.duration else 0.0
        axes = [pos[i] for i in q]
        if dur:
            for i, ax in zip(q, axes):
                t1 = getattr(qp[i], "t1", None) if i < len(qp) and qp[i] is not None else None
                if t1:
                    cost += dur / t1 * float(np.sum(np.abs(np.take(psi, 1, axis=ax)) ** 2))
        try:
            mat = np.asarray(op.to_matrix(), dtype=complex)
        except Exception:
            return None
        m = len(axes)
        rev = axes[::-1]  # Qiskit's matrix index is little-endian: the first qarg is the least significant bit
        psi = np.tensordot(mat.reshape((2,) * (2 * m)), psi, axes=(list(range(m, 2 * m)), rev))
        psi = np.moveaxis(psi, list(range(m)), rev)
    return cost


def _final_resynthesis(out, target, max_error):
    """Item 35: re-synthesise every two-qubit block of a finished circuit with Qiskit, on the target, exactly.
    Returns the original circuit if the result has an instruction the target does not provide or a two-qubit
    gate in a direction the target reports failed."""
    pm = PassManager([ConsolidateBlocks(force_consolidate=True, approximation_degree=1.0, target=target),
                      UnitarySynthesis(approximation_degree=1.0, target=target),
                      Optimize1qGatesDecomposition(target=target)])
    new = pm.run(out)
    new._layout = out._layout
    for ins in new.data:
        name = ins.operation.name
        if name in ("barrier", "measure", "delay"):
            continue
        q = tuple(new.find_bit(b).index for b in ins.qubits)
        props = target[name].get(q, None) if name in target.operation_names else None
        if (name not in target.operation_names or q not in target[name]
                or (len(q) == 2 and props is not None and props.error is not None and props.error >= max_error)):
            RESYNTH_STATS["kept_original"] += 1
            return out
    RESYNTH_STATS["applied"] += 1
    return new


COMPARE_STATS = {"psf": 0, "level3": 0, "level3_refused": 0, "not_estimable": 0, "floor": 0, "floor_refused": 0}


def decoherence_floor(target, qubits, duration) -> float:
    """Average gate infidelity of thermal relaxation on `qubits` for `duration` (item 34, carried over by item 37):
    per qubit the process fidelity is (1 + 2 exp(-t/T2) + exp(-t/T1)) / 4 with T2 capped at 2 T1, the product over
    qubits is F, and F_avg = (d F + 1) / (d + 1). 0 when the duration or T1 is unknown."""
    import math
    if not duration:
        return 0.0
    qp = getattr(target, "qubit_properties", None)
    if not qp:
        return 0.0
    f = 1.0
    for q in qubits:
        p = qp[q] if q < len(qp) else None
        t1 = getattr(p, "t1", None) if p is not None else None
        t2 = getattr(p, "t2", None) if p is not None else None
        if not t1:
            return 0.0
        t2 = min(t2, 2 * t1) if t2 else 2 * t1
        f *= (1.0 + 2.0 * math.exp(-duration / t2) + math.exp(-duration / t1)) / 4.0
    d = 2 ** len(qubits)
    return 1.0 - (d * f + 1.0) / (d + 1.0)


def floor_aware_target(target):
    """A copy of `target` in which every instruction's error is max(reported error, decoherence floor) (item 34,
    carried over by item 37). The original is not modified."""
    import copy
    from qiskit.transpiler import InstructionProperties
    out = copy.deepcopy(target)
    for name in list(out.operation_names):
        if name in ("measure", "delay", "reset", "barrier"):
            continue
        for qargs, props in list(out[name].items()):
            if qargs is None or props is None or props.error is None:
                continue
            fl = decoherence_floor(out, list(qargs), props.duration)
            if fl > props.error:
                out.update_instruction_properties(name, qargs, InstructionProperties(duration=props.duration, error=fl))
    return out


def pauli_cost(circ, target):
    """Item 37's state-aware first-order Pauli estimate (see the changelog). None if more than RESYNTH_MAX_QUBITS
    qubits are touched or an instruction has no matrix (other than barrier, measure, delay)."""
    import math
    ops = [(ins.operation, tuple(circ.find_bit(b).index for b in ins.qubits)) for ins in circ.data
           if ins.operation.name not in ("barrier", "measure", "delay")]
    active = sorted({i for _, q in ops for i in q})
    if len(active) > RESYNTH_MAX_QUBITS:
        return None
    k = max(len(active), 1)
    pos = {p: j for j, p in enumerate(active)}
    qp = getattr(target, "qubit_properties", None) or []
    psi = np.zeros((2,) * k, dtype=complex)
    psi[(0,) * k] = 1.0
    cost = 0.0
    for op, q in ops:
        try:
            mat = np.asarray(op.to_matrix(), dtype=complex)
        except Exception:
            return None
        axes = [pos[i] for i in q]
        m = len(axes)
        rev = axes[::-1]  # Qiskit's matrix index is little-endian: the first qarg is the least significant bit
        psi = np.tensordot(mat.reshape((2,) * (2 * m)), psi, axes=(list(range(m, 2 * m)), rev))
        psi = np.moveaxis(psi, list(range(m)), rev)
        props = target[op.name].get(q, None) if op.name in target.operation_names else None
        if props is None:
            continue
        e = props.error or 0.0
        t = props.duration or 0.0
        thermal_f = 1.0
        for i, ax in zip(q, axes):
            t1 = getattr(qp[i], "t1", None) if i < len(qp) and qp[i] is not None else None
            t2 = getattr(qp[i], "t2", None) if i < len(qp) and qp[i] is not None else None
            if not t or not t1:
                continue
            t2 = min(t2, 2 * t1) if t2 else 2 * t1
            px = (1.0 - math.exp(-t / t1)) / 4.0
            pz = max((1.0 - math.exp(-t / t2)) / 2.0 - px, 0.0)
            mm = np.moveaxis(psi, ax, 0).reshape(2, -1)
            rho = mm @ mm.conj().T
            ex, ey, ez = 2 * rho[0, 1].real, -2 * rho[0, 1].imag, (rho[0, 0] - rho[1, 1]).real
            cost += px * (1 - ex * ex) + px * (1 - ey * ey) + pz * (1 - ez * ez)
            thermal_f *= (1.0 + 2.0 * math.exp(-t / t2) + math.exp(-t / t1)) / 4.0
        d = 2 ** m
        cost += max(e - (1.0 - (d * thermal_f + 1.0) / (d + 1.0)), 0.0) * (d + 1) / d
    return cost


def hybrid_cost(circ, target):
    """Item 38's estimate: amplitude damping as `excitation_cost` counts it, pure dephasing as the Z part of
    `pauli_cost`, and the reported error above the thermal floor (see the changelog). None if more than
    RESYNTH_MAX_QUBITS qubits are touched or an instruction has no matrix (other than barrier, measure, delay)."""
    import math
    ops = [(ins.operation, tuple(circ.find_bit(b).index for b in ins.qubits)) for ins in circ.data
           if ins.operation.name not in ("barrier", "measure", "delay")]
    active = sorted({i for _, q in ops for i in q})
    if len(active) > RESYNTH_MAX_QUBITS:
        return None
    k = max(len(active), 1)
    pos = {p: j for j, p in enumerate(active)}
    qp = getattr(target, "qubit_properties", None) or []
    psi = np.zeros((2,) * k, dtype=complex)
    psi[(0,) * k] = 1.0

    def thermal(i, t):
        p = qp[i] if i < len(qp) else None
        t1 = getattr(p, "t1", None) if p is not None else None
        if not t or not t1:
            return None
        t2 = getattr(p, "t2", None)
        return t1, (min(t2, 2 * t1) if t2 else 2 * t1)

    def rho1(ax):
        mm = np.moveaxis(psi, ax, 0).reshape(2, -1)
        return mm @ mm.conj().T

    cost = 0.0
    for op, q in ops:
        try:
            mat = np.asarray(op.to_matrix(), dtype=complex)
        except Exception:
            return None
        props = target[op.name].get(q, None) if op.name in target.operation_names else None
        axes = [pos[i] for i in q]
        m = len(axes)
        t = (props.duration or 0.0) if props is not None else 0.0
        if props is not None:
            for i, ax in zip(q, axes):
                th = thermal(i, t)
                if th:
                    cost += t / th[0] * float(rho1(ax)[1, 1].real)
        rev = axes[::-1]  # Qiskit's matrix index is little-endian: the first qarg is the least significant bit
        psi = np.tensordot(mat.reshape((2,) * (2 * m)), psi, axes=(list(range(m, 2 * m)), rev))
        psi = np.moveaxis(psi, list(range(m)), rev)
        if props is None:
            continue
        thermal_f = 1.0
        for i, ax in zip(q, axes):
            th = thermal(i, t)
            if not th:
                continue
            t1, t2 = th
            rate = max(1.0 / t2 - 1.0 / (2.0 * t1), 0.0)
            r = rho1(ax)
            ez = float((r[0, 0] - r[1, 1]).real)
            cost += (1.0 - math.exp(-t * rate)) / 2.0 * (1.0 - ez * ez)
            thermal_f *= (1.0 + 2.0 * math.exp(-t / t2) + math.exp(-t / t1)) / 4.0
        d = 2 ** m
        cost += max((props.error or 0.0) - (1.0 - (d * thermal_f + 1.0) / (d + 1.0)), 0.0) * (d + 1) / d
    return cost


def _choose(cands, target, score):
    """Items 37-38: the (name, circuit) with the lowest estimate; the first (the release's circuit) on ties or when
    any estimate cannot be made."""
    f = {"pauli": pauli_cost, "hybrid": hybrid_cost}.get(score, excitation_cost)
    costs = [f(c, target) for _, c in cands]
    if any(c is None for c in costs):
        COMPARE_STATS["not_estimable"] += 1
        return cands[0][1]
    best = min(range(len(cands)), key=lambda j: (costs[j], j))
    COMPARE_STATS[cands[best][0]] += 1
    return cands[best][1]


def _acceptable(circ, target, max_error) -> bool:
    """Item 36: every instruction on the target, no failed qubit, no two-qubit gate in a failed direction."""
    edges, qubits = _failed_elements(target, max_error)
    for ins in circ.data:
        name = ins.operation.name
        if name in ("barrier", "measure", "delay"):
            continue
        q = tuple(circ.find_bit(b).index for b in ins.qubits)
        if name not in target.operation_names or q not in target[name]:
            return False
        if any(i in qubits for i in q) or (len(q) == 2 and q in edges):
            return False
    return True


def _compare_level3(qc, out, target, max_error, seed_transpiler):
    """Item 36: the release's circuit `out` or Qiskit level 3's, whichever has the lower excitation_cost."""
    l3 = transpile(qc, target=target, optimization_level=3, seed_transpiler=seed_transpiler, approximation_degree=1.0)
    if not _acceptable(l3, target, max_error):
        COMPARE_STATS["level3_refused"] += 1
        return out
    a, b = excitation_cost(out, target), excitation_cost(l3, target)
    if a is None or b is None:
        COMPARE_STATS["not_estimable"] += 1
        return out
    if b < a:
        COMPARE_STATS["level3"] += 1
        return l3
    COMPARE_STATS["psf"] += 1
    return out


def _select_resynthesis(out, target, max_error):
    """Item 35, "select": the re-synthesised circuit if its excitation_cost is lower, else the original."""
    new = _final_resynthesis(out, target, max_error)
    if new is out:
        return out
    a, b = excitation_cost(out, target), excitation_cost(new, target)
    if a is None or b is None:
        RESYNTH_STATS["not_estimable"] += 1
        return out
    if b < a:
        RESYNTH_STATS["selected_resynthesised"] += 1
        return new
    RESYNTH_STATS["selected_original"] += 1
    return out


def _refine_found(property_set):
    return property_set["VF2PostLayout_stop_reason"] is VF2PostLayoutStopReason.SOLUTION_FOUND


def _failed_elements(target, max_error: float, gate_names=("cz", "ecr", "cx"), qubit_gate: str = "sx"):
    """(set of directed failed edges, set of failed qubits) as reported by `target` (changelog item 31)."""
    name = next((g for g in gate_names if g in target.operation_names), None)
    edges = set()
    if name is not None:
        for qargs, props in target[name].items():
            if qargs is not None and props is not None and props.error is not None and props.error >= max_error:
                edges.add(tuple(qargs))
    qubits = {q for q, e in qubit_errors_from_target(target, qubit_gate).items() if e >= max_error}
    return edges, qubits


def _uses_failed(circ, edges, qubits) -> bool:
    for inst in circ.data:
        idx = tuple(circ.find_bit(q).index for q in inst.qubits)
        if any(i in qubits for i in idx):
            return True
        if len(idx) == 2 and (idx in edges or idx[::-1] in edges):
            return True
    return False


def prune_coupling_map(coupling_map: CouplingMap, target, max_error: float = 0.5,
                       gate_names=("cz", "ecr", "cx"), qubit_gate: str = "sx") -> CouplingMap:
    """A copy of `coupling_map` without the edges the `target` reports as failed (changelog item 31).

    A directed edge (a, b) is removed when the error of the target's native 2-qubit gate on (a, b) -- or on
    (b, a) if (a, b) is not listed -- is >= `max_error`, or when either endpoint's `qubit_gate` error is
    >= `max_error`. Edges without an error value are kept. Every physical qubit is kept, so indices and
    `size()` do not change."""
    name = next((g for g in gate_names if g in target.operation_names), None)
    gate_err = {}
    if name is not None:
        for qargs, props in target[name].items():
            if qargs is not None and props is not None and props.error is not None:
                gate_err[tuple(qargs)] = props.error
    bad_q = {q for q, e in qubit_errors_from_target(target, qubit_gate).items() if e >= max_error}
    out = CouplingMap()
    for q in range(coupling_map.size()):
        out.add_physical_qubit(q)
    removed = 0
    for a, b in coupling_map.get_edges():
        e = gate_err.get((a, b), gate_err.get((b, a)))
        if (e is not None and e >= max_error) or a in bad_q or b in bad_q:
            removed += 1
            continue
        out.add_edge(a, b)
    PRUNE_STATS["calls"] += 1
    PRUNE_STATS["edges_removed"] += removed
    PRUNE_STATS["qubits_isolated"] += sum(1 for q in range(out.size())
                                          if not out.graph.out_degree(q) and not out.graph.in_degree(q)
                                          and coupling_map.graph.out_degree(q) + coupling_map.graph.in_degree(q))
    return out


def _edge_weights_from_errors(edge_errors: dict, qubit_errors: dict | None = None,
                              n2: float = 1.0, n1: float = 0.0) -> dict:
    """Integer matching weights, larger is better:
    round(1e6 * (K + n2 log(1 - e_pq) + n1 (log(1 - e_p) + log(1 - e_q)))),
    every error capped at 0.5 and K chosen so that every weight is positive.
    The total weight of a matching then tracks the log of the product of the
    fidelities of the gates it will carry. Edges absent from `edge_errors`
    get weight 0 in the matching (treated as worst); qubits absent from
    `qubit_errors` contribute nothing. With the defaults (n2=1, n1=0, no
    qubit errors) edges rank exactly as in the 2026-09-26.3 weighting
    (the constant offset differs)."""
    import math

    def lf(err):
        return math.log(1.0 - min(float(err), 0.5))

    k = 1.0 + (n2 + 2.0 * n1) * math.log(2.0)
    out = {}
    for (p, q), err in edge_errors.items():
        total = n2 * lf(err)
        if qubit_errors is not None and n1:
            total += n1 * (lf(qubit_errors.get(p, 0.0)) + lf(qubit_errors.get(q, 0.0)))
        out[(p, q)] = int(round(1e6 * (k + total)))
    return out


def _layout_weights(qc_compressed: QuantumCircuit, pairs, edge_errors: dict,
                    qubit_errors: dict | None) -> dict:
    """Matching weights for `compile_for_hardware` (changelog items 16, 19).
    Without qubit errors: the 2026-09-26.3 weighting. With them: n2 = mean
    number of 2-qubit gates per interacting pair in the compressed circuit,
    n1 = 2 n2 + 1 estimated `sx` per qubit."""
    if qubit_errors is None:
        return _edge_weights_from_errors(edge_errors)  # item 16 behaviour
    n_twoq = sum(1 for inst in qc_compressed.data if len(inst.qubits) == 2)
    n2 = n_twoq / max(1, len(pairs))
    return _edge_weights_from_errors(edge_errors, qubit_errors, n2=n2, n1=2.0 * n2 + 1.0)


def _layout_map_to_list(layout_map: dict, num_logical: int, num_physical: int) -> list[int]:
    """Convert `smart_vf2_layout()`'s `{logical: physical}` dict to the
    virtual-qubit-index-ordered list `transpile(initial_layout=...)` expects.

    Inlined from `benchmarks/benchmark_smart_layout_vs_default.py`'s
    `layout_map_to_list()` (same logic, copied rather than imported so this
    module does not depend on a benchmark script's internals) -- see that
    file's own docstring for why unused logical qubits get the remaining
    physical qubits assigned in index order rather than left unassigned.
    """
    used = set(layout_map.values())
    spare_iter = (p for p in range(num_physical) if p not in used)
    out = []
    for q in range(num_logical):
        if q in layout_map:
            out.append(layout_map[q])
        else:
            out.append(next(spare_iter))
    return out


class _CarryPermutation(AnalysisPass):
    """Re-inserts the `virtual_permutation_layout` found by an earlier ElidePermutations run, so the preset
    pipeline folds it into the output's final layout exactly as it does at optimization levels 2 and 3."""

    def __init__(self, layout):
        super().__init__()
        self._layout = layout

    def run(self, dag):
        self.property_set["virtual_permutation_layout"] = self._layout


def _resolve_auto(value, entangling_basis, name):
    if value == "auto":
        return entangling_basis == "cx"
    if isinstance(value, bool):
        return value
    raise ValueError(f"{name} must be True, False or 'auto', got {value!r}")


class _AbsorbRoutingSwaps(TransformationPass):
    """Candidate 2026-10-01.c2, run in the preset pipeline's post_routing stage (before translation).
    Consolidates every 2-qubit block that holds a SWAP inserted by routing and whose optimal CX count is below
    its cost as written (SWAP 3 CX, so SWAP + any 2-qubit gate on the same pair always qualifies), and
    re-synthesises it with the PSF-Zero core. Blocks without a SWAP are left exactly as routed, which keeps
    this pass cheap on circuits that routing did not touch."""

    def __init__(self, verify, tol, on_unsupported):
        super().__init__()
        self._verify, self._tol, self._on_unsupported = verify, tol, on_unsupported

    def run(self, dag):
        if "swap" not in dag.count_ops():
            return dag
        qc = dag_to_circuit(dag)
        pm = PassManager([
            Collect2qBlocks(filter_fn=lambda d, block: (any(n.op.name == "swap" for n in block)
                                                        and _block_saves_cx(block))),
            ConsolidateBlocks(kak_basis_gate=None, force_consolidate=True),
        ])
        blocked = pm.run(qc)
        idx = [i for i, inst in enumerate(blocked.data)
               if len(inst.qubits) == 2 and inst.operation.name == "unitary"]
        if not idx:
            return dag
        synth = SU4GeodesicPSFSynthesizer(
            GeodesicPSFHyper(tol=self._tol, on_unsupported=self._on_unsupported, entangling_basis="cx"),
            verify=self._verify)
        done = dict(zip(idx, synth.synthesize_many([blocked.data[i].operation.to_matrix() for i in idx])))
        out = blocked.copy_empty_like()
        out.global_phase = blocked.global_phase
        for i, inst in enumerate(blocked.data):
            if i in done:
                out.compose(done[i][0], inst.qubits, inplace=True)
            else:
                out.append(inst.operation, inst.qubits, inst.clbits)
        new = circuit_to_dag(out)
        return new


def compile_for_hardware(
    qc: QuantumCircuit,
    coupling_map: CouplingMap,
    basis_gates: list[str] | None = None,
    block_gate_floor: int = DEFAULT_BLOCK_GATE_FLOOR,
    routing_optimization_level: int = 1,
    verify: Union[bool, str] = True,
    entangling_basis: str = "canonical",
    seed_transpiler: int | None = None,
    initial_layout: list[int] | None = None,
    on_unsupported: str = "keep",
    tol: float = 1e-5,
    layout_search: bool = False,
    layout_search_time_budget_s: float = 2.0,
    layout_search_call_limit: int = 50_000,
    layout_search_fallback_call_limit: int = 2_000_000,
    layout_search_use_fallback: bool = True,
    layout_edge_errors: dict | None = None,
    layout_qubit_errors: dict | None = None,
    callback=None,
    elide_permutations: Union[bool, str] = "auto",
    post_routing_resynthesis: Union[bool, str] = "auto",
    target=None,
    prune_max_error: float = 0.5,
    placement_refine: bool = False,
    placement_call_limit: int = 300_000,
    placement_max_trials: int = 2_500,
    final_resynthesis: Union[bool, str] = False,
    compare_level3: bool = False,
    compare_floor: bool = False,
    candidate_score: str = "excitation",
    _refine_target=None,
) -> QuantumCircuit:
    """Compress with PSF-Zero, then route (and, if `basis_gates` is given,
    translate) with Qiskit.

    `basis_gates` matters more than it looks: with only a `coupling_map`,
    `transpile()` lays out and routes but never targets a gate set, so the
    RXX/RYY/RZZ this pass emits pass straight through and the result is not
    ISA-submittable.

    `routing_optimization_level` defaults to 1. It used to default to 2, on the
    stated grounds that translation "only happens" at level 2 -- that is not
    true, and was measured: with `basis_gates=["rz","sx","x","cx"]` the output
    contains nothing outside that set at level 0, 1 and 2 alike. What level 2
    actually does here is undo this pass. Qiskit's preset pipeline re-runs
    ConsolidateBlocks (init stage) and UnitarySynthesis (init + translation
    stages) over input it has no reason to trust, so at level 2 the result is
    bit-identical to plain `transpile(optimization_level=2)` -- verified gate
    for gate, qubit for qubit, parameter for parameter at 4, 6 and 7 qubits --
    for essentially the same wall time (100q dense-pair blocks: 1224 ms here
    vs 1232 ms for plain Qiskit). Every millisecond PSF-Zero spends is thrown
    away at that level. Addenda 24-25 (spare-qubit-cliff series) measured the
    same effect on a saturated coupling map specifically: raising this
    parameter from 1 toward 3 makes PSF-Zero's own spare-qubit cliff grow from
    ~7-8x to ~250-275x -- landing inside plain Qiskit L3's own 263-300x cliff
    -- and by level 3, `compile_for_hardware` is no longer faster than plain
    Qiskit L3 anywhere in that sweep, cliff or not (0.83x-0.96x). That result
    is what motivated `layout_search` below: rather than relying on level 1's
    internal routing call inheriting a cheaper `VF2Layout` failure, address
    the failure itself.

    The trade at level 1, measured on 50-156 qubit dense-pair-block circuits
    over a grid coupling map, is: the same 2-qubit gate count as Qiskit's
    optimization_level 2 and 3 (150 at 100q, 240 at 156q) for 1/20th to 1/59th
    of their time, at roughly 30-40% more depth (23 vs 16 at 100q, 44 vs 35 at
    156q). Against optimization_level 1 it wins outright: 1.8x faster, 20x
    fewer 2-qubit gates, 10x shallower. If minimum depth is what matters and
    compile time is not a constraint, call Qiskit's optimization_level=2
    directly -- routing_optimization_level=2 here gives you exactly that
    circuit and charges PSF-Zero's synthesis on top.

    This pass only helps circuits with deep interaction on the same qubit pair,
    the ones `Collect2qBlocks` can gather into blocks longer than
    `block_gate_floor`. On wide-and-shallow input (`random_circuit`, say) it
    finds few or no blocks, returns the circuit largely untouched by design,
    and is then mostly overhead ahead of the Qiskit call.

    `tol` is forwarded to `compile()`. It previously was not, so the
    acceptance threshold was pinned at its default on this path with no way to
    reach it -- `seed_transpiler` and `on_unsupported` were both plumbed
    through and this one was simply missed.

    `seed_transpiler` pins the internal routing search. Leaving it unset means
    an `optimization_level >= 2` transpile returns a different circuit, and
    takes a different amount of time, on every call for identical input --
    which is exactly the effect that made this project's own coupling-map
    timings unreproducible until it was pinned on the comparison side.

    `initial_layout` pins the logical-to-physical qubit assignment, bypassing
    Qiskit's own layout search entirely. It is forwarded verbatim to
    `transpile()`, so it takes the same forms `transpile()` accepts -- most
    usefully a list of physical qubit indices ordered by virtual qubit index.

    **Passing this skips the whole layout stage, not just the layout search.**
    Measured with `transpile(callback=...)` on 2026-09-14:

        default:              SetLayout, VF2Layout, SabreLayout, VF2PostLayout
        with initial_layout:  SetLayout, ApplyLayout

    Note `VF2PostLayout` is skipped too. With only a `coupling_map` (no error
    rates) that pass has nothing to score against and its absence changes
    nothing, which is the configuration this project's benchmarks use. **With
    a real hardware `Target` carrying error rates it does have something to
    score, and skipping it can therefore cost fidelity** -- a layout that is
    valid (every interacting pair lands on a coupled edge) is not
    automatically a layout that is good on a noisy device. Supplying this
    argument moves that judgement to the caller. `layout_search=True` below
    also skips `VF2PostLayout` whenever it finds a layout, for the same
    reason: it, too, threads its result through `initial_layout`.

    The motivating use is feeding in a layout found by a separate search: this
    project's `psf_smart_layout.smart_vf2_layout()` returns `{logical:
    physical}`, which `layout_map_to_list()` in
    `benchmarks/benchmark_smart_layout_vs_default.py` converts to the list
    form expected here. Until this parameter existed there was no way to do
    that, so the end-to-end comparison that motivated the search could not be
    run at all (recorded in spare-qubit-cliff addendum 15, section 4).

    `layout_search` (new, item 12 in this file's changelog): when True, runs
    that same search *inside* this call instead of requiring the caller to
    invoke `smart_vf2_layout()` and build `initial_layout` by hand. The
    interaction graph handed to the search is read directly off the
    PSF-Zero-compressed circuit's own 2-qubit instructions (deduplicated,
    undirected pairs of qubit indices) -- the same circuit that is about to be
    routed, not the pre-compression input, since consolidation does not
    change which qubit pairs interact. Mutually exclusive with passing
    `initial_layout` directly (raises `ValueError` if both are given -- there
    is no principled way to silently prefer one over the other). Requires
    `psf_smart_layout` to be importable (raises `ImportError` naming the
    problem if it is not, rather than silently skipping the search); this is
    a new hard dependency only for callers who opt into `layout_search=True`.
    `layout_search_time_budget_s`, `layout_search_call_limit` and
    `layout_search_fallback_call_limit` are forwarded to
    `smart_vf2_layout()`'s `time_budget_s`, `per_attempt_call_limit` and
    `fallback_call_limit` respectively; `layout_search_use_fallback` disables
    its more expensive stage-2 search (see `psf_smart_layout.py`) if set to
    False. If the search does not find a layout within budget, this function
    falls through to Qiskit's own default layout stage exactly as if
    `layout_search` had been False -- the time already spent searching is
    real and is included in this call's own wall-clock cost, not hidden.

    `layout_edge_errors` (new, item 16): `{(p, q): error}` for the native
    2-qubit gate; see the changelog. Used only with `layout_search=True` and
    only when the interaction graph is a set of disjoint pairs.
    `layout_qubit_errors` (new, item 19): `{q: error}` of `sx`, added to the
    same weights; ignored unless `layout_edge_errors` is also given.

    `elide_permutations` and `post_routing_resynthesis` (candidate 2026-10-01.c2, items 29-30): True,
    False or "auto" (on for `entangling_basis="cx"` only). When either is active the routing call is
    made through `generate_preset_pass_manager(...)` with the same arguments as the `transpile()` call
    below, plus a `pre_init` pass (the permutation) and/or a `post_routing` pass (SWAP absorption).
    `target` and `prune_max_error` (candidate 2026-10-02.c3, item 31): with a device `Target`, a result that
    uses a failed coupler or qubit (error >= `prune_max_error`) is recompiled on the coupling map without
    them; any other result is returned exactly as without `target`. `None` (default) changes nothing.
    `placement_refine`, `placement_call_limit`, `placement_max_trials` (candidate 2026-10-02.c5, item 33):
    with a `target`, end the routing with Qiskit level 3's exact re-placement (`VF2PostLayout` with
    `strict_direction=True`), which relabels physical qubits when that lowers the summed -log(1 - error) of the
    gates as placed. Requires `target`; False (default) changes nothing. `_refine_target` is internal.
    `final_resynthesis` (candidate 2026-10-03.c8, item 35): True re-synthesises every two-qubit block of the
    finished circuit with Qiskit (exact, on the target); "select" does so only when `excitation_cost` says the
    result is better. Requires `target`; False (default) changes nothing.
    `compare_level3` (candidate 2026-10-03.c9, item 36): True also compiles the input with Qiskit's level 3 on the
    target and returns that circuit instead when its `excitation_cost` is lower and it uses no failed element.
    Requires `target`; False (default) changes nothing.
    `compare_floor`, `candidate_score` (candidate 2026-10-03.c10, item 37): `compare_floor=True` adds the
    release's pipeline re-placed on `floor_aware_target(target)` as a candidate; `candidate_score="pauli"` chooses
    among the candidates by `pauli_cost` instead of `excitation_cost`. Require `target`; the defaults change
    nothing. `candidate_score="hybrid"` (candidate 2026-10-04.c11, item 38) chooses by `hybrid_cost`.
    `callback` (new, item 13): forwarded verbatim to the internal
    `transpile()` call below, unchanged from what plain `transpile(callback=
    ...)` accepts. `None` by default -- passing nothing here changes nothing
    about this function's behavior or cost. See item 13 in this file's
    changelog for why this exists.
    """
    if placement_refine and target is None:
        raise ValueError("placement_refine=True needs the device `target` (changelog item 33)")
    if final_resynthesis not in (False, True, "select"):
        raise ValueError('final_resynthesis must be False, True or "select" (changelog item 35)')
    if final_resynthesis and target is None:
        raise ValueError("final_resynthesis needs the device `target` (changelog item 35)")
    if compare_level3 and target is None:
        raise ValueError("compare_level3=True needs the device `target` (changelog item 36)")
    if compare_floor and target is None:
        raise ValueError("compare_floor=True needs the device `target` (changelog item 37)")
    if candidate_score not in ("excitation", "pauli", "hybrid"):
        raise ValueError('candidate_score must be "excitation", "pauli" or "hybrid" (changelog items 37-38)')
    if layout_search and initial_layout is not None:
        raise ValueError(
            "layout_search=True and an explicit initial_layout were both "
            "given -- ambiguous which should be used. Pass one or the other, "
            "not both."
        )

    # Candidate 2026-10-02.c3 (item 31): compile as without a target; recompile on the pruned coupling map
    # only if that result uses a failed coupler or qubit.
    if target is not None:
        args = dict(qc=qc, coupling_map=coupling_map, basis_gates=basis_gates, block_gate_floor=block_gate_floor,
                    routing_optimization_level=routing_optimization_level, verify=verify,
                    entangling_basis=entangling_basis, seed_transpiler=seed_transpiler,
                    initial_layout=initial_layout, on_unsupported=on_unsupported, tol=tol,
                    layout_search=layout_search, layout_search_time_budget_s=layout_search_time_budget_s,
                    layout_search_call_limit=layout_search_call_limit,
                    layout_search_fallback_call_limit=layout_search_fallback_call_limit,
                    layout_search_use_fallback=layout_search_use_fallback, layout_edge_errors=layout_edge_errors,
                    layout_qubit_errors=layout_qubit_errors, callback=callback,
                    elide_permutations=elide_permutations, post_routing_resynthesis=post_routing_resynthesis,
                    placement_call_limit=placement_call_limit, placement_max_trials=placement_max_trials,
                    _refine_target=target if placement_refine else None)
        out = compile_for_hardware(**args)
        edges, qubits = _failed_elements(target, prune_max_error)
        PRUNE_STATS["checked"] += 1
        if _uses_failed(out, edges, qubits):
            PRUNE_STATS["recompiled"] += 1
            args["coupling_map"] = prune_coupling_map(coupling_map, target, prune_max_error)
            out = compile_for_hardware(**args)
        if final_resynthesis == "select":
            out = _select_resynthesis(out, target, prune_max_error)
        elif final_resynthesis:
            out = _final_resynthesis(out, target, prune_max_error)
        if not compare_floor and candidate_score == "excitation":
            if compare_level3:
                out = _compare_level3(qc, out, target, prune_max_error, seed_transpiler)
            return out
        # Item 37: up to three candidates, chosen by candidate_score.
        cands = [("psf", out)]
        if compare_floor:
            fargs = dict(args, coupling_map=coupling_map, _refine_target=floor_aware_target(target))
            fo = compile_for_hardware(**fargs)
            if _uses_failed(fo, edges, qubits):
                fargs["coupling_map"] = prune_coupling_map(coupling_map, target, prune_max_error)
                fo = compile_for_hardware(**fargs)
            if final_resynthesis == "select":
                fo = _select_resynthesis(fo, target, prune_max_error)
            elif final_resynthesis:
                fo = _final_resynthesis(fo, target, prune_max_error)
            if _acceptable(fo, target, prune_max_error):
                cands.append(("floor", fo))
            else:
                COMPARE_STATS["floor_refused"] += 1
        if compare_level3:
            l3 = transpile(qc, target=target, optimization_level=3, seed_transpiler=seed_transpiler,
                           approximation_degree=1.0)
            if _acceptable(l3, target, prune_max_error):
                cands.append(("level3", l3))
            else:
                COMPARE_STATS["level3_refused"] += 1
        return _choose(cands, target, candidate_score) if len(cands) > 1 else out

    # Candidate 2026-10-01.c2: both default to on only for entangling_basis="cx" (their saving is counted in
    # CX); post-routing re-synthesis also needs basis_gates to translate its output.
    elide_permutations = _resolve_auto(elide_permutations, entangling_basis, "elide_permutations")
    post_routing_resynthesis = _resolve_auto(post_routing_resynthesis, entangling_basis, "post_routing_resynthesis")
    permutation = None
    if elide_permutations:
        # Split2QUnitaries(split_swap=True) also catches a SWAP written as a 2-qubit `unitary` (how PennyLane
        # tapes arrive), which ElidePermutations alone does not recognise.
        epm = PassManager([ElidePermutations(), Split2QUnitaries(split_swap=True)])
        seen = {}
        qc_elided = epm.run(qc, callback=lambda **kw: seen.update(
            vpl=kw["property_set"]["virtual_permutation_layout"]))
        permutation = seen.get("vpl")
        if permutation is not None:
            qc_elided._layout = None
            qc = qc_elided
    qc_compressed = compile(
        qc,
        block_gate_floor=block_gate_floor,
        verify=verify,
        entangling_basis=entangling_basis,
        on_unsupported=on_unsupported,
        tol=tol,
    )

    if layout_search:
        try:
            from psf_smart_layout import smart_vf2_layout
        except ImportError as exc:
            raise ImportError(
                "layout_search=True requires psf_smart_layout.py to be "
                "importable (it is normally under prototypes/ in this "
                "project's repository layout) -- it was not found. Either "
                "make it importable (e.g. on PYTHONPATH or copied next to "
                "this file) or call with layout_search=False."
            ) from exc

        pairs = set()
        for inst in qc_compressed.data:
            if len(inst.qubits) == 2:
                i = qc_compressed.find_bit(inst.qubits[0]).index
                j = qc_compressed.find_bit(inst.qubits[1]).index
                if i != j:
                    pairs.add((i, j) if i < j else (j, i))

        layout_map, _search_info = smart_vf2_layout(
            coupling_map,
            sorted(pairs),
            qc_compressed.num_qubits,
            per_attempt_call_limit=layout_search_call_limit,
            time_budget_s=layout_search_time_budget_s,
            fallback_call_limit=layout_search_fallback_call_limit,
            use_fallback=layout_search_use_fallback,
            **({} if layout_edge_errors is None
               else {"edge_weights": _layout_weights(qc_compressed, pairs, layout_edge_errors,
                                                     layout_qubit_errors)}),
        )
        if layout_map is not None:
            initial_layout = _layout_map_to_list(
                layout_map, qc_compressed.num_qubits, coupling_map.size()
            )
        # else: fall through to Qiskit's own default layout stage below,
        # exactly as if layout_search had been False. The search time already
        # spent is real and is not subtracted from this call's wall time.

    if permutation is None and not post_routing_resynthesis and _refine_target is None:
        return transpile(
            qc_compressed,
            coupling_map=coupling_map,
            basis_gates=basis_gates,
            optimization_level=routing_optimization_level,
            seed_transpiler=seed_transpiler,
            initial_layout=initial_layout,
            callback=callback,
        )
    pm = generate_preset_pass_manager(
        optimization_level=routing_optimization_level,
        coupling_map=coupling_map,
        basis_gates=basis_gates,
        seed_transpiler=seed_transpiler,
        initial_layout=initial_layout,
    )
    if permutation is not None:
        if pm.pre_init is None:
            pm.pre_init = PassManager([_CarryPermutation(permutation)])
        else:
            pm.pre_init.append(_CarryPermutation(permutation))
    if post_routing_resynthesis:
        absorb = _AbsorbRoutingSwaps(_validate_verify(verify), tol, on_unsupported)
        if pm.post_routing is None:
            pm.post_routing = PassManager([absorb])
        else:
            pm.post_routing.append(absorb)
    if _refine_target is not None:
        # Item 33: the exact re-placement that Qiskit level 3 runs at the end of its optimization stage.
        refine = [VF2PostLayout(target=_refine_target, seed=-1, call_limit=placement_call_limit,
                                max_trials=placement_max_trials, strict_direction=True),
                  _RecordRefine(),
                  ConditionalController(ApplyLayout(), condition=_refine_found)]
        if pm.optimization is None:
            pm.optimization = PassManager(refine)
        else:
            pm.optimization.append(refine)
    return pm.run(qc_compressed, callback=callback)
