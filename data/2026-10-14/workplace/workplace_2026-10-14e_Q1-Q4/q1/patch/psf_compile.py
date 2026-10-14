"""PSF-Zero -- the compiler. **This file is the latest version of it.**

VERSION: 2026-10-14.c53+91 -- workplace proposal on prototype c53 (c43 = release 2026-10-11.4): items 75 revised, 77-80, 82-91

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
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler.basepasses import AnalysisPass, TransformationPass
from qiskit.transpiler.passes import (BasisTranslator, Collect2qBlocks, ConsolidateBlocks, ElidePermutations,
                                      Optimize1qGatesDecomposition, Split2QUnitaries, UnitarySynthesis)
from qiskit.transpiler import ConditionalController
from qiskit.transpiler.passes import ApplyLayout, VF2PostLayout, Unroll3qOrMore
from qiskit.exceptions import QiskitError
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

VERSION = "2026-10-14.c53+91"  # workplace proposal, item 91 (embeddable graphs by edge colouring) on prototype: items 87-88 (workplace: chain brickwork, rzz in a Pauli frame) + items 89 (diagonal Clifford before measurement dropped) and 90 (estimate-based defaults off, item 75 scored by duration) + item 86 revised (native fractional gates, every rzz folded) + items 84 (choices by duration) and 85 (failed qubits cut first) + item 83 (time-aware block order) + item 82 (mirrored layers) + release 2026-10-11.4 + item 75 revised (search order) + item 77 (no time budget in the layout search) + items 78-80 (re-placement at the routing stage, basis from the Target, Qiskit level 3 as a candidate)
__version__ = VERSION

# Changelog item 27: the version string of the loaded Rust core, for logs.
# Cores built before 2026-09-28 do not define it.
CORE_VERSION = getattr(psf_zero_core, "CORE_VERSION", None)

try:  # item 57b: optional; without it the estimates and checks run in Python, as before
    import psf_zero_core57 as _CORE57
except ImportError:
    _CORE57 = None
CORE57_VERSION = getattr(_CORE57, "CORE57_VERSION", None)

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


class FailedElementsError(TranspilerError):
    """Item 58: no placement of the circuit avoids the couplers and qubits the device reports as failed."""


WARN_WITHOUT_TARGET = True  # item 58: warn once per process when compile_for_hardware has no device Target
_WARNED = {"no_target": False}
_BACKEND_BASIS = ("cx", "cz", "ecr", "rz", "sx", "x", "id")  # item 58: the basis taken from a backend's Target


def readout_errors_from_target(target) -> dict:
    """Item 58: `{q: error}` of `measure`, for the qubits that have an error value."""
    out = {}
    if "measure" not in target.operation_names:
        return out
    for qargs, props in target["measure"].items():
        if qargs is None or props is None or props.error is None:
            continue
        out[qargs[0]] = props.error
    return out


PRUNE_STATS = {"calls": 0, "edges_removed": 0, "qubits_isolated": 0, "checked": 0, "recompiled": 0,
               "unavoidable": 0, "still_failed": 0, "raised": 0, "pruned_first": 0}
PRUNE_FIRST = True  # item 61: with failed elements in the Target, compile once on the pruned map
ABSORB_SYNTH = "qiskit"  # items 62-63: "qiskit" (default) or "psf": how item 30's absorbed blocks are synthesised
COMPRESS = True  # item 62: False routes the input without PSF-Zero's compression (a switch for measurement)


def _recompile_pruned(args, first, edges=None, qubits=None, mode="keep"):
    """Item 43: item 31's recompile on the pruned map, or `first` (with a warning) when no placement exists.
    Item 58: with `edges` and `qubits`, the recompiled circuit is checked again; with `mode="raise"`, a circuit that
    cannot avoid the failed elements raises FailedElementsError instead of being returned."""
    try:
        out = compile_for_hardware(**args)
    except QiskitError as exc:  # item 58: TranspilerError, and CouplingError on a map left without edges
        PRUNE_STATS["unavoidable"] += 1
        if mode == "raise":
            PRUNE_STATS["raised"] += 1
            raise FailedElementsError(f"no placement avoids the failed couplers and qubits ({exc}) (changelog "
                                      f"item 58)") from exc
        warnings.warn(f"no placement on the coupling map without the failed elements ({exc}); keeping the output "
                      f"that uses them (changelog item 43)", RuntimeWarning, stacklevel=3)
        return first
    if edges is not None and _uses_failed(out, edges, qubits):
        PRUNE_STATS["still_failed"] += 1
        if mode == "raise":
            PRUNE_STATS["raised"] += 1
            raise FailedElementsError("the circuit compiled on the map without the failed couplers still uses a "
                                      "failed qubit (changelog item 58)")
        warnings.warn("the output still uses a failed qubit; keeping it (changelog item 58)", RuntimeWarning,
                      stacklevel=3)
    return out
REFINE_STATS = {"calls": 0, "applied": 0}  # changelog item 33


class _RecordRefine(AnalysisPass):
    """Counts how often the exact re-placement (changelog item 33) found a better placement."""

    def run(self, dag):
        REFINE_STATS["calls"] += 1
        if self.property_set["VF2PostLayout_stop_reason"] is VF2PostLayoutStopReason.SOLUTION_FOUND:
            REFINE_STATS["applied"] += 1


RESYNTH_STATS = {"applied": 0, "kept_original": 0, "selected_resynthesised": 0, "selected_original": 0,
                 "not_estimable": 0, "not_checked": 0}
RESYNTH_MAX_QUBITS = 16


CORE57_STATS = {"rust": 0, "python": 0, "fallback": 0}  # item 57b: calls made in Rust, in Python, and refused
_NOT_DONE = object()  # item 57b: the core did not take the call; the Python code runs
_CORE57_MAGIC, _CORE57_BUFFER = 0x37354650, 1


def _core57_buffer(kind, rows, k, t1, t2, state=None):
    """Item 57b: the byte buffer of psf_zero_core57 (its `statevec::parse`): rows are (matrix, positions, error,
    duration, has_props), NaN for a value not reported."""
    import struct
    ms = bytes(len(p) for _, p, _, _, _ in rows)
    ps = bytes(x for _, p, _, _, _ in rows for x in p)
    hp = bytes(1 if h else 0 for _, _, _, _, h in rows)
    err = np.array([e for _, _, e, _, _ in rows], dtype="<f8")
    dur = np.array([d for _, _, _, d, _ in rows], dtype="<f8")
    mat = (np.concatenate([np.ascontiguousarray(m, dtype=complex).ravel() for m, _, _, _, _ in rows]).view("<f8")
           if rows else np.zeros(0, dtype="<f8"))
    out = [struct.pack("<7I", _CORE57_MAGIC, _CORE57_BUFFER, kind, k, len(rows), len(ps), mat.size),
           np.asarray(t1, dtype="<f8").tobytes(), np.asarray(t2, dtype="<f8").tobytes(), ms, ps, hp,
           err.tobytes(), dur.tobytes(), mat.astype("<f8", copy=False).tobytes()]
    if state is not None:
        out.append(np.ascontiguousarray(state, dtype=complex).ravel().view("<f8").astype("<f8", copy=False).tobytes())
    return b"".join(out)


def _core57_estimate(kind, ops, active, pos, qp, target):
    """Item 57b: `excitation_cost` (kind 0) or `hybrid_cost`'s gate terms (kind 1) in psf_zero_core57. None where
    the Python code returns None (an instruction without a matrix); _NOT_DONE where the core cannot take the call."""
    nan = float("nan")
    names = target.operation_names
    rows = []
    for op, q in ops:
        if len(q) > 6:
            return _NOT_DONE
        props = target[op.name].get(q, None) if op.name in names else None
        try:
            mat = _gate_matrix(op)
        except Exception:
            return None
        if props is None:
            rows.append((mat, [pos[i] for i in q], nan, nan, False))
        else:
            e, d = props.error, props.duration
            rows.append((mat, [pos[i] for i in q], nan if e is None else float(e), nan if d is None else float(d),
                         True))
    t1, t2 = [], []
    for i in active:
        p = qp[i] if i < len(qp) else None
        a = getattr(p, "t1", None) if p is not None else None
        b = getattr(p, "t2", None) if p is not None else None
        t1.append(nan if a is None else float(a))
        t2.append(nan if b is None else float(b))
    try:
        value = _CORE57.estimate57(_core57_buffer(kind, rows, len(active), t1, t2))
    except ValueError:
        CORE57_STATS["fallback"] += 1
        return _NOT_DONE
    CORE57_STATS["rust"] += 1
    return value


def _core57_apply(psi, ops, pos):
    """Item 57b: `_apply_ops` in psf_zero_core57; None where the core cannot take the call."""
    if any(len(q) > 6 for _, q in ops) or psi.ndim > RESYNTH_MAX_QUBITS:
        return None
    nan = float("nan")
    rows = [(mat, [pos[i] for i in q], nan, nan, False) for mat, q in ops]
    try:
        out = _CORE57.apply_ops57(_core57_buffer(2, rows, psi.ndim, [nan] * psi.ndim, [nan] * psi.ndim, psi))
    except ValueError:
        CORE57_STATS["fallback"] += 1
        return None
    CORE57_STATS["rust"] += 1
    return np.frombuffer(out, dtype="<c16").reshape(psi.shape)


def excitation_cost(circ, target):
    """Item 35's estimate for "select": sum over gates of -log(1 - reported error), plus duration / T1 x P(1) on each
    of the gate's qubits, P(1) from the noiseless state (from |0...0>) just before the gate. None if more than
    RESYNTH_MAX_QUBITS qubits are touched or an instruction has no matrix (other than barrier, measure, delay).
    The state is a plain numpy tensor (axis j = j-th touched qubit); gate matrices are Qiskit's (little-endian).
    Item 49: P(1) is read from each qubit's 2x2 reduced state, which follows single-qubit gates without the state."""
    import math
    ops = [(ins.operation, tuple(circ.find_bit(b).index for b in ins.qubits)) for ins in circ.data
           if ins.operation.name not in ("barrier", "measure", "delay")]
    active = sorted({i for _, q in ops for i in q})
    if len(active) > RESYNTH_MAX_QUBITS:
        return None
    k = max(len(active), 1)
    pos = {p: j for j, p in enumerate(active)}
    qp = getattr(target, "qubit_properties", None) or []
    if _CORE57 is not None:  # item 57b
        r = _core57_estimate(0, ops, active, pos, qp, target)
        if r is not _NOT_DONE:
            return r
    CORE57_STATS["python"] += 1
    psi = np.zeros((2,) * k, dtype=complex)
    psi[(0,) * k] = 1.0
    cost = 0.0
    rho = {}   # item 49: qubit -> its 2x2 reduced state, single-qubit gates in `pend` included (absent: |0><0|)
    pend = {}  # item 49: qubit -> product of its single-qubit gates not yet applied to the state
    for op, q in ops:
        props = target[op.name].get(q, None) if op.name in target.operation_names else None
        try:
            mat = _gate_matrix(op)
        except Exception:
            return None
        if props is not None and props.error is not None:
            cost += -math.log(max(1.0 - props.error, 1e-300))
        dur = props.duration if props is not None and props.duration else 0.0
        if dur:
            for i in q:
                t1 = getattr(qp[i], "t1", None) if i < len(qp) and qp[i] is not None else None
                if t1:
                    cost += dur / t1 * _p1(rho, i)
        if len(q) == 1:
            i = q[0]
            p = pend.get(i)
            pend[i] = mat if p is None else mat @ p
            r = rho.get(i, _RHO0)
            rho[i] = mat @ r @ mat.conj().T
            continue
        pre = {x: pend.pop(x) for x in q if x in pend}
        if pre:
            mat = mat @ _embed_1q(pre, q)
        axes = [pos[i] for i in q]
        m = len(axes)
        rev = axes[::-1]  # Qiskit's matrix index is little-endian: the first qarg is the least significant bit
        psi = np.tensordot(mat.reshape((2,) * (2 * m)), psi, axes=(list(range(m, 2 * m)), rev))
        psi = np.moveaxis(psi, list(range(m)), rev)
        for i, ax in zip(q, axes):
            rho[i] = _rho1(psi, ax)
    return cost


EXACT_TOL = 1e-6  # item 39: largest state infidelity accepted (B17's failure line; exact paths stay below 1e-7)
EXACT_SEEDS = (39_001, 39_002)
EXACT_STATS = {"checked": 0, "refused_resynthesis": 0, "refused_floor": 0, "refused_level3": 0, "not_checkable": 0}
EXACT_SEEN = []  # item 59a: (where, largest state infidelity over the seeds) per check that was made
EXACT_SEEN_MAX = 10_000
_EXACT_SKIP = ("barrier", "measure", "delay")


def _product_state(rng):
    """A random single-qubit pure state (Haar), as a length-2 complex vector."""
    v = rng.normal(size=2) + 1j * rng.normal(size=2)
    return v / np.linalg.norm(v)


def _apply_ops(psi, ops, pos):
    """Applies [(matrix, physical qubits)] to the tensor `psi` (axis pos[q] = qubit q); Qiskit's little-endian
    matrices, as in excitation_cost. Item 57b: in psf_zero_core57 when it is installed and can take the call."""
    if _CORE57 is not None and ops:
        out = _core57_apply(psi, ops, pos)
        if out is not None:
            return out
    CORE57_STATS["python"] += 1
    for mat, q in ops:
        axes = [pos[i] for i in q]
        m = len(axes)
        rev = axes[::-1]
        psi = np.tensordot(mat.reshape((2,) * (2 * m)), psi, axes=(list(range(m, 2 * m)), rev))
        psi = np.moveaxis(psi, list(range(m)), rev)
    return psi


EXACT_MAX_GATE_QUBITS = 6  # item 44: wider instructions are expanded through their definition, never made a matrix
EXACT_MAX_OPS = 200_000  # item 44: a longer expansion makes the check one that cannot be made


def _touched_qubits(circ):
    """Item 44: indices of the qubits acted on by instructions other than barrier, measure and delay."""
    return {circ.find_bit(b).index for ins in circ.data if ins.operation.name not in _EXACT_SKIP for b in ins.qubits}


FEASIBILITY_STATS = {"candidate": 0, "level3": 0, "input": 0, "resynthesis": 0}  # item 56: what was skipped


def _narrow(ins):
    """Item 56: True for an instruction `_ops_of` turns into one matrix (or gives up on)."""
    return ins.operation.name not in _EXACT_SKIP and ins.operation.num_qubits <= EXACT_MAX_GATE_QUBITS


def _n_narrow(circ):
    """Item 56: the instructions `_ops_of` turns into one matrix each. Each adds one entry to its list or makes it
    give up, and it gives up rather than hold more than EXACT_MAX_OPS: so if this exceeds EXACT_MAX_OPS,
    `_ops_of(circ)` is None."""
    return sum(1 for ins in circ.data if _narrow(ins))


def _narrow_touched(circ):
    """Item 56: the qubits of the instructions `_n_narrow` counts: all among the qubits of `_ops_of(circ)`'s list
    when it is not None (`_fuse_ops` keeps every qubit)."""
    return {circ.find_bit(b).index for ins in circ.data if _narrow(ins) for b in ins.qubits}


def _cannot_implement(qc, out):
    """Item 56: True only if `_implements(qc, out)` would return False, decided from counts, without a matrix. False
    whenever that is not certain, including when a count raises."""
    try:
        n = qc.num_qubits
        if n > RESYNTH_MAX_QUBITS or _n_narrow(qc) > EXACT_MAX_OPS or _n_narrow(out) > EXACT_MAX_OPS:
            return True
        lay = getattr(out, "layout", None)
        if lay is not None:
            init = set(lay.initial_index_layout(filter_ancillas=True)[:n])
            fin = set(lay.final_index_layout(filter_ancillas=True)[:n])
        else:
            init = fin = set(range(n))
        return len(_narrow_touched(out) | init | fin) > RESYNTH_MAX_QUBITS
    except Exception:  # noqa: BLE001 - not certain, so not skipped
        return False


def _ops_of(circ):
    """[(matrix, qubit indices)] for every instruction except barrier, measure and delay; None if one has no
    matrix. Item 44: an instruction on more than EXACT_MAX_GATE_QUBITS qubits is expanded through its `definition`
    (recursively) instead of being made a matrix; None if it has none or the expansion exceeds EXACT_MAX_OPS.
    Global phases of definitions are dropped: every check ignores the global phase."""
    out = []

    def add(op, qubits, depth):
        if op.name in _EXACT_SKIP:
            return True
        if op.num_qubits > EXACT_MAX_GATE_QUBITS:
            try:
                d = op.definition
            except Exception:  # noqa: BLE001 - a definition that cannot be built means "not checkable"
                return False
            if d is None or depth >= 32:
                return False
            return all(add(sub.operation, tuple(qubits[d.find_bit(b).index] for b in sub.qubits), depth + 1)
                       for sub in d.data)
        if len(out) >= EXACT_MAX_OPS:
            return False
        try:
            mat = _gate_matrix(op)
        except Exception:
            return False
        out.append((mat, qubits))
        return True

    for ins in circ.data:
        if not add(ins.operation, tuple(circ.find_bit(b).index for b in ins.qubits), 0):
            return None
    return _fuse_ops(out)  # item 46


_EYE2 = np.eye(2, dtype=complex)
_EYE2.flags.writeable = False


def _embed_1q(mats, qubits):
    """Item 46: the matrix on `qubits` (Qiskit's little-endian order, qubits[0] least significant) that applies
    mats[q] to each qubit q in `mats` and the identity elsewhere.

    Item 53: on one qubit, the matrix itself; on two, the Kronecker product as one broadcast multiplication (the same
    products np.kron forms, without its overhead). Wider: as before."""
    if len(qubits) == 1:
        return mats.get(qubits[0], _EYE2)
    if len(qubits) == 2:
        a, b = mats.get(qubits[1], _EYE2), mats.get(qubits[0], _EYE2)
        return (a[:, None, :, None] * b[None, :, None, :]).reshape(4, 4)
    m = np.ones((1, 1), dtype=complex)
    for q in reversed(qubits):
        m = np.kron(m, mats.get(q, _EYE2))
    return m


_GATE_MATRICES = OrderedDict()  # item 53: (name, parameters) -> read-only matrix of a standard gate
_GATE_MATRICES_MAX = 4096
_STANDARD_GATES = None


def _gate_matrix(op):
    """Item 53: np.asarray(op.to_matrix(), dtype=complex). For a standard gate (its class is Qiskit's standard gate
    of that name) with numeric parameters, the matrix is kept for the last _GATE_MATRICES_MAX (name, parameters) and
    returned read-only: the same values every time, without rebuilding them. Anything else is built as before, and
    raises as before."""
    global _STANDARD_GATES
    if _STANDARD_GATES is None:
        from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
        _STANDARD_GATES = {name: getattr(g, "base_class", type(g)) for name, g in get_standard_gate_name_mapping().items()}
    cls = _STANDARD_GATES.get(op.name)
    key = None
    if cls is not None and getattr(op, "base_class", type(op)) is cls:
        try:
            key = (op.name, tuple(float(p) for p in op.params))
        except (TypeError, ValueError):
            key = None
    if key is None:
        return np.asarray(op.to_matrix(), dtype=complex)
    mat = _GATE_MATRICES.get(key)
    if mat is None:
        mat = np.asarray(op.to_matrix(), dtype=complex).copy()
        mat.flags.writeable = False
        _GATE_MATRICES[key] = mat
        while len(_GATE_MATRICES) > _GATE_MATRICES_MAX:
            _GATE_MATRICES.popitem(last=False)
    else:
        _GATE_MATRICES.move_to_end(key)
    return mat


def _fuse_ops(ops):
    """Item 46: [(matrix, qubits)] with the same action as `ops`: each single-qubit gate multiplied into the next gate
    on its qubit that acts on more qubits, consecutive gates on the same qubits multiplied into one, and single-qubit
    gates with no later gate on their qubit kept (in the order of their qubits' first appearance)."""
    out, pend = [], {}
    for mat, q in ops:
        if len(q) == 1:
            p = pend.get(q[0])
            pend[q[0]] = mat if p is None else mat @ p
            continue
        pre = {x: pend.pop(x) for x in q if x in pend}
        if pre:
            mat = mat @ _embed_1q(pre, q)
        if out and out[-1][1] == q:
            out[-1] = (mat @ out[-1][0], q)
        else:
            out.append((mat, q))
    out.extend((u, (x,)) for x, u in pend.items())
    return out


ESTIMATE_TIE_TOL = 1e-12  # item 46: estimates within this relative difference are equal


def _lower(b, a):
    """Item 46: True if estimate `b` is lower than `a` by more than ESTIMATE_TIE_TOL (relative). Two estimates closer
    than that are a tie, and a tie keeps the earlier candidate, as an exact tie always did."""
    return b < a - ESTIMATE_TIE_TOL * max(abs(a), abs(b))


_RHO0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)  # item 49: |0><0|


def _rho1(psi, ax):
    """Item 49: the 2x2 reduced density matrix of the qubit on axis `ax` of the state tensor `psi`."""
    m = np.moveaxis(psi, ax, 0).reshape(2, -1)
    a, b = m[0], m[1]
    c = np.vdot(b, a)  # <0|rho|1> = sum a conj(b)
    return np.array([[np.vdot(a, a).real, c], [np.conj(c), np.vdot(b, b).real]], dtype=complex)


def _p1(rho, i):
    """Item 49: P(1) of qubit `i` from its kept 2x2 reduced state (|0><0| if none is kept)."""
    r = rho.get(i)
    return 0.0 if r is None else float(r[1, 1].real)


def _populations(psi, ax):
    """Item 46: (P(0), P(1)) of the qubit on axis `ax` of the state tensor `psi`: the diagonal of its reduced density
    matrix, which is all `hybrid_cost` reads of it."""
    m = np.moveaxis(psi, ax, 0).reshape(2, -1)
    return float(np.vdot(m[0], m[0]).real), float(np.vdot(m[1], m[1]).real)


def _free_diagonal(mat, props, q):
    """Item 46: True for a single-qubit gate with a diagonal matrix, no reported error and no duration (it adds
    nothing to `excitation_cost` or `hybrid_cost`, and changes no population they read)."""
    if len(q) != 1 or mat[0, 1] != 0 or mat[1, 0] != 0:
        return False
    return props is None or (not props.error and not props.duration)


def _seen(where, value):
    """Item 59a: records a check's largest infidelity."""
    if len(EXACT_SEEN) < EXACT_SEEN_MAX:
        EXACT_SEEN.append((where, float(value)))


def _same_action(ref, new, tol=EXACT_TOL):
    """Item 39: True if `new` and `ref` (same physical qubits and layout) map two seeded random product states of
    every touched qubit to the same state, up to global phase. False otherwise or if the check cannot be made."""
    EXACT_STATS["checked"] += 1
    if len(_touched_qubits(ref) | _touched_qubits(new)) > RESYNTH_MAX_QUBITS:  # item 44: before any matrix
        EXACT_STATS["not_checkable"] += 1
        return False
    a, b = _ops_of(ref), _ops_of(new)
    if a is None or b is None:
        EXACT_STATS["not_checkable"] += 1
        return False
    touched = sorted({i for _, q in a + b for i in q})
    if len(touched) > RESYNTH_MAX_QUBITS:
        EXACT_STATS["not_checkable"] += 1
        return False
    if not touched:
        return True
    pos = {p: j for j, p in enumerate(touched)}
    worst = 0.0
    for seed in EXACT_SEEDS:
        rng = np.random.default_rng(seed)
        psi = np.array(1.0 + 0j)
        for _ in touched:
            psi = np.multiply.outer(psi, _product_state(rng))
        x, y = _apply_ops(psi, a, pos), _apply_ops(psi, b, pos)
        inf = 1.0 - abs(np.vdot(x.ravel(), y.ravel())) ** 2
        worst = max(worst, inf)
        if inf > tol:
            _seen("same_action", worst)
            return False
    _seen("same_action", worst)
    return True


def _implements(qc, out, tol=EXACT_TOL):
    """Item 39: True if the compiled circuit `out` implements the logical circuit `qc`: two seeded random product
    states of qc's qubits, placed at out's initial layout (other touched qubits in |0>), are mapped to qc's output
    states at out's final layout, with the other touched qubits back in |0> (state infidelity <= tol, global phase
    ignored). False otherwise or if the check cannot be made."""
    EXACT_STATS["checked"] += 1
    n = qc.num_qubits
    if n > RESYNTH_MAX_QUBITS:  # item 44: before any matrix
        EXACT_STATS["not_checkable"] += 1
        return False
    a, b = _ops_of(qc), _ops_of(out)
    if a is None or b is None:
        EXACT_STATS["not_checkable"] += 1
        return False
    lay = getattr(out, "layout", None)
    if lay is not None:
        init = list(lay.initial_index_layout(filter_ancillas=True)[:n])
        fin = list(lay.final_index_layout(filter_ancillas=True)[:n])
    else:
        init = fin = list(range(n))
    touched = sorted({i for _, q in b for i in q} | set(init) | set(fin))
    if len(touched) > RESYNTH_MAX_QUBITS or n > RESYNTH_MAX_QUBITS:
        EXACT_STATS["not_checkable"] += 1
        return False
    pos = {p: j for j, p in enumerate(touched)}
    zero = np.array([1.0 + 0j, 0.0])
    rest = [pos[p] for p in touched if p not in set(fin)]
    worst = 0.0
    for seed in EXACT_SEEDS:
        rng = np.random.default_rng(seed)
        states = [_product_state(rng) for _ in range(n)]
        ideal = np.array(1.0 + 0j)
        for s in states:
            ideal = np.multiply.outer(ideal, s)
        ideal = _apply_ops(ideal, a, {v: v for v in range(n)})
        at = {p: v for v, p in enumerate(init)}
        psi = np.array(1.0 + 0j)
        for p in touched:
            psi = np.multiply.outer(psi, states[at[p]] if p in at else zero)
        psi = _apply_ops(psi, b, pos)
        idx = tuple(0 if j in rest else slice(None) for j in range(len(touched)))
        red = psi[idx]                                   # axes: final positions, in touched order
        kept = [p for p in touched if p in set(fin)]
        red = np.moveaxis(red, [kept.index(fin[v]) for v in range(n)], list(range(n)))
        inf = 1.0 - abs(np.vdot(ideal.ravel(), red.ravel())) ** 2
        worst = max(worst, inf)
        if inf > tol:
            _seen("implements", worst)
            return False
    _seen("implements", worst)
    return True


def _final_resynthesis(out, target, max_error):
    """Item 35: re-synthesise every two-qubit block of a finished circuit with Qiskit, on the target, exactly.
    Returns the original circuit if the result has an instruction the target does not provide or a two-qubit
    gate in a direction the target reports failed, or if it fails item 39's check."""
    new = _resynthesis_candidate(out, target, max_error)
    if new is out:
        return out
    if not _same_action(out, new):  # item 39
        EXACT_STATS["refused_resynthesis"] += 1
        RESYNTH_STATS["kept_original"] += 1
        return out
    RESYNTH_STATS["applied"] += 1
    return new


def _resynthesis_candidate(out, target, max_error):
    """Item 47: item 35's re-synthesised circuit, checked against the target but not yet by item 39; the original
    circuit (counted in RESYNTH_STATS["kept_original"]) if the target check fails."""
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
    return new


COMPARE_STATS = {"psf": 0, "level3": 0, "level3_refused": 0, "not_estimable": 0, "floor": 0, "floor_refused": 0,
                 "not_checked": 0}


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
            mat = _gate_matrix(op)
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


def readout_cost(circ, target) -> float:
    """Item 40: sum of the Target's measure error over the qubits the circuit measures (each qubit once)."""
    if "measure" not in target.operation_names:
        return 0.0
    seen, cost = set(), 0.0
    for ins in circ.data:
        if ins.operation.name != "measure":
            continue
        q = circ.find_bit(ins.qubits[0]).index
        if q in seen:
            continue
        seen.add(q)
        props = target["measure"].get((q,), None)
        if props is not None and props.error is not None:
            cost += float(props.error)
    return cost


def hybrid_cost(circ, target):
    """Item 38's estimate: amplitude damping as `excitation_cost` counts it, pure dephasing as the Z part of
    `pauli_cost`, and the reported error above the thermal floor (see the changelog). None if more than
    RESYNTH_MAX_QUBITS qubits are touched or an instruction has no matrix (other than barrier, measure, delay).
    Item 49: populations are read from each qubit's 2x2 reduced state, which follows single-qubit gates without the
    state."""
    import math
    ops = [(ins.operation, tuple(circ.find_bit(b).index for b in ins.qubits)) for ins in circ.data
           if ins.operation.name not in ("barrier", "measure", "delay")]
    active = sorted({i for _, q in ops for i in q})
    if len(active) > RESYNTH_MAX_QUBITS:
        return None
    k = max(len(active), 1)
    pos = {p: j for j, p in enumerate(active)}
    qp = getattr(target, "qubit_properties", None) or []
    if _CORE57 is not None:  # item 57b
        r = _core57_estimate(1, ops, active, pos, qp, target)
        if r is not _NOT_DONE:
            return None if r is None else r + readout_cost(circ, target)  # item 40, as below
    CORE57_STATS["python"] += 1
    psi = np.zeros((2,) * k, dtype=complex)
    psi[(0,) * k] = 1.0

    def thermal(i, t):
        p = qp[i] if i < len(qp) else None
        t1 = getattr(p, "t1", None) if p is not None else None
        if not t or not t1:
            return None
        t2 = getattr(p, "t2", None)
        return t1, (min(t2, 2 * t1) if t2 else 2 * t1)

    cost = 0.0
    rho = {}   # item 49: qubit -> its 2x2 reduced state, single-qubit gates in `pend` included (absent: |0><0|)
    pend = {}  # item 49: qubit -> product of its single-qubit gates not yet applied to the state
    for op, q in ops:
        try:
            mat = _gate_matrix(op)
        except Exception:
            return None
        props = target[op.name].get(q, None) if op.name in target.operation_names else None
        m = len(q)
        t = (props.duration or 0.0) if props is not None else 0.0
        if props is not None:
            for i in q:
                th = thermal(i, t)
                if th:
                    cost += t / th[0] * _p1(rho, i)
        if m == 1:
            i = q[0]
            p = pend.get(i)
            pend[i] = mat if p is None else mat @ p
            rho[i] = mat @ rho.get(i, _RHO0) @ mat.conj().T
        else:
            pre = {x: pend.pop(x) for x in q if x in pend}
            if pre:
                mat = mat @ _embed_1q(pre, q)
            axes = [pos[i] for i in q]
            rev = axes[::-1]  # Qiskit's matrix index is little-endian: the first qarg is the least significant bit
            psi = np.tensordot(mat.reshape((2,) * (2 * m)), psi, axes=(list(range(m, 2 * m)), rev))
            psi = np.moveaxis(psi, list(range(m)), rev)
            for i, ax in zip(q, axes):
                rho[i] = _rho1(psi, ax)
        if props is None:
            continue
        thermal_f = 1.0
        for i in q:
            th = thermal(i, t)
            if not th:
                continue
            t1, t2 = th
            rate = max(1.0 / t2 - 1.0 / (2.0 * t1), 0.0)
            ez = float(rho[i][0, 0].real - rho[i][1, 1].real) if i in rho else 1.0  # <Z> after the gate
            cost += (1.0 - math.exp(-t * rate)) / 2.0 * (1.0 - ez * ez)
            thermal_f *= (1.0 + 2.0 * math.exp(-t / t2) + math.exp(-t / t1)) / 4.0
        d = 2 ** m
        cost += max((props.error or 0.0) - (1.0 - (d * thermal_f + 1.0) / (d + 1.0)), 0.0) * (d + 1) / d
    return cost + readout_cost(circ, target)  # item 40


def kraus_cost(circ, target):
    """Item 42's estimate (see the changelog; Addendum 339's `kraus_pur`). None if more than RESYNTH_MAX_QUBITS qubits
    are touched or an instruction has no matrix (other than barrier, measure, delay)."""
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

    def rho(axes):
        mm = np.moveaxis(psi, list(axes), list(range(len(axes)))).reshape(2 ** len(axes), -1)
        return mm @ mm.conj().T

    cost = 0.0
    for op, q in ops:
        try:
            mat = _gate_matrix(op)
        except Exception:
            return None
        props = target[op.name].get(q, None) if op.name in target.operation_names else None
        axes = [pos[i] for i in q]
        m = len(axes)
        t = (props.duration or 0.0) if props is not None else 0.0
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
            g = 1.0 - math.exp(-t / t1)
            rate = max(1.0 / t2 - 1.0 / (2.0 * t1), 0.0)
            pphi = (1.0 - math.exp(-t * rate)) / 2.0
            r = rho([ax])
            # Tr(rho K) for K0 = diag(1, sqrt(1 - g)), K1 = sqrt(g)|0><1|, and the same after Z (closed form)
            s = math.sqrt(1.0 - g)
            off = g * abs(r[1, 0]) ** 2
            f = (1.0 - pphi) * (abs(r[0, 0] + s * r[1, 1]) ** 2 + off) + pphi * (abs(r[0, 0] - s * r[1, 1]) ** 2 + off)
            cost += 1.0 - f
            thermal_f *= (1.0 + 2.0 * math.exp(-t / t2) + math.exp(-t / t1)) / 4.0
        d = 2 ** m
        excess = max((props.error or 0.0) - (1.0 - (d * thermal_f + 1.0) / (d + 1.0)), 0.0)
        purity = float(np.sum(np.abs(rho(axes)) ** 2))  # Tr(rho^2) for a Hermitian rho
        cost += excess * d / (d - 1) * (1.0 - purity / d)
    return cost


def _choose(cands, target, score):
    """Items 37-38: the (name, circuit) with the lowest estimate; the first (the release's circuit) on ties or when
    any estimate cannot be made. Item 47: `compile_for_hardware` now chooses with `_choose_lazy`, which returns what
    this function returns on the candidates that pass item 39's check; this function is kept unchanged."""
    f = {"pauli": pauli_cost, "hybrid": hybrid_cost, "kraus": kraus_cost}.get(score, excitation_cost)
    costs = [f(c, target) for _, c in cands]
    if any(c is None for c in costs):
        COMPARE_STATS["not_estimable"] += 1
        return cands[0][1]
    low = min(costs)
    best = next(j for j in range(len(cands)) if not _lower(low, costs[j]))  # item 46: the first within the tie band
    COMPARE_STATS[cands[best][0]] += 1
    return cands[best][1]


def _cost_fn(score):
    return {"pauli": pauli_cost, "hybrid": hybrid_cost, "kraus": kraus_cost}.get(score, excitation_cost)


def _choose_lazy(qc, out, others, target, score):
    """Item 47: `_choose` over the release's circuit `out` and those of `others` -- (name, circuit) pairs, each
    already acceptable on the target (item 36) -- that pass item 39's `_implements`, making that check only where its
    result can change the choice. Returns exactly what 2026-10-07.c19 returned:
    - if `out`'s estimate is None, `out` (c19 returned it whether or not another candidate passed);
    - a candidate whose estimate is None and which passes the check makes the choice `out` (c19's "not_estimable");
    - a candidate whose estimate is higher than `out`'s by more than ESTIMATE_TIE_TOL is never chosen and does not
      change the lowest estimate, so it is not checked;
    - otherwise the lowest among `out` and the candidates that pass, the first within the tie band (`_lower`).
    Item 56: a candidate whose check certainly cannot be made is dropped first; in the loop below it could only be
    skipped or refused, neither of which changes the result."""
    kept = [(name, c) for name, c in others if not _cannot_implement(qc, c)]
    FEASIBILITY_STATS["candidate"] += len(others) - len(kept)
    others = kept
    if not others:
        return out
    f = _cost_fn(score)
    c0 = f(out, target)
    if c0 is None:
        COMPARE_STATS["not_estimable"] += 1
        return out
    costs = [f(c, target) for _, c in others]
    valid, skipped = [], False  # valid: (name, circuit, cost) in the original order
    for (name, c), x in zip(others, costs):
        if x is not None and _lower(c0, x):
            COMPARE_STATS["not_checked"] += 1
            skipped = True
            continue
        if not _implements(qc, c):  # item 39
            EXACT_STATS["refused_" + name] += 1
            COMPARE_STATS[name + "_refused"] += 1
            continue
        if x is None:
            COMPARE_STATS["not_estimable"] += 1
            return out
        valid.append((name, c, x))
    if not valid:
        if skipped:  # the release's circuit won against a candidate that was not checked
            COMPARE_STATS["psf"] += 1
        return out
    cands = [("psf", out, c0)] + valid
    low = min(x for _, _, x in cands)
    best = next(j for j in range(len(cands)) if not _lower(low, cands[j][2]))  # item 46: the first within the band
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
    if _cannot_implement(qc, l3):  # item 56: every path below would return `out`
        FEASIBILITY_STATS["level3"] += 1
        COMPARE_STATS["level3_refused"] += 1
        return out
    a, b = excitation_cost(out, target), excitation_cost(l3, target)  # item 47: the estimates first
    if a is None or b is None:
        COMPARE_STATS["not_estimable"] += 1
        return out
    if not _lower(b, a):  # item 46: a tie keeps the release's circuit
        COMPARE_STATS["not_checked"] += 1
        COMPARE_STATS["psf"] += 1
        return out
    if not _implements(qc, l3):  # item 39, made only when level 3's circuit would be returned (item 47)
        EXACT_STATS["refused_level3"] += 1
        COMPARE_STATS["level3_refused"] += 1
        return out
    COMPARE_STATS["level3"] += 1
    return l3


SKIP_STATS = {"resynthesis": 0, "floor": 0, "level3": 0}  # item 45
UNROLL_STATS = {"unrolled": 0, "failed": 0}  # item 48
CANCEL_STATS = {"tried": 0, "cancelled_kept": 0, "original_kept": 0, "failed": 0}  # item 50


def _count_2q(qc):
    """Item 50: the number of instructions on two qubits (barriers aside)."""
    return sum(1 for ins in qc.data if len(ins.qubits) == 2 and ins.operation.name != "barrier")


def _cancel_candidate(qc):
    """Item 50: `qc` after Qiskit's CommutativeCancellation if that removes at least one two-qubit gate, else None."""
    from qiskit.transpiler.passes import CommutativeCancellation
    n = _count_2q(qc)
    if n == 0:
        return None
    try:
        out = PassManager([CommutativeCancellation()]).run(qc)
    except QiskitError:
        CANCEL_STATS["failed"] += 1
        return None
    return out if _count_2q(out) < n else None


def _has_wide(qc):
    """Item 48: True if `qc` has an instruction on more than two qubits other than a barrier."""
    return any(len(ins.qubits) > 2 and ins.operation.name != "barrier" for ins in qc.data)


def _unroll_wide(qc, basis_gates):
    """Item 48: `qc` with every instruction on more than two qubits (barriers aside, and gates named in
    `basis_gates`) expanded through its definition by Qiskit's Unroll3qOrMore; `qc` itself, unchanged, if it has
    none or if one cannot be expanded (with a warning)."""
    if not _has_wide(qc):
        return qc
    try:
        out = PassManager([Unroll3qOrMore(basis_gates=list(basis_gates) if basis_gates else None)]).run(qc)
    except QiskitError as exc:
        UNROLL_STATS["failed"] += 1
        warnings.warn(f"an instruction on more than two qubits could not be expanded ({exc}); the input is compiled "
                      "as given (changelog item 48)")
        return qc
    UNROLL_STATS["unrolled"] += 1
    return out


def _checkable_logical(qc):
    """Item 45: True if item 39's `_implements` can be made for `qc` (at most RESYNTH_MAX_QUBITS logical qubits).
    Item 56: and if `_ops_of(qc)` does not certainly give up."""
    if qc.num_qubits > RESYNTH_MAX_QUBITS:
        return False
    if _n_narrow(qc) > EXACT_MAX_OPS:
        FEASIBILITY_STATS["input"] += 1
        return False
    return True


def _resynthesise(circ, mode, target, max_error):
    """Item 35 by `mode` ("select", or any other true value for always); unchanged circuit if `mode` is false.
    Item 45: not built when the circuit touches more than RESYNTH_MAX_QUBITS qubits, where item 39's
    `_same_action` cannot be made and would refuse it."""
    if not mode:
        return circ
    if len(_touched_qubits(circ)) > RESYNTH_MAX_QUBITS:
        SKIP_STATS["resynthesis"] += 1
        return circ
    if _n_narrow(circ) > EXACT_MAX_OPS:  # item 56: _same_action(circ, ...) would return False
        FEASIBILITY_STATS["resynthesis"] += 1
        return circ
    if mode == "select":
        return _select_resynthesis(circ, target, max_error)
    return _final_resynthesis(circ, target, max_error)


def _select_resynthesis(out, target, max_error):
    """Item 35, "select": the re-synthesised circuit if its excitation_cost is lower and it passes item 39's check,
    else the original. Item 47: the check is made only when the estimate would select the re-synthesised circuit."""
    new = _resynthesis_candidate(out, target, max_error)
    if new is out:
        return out
    a, b = excitation_cost(out, target), excitation_cost(new, target)
    if a is None or b is None:
        RESYNTH_STATS["not_estimable"] += 1
        return out
    if not _lower(b, a):  # item 46: a tie keeps the original
        RESYNTH_STATS["not_checked"] += 1
        RESYNTH_STATS["selected_original"] += 1
        return out
    if not _same_action(out, new):  # item 39
        EXACT_STATS["refused_resynthesis"] += 1
        RESYNTH_STATS["kept_original"] += 1
        return out
    RESYNTH_STATS["applied"] += 1
    RESYNTH_STATS["selected_resynthesised"] += 1
    return new


def _refine_found(property_set):
    return property_set["VF2PostLayout_stop_reason"] is VF2PostLayoutStopReason.SOLUTION_FOUND


# ---------------------------------------------------------------- item 68: STEINER6 (Addendum 444)
import heapq  # noqa: E402
import math  # noqa: E402
from collections import deque  # noqa: E402

_ST_FAILED = 0.5


def _st_working_graph(target):
    """(adjacency of the working couplers, failed qubits): couplers and qubits with error >= _ST_FAILED, and qubits whose
    measurement fails, are left out."""
    g2 = next(g for g in ("cz", "ecr", "cx") if g in target.operation_names)
    bad_q = set()
    for name in ("sx", "measure"):
        if name in target.operation_names:
            for qa, p in target[name].items():
                if qa is not None and p is not None and p.error is not None and p.error >= _ST_FAILED:
                    bad_q.add(qa[0])
    adj = {q: set() for q in range(target.num_qubits) if q not in bad_q}
    for qa, p in target[g2].items():
        if qa is None or (p is not None and p.error is not None and p.error >= _ST_FAILED):
            continue
        a, b = qa
        if a in adj and b in adj:
            adj[a].add(b)
            adj[b].add(a)
    return adj, bad_q


def _st_bfs_all(adj):
    """Hop distances and BFS parents from every working qubit."""
    dist, par = {}, {}
    for s in adj:
        d, p, dq = {s: 0}, {s: None}, deque([s])
        while dq:
            u = dq.popleft()
            for w in sorted(adj[u]):
                if w not in d:
                    d[w], p[w] = d[u] + 1, u
                    dq.append(w)
        dist[s], par[s] = d, p
    return dist, par


def _st_steiner_tree(terminals, dist, par):
    """An approximate Steiner tree (edges as a parent map rooted at terminals[0]) spanning the terminals: Prim's tree on
    their hop distances, each edge expanded into a shortest path, then a BFS tree of the union with non-terminal
    leaves pruned."""
    ts = list(dict.fromkeys(terminals))
    if len(ts) == 1:
        return ts[0], {}
    inside, nodes, edges = {ts[0]}, {ts[0]}, set()
    while len(inside) < len(ts):
        best = min(((dist[u][v], u, v) for u in inside for v in ts if v not in inside), key=lambda x: x[:3])
        _, u, v = best
        x = v
        while x != u:  # walk v -> u along u's BFS tree
            y = par[u][x]
            edges.add((min(x, y), max(x, y)))
            nodes.update((x, y))
            x = y
        inside.add(v)
    adj = {n: set() for n in nodes}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    root, parent, dq = ts[0], {ts[0]: None}, deque([ts[0]])
    while dq:
        u = dq.popleft()
        for w in sorted(adj[u]):
            if w not in parent:
                parent[w] = u
                dq.append(w)
    term = set(ts)
    changed = True
    while changed:  # prune non-terminal leaves
        changed = False
        kids = {}
        for c, p in parent.items():
            if p is not None:
                kids.setdefault(p, []).append(c)
        for c in list(parent):
            if c not in term and not kids.get(c) and parent[c] is not None:
                del parent[c]
                changed = True
    return root, {c: p for c, p in parent.items() if p is not None}


def _st_tree_cost(root, parent, terminals):
    steiner = sum(1 for c in parent if c not in terminals)
    return 2 * (len(parent) + steiner)


def _st_placement(n, terms, adj, dist, start):
    """Greedy: logical qubits in order of their co-occurrence with those already placed, each on the free working
    qubit closest (by co-occurrence-weighted hop distance) to its placed partners."""
    w = {}
    for lb, _ in terms:
        sup = [j for j, ch in enumerate(reversed(lb)) if ch != "I"]
        for i in range(len(sup)):
            for k in range(i + 1, len(sup)):
                a, b = sup[i], sup[k]
                w[(a, b)] = w.get((a, b), 0) + 1
                w[(b, a)] = w.get((b, a), 0) + 1
    deg = [sum(w.get((i, j), 0) for j in range(n)) for i in range(n)]
    order = [max(range(n), key=lambda i: (deg[i], -i))]
    phi, free = {order[0]: start}, set(adj) - {start}
    while len(phi) < n:
        cand = [i for i in range(n) if i not in phi]
        nxt = max(cand, key=lambda i: (sum(w.get((i, j), 0) for j in phi), deg[i], -i))
        best = min(free, key=lambda v: (sum(w.get((nxt, j), 0) * dist[phi[j]].get(v, 10 ** 6) for j in phi),
                                        -len(adj[v]), v))
        phi[nxt] = best
        free.discard(best)
    return [phi[i] for i in range(n)]


def _st_gadget_ops(label, coeff, t, phys, dist, par):
    """The gates of exp(-i t c P) on the device: [(name, qubits, params)]."""
    sup = [(j, ch) for j, ch in enumerate(reversed(label)) if ch != "I"]
    if not sup:
        return [("gphase", (), (-t * coeff,))]
    pre, post = [], []
    for j, ch in sup:
        q = phys[j]
        if ch == "X":
            pre.append(("h", (q,), ()))
            post.append(("h", (q,), ()))
        elif ch == "Y":
            pre += [("sdg", (q,), ()), ("h", (q,), ())]
            post += [("h", (q,), ()), ("s", (q,), ())]
    terms_q = [phys[j] for j, _ in sup]
    root, parent = _st_steiner_tree(terms_q, dist, par)
    kids = {}
    for c, p in parent.items():
        kids.setdefault(p, []).append(c)
    term = set(terms_q)
    collect = []

    def visit(v):
        p = parent.get(v)
        if p is not None and v not in term:
            collect.append(("cx", (v, p), ()))  # a Steiner qubit's own bit is added twice to its parent: it cancels
        for c in sorted(kids.get(v, [])):
            visit(c)
        if p is not None:
            collect.append(("cx", (v, p), ()))
    for c in sorted(kids.get(root, [])):
        visit(c)
    return pre + collect + [("rz", (root,), (2 * t * coeff,))] + collect[::-1] + post


def _st_error_distances(adj, target):
    """All-pairs Dijkstra on the working couplers, a coupler costing -ln(1 - its two-qubit error) (the larger of the
    two directions); (dist, par) with STEINER3's interface (par[s][x] = x's parent in the tree from s)."""
    g2 = next(g for g in ("cz", "ecr", "cx") if g in target.operation_names)
    err = {}
    for qa, p in target[g2].items():
        if qa is None or p is None or p.error is None:
            continue
        k = (min(qa), max(qa))
        err[k] = max(err.get(k, 0.0), p.error)
    w = {k: -math.log(1.0 - min(e, 0.999)) + 1e-9 for k, e in err.items()}
    dist, par = {}, {}
    for s in adj:
        d, p, heap = {s: 0.0}, {s: None}, [(0.0, s)]
        while heap:
            du, u = heapq.heappop(heap)
            if du > d[u]:
                continue
            for v in sorted(adj[u]):
                nd = du + w.get((min(u, v), max(u, v)), 1.0)
                if nd < d.get(v, math.inf):
                    d[v], p[v] = nd, u
                    heapq.heappush(heap, (nd, v))
        dist[s], par[s] = d, p
    return dist, par, w


def _st_interaction(n, terms):
    w = {}
    for lb, _ in terms:
        sup = [j for j, ch in enumerate(reversed(lb)) if ch != "I"]
        for i in range(len(sup)):
            for k in range(i + 1, len(sup)):
                e = (sup[i], sup[k])
                w[e] = w.get(e, 0) + 1
    return w


def _st_embeddings(n, terms, adj, dist):
    """Up to _ST_KEEP placements from VF2 _st_embeddings of the heaviest embeddable part of the _st_interaction graph."""
    import rustworkx as rx
    nodes = sorted(adj)
    idx = {q: i for i, q in enumerate(nodes)}
    dev = rx.PyGraph()
    dev.add_nodes_from(nodes)
    dev.add_edges_from_no_data(sorted({(idx[a], idx[b]) for a in adj for b in adj[a] if a < b}))
    w = _st_interaction(n, terms)
    keep = []
    for e in sorted(w, key=lambda e: (-w[e], e)):
        pat = rx.PyGraph()
        pat.add_nodes_from(range(n))
        pat.add_edges_from_no_data(keep + [e])
        if max(pat.degree(i) for i in range(n)) > max(dev.degree(i) for i in range(len(nodes))):
            continue
        if rx.is_subgraph_isomorphic(dev, pat, id_order=False, induced=False, call_limit=_ST_VF2_CHECK):
            keep.append(e)
    pat = rx.PyGraph()
    pat.add_nodes_from(range(n))
    pat.add_edges_from_no_data(keep)
    scored = []
    try:
        it = rx.vf2_mapping(dev, pat, id_order=False, subgraph=True, induced=False, call_limit=_ST_VF2_CHECK * 10)
        for k, m in enumerate(it):
            if k >= _ST_VF2_MAPS:
                break
            phys = [None] * n
            for dnode, lnode in m.items():
                phys[lnode] = nodes[dnode]
            if any(p is None for p in phys):
                continue
            score = sum(c * dist[phys[a]].get(phys[b], 1e6) for (a, b), c in w.items())
            scored.append((score, phys))
    except Exception:  # noqa: BLE001 - no embedding found within the call limit: the greedy _st_placement remains
        pass
    scored.sort(key=lambda x: x[0])
    out, seen = [], set()
    for _, phys in scored:
        if tuple(phys) not in seen:
            seen.add(tuple(phys))
            out.append(phys)
        if len(out) >= _ST_KEEP:
            break
    return out, len(keep), len(w)


def _st_placement_cost(phys, terms, dist, par, edge_w):
    """Hop arm (edge_w None): STEINER3's tree cost summed. Error arm: the trees' couplers' costs x CNOTs on them."""
    cost = 0.0
    for lb, _ in terms:
        sup = [phys[j] for j, ch in enumerate(reversed(lb)) if ch != "I"]
        if len(sup) < 2:
            continue
        if any(dist[sup[0]].get(v) is None for v in sup):
            return math.inf
        r, p = _st_steiner_tree(sup, dist, par)
        if edge_w is None:
            cost += _st_tree_cost(r, p, set(sup))
        else:
            term = set(sup)
            cost += sum(edge_w.get((min(c, q), max(c, q)), 1.0) * (2 if c in term else 4) for c, q in p.items())
    return cost


DET_STATS = {"calls": 0, "applied": 0, "trials": 0, "skipped": 0}  # item 75
DET_TRIALS_CAP, DET_CALL_CAP = 25_000, 3_000_000  # item 78: per search order, whatever Qiskit's preset asks for
DET_FAILED_ERROR = 0.5  # item 78: at the routing stage, an element at or above this error is never placed on

# Item 78: at Qiskit's routing stage the circuit is not yet in the device's gates. A two-qubit gate is scored as this
# many native two-qubit gates on its coupler (2 if not listed: a general two-qubit gate needs 3, most controlled
# rotations 2); a single-qubit gate as this many sx gates on its qubit (2 if not listed; diagonal gates are free, as rz
# is virtual on these devices).
_NATIVE2_COUNT = {"swap": 3, "cx": 1, "cz": 1, "ecr": 1, "cy": 1, "ch": 1, "dcx": 2, "iswap": 2}
_SX_COUNT = {"rz": 0, "p": 0, "u1": 0, "z": 0, "s": 0, "sdg": 0, "t": 0, "tdg": 0, "id": 0,
             "x": 1, "sx": 1, "sxdg": 1, "h": 1, "y": 1, "u2": 1}


class _DeterministicPostLayout(AnalysisPass):
    """Item 75: the exact re-placement of a routed circuit by the device's errors, in a fixed order: subgraph matchings
    of its two-qubit interaction graph into the device's coupling graph from rustworkx's VF2 with `id_order=True`, each
    scored by the summed -ln(1 - error) of every operation as placed; qubits with no two-qubit gate then go, one by one
    in index order, to the free physical qubit where their own operations cost least. The lowest score is applied if it
    is below the current placement's (the first found on a tie). It sets `post_layout` and
    `VF2PostLayout_stop_reason` as Qiskit's VF2PostLayout does, so the passes after it are unchanged. Qiskit's pass
    returns different layouts from call to call with a fixed seed (Addendum 450); this one does not."""

    def __init__(self, target, call_limit=300_000, max_trials=2_500, strict_direction=True):
        super().__init__()
        self.target = target
        self.call_limit = call_limit
        self.max_trials = max_trials
        self.strict = strict_direction
        self.proxy = False  # item 78: set by _DeterministicVF2(target, routing_stage=True) for the non-strict call

    def run(self, dag):
        import rustworkx as rx
        from qiskit.transpiler import Layout
        DET_STATS["calls"] += 1
        self.property_set["VF2PostLayout_stop_reason"] = VF2PostLayoutStopReason.NO_SOLUTION_FOUND
        t = self.target
        N = t.num_qubits
        if len(dag.qubits) != N:
            DET_STATS["skipped"] += 1
            return
        idx = {q: i for i, q in enumerate(dag.qubits)}
        one, two = {}, {}
        for node in dag.op_nodes():
            name = node.op.name
            if name in ("barrier", "delay"):
                continue
            if getattr(node.op, "blocks", None) is not None or len(node.qargs) > 2 or node.op.num_clbits and \
                    name != "measure":
                DET_STATS["skipped"] += 1
                return
            qs = tuple(idx[q] for q in node.qargs)
            if len(qs) == 2:
                two[(name, qs)] = two.get((name, qs), 0) + 1
            elif len(qs) == 1:
                one[(name, qs[0])] = one.get((name, qs[0]), 0) + 1
        weights = {}
        native2 = next((g for g in ("cz", "ecr", "cx") if g in t.operation_names), None)
        proxy = {}  # item 78: names the device does not run, scored as what they translate to (non-strict only)
        for name in {n for n, _ in one} | {n for n, _ in two} | ({n for n in (native2, "sx") if n in t.operation_names} if self.proxy else set()):
            if name not in t.operation_names or self.proxy and name not in (native2, "sx", "measure"):
                arity = 2 if any(n == name for n, _ in two) else 1
                if not self.proxy or (arity == 2 and native2 is None) or (arity == 1 and "sx" not in t.operation_names):
                    DET_STATS["skipped"] += 1
                    return
                proxy[name] = (native2, _NATIVE2_COUNT.get(name, 2)) if arity == 2 else ("sx", _SX_COUNT.get(name, 2))
                continue
            table = {}
            for qargs, props in t[name].items():
                e = props.error if props is not None and props.error is not None else 0.0
                failed = e >= 1.0 or self.proxy and e >= DET_FAILED_ERROR  # item 78
                if PLACE_SCORE == "duration":  # item 90: the placed operation's duration, not its error
                    d = props.duration if props is not None and props.duration is not None else 0.0
                    table[None if qargs is None else tuple(qargs)] = math.inf if failed else d
                else:
                    table[None if qargs is None else tuple(qargs)] = math.inf if failed else -math.log1p(-max(e, 0.0))
            weights[name] = table

        def w(name, qargs):
            k = 1
            if name in proxy:
                name, k = proxy[name]
                if k == 0:
                    return 0.0
            table = weights[name]
            v = table.get(qargs)
            if v is None and not self.strict and len(qargs) == 2:
                v = table.get(qargs[::-1])
            if v is None:
                v = 0.0 if None in table else math.inf
            return k * v

        active = sorted({q for _, q in one} | {q for _, qs in two for q in qs})
        if not active:
            return
        cost1 = {}
        for (name, q), c in one.items():
            row = cost1.setdefault(q, [0.0] * N)
            for p in range(N):
                row[p] += c * w(name, (p,))
        if self.proxy:  # item 78: no qubit with any operation goes to a failed qubit (gates come with translation)
            bad_p = set()
            for name in [n for n in ("sx", "x", "measure") if n in t.operation_names]:
                for qargs, props in t[name].items():
                    if qargs is not None and props is not None and props.error is not None and \
                            props.error >= DET_FAILED_ERROR:
                        bad_p.add(qargs[0])
            for q in active:
                row = cost1.setdefault(q, [0.0] * N)
                for p in bad_p:
                    row[p] = math.inf
        zero = [0.0] * N
        conn = sorted({q for _, qs in two for q in qs})
        iso = [q for q in active if q not in set(conn)]
        pos = {q: i for i, q in enumerate(conn)}
        names2 = sorted({n for n, _ in two})

        def place_isolated(taken, total):
            m = {}
            for q in iso:
                row = cost1.get(q, zero)
                best_p, best_c = None, math.inf
                for p in range(N):
                    if p not in taken and row[p] < best_c - 1e-15:
                        best_p, best_c = p, row[p]
                if best_p is None:
                    return None, math.inf
                m[q] = best_p
                taken.add(best_p)
                total += best_c
            return m, total

        def score(mc):  # mc: logical connected qubit -> physical
            total = 0.0
            for q, p in mc.items():
                total += cost1.get(q, zero)[p]
            for (name, (a, b)), c in two.items():
                total += c * w(name, (mc[a], mc[b]))
                if total == math.inf:
                    return total
            return total

        current = sum(cost1.get(q, zero)[q] for q in active) + sum(c * w(n, qs) for (n, qs), c in two.items())
        best, best_map = current, None
        trials = 0
        if conn:
            dev_edges = None
            for name in names2:
                name = proxy.get(name, (name, 1))[0]  # item 78
                ok = {tuple(q) for q in t[name] if q is not None}
                if not self.strict:
                    ok |= {(b, a) for a, b in ok}
                dev_edges = ok if dev_edges is None else dev_edges & ok
            # item 75 (revised in c44): c43's index order, then the device's qubits best first and the circuit's
            # busiest first, then VF2's own order; each up to the trial limit
            own = {}
            for name in [n for n in ("sx", "measure") if n in t.operation_names]:
                for qargs, props in t[name].items():
                    if qargs is None or props is None or props.error is None:
                        continue
                    e = props.error
                    if PLACE_SCORE == "duration":  # item 90: search order by the qubit's own durations
                        v = (props.duration or 0.0) if e < 1.0 else 1e3
                    else:
                        v = -math.log1p(-e) if e < 1.0 else 1e3
                    own[qargs[0]] = own.get(qargs[0], 0.0) + v
            nbr = {p: [] for p in range(N)}
            for a, b in dev_edges:
                v = w(names2[0], (a, b))
                nbr[a].append(v if v < math.inf else 1e3)
                nbr[b].append(v if v < math.inf else 1e3)
            quality = sorted(range(N), key=lambda p: (own.get(p, 0.0) + (sum(nbr[p]) / len(nbr[p]) if nbr[p] else 1e3), p))
            load = {}
            for (_, (a, b)), c in two.items():
                load[a] = load.get(a, 0) + c
                load[b] = load.get(b, 0) + c
            busy = sorted(conn, key=lambda q: (-load.get(q, 0), q))
            cedges0 = {(a, b) for _, (a, b) in two}
            if not self.strict:
                cedges0 |= {(b, a) for a, b in cedges0}
            for phys_order, circ_order, id_order in ((list(range(N)), conn, True), (quality, busy, True),
                                                     (list(range(N)), conn, False)):
                at_d = {p: i for i, p in enumerate(phys_order)}
                at_c = {q: i for i, q in enumerate(circ_order)}
                dg, cg = rx.PyDiGraph(), rx.PyDiGraph()
                dg.add_nodes_from(phys_order)
                dg.add_edges_from_no_data(sorted((at_d[a], at_d[b]) for a, b in dev_edges))
                cg.add_nodes_from(circ_order)
                cg.add_edges_from_no_data(sorted((at_c[a], at_c[b]) for a, b in cedges0))
                n_here = 0
                for mapping in rx.vf2_mapping(dg, cg, subgraph=True, induced=False, id_order=id_order,
                                              call_limit=self.call_limit):
                    trials += 1
                    n_here += 1
                    mc = {circ_order[cn]: phys_order[dn] for dn, cn in mapping.items()}
                    sc = score(mc)
                    if sc < best:
                        mi, total = place_isolated(set(mc.values()), sc)
                        if total < best - 1e-12:
                            best, best_map = total, {**mc, **mi}
                    if n_here >= self.max_trials:
                        break
        else:
            mi, total = place_isolated(set(), 0.0)
            if total < best - 1e-12:
                best, best_map = total, mi
        DET_STATS["trials"] += trials
        if best_map is None:
            self.property_set["VF2PostLayout_stop_reason"] = VF2PostLayoutStopReason.NO_BETTER_SOLUTION_FOUND
            return
        used = set(best_map.values())
        free = iter([p for p in range(N) if p not in used])
        post = Layout()
        for i, q in enumerate(dag.qubits):
            post[q] = best_map[i] if i in best_map else next(free)
        self.property_set["post_layout"] = post
        self.property_set["VF2PostLayout_stop_reason"] = VF2PostLayoutStopReason.SOLUTION_FOUND
        DET_STATS["applied"] += 1


class _NoPostLayout(AnalysisPass):
    """Item 75: in place of Qiskit's VF2PostLayout when the call has no device Target: there are no errors to place
    by, and Qiskit's pass then chooses among equal layouts differently from call to call."""

    def run(self, dag):
        self.property_set["VF2PostLayout_stop_reason"] = VF2PostLayoutStopReason.NO_SOLUTION_FOUND


class _DeterministicVF2:
    """Item 75: while active, Qiskit's preset pass managers built in this block get item 75's re-placement (scored on
    `target`, the device's own Target) or, without a Target, none, in place of VF2PostLayout."""

    def __init__(self, target, routing_stage=False):
        self.target = target
        self.routing_stage = routing_stage  # item 78: score the routing stage's gates by their translated cost

    def __enter__(self):
        import qiskit.transpiler.preset_passmanagers.builtin_plugins as plugins
        import qiskit.transpiler.preset_passmanagers.common as common
        self.mods = (plugins, common)
        self.saved = [m.VF2PostLayout for m in self.mods]
        target = self.target
        routing_stage = self.routing_stage

        def make(*args, call_limit=300_000, max_trials=2_500, strict_direction=False, **kw):
            if target is None:
                return _NoPostLayout()
            if not routing_stage:
                return _DeterministicPostLayout(target, call_limit or 300_000, max_trials or 2_500, strict_direction)
            p = _DeterministicPostLayout(target, min(call_limit or 300_000, DET_CALL_CAP),  # item 78
                                         min(max_trials or 2_500, DET_TRIALS_CAP), strict_direction)
            p.proxy = not strict_direction
            return p
        for m in self.mods:
            m.VF2PostLayout = make
        return self

    def __exit__(self, *exc):
        for m, v in zip(self.mods, self.saved):
            m.VF2PostLayout = v
        return False


NEAR_STATS = {"checked": 0, "tried": 0, "kept_alternative": 0, "kept_first": 0, "no_alternative": 0}  # item 74

L3C_STATS = {"tried": 0, "kept_psf": 0, "kept_level3": 0, "failed_elements": 0, "not_exact": 0, "unverified": 0,
             "error": 0}  # item 80


def _level3_candidate(qc, out, target, max_error, seed_transpiler):
    """Item 80: `out`, or Qiskit's level-3 preset on the device's Target when its summed -ln(1 - error) on the device
    (item 69's measure) is lower. Qiskit's circuit is built under item 75's `_DeterministicVF2(target)`, so it is the
    same on every call; it is refused if it uses a failed element, or, where item 39's check can be made, if it does
    not implement the input. A tie keeps `out`."""
    L3C_STATS["tried"] += 1
    try:
        with _DeterministicVF2(target, routing_stage=not MAP_FREE_DEFAULTS):  # item 78 (off by item 90)
            pm = generate_preset_pass_manager(optimization_level=3, target=target, seed_transpiler=seed_transpiler)
        l3 = pm.run(qc)
    except (TranspilerError, QiskitError, ValueError):
        L3C_STATS["error"] += 1
        return out
    edges, qubits = _failed_elements(target, max_error)
    if _uses_failed(l3, edges, qubits):
        L3C_STATS["failed_elements"] += 1
        return out
    if not _better(l3, out, target):  # item 84
        L3C_STATS["kept_psf"] += 1
        return out
    if _checkable_logical(qc) and not any(i.operation.name == "measure" for i in qc.data):
        if not _implements(qc, l3):
            L3C_STATS["not_exact"] += 1
            return out
    else:
        L3C_STATS["unverified"] += 1
    L3C_STATS["kept_level3"] += 1
    return l3


def _near_used(circ, target, near, max_error):
    """Item 74: True if `circ` uses an element whose error is in [near, max_error): a two-qubit gate on such a coupler
    (either direction for symmetric gates), any gate on a qubit whose sx error is in that range, or a measurement of
    a qubit whose readout error is."""
    name = next((g for g in ("cz", "ecr", "cx") if g in target.operation_names), None)
    edges = set()
    if name is not None:
        for qargs, props in target[name].items():
            if qargs is not None and props is not None and props.error is not None and near <= props.error < max_error:
                edges.add(tuple(qargs))
    gate_q = {q for q, e in qubit_errors_from_target(target, "sx").items() if near <= e < max_error}
    read_q = {q for q, e in readout_errors_from_target(target).items() if near <= e < max_error}
    if not (edges or gate_q or read_q):
        return False
    pos = {q: i for i, q in enumerate(circ.qubits)}
    for inst in circ.data:
        idx = tuple(pos[q] for q in inst.qubits)
        if inst.operation.name == "measure":
            if idx[0] in read_q or idx[0] in gate_q:
                return True
            continue
        if inst.operation.name in ("barrier", "delay"):
            continue
        if any(i in gate_q for i in idx):
            return True
        if len(idx) == 2 and (idx in edges or idx[::-1] in edges):
            return True
    return False


def _near_alternative(out, call_args, target, near, max_error):
    """Item 74 (revised in prototype c43): `out`, or the input compiled with the near-failed elements `out` used
    avoided as well, when that output is better on the device (lower summed -ln(1 - error), item 69's measure). The
    second compile sees a copy of the Target in which those elements, and only those, are marked failed; the choice is
    made on the real Target. Two fixed constructions compared by the device's own error table, not a search."""
    NEAR_STATS["checked"] += 1
    if near is None or near >= max_error:
        return out
    used = _near_used_elements(out, target, near, max_error)
    if not used:
        return out
    NEAR_STATS["tried"] += 1
    import copy
    from qiskit.transpiler import InstructionProperties
    marked = copy.deepcopy(target)
    for name, qargs in used:
        old = marked[name][qargs]
        marked.update_instruction_properties(name, qargs, InstructionProperties(
            duration=getattr(old, "duration", None), error=max_error))
    try:
        alt = compile_for_hardware(**dict(call_args, backend=None, target=marked, prune_max_error=max_error,
                                          near_failed_error=None, steiner=False, on_failed_elements="raise",
                                          level3_candidate=False))  # item 80 runs once, on the real Target
    except (FailedElementsError, TranspilerError, ValueError):
        alt = None
    if alt is None:
        NEAR_STATS["no_alternative"] += 1
        return out
    if _better(alt, out, target):  # item 84
        NEAR_STATS["kept_alternative"] += 1
        return alt
    NEAR_STATS["kept_first"] += 1
    return out


def _near_used_elements(circ, target, near, max_error):
    """Item 74 (prototype c43): the near-failed elements (error in [near, max_error)) that `circ` uses, as
    [(instruction name, qargs)] of the Target: the coupler of each two-qubit gate on one (both directions), the sx
    (and x) of a qubit with any gate on it whose sx error is in range, the measurement of a measured qubit whose readout
    error is."""
    name2 = next((g for g in ("cz", "ecr", "cx") if g in target.operation_names), None)
    edges = set()
    if name2 is not None:
        for qargs, props in target[name2].items():
            if qargs is not None and props is not None and props.error is not None and near <= props.error < max_error:
                edges.add(tuple(qargs))
    gate_q = {q for q, e in qubit_errors_from_target(target, "sx").items() if near <= e < max_error}
    read_q = {q for q, e in readout_errors_from_target(target).items() if near <= e < max_error}
    if not (edges or gate_q or read_q):
        return []
    pos = {q: i for i, q in enumerate(circ.qubits)}
    out = set()
    for inst in circ.data:
        idx = tuple(pos[q] for q in inst.qubits)
        name = inst.operation.name
        if name in ("barrier", "delay"):
            continue
        if name == "measure":
            if idx[0] in read_q:
                out.add(("measure", idx))
        for q in idx:
            if q in gate_q:
                for g in ("sx", "x"):
                    if g in target.operation_names and (q,) in target[g]:
                        out.add((g, (q,)))
        if len(idx) == 2:
            for e in (idx, idx[::-1]):
                if e in edges:
                    out.add((name2, e))
                    if e[::-1] in target[name2]:
                        out.add((name2, e[::-1]))
    return sorted(out)


def _near_alternative_c42(out, call_args, target, near, max_error):  # item 74 as in c42, kept for the record
    """Item 74: `out`, or the input compiled with every element of error >= `near` avoided when that output is
    better on the device (lower summed -ln(1 - error), item 69's measure). Two fixed constructions compared by the
    device's own error table, not a search."""
    NEAR_STATS["checked"] += 1
    if near is None or near >= max_error or not _near_used(out, target, near, max_error):
        return out
    NEAR_STATS["tried"] += 1
    try:
        alt = compile_for_hardware(**dict(call_args, prune_max_error=near, near_failed_error=None, steiner=False,
                                          on_failed_elements="raise"))
    except (FailedElementsError, TranspilerError, ValueError):
        alt = None
    if alt is None:
        NEAR_STATS["no_alternative"] += 1
        return out
    if _st_cost(alt, target) < _st_cost(out, target):
        NEAR_STATS["kept_alternative"] += 1
        return alt
    NEAR_STATS["kept_first"] += 1
    return out


def _st_term_masks(label):
    """(x mask, z mask, support) of a label; character j from the right acts on logical qubit j."""
    x = z = 0
    sup = []
    for j, ch in enumerate(reversed(label)):
        if ch == "I":
            continue
        sup.append(j)
        if ch in "XY":
            x |= 1 << j
        if ch in "ZY":
            z |= 1 << j
    return x, z, tuple(sup)


def _st_commute(a, b):
    return (bin((a[0] & b[1]) ^ (a[1] & b[0])).count("1") & 1) == 0


class _StMover:
    def __init__(self, n, triples, adj, dist, par, pi0, w=64, f=12, e=24, err=None):
        """err: None (hop counts only) or (error distances, their parents, coupler weight -ln(1 - e) by sorted pair):
        bridged terms then take their trees on the error distances, and a SWAP's cost is scaled by its coupler's
        weight over the median coupler's."""
        self.n, self.adj, self.dist, self.par = n, adj, dist, par
        self.err = err
        self._memo = {}
        self._pmemo = {}
        if err is not None:
            ws = sorted(err[2].values())
            self.w_med = ws[len(ws) // 2] if ws else 1.0
        self.triples = triples
        self.masks = [_st_term_masks(lb) for lb, _, _ in triples]
        self.pi = list(pi0)
        self.inv = {p: i for i, p in enumerate(self.pi)}
        self.w, self.f, self.e = w, f, e
        self.ops = []  # (name, physical qubits, params)
        self.gphase = 0.0
        self.stats = dict(swaps=0, merged_swaps=0, bridged_terms=0, bridge_excess=0, terms=0)
        self.order = []
        self.last2q = {}  # physical qubit -> the pair of the last two-qubit gate on it (single-qubit gates join blocks)
        self.st_blocks, self._lastb = [], {}  # item 83: the emitted blocks; physical qubit -> its last block

    # costs ---------------------------------------------------------
    def cost(self, j, pi=None):
        pi = self.pi if pi is None else pi
        sup = self.masks[j][2]
        if len(sup) <= 1:
            return 0
        key = tuple(sorted(pi[q] for q in sup))
        c = self._memo.get(key)
        if c is None:
            r, p = _st_steiner_tree(list(key), self.dist, self.par)
            c = self._memo[key] = _st_tree_cost(r, p, set(key))
        return c

    def excess(self, j, pi=None):
        k = len(self.masks[j][2])
        return 0 if k <= 1 else self.cost(j, pi) - 2 * (k - 1)

    def proxy_excess(self, j, pi):
        """A cheap stand-in for the excess when scoring SWAPs: for two qubits, 2 (d - 1) x 2 (a path of d hops costs
        2 d + 2 (d - 1) CNOTs); for more, the same on the minimum spanning tree of their hop distances."""
        sup = self.masks[j][2]
        k = len(sup)
        if k <= 1:
            return 0
        phys = [pi[q] for q in sup]
        if k == 2:
            d = self.dist[phys[0]][phys[1]]
            return 4 * (d - 1)
        key = tuple(sorted(phys))
        c = self._pmemo.get(key)
        if c is not None:
            return c
        c = self._pmemo[key] = self._mst_excess(list(key))
        return c

    def _mst_excess(self, phys):
        inside = {phys[0]}
        rest = set(phys[1:])
        tot = 0
        best = {v: self.dist[phys[0]][v] for v in rest}
        while rest:
            v = min(rest, key=lambda u: (best[u], u))
            tot += best[v] - 1
            rest.discard(v)
            inside.add(v)
            for u in rest:
                du = self.dist[v][u]
                if du < best[u]:
                    best[u] = du
        return 4 * tot

    # emission ------------------------------------------------------
    def run_term(self, j):
        lb, c, t = self.triples[j]
        first_op = len(self.ops)  # item 83
        pauli = {self.pi[q]: ch for q, ch in enumerate(reversed(lb)) if ch != "I"}
        ex = self.excess(j)
        dist, par = (self.dist, self.par) if (self.err is None or ex == 0) else (self.err[0], self.err[1])
        for name, q, prm in _st_gadget_ops(lb, c, t, self.pi, dist, par):
            if name == "gphase":
                self.gphase += prm[0]
            else:
                self.ops.append((name, q, prm))
                if len(q) == 2:
                    pair = frozenset(q)
                    self.last2q[q[0]] = self.last2q[q[1]] = pair
        if ex > 0:
            self.stats["bridged_terms"] += 1
            self.stats["bridge_excess"] += ex
        self.stats["terms"] += 1
        self.order.append(j)
        _st_record_term(self, first_op, pauli)  # item 83

    def mergeable(self, p, q):
        """A SWAP on (p, q) joins the block of the last two-qubit gates there: Qiskit's level 3 fuses them into one
        two-qubit unitary of at most 3 CNOTs, so the SWAP adds at most 1 CNOT to a block that had 2 (a ZZ term)."""
        pair = frozenset((p, q))
        return self.last2q.get(p) == pair and self.last2q.get(q) == pair

    def swap(self, p, q):
        if self.mergeable(p, q):
            self.stats["merged_swaps"] += 1
        self.ops.append(("swap", (p, q), ()))
        _st_record_swap(self, len(self.ops) - 1, p, q)  # item 83
        self.last2q[p] = self.last2q[q] = frozenset((p, q))
        a, b = self.inv.get(p), self.inv.get(q)
        if a is not None:
            self.pi[a] = q
        if b is not None:
            self.pi[b] = p
        self.inv = {}
        for i, v in enumerate(self.pi):
            self.inv[v] = i
        self.stats["swaps"] += 1

    # scheduling ----------------------------------------------------
    def build(self):
        T = len(self.triples)
        unrun = list(range(T))  # in order
        pos = 0
        guard = 0
        stuck_swaps = 0
        while unrun:
            guard += 1
            if guard > 50 * T + 10000:
                raise RuntimeError("no progress")
            window = unrun[: self.w]
            elig = []
            for idx, j in enumerate(window):
                mj = self.masks[j]
                if all(_st_commute(self.masks[i], mj) for i in window[:idx]):
                    elig.append(j)
            # run every eligible term that needs no bridge
            free = [j for j in elig if self.excess(j) == 0]
            if free:
                for j in free:
                    self.run_term(j)
                done = set(free)
                unrun = [j for j in unrun if j not in done]
                stuck_swaps = 0
                continue
            front = elig[: self.f]
            ext = elig[self.f: self.f + self.e]
            # beyond the window: the next terms in order, as extended set when the window is small
            h0 = self.score(front, ext, self.pi)
            best = None
            cand = set()
            for j in front:
                for q in self.masks[j][2]:
                    p = self.pi[q]
                    for v in self.adj[p]:
                        cand.add((min(p, v), max(p, v)))
            for (p, v) in sorted(cand):
                pi2 = list(self.pi)
                a, b = self.inv.get(p), self.inv.get(v)
                if a is not None:
                    pi2[a] = v
                if b is not None:
                    pi2[b] = p
                merged = self.mergeable(p, v)
                c_swap = 1 if merged else 3
                if self.err is not None:
                    c_swap *= self.err[2].get((p, v), self.w_med) / self.w_med
                gain = h0 - self.score(front, ext, pi2)
                key = (gain - c_swap, merged, -p, -v)
                if best is None or key > best[0]:
                    best = (key, p, v, merged, gain, c_swap)
            min_ex = min((self.excess(j), j) for j in front)
            if best is not None and best[4] > best[5] and stuck_swaps < 4 * self.n:
                _, p, v, merged, _, _ = best
                self.swap(p, v)
                stuck_swaps += 1
                continue
            # no SWAP pays: run the cheapest eligible term with its bridges
            j = min_ex[1]
            self.run_term(j)
            unrun = [x for x in unrun if x != j]
            stuck_swaps = 0
        return self

    def score(self, front, ext, pi):
        h = sum(self.proxy_excess(j, pi) for j in front)
        if ext:
            h += 0.5 * sum(self.proxy_excess(j, pi) for j in ext) / len(ext) * len(front) / max(1, len(front))
        return h


class _StMover2(_StMover):
    """STEINER6 v4: every unrun term that commutes with all earlier unrun terms is eligible (not only those in a
    window); the anticommutation between terms is computed once. A SWAP is scored on every eligible term that
    contains one of the two logical qubits it moves (weight 1) and on the blocked ones (weight LAMBDA), by the change
    of their proxy excess; it is made if that gain exceeds its cost (3 CNOTs, 1 when it joins the last block on the
    pair, scaled by the coupler's error weight in the error mode). Otherwise the eligible term of the least excess
    (earliest on a tie) runs with its bridges."""
    LAMBDA = 0.1
    LOOK = 200
    MU = 0.03

    def build(self):
        import numpy as np
        T = len(self.triples)
        n = self.n
        X = np.zeros((T, n), dtype=np.uint8)
        Z = np.zeros((T, n), dtype=np.uint8)
        for j, (x, z, sup) in enumerate(self.masks):
            for q in sup:
                X[j, q] = (x >> q) & 1
                Z[j, q] = (z >> q) & 1
        succ = [[] for _ in range(T)]
        blocked = [0] * T
        B = 512
        for i0 in range(0, T, B):
            a = ((X[i0:i0 + B].astype(np.int32) @ Z.T.astype(np.int32)) + (Z[i0:i0 + B].astype(np.int32)
                 @ X.T.astype(np.int32))) & 1
            for ii in range(a.shape[0]):
                i = i0 + ii
                later = np.nonzero(a[ii, i + 1:])[0] + i + 1
                succ[i] = later.tolist()
                for j in succ[i]:
                    blocked[j] += 1
        terms_of = [[] for _ in range(n)]
        for j, (_, _, sup) in enumerate(self.masks):
            for q in sup:
                terms_of[q].append(j)
        unrun = [True] * T
        elig = set(j for j in range(T) if blocked[j] == 0)
        ex = {j: self.excess(j) for j in elig}
        left = T
        stuck = 0
        guard = 0
        ptr = 0
        while left:
            guard += 1
            if guard > 20 * T + 100000:
                raise RuntimeError("no progress")
            free = sorted(j for j in elig if ex[j] == 0)
            if free:
                small = [k for k in free if len(self.masks[k][2]) <= 2]
                if small:  # terms on one or two qubits: in the given order, all at once
                    for k in small:
                        self._run(k, unrun, elig, ex, succ, blocked)
                        left -= 1
                else:  # terms on 3 or more qubits commute pairwise: the most similar to the last one next
                    j = max(free, key=lambda k: (self.similar(self.last_term, k), -k))
                    self._run(j, unrun, elig, ex, succ, blocked)
                    left -= 1
                stuck = 0
                continue
            # the blocked terms looked at: the next LOOK unrun ones in the given order
            while ptr < T and not unrun[ptr]:
                ptr += 1
            look = set()
            k = ptr
            while k < T and len(look) < self.LOOK:
                if unrun[k] and k not in elig:
                    look.add(k)
                k += 1
            # SWAP candidates: couplers next to the qubits of the eligible terms of least excess
            order = sorted(elig, key=lambda j: (ex[j], j))
            focus = order[: self.f]
            cand = set()
            for j in focus:
                for q in self.masks[j][2]:
                    p = self.pi[q]
                    for v in self.adj[p]:
                        cand.add((min(p, v), max(p, v)))
            best = None
            for (p, v) in sorted(cand):
                a, b = self.inv.get(p), self.inv.get(v)
                pi2 = list(self.pi)
                if a is not None:
                    pi2[a] = v
                if b is not None:
                    pi2[b] = p
                touched = set()
                for lq in (a, b):
                    if lq is not None:
                        touched.update(terms_of[lq])
                gain = 0.0
                fset = set(focus)
                for j in touched:
                    if not unrun[j] or not (j in elig or j in look):
                        continue
                    wgt = 1.0 if j in fset else (self.MU if j in elig else self.LAMBDA)
                    gain += wgt * (self.proxy_excess(j, self.pi) - self.proxy_excess(j, pi2))
                merged = self.mergeable(p, v)
                c_swap = 1 if merged else 3
                if self.err is not None:
                    c_swap *= self.err[2].get((p, v), self.w_med) / self.w_med
                key = (gain - c_swap, merged, -p, -v)
                if best is None or key > best[0]:
                    best = (key, p, v, a, b)
            if best is not None and best[0][0] > 0 and stuck < 4 * n:
                _, p, v, a, b = best
                self.swap(p, v)
                for lq in (a, b):
                    if lq is not None:
                        for j in terms_of[lq]:
                            if j in elig:
                                ex[j] = self.excess(j)
                stuck += 1
                continue
            j = order[0]
            self._run(j, unrun, elig, ex, succ, blocked)
            left -= 1
            stuck = 0
        return self

    last_term = None

    def similar(self, i, k):
        """Qubits on which terms i and k carry the same Pauli, minus half those where they differ (0 for no i)."""
        if i is None:
            return 0
        xi, zi, si = self.masks[i]
        xk, zk, sk = self.masks[k]
        both = set(si) & set(sk)
        same = sum(1 for q in both if ((xi >> q) & 1) == ((xk >> q) & 1) and ((zi >> q) & 1) == ((zk >> q) & 1))
        return same - 0.5 * (len(both) - same)

    def _run(self, j, unrun, elig, ex, succ, blocked):
        self.last_term = j
        self.run_term(j)
        unrun[j] = False
        elig.discard(j)
        ex.pop(j, None)
        for k in succ[j]:
            blocked[k] -= 1
            if blocked[k] == 0 and unrun[k]:
                elig.add(k)
                ex[k] = self.excess(k)


# ------------------------------------------------------------------ item 83: time-aware order of STEINER6's blocks

ST_SCHEDULE = True  # item 83: re-emit STEINER6's blocks by list scheduling when that shortens the schedule
_ST_DUR = {"cx": 1.0, "swap": 3.0, "rz": 0.0}  # item 83: durations in CX units for the ordering (rz is virtual)
_ST_DUR_1Q = 0.3
_ST_REORDER_PAIRS = 5_000_000  # item 83: commutation checks allowed per circuit; beyond it the mover's order is kept


def _st_record_term(mv, first_op, pauli):
    """Item 83: record the gadget emitted at mv.ops[first_op:] as a block; pauli: its term on physical qubits."""
    idx = list(range(first_op, len(mv.ops)))
    mv.st_blocks.append(["g", idx, pauli])
    k = len(mv.st_blocks) - 1
    for i in idx:
        for x in mv.ops[i][1]:
            mv._lastb[x] = k


def _st_record_swap(mv, first_op, p, q):
    """Item 83: record the SWAP emitted at mv.ops[first_op:]. When the last block on both of its qubits is one gadget
    on exactly that pair, the SWAP is fused to it (Qiskit merges the two into one two-qubit unitary; the fusion keeps
    them together)."""
    idx = list(range(first_op, len(mv.ops)))
    k = mv._lastb.get(p)
    if k is not None and mv._lastb.get(q) == k and mv.st_blocks[k][0] == "g":
        touched = {x for i in mv.st_blocks[k][1] for x in mv.ops[i][1]}
        if touched <= {p, q}:
            mv.st_blocks[k] = ["f", mv.st_blocks[k][1] + idx, (p, q)]
            return
    mv.st_blocks.append(["s", idx, (p, q)])
    mv._lastb[p] = mv._lastb[q] = len(mv.st_blocks) - 1


def _st_blocks_commute(a, b):
    """Item 83: True if blocks a and b (sharing a qubit) commute as unitaries. A gadget is exp(-i t c P) on its
    terminals and the identity on its Steiner qubits; a fused block is taken to commute with nothing it touches."""
    ka, kb = a[0], b[0]
    if ka == "f" or kb == "f":
        return False
    if ka == "g" and kb == "g":
        pa, pb = a[2], b[2]
        anti = 0
        for q, ch in pa.items():
            o = pb.get(q)
            if o is not None and o != ch:
                anti ^= 1
        return anti == 0
    if ka == "s" and kb == "s":
        return set(a[2]) == set(b[2])
    g, s = (a, b) if ka == "g" else (b, a)
    p, q = s[2]
    return g[2].get(p, "I") == g[2].get(q, "I")


def _st_op_dur(name):
    return _ST_DUR.get(name, _ST_DUR_1Q)


def _st_asap(ops):
    """Item 83: the makespan of `ops` (in order, each as soon as its qubits are free) in CX units."""
    clock = {}
    for name, q, _ in ops:
        s = max(clock.get(x, 0.0) for x in q)
        for x in q:
            clock[x] = s + _st_op_dur(name)
    return max(clock.values()) if clock else 0.0


def _st_reorder(ops, blocks):
    """Item 83: `ops` re-emitted block by block by list scheduling, or None when the blocks do not cover the ops
    exactly once. A block is ready once every earlier block (in the mover's order) that shares a qubit with it and does
    not commute with it has been emitted; of the ready blocks, the one that can start earliest goes next (ties: the
    longer chain of dependent work after it, then the mover's order). The blocks and their gates are unchanged, and
    every pair that does not commute keeps its order, so the circuit's action is unchanged."""
    if not blocks:
        return None
    B = len(blocks)
    if sorted(i for b in blocks for i in b[1]) != list(range(len(ops))):
        return None
    touched = [tuple(sorted({x for i in b[1] for x in ops[i][1]})) for b in blocks]
    dur = [sum(_st_op_dur(ops[i][0]) for i in b[1]) for b in blocks]
    first = [min(b[1]) for b in blocks]
    order0 = sorted(range(B), key=lambda i: first[i])
    on = {}
    for i in order0:
        for x in touched[i]:
            on.setdefault(x, []).append(i)
    if sum(len(lst) * (len(lst) - 1) // 2 for lst in on.values()) > _ST_REORDER_PAIRS:
        return None
    preds = [set() for _ in range(B)]
    for lst in on.values():
        for k, i in enumerate(lst):
            for h in range(k - 1, -1, -1):
                j = lst[h]
                if j not in preds[i] and not _st_blocks_commute(blocks[j], blocks[i]):
                    preds[i].add(j)
    succ = [[] for _ in range(B)]
    indeg = [len(p) for p in preds]
    for i in range(B):
        for j in preds[i]:
            succ[j].append(i)
    blevel = [0.0] * B
    for i in reversed(order0):
        blevel[i] = dur[i] + max((blevel[k] for k in succ[i]), default=0.0)
    clock = {}
    ready = {i for i in range(B) if indeg[i] == 0}
    out = []
    while ready:
        i = min(ready, key=lambda k: (max(clock.get(x, 0.0) for x in touched[k]), -blevel[k], first[k]))
        ready.discard(i)
        for k in sorted(blocks[i][1]):
            name, q, _ = ops[k]
            s = max(clock.get(x, 0.0) for x in q)
            for x in q:
                clock[x] = s + _st_op_dur(name)
            out.append(ops[k])
        for k in succ[i]:
            indeg[k] -= 1
            if indeg[k] == 0:
                ready.add(k)
    if len(out) != len(ops):  # a cycle cannot arise (edges go forwards in the mover's order); kept as a guard
        return None
    return out



_ST_FAILED = 0.5  # set from prune_max_error by _st_compile
_ST_VF2_CHECK, _ST_VF2_MAPS, _ST_KEEP = 20_000, 200, 6
_ST_EMBED = 2  # STEINER5's VF2 embeddings added per distance mode as initial placements
_ST_BIG_TERMS = 2000  # above this many terms: the error mode only, one greedy start and one embedding
STEINER_STATS = {"not_applicable": 0, "not_commuting": 0, "used": 0, "fallback": 0, "too_large": 0, "chosen": 0,
                 "kept_usual": 0, "gate_level": 0, "permuted": 0, "mirror_tried": 0,
                 "mirror_kept": 0, "reordered": 0, "reorder_kept_order": 0,
                 "diag_dropped": 0}  # changelog items 68-76, 82, 83, 89
DROP_DIAGONAL_BEFORE_MEASURE = True  # item 89
MAP_FREE_DEFAULTS = True  # item 90: items 33/60, 64 and 78 off by default
PLACE_SCORE = "duration"  # item 90: item 75 scored by the placed operations' durations ("esp": c51)
_ST_MAX_TERMS = 2000  # item 70


def _st_cost(circ, target):
    """Item 69: the summed -ln(1 - error) of every operation of `circ` on the device (-ln ESP); inf if one is not
    offered there or has error >= 1. Barriers and delays count 0, and so does control flow (no error reported)."""
    pos = {q: i for i, q in enumerate(circ.qubits)}
    total = 0.0
    for ins in circ.data:
        name = ins.operation.name
        if name in ("barrier", "delay"):
            continue
        if name not in target.operation_names:
            return math.inf
        entries = target[name]
        qargs = tuple(pos[q] for q in ins.qubits)
        if qargs in entries:
            props = entries[qargs]
        elif None in entries:
            continue
        else:
            return math.inf
        e = props.error if props is not None and props.error is not None else 0.0
        if e >= 1.0:
            return math.inf
        total += -math.log1p(-max(e, 0.0))
    return total


# ---------------------------------------------------------------------------------------------------------------------
# Prototype c49: item 84 (choices by duration, not by the device's error table) and item 85 (failed qubits cut first)
# ---------------------------------------------------------------------------------------------------------------------

SELECT_BY = "duration"  # item 84: "duration" (c49) or "esp" (c48's summed -ln(1 - error))
SELECT_STATS = {"duration": 0, "cx_units": 0, "tie_2q": 0, "tie_kept": 0}
CUT_FAILED_QUBITS = True  # item 85: compile on the device without its failed qubits
CUT_STATS = {"cut": 0, "qubits_cut": 0, "too_few": 0, "no_placement": 0}


def _duration_s(circ, target):
    """Item 84: the as-soon-as-possible makespan of `circ` in seconds on the durations `target` reports (each
    operation starts when all its qubits are free; barriers take no time), or None when an operation has no reported
    duration on its qubits (then both circuits are compared in CX units)."""
    pos = {q: i for i, q in enumerate(circ.qubits)}
    clock = {}
    names = set(target.operation_names)
    for ins in circ.data:
        name = ins.operation.name
        if name == "barrier":
            continue
        if name not in names:
            return None
        qargs = tuple(pos[q] for q in ins.qubits)
        entries = target[name]
        props = entries.get(qargs) if qargs in entries else entries.get(None)
        d = getattr(props, "duration", None) if props is not None else None
        if d is None:
            if name == "rz":  # virtual on every IBM device
                d = 0.0
            else:
                return None
        s = max((clock.get(x, 0.0) for x in qargs), default=0.0)
        for x in qargs:
            clock[x] = s + d
    return max(clock.values()) if clock else 0.0


def _duration_cx(circ):
    """Item 84: the same makespan in item 83's CX units (CX 1, SWAP 3, rz 0, any other single-qubit gate 0.3, any
    other two-qubit gate 1), for a Target without durations."""
    pos = {q: i for i, q in enumerate(circ.qubits)}
    clock = {}
    for ins in circ.data:
        name = ins.operation.name
        if name == "barrier":
            continue
        qargs = tuple(pos[q] for q in ins.qubits)
        if name in _ST_DUR:
            d = _ST_DUR[name]
        else:
            d = 1.0 if len(qargs) >= 2 else _ST_DUR_1Q
        s = max((clock.get(x, 0.0) for x in qargs), default=0.0)
        for x in qargs:
            clock[x] = s + d
    return max(clock.values()) if clock else 0.0


def _better(a, b, target):
    """Item 84: True if circuit `a` is to be returned instead of `b` (both compiled for `target`).

    "duration" (default): a circuit with an operation the device does not offer, or one reported at error >= 1, loses
    (item 69's safety, unchanged). Otherwise the shorter as-soon-as-possible schedule on the device's reported gate
    durations wins; on equal durations (within 1e-12 s), fewer two-qubit gates; on a full tie, `b` (the caller's
    default) is kept. No error value of the device is used beyond that. "esp" restores c48's choice (lower summed
    -ln(1 - error))."""
    ca, cb = _st_cost(a, target), _st_cost(b, target)
    if SELECT_BY == "esp":
        return ca < cb
    fa, fb = math.isinf(ca), math.isinf(cb)
    if fa != fb:
        return fb
    da, db = _duration_s(a, target), _duration_s(b, target)
    if da is None or db is None:
        da, db, tol = _duration_cx(a), _duration_cx(b), 1e-9
        SELECT_STATS["cx_units"] += 1
    else:
        tol = 1e-12
        SELECT_STATS["duration"] += 1
    if abs(da - db) > tol:
        return da < db
    na, nb = _count_2q(a), _count_2q(b)
    if na != nb:
        SELECT_STATS["tie_2q"] += 1
        return na < nb
    SELECT_STATS["tie_kept"] += 1
    return False


def _working_target(target, cut):
    """Item 85: (a Target of the device without the qubits in `cut`, renumbered in order, and the list of the
    device's qubits it keeps). Every instruction, property (errors and durations as reported) and qubit property
    of the kept qubits is copied; operations offered on all qubits stay global."""
    from qiskit.transpiler import Target
    keep = [q for q in range(target.num_qubits) if q not in cut]
    idx = {q: i for i, q in enumerate(keep)}
    qprops = None
    if target.qubit_properties is not None:
        qprops = [target.qubit_properties[q] for q in keep]
    sub = Target(description=getattr(target, "description", None), num_qubits=len(keep), dt=target.dt,
                 granularity=target.granularity, min_length=target.min_length,
                 pulse_alignment=target.pulse_alignment, acquire_alignment=target.acquire_alignment,
                 qubit_properties=qprops)
    for name in target.operation_names:
        op = target.operation_from_name(name)
        if isinstance(op, type):  # variadic or control-flow operations: offered on every qubit
            sub.add_instruction(op, name=name)
            continue
        props = {}
        for qargs, p in target[name].items():
            if qargs is None:
                props[None] = p
            elif all(q in idx for q in qargs):
                props[tuple(idx[q] for q in qargs)] = p
        if not props:
            continue
        if None in props and len(props) == 1:
            sub.add_instruction(op, None, name=name)
        else:
            sub.add_instruction(op, props, name=name)
    return sub, keep


def _embed_working(o, keep, N):
    """Item 85: circuit `o`, compiled on `_working_target`'s renumbered device, written on the device's own N qubits
    (qubit i of `o` on keep[i]), with its layout carried over: the cut qubits hold fresh ancillas that no operation
    touches and that do not move."""
    from qiskit.circuit import AncillaRegister, QuantumRegister
    from qiskit.transpiler import Layout, TranspileLayout
    full = QuantumCircuit(QuantumRegister(N, "q"), *o.cregs, name=o.name, global_phase=o.global_phase,
                          metadata=dict(o.metadata or {}))
    loose = [c for c in o.clbits if not o.find_bit(c).registers]
    if loose:
        full.add_bits(loose)
    qmap = {o.qubits[i]: full.qubits[keep[i]] for i in range(len(keep))}
    for ins in o.data:
        full.append(ins.operation, [qmap[q] for q in ins.qubits], ins.clbits, copy=False)
    lay = getattr(o, "layout", None)
    if lay is None:
        return full
    cut = [q for q in range(N) if q not in set(keep)]
    anc = AncillaRegister(len(cut), "cut_failed") if cut else None
    v2p = {v: keep[p] for v, p in lay.initial_layout.get_virtual_bits().items()}
    if anc is not None:
        v2p.update({anc[k]: cut[k] for k in range(len(cut))})
    initial = Layout(v2p)
    for reg in lay.initial_layout.get_registers():
        initial.add_register(reg)
    if anc is not None:
        initial.add_register(anc)
    mapping = dict(lay.input_qubit_mapping)
    base = len(mapping)
    if anc is not None:
        mapping.update({anc[k]: base + k for k in range(len(cut))})
    final = None
    if lay.final_layout is not None:
        opos = {q: i for i, q in enumerate(o.qubits)}
        fl = {full.qubits[keep[opos[q]]]: keep[p] for q, p in lay.final_layout.get_virtual_bits().items()}
        fl.update({full.qubits[c]: c for c in cut})
        final = Layout(fl)
    full._layout = TranspileLayout(initial_layout=initial, input_qubit_mapping=mapping, final_layout=final,
                                   _input_qubit_count=lay._input_qubit_count, _output_qubit_list=list(full.qubits))
    return full


def _cut_failed_qubits(kw):
    """Item 85: the output of compile_for_hardware(**kw) compiled on the device without its failed qubits and written
    back on the device's own indices, or None when the device has no failed qubit (or too few working qubits and
    on_failed_elements="keep": the call then goes on as in c48). `kw` is the call's arguments, the device's Target
    resolved."""
    target, qc = kw["target"], kw["qc"]
    cut = sorted(_failed_elements(target, kw["prune_max_error"])[1])
    if not cut:
        return None
    if target.num_qubits - len(cut) < qc.num_qubits:
        CUT_STATS["too_few"] += 1
        if kw["on_failed_elements"] == "raise":
            raise FailedElementsError(
                f"the device has {target.num_qubits - len(cut)} working qubits (failed: {cut}) and the circuit "
                f"{qc.num_qubits} (changelog item 85)")
        return None
    sub, keep = _working_target(target, set(cut))
    CUT_STATS["cut"] += 1
    CUT_STATS["qubits_cut"] += len(cut)
    try:
        out = compile_for_hardware(**dict(kw, target=sub, backend=None))
    except TranspilerError as exc:  # FailedElementsError included: no placement on the working qubits
        if isinstance(exc, FailedElementsError):
            raise
        CUT_STATS["no_placement"] += 1
        if kw["on_failed_elements"] == "raise":
            raise FailedElementsError(f"no placement on the device's working qubits (failed: {cut}; changelog "
                                      f"item 85): {exc}") from exc
        return None
    return _embed_working(out, keep, target.num_qubits)


def _st_evolution(qc):
    """Item 68: (PauliEvolutionGate instruction, [(qubit, clbit)] of the final measurements) when `qc` is one
    PauliEvolutionGate on all its qubits in order (barriers and final measurements aside), with Qiskit's default
    first-order product formula (LieTrotter, one repetition), a bound time and real coefficients; else None."""
    evo, meas = None, []
    for ins in qc.data:
        name = ins.operation.name
        if name == "barrier":
            continue
        if name == "PauliEvolution" and evo is None and not meas:
            evo = ins
        elif name == "measure" and evo is not None:
            meas.append((qc.find_bit(ins.qubits[0]).index, qc.find_bit(ins.clbits[0]).index))
        else:
            return None
    if evo is None:
        return None
    gate = evo.operation
    if isinstance(gate.operator, list):
        return None
    syn = getattr(gate, "synthesis", None)
    if syn is not None and (type(syn).__name__ != "LieTrotter" or getattr(syn, "reps", 1) != 1):
        return None
    if [qc.find_bit(q).index for q in evo.qubits] != list(range(qc.num_qubits)):
        return None
    try:
        t = float(gate.time)
    except (TypeError, ValueError):
        return None
    coeffs = np.asarray(gate.operator.coeffs)
    if coeffs.dtype == object or np.any(np.abs(np.imag(coeffs)) > 1e-12):
        return None
    triples = [(lb, float(np.real(c)), t) for lb, c in gate.operator.to_list()]
    return evo, meas, triples


def _st_all_commute(triples):
    """Item 68: True if every two terms commute (checked on the Pauli bit matrices, in blocks)."""
    labels = [lb for lb, _, _ in triples]
    n = len(labels[0]) if labels else 0
    X = np.array([[ch in "XY" for ch in reversed(lb)] for lb in labels], dtype=np.int32).reshape(len(labels), n)
    Z = np.array([[ch in "ZY" for ch in reversed(lb)] for lb in labels], dtype=np.int32).reshape(len(labels), n)
    for i0 in range(0, len(labels), 512):
        a = (X[i0:i0 + 512] @ Z.T + Z[i0:i0 + 512] @ X.T) & 1
        if a.any():
            return False
    return True


def _st_construct(n, triples, target, meas):
    """Item 68: STEINER6's construction (Addendum 444, run v6): candidates from the greedy placements (hop and error
    distances, two starts each) and STEINER5's embeddings, each run by _StMover2; the shortest schedule kept (item 84;
    the lowest estimated -ln ESP with SELECT_BY = "esp").
    (mover, initial placement) or None."""
    adj, _ = _st_working_graph(target)
    if len(adj) < n:
        return None
    dist, par = _st_bfs_all(adj)
    derr, perr, wedge = _st_error_distances(adj, target)
    if not wedge:
        return None
    readout = {}
    if "measure" in target.operation_names:
        for qa, pr in target["measure"].items():
            if qa is not None and pr is not None and pr.error is not None:
                readout[qa[0]] = -math.log(1.0 - min(pr.error, 0.999))
    terms = [(lb, c) for lb, c, _ in triples]
    w_med = sorted(wedge.values())[len(wedge) // 2]
    starts_h = sorted(adj, key=lambda v: (-len(adj[v]), v))[:2]
    nb = {v: sum(wedge.get((min(v, u), max(v, u)), w_med) for u in adj[v]) / max(1, len(adj[v])) for v in adj}
    starts_e = sorted((v for v in adj if len(adj[v]) >= 2), key=lambda v: (nb[v], v))[:2]
    inits = {"H": [("greedy", _st_placement(n, terms, adj, dist, s0)) for s0 in starts_h],
             "E": [("greedy", _st_placement(n, terms, adj, derr, s0)) for s0 in starts_e]}
    for mode, (dd, pp, ew) in (("H", (dist, par, None)), ("E", (derr, perr, wedge))):
        try:
            emb, _, _ = _st_embeddings(n, terms, adj, dd)
        except Exception:  # noqa: BLE001 - no embedding within the call limit: the greedy placements remain
            emb = []
        scored = sorted(((_st_placement_cost(e, terms, dd, pp, ew), k) for k, e in enumerate(emb)))[:_ST_EMBED]
        inits[mode] += [("vf2", emb[k]) for _, k in scored]
    modes = ("H", "E")
    if len(triples) > _ST_BIG_TERMS:
        modes = ("E",)
        inits["E"] = inits["E"][:1] + [x for x in inits["E"] if x[0] == "vf2"][:1]
    cands = []
    for mode in modes:
        for label, pi0 in inits[mode]:
            if any(p not in adj for p in pi0):
                continue
            m = _StMover2(n, triples, adj, dist, par, pi0, err=None if mode == "H" else (derr, perr, wedge)).build()
            if SELECT_BY == "esp":
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
            cands.append((est, mode, label, m, pi0))
    if not cands:
        return None
    best = min(cands, key=lambda c: (c[0], c[1], c[2]))
    return best[3], list(best[4])


_ST_ROT = {"rz": "Z", "rx": "X", "ry": "Y", "rzz": "ZZ", "rxx": "XX", "ryy": "YY"}  # item 72: exp(-i theta/2 P)
_ST_FORMULAS = ("LieTrotter", "SuzukiTrotter")  # item 72: deterministic product formulas (not QDrift)


def _st_flatten(qc, depth=0):
    """Item 72: ([(operation, qubit indices, clbit indices)], global phase) of `qc`, with composite instructions that
    are not read directly (a PauliEvolutionGate, a rotation, a single-qubit gate, measure, barrier) replaced by their
    definitions, up to three levels."""
    out, phase = [], float(qc.global_phase) if not hasattr(qc.global_phase, "parameters") else None
    if phase is None:
        return None, None
    for ins in qc.data:
        op = ins.operation
        qs = [qc.find_bit(q).index for q in ins.qubits]
        cs = [qc.find_bit(c).index for c in ins.clbits]
        leaf = (op.name in ("barrier", "measure", "PauliEvolution") or op.name in _ST_ROT
                or (len(qs) == 1 and not cs and getattr(op, "blocks", None) is None))
        if not leaf and depth < 3 and not cs and getattr(op, "definition", None) is not None:
            inner, ph = _st_flatten(op.definition, depth + 1)
            if inner is None:
                return None, None
            phase += ph
            out += [(o, [qs[i] for i in q2], c2) for o, q2, c2 in inner]
        else:
            out.append((op, qs, cs))
    return out, phase


def _st_pauli_terms(gate, qs, n):
    """Item 72: the terms a PauliEvolutionGate on qubits `qs` applies, in order, as [(label on n qubits, angle, 1.0)]
    with each term exp(-i angle P), from its product formula's own expansion (LieTrotter or SuzukiTrotter, any order and
    repetitions); and the global phase of identity terms. None if not readable (a list of operators, an unbound time,
    complex coefficients, another formula)."""
    from qiskit.synthesis import LieTrotter
    if isinstance(gate.operator, list):
        return None
    try:
        float(gate.time)
    except (TypeError, ValueError):
        return None
    coeffs = np.asarray(gate.operator.coeffs)
    if coeffs.dtype == object or np.any(np.abs(np.imag(coeffs)) > 1e-12):
        return None
    syn = gate.synthesis if gate.synthesis is not None else LieTrotter()
    if type(syn).__name__ not in _ST_FORMULAS:
        return None
    try:
        seq = syn.expand(gate)
    except Exception:  # noqa: BLE001 - not expandable: the usual path
        return None
    out, phase = [], 0.0
    for item in seq:
        pl, idx, coeff = item[0], list(item[1]), float(np.real(item[2]))
        lab = ["I"] * n
        for ch, i in zip(pl, idx):  # character k acts on the gate's qubit idx[k]
            if ch != "I":
                lab[n - 1 - qs[i]] = ch
        lab = "".join(lab)
        if set(lab) <= {"I"}:
            phase -= coeff / 2
            continue
        out.append((lab, coeff / 2, 1.0))
    return out, phase


def _st_parse(qc):
    """Item 72: (leading single-qubit gates [(operation, qubit)], final measurements [(qubit, clbit)], terms
    [(label, angle, 1.0)] in order, global phase) when `qc` is: single-qubit gates first, then only Pauli evolutions and
    rotation gates (rz, rx, ry, rzz, rxx, ryy), then measurements (barriers anywhere); else None."""
    flat, phase = _st_flatten(qc)
    if flat is None:
        return None
    n = qc.num_qubits
    prefix, meas, terms = [], [], []
    started = False
    for op, qs, cs in flat:
        name = op.name
        if name == "barrier":
            continue
        if name == "measure":
            meas.append((qs[0], cs[0]))
            continue
        if meas:
            return None  # an operation after a measurement
        if name == "PauliEvolution":
            got = _st_pauli_terms(op, qs, n)
            if got is None:
                return None
            terms += got[0]
            phase += got[1]
            started = True
            continue
        if name in _ST_ROT and len(op.params) == 1:
            try:
                angle = float(op.params[0])
            except (TypeError, ValueError):
                return None
            lab = ["I"] * n
            for ch, q in zip(_ST_ROT[name], qs):
                lab[n - 1 - q] = ch
            terms.append(("".join(lab), angle / 2, 1.0))
            started = True
            continue
        if len(qs) == 1 and not cs and not started:
            try:
                op.to_matrix()
            except Exception:  # noqa: BLE001 - not a unitary gate
                return None
            prefix.append((op, qs[0]))
            continue
        return None
    if not terms:
        return None
    return prefix, meas, terms, phase


def _st_merge(terms):
    """Item 72: repeated terms joined where exact: a term is added to an earlier one with the same Pauli if no term in
    between anticommutes with it (it could then be moved next to it). [(label, angle, 1.0)] in order."""
    out, open_ = [], {}
    for lb, c, t in terms:
        x, z, _ = _st_term_masks(lb)
        if lb in open_:
            out[open_[lb]][1] += c * t
            continue
        for other in [k for k, i in open_.items() if bin((out[i][2] & z) ^ (out[i][3] & x)).count("1") & 1]:
            del open_[other]
        out.append([lb, c * t, x, z])
        open_[lb] = len(out) - 1
    return [(lb, a, 1.0) for lb, a, _, _ in out]


_ST_GATE_LIMIT = 200_000  # item 73: circuits with more instructions are not read
_ST_PAULI_1Q = {"I": (0, 0), "X": (1, 0), "Y": (1, 1), "Z": (0, 1)}
_ST_CONJ_CACHE = {}
_ST_CLIFF1Q = None


def _st_pmul(a, b):
    """Item 73: the product a.b of two Paulis written (x mask, z mask, k) for i^k X^x Z^z."""
    return a[0] ^ b[0], a[1] ^ b[1], (a[2] + b[2] + 2 * bin(a[1] & b[0]).count("1")) % 4


def _st_conj(op):
    """Item 73: for a gate on k <= 2 qubits whose conjugation maps Paulis to Paulis (a Clifford gate), the images
    G^dagger P G of X and Z on each of its qubits, as [(label on its k qubits, +1 or -1)] in the order X0, Z0, X1, Z1;
    None if it is not Clifford or has no matrix. From the gate's own matrix, cached by name and parameters."""
    try:
        key = (op.name, tuple(float(p) for p in op.params))
    except (TypeError, ValueError):
        return None
    if key in _ST_CONJ_CACHE:
        return _ST_CONJ_CACHE[key]
    from itertools import product
    from qiskit.quantum_info import Pauli
    k = op.num_qubits
    try:
        m = np.asarray(op.to_matrix(), dtype=complex)
    except Exception:  # noqa: BLE001 - no matrix
        _ST_CONJ_CACHE[key] = None
        return None
    basis = ["".join(t) for t in product("IXYZ", repeat=k)]
    mats = {lb: Pauli(lb).to_matrix() for lb in basis}
    out = []
    for j in range(k):
        for ch in "XZ":
            lb = ["I"] * k
            lb[k - 1 - j] = ch
            img = m.conj().T @ mats["".join(lb)] @ m
            hit = None
            for r in basis:
                c = np.trace(mats[r].conj().T @ img) / 2 ** k
                if abs(c) > 1e-6:
                    if hit is not None or abs(abs(c) - 1) > 1e-9 or abs(c.imag) > 1e-9:
                        hit = None
                        break
                    hit = (r, 1 if c.real > 0 else -1)
            if hit is None:
                _ST_CONJ_CACHE[key] = None
                return None
            out.append(hit)
    _ST_CONJ_CACHE[key] = out
    return out


def _st_cliff1q():
    """Item 73: {(image of X, image of Z) as (character, sign): [gate, ...]} for the 24 single-qubit Cliffords, from
    the shortest words in H and S."""
    global _ST_CLIFF1Q
    if _ST_CLIFF1Q is None:
        from qiskit.circuit.library import HGate, SGate
        table, frontier = {}, [[]]
        while frontier and len(table) < 24:
            nxt = []
            for word in frontier:
                m = np.eye(2, dtype=complex)
                for g in word:
                    m = g.to_matrix() @ m
                c = _st_conj(type("W", (), {"name": "w" + "".join(g.name for g in word), "params": [],
                                            "num_qubits": 1, "to_matrix": lambda self, m=m: m})())
                key = (c[0], c[1])
                if key not in table:
                    table[key] = word
                    nxt += [word + [HGate()], word + [SGate()]]
            frontier = nxt
        _ST_CLIFF1Q = table
    return _ST_CLIFF1Q


def _st_parse_gates(qc):
    """Item 73: (leading single-qubit gates [(operation, qubit)], final measurements [(qubit, clbit)], terms
    [(label, angle, 1.0)] in order, global phase, final single-qubit Cliffords [(operation, qubit)]) such that the
    circuit equals, up to a global phase, the leading gates, then the terms exp(-i angle P), then the final gates, then
    the measurements; None if it cannot be written so.

    Single-qubit gates on a qubit before its first multi-qubit gate are kept as the leading gates. After that, Clifford
    gates (any gate on one or two qubits that maps Paulis to Paulis, from its matrix, including rotations by multiples
    of pi/2) are carried to the end: a later rotation exp(-i a P) becomes exp(-i a C^dagger P C), where C is the product
    of the Clifford gates so far. Rotations rz, rx, ry, rzz, rxx, ryy, p, u1, t, tdg and Pauli evolutions give terms; any
    other single-qubit gate is written as rz.ry.rz; other gates are read through their definitions (three levels). The
    Clifford gates left at the end must act on each qubit alone (a product of single-qubit Cliffords)."""
    from qiskit.synthesis import OneQubitEulerDecomposer
    n = qc.num_qubits
    if n == 0 or len(qc.data) > _ST_GATE_LIMIT:
        return None
    try:
        phase = float(qc.global_phase)
    except (TypeError, ValueError):
        return None
    frame = [[(1 << q, 0, 0), (0, 1 << q, 0)] for q in range(n)]  # C^dagger X_q C, C^dagger Z_q C
    touched = [False] * n
    prefix, meas, terms = [], [], []
    count = [0]
    euler = OneQubitEulerDecomposer("ZYZ")

    def image(chars):  # chars: {qubit: "X" | "Y" | "Z"} -> C^dagger P C as (x, z, k)
        acc = (0, 0, 0)
        for q, ch in chars.items():
            if ch == "X":
                acc = _st_pmul(acc, frame[q][0])
            elif ch == "Z":
                acc = _st_pmul(acc, frame[q][1])
            elif ch == "Y":
                acc = _st_pmul(acc, _st_pmul((0, 0, 1), _st_pmul(frame[q][0], frame[q][1])))
        return acc

    def term(chars, angle):
        x, z, k = image(chars)
        if not x and not z:
            return None
        sign = (k - bin(x & z).count("1")) % 4
        if sign not in (0, 2):
            return None
        lab = ["I"] * n
        both = x | z
        while both:
            low = both & -both
            j = low.bit_length() - 1
            lab[n - 1 - j] = "Y" if x & low and z & low else ("X" if x & low else "Z")
            both ^= low
        terms.append(("".join(lab), angle if sign == 0 else -angle, 1.0))
        return True

    def clifford(op, qs):
        c = _st_conj(op)
        if c is None:
            return False
        new = []
        for j in range(len(qs)):
            row = []
            for r, sg in c[2 * j: 2 * j + 2]:
                chars = {qs[len(qs) - 1 - i]: ch for i, ch in enumerate(r) if ch != "I"}
                acc = image(chars)
                if sg < 0:
                    acc = (acc[0], acc[1], (acc[2] + 2) % 4)
                row.append(acc)
            new.append(row)
        for j, q in enumerate(qs):
            frame[q] = new[j]
        return True

    def one(op, qs, depth):
        nonlocal phase
        count[0] += 1
        if count[0] > _ST_GATE_LIMIT:
            return False
        name = op.name
        if len(qs) == 1 and not touched[qs[0]] and name != "PauliEvolution":
            try:
                op.to_matrix()
            except Exception:  # noqa: BLE001 - not a unitary gate
                return False
            prefix.append((op, qs[0]))
            return True
        if name == "PauliEvolution":
            got = _st_pauli_terms(op, qs, n)
            if got is None:
                return False
            for q in qs:
                touched[q] = True
            phase += got[1]
            for lb, a, _ in got[0]:
                chars = {n - 1 - i: ch for i, ch in enumerate(lb) if ch != "I"}
                if term(chars, a) is None:
                    return False
            return True
        for q in qs:
            touched[q] = True
        if name in ("p", "u1", "t", "tdg") and len(qs) == 1:
            try:
                lam = {"t": math.pi / 4, "tdg": -math.pi / 4}.get(name) or float(op.params[0])
            except (TypeError, ValueError):
                return False
            phase += lam / 2
            from qiskit.circuit.library import RZGate
            return one(RZGate(lam), qs, depth)
        if name in _ST_ROT and len(op.params) == 1:
            try:
                angle = float(op.params[0])
            except (TypeError, ValueError):
                return False
            if len(qs) == 1 and abs(math.remainder(angle, math.pi / 2)) < 1e-12:
                return clifford(op, qs)
            return term({q: ch for q, ch in zip(qs, _ST_ROT[name])}, angle / 2) is not None
        if len(qs) <= 2 and getattr(op, "blocks", None) is None and _st_conj(op) is not None:
            return clifford(op, qs)
        if len(qs) == 1:
            try:
                sub = euler(op.to_matrix())
            except Exception:  # noqa: BLE001 - not a unitary gate
                return False
            phase += float(sub.global_phase)
            return all(one(i.operation, qs, depth) for i in sub.data)
        definition = getattr(op, "definition", None)
        if depth >= 3 or definition is None or getattr(op, "blocks", None) is not None:
            return False
        try:
            phase += float(definition.global_phase)
        except (TypeError, ValueError):
            return False
        for i in definition.data:
            if i.clbits:
                return False
            if not one(i.operation, [qs[definition.find_bit(b).index] for b in i.qubits], depth + 1):
                return False
        return True

    for ins in qc.data:
        op = ins.operation
        name = op.name
        if name == "barrier":
            continue
        qs = [qc.find_bit(q).index for q in ins.qubits]
        if name == "measure":
            meas.append((qs[0], qc.find_bit(ins.clbits[0]).index))
            continue
        if meas or ins.clbits:
            return None
        if not one(op, qs, 0):
            return None
    if not terms:
        return None
    if DROP_DIAGONAL_BEFORE_MEASURE and meas and {q for q, _ in meas} == set(range(n)):  # item 89
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
    perm = list(range(n))  # item 76: perm[q] is the wire whose state logical qubit q holds at the end
    seen = set()
    for q in range(n):
        if frame[q] == [(1 << q, 0, 0), (0, 1 << q, 0)]:
            seen.add(q)
            continue
        key = []
        wire = None
        for x, z, k in frame[q]:
            both = x | z
            if not both or both & (both - 1):
                return None  # the Clifford left at the end acts on more than one qubit here
            if wire is None:
                wire = both.bit_length() - 1
            elif both != 1 << wire:
                return None
            ch = "Y" if x and z else ("X" if x else "Z")
            sg = (k - (1 if ch == "Y" else 0)) % 4
            if sg not in (0, 2):
                return None
            key.append((ch, 1 if sg == 0 else -1))
        if wire in seen:
            return None
        seen.add(wire)
        perm[q] = wire
        table = table or _st_cliff1q()
        word = table.get(tuple(key))
        if word is None:
            return None
        suffix += [(g, wire) for g in word]
    if len(seen) != n:
        return None
    STEINER_STATS["gate_level"] += 1
    if perm != list(range(n)):
        STEINER_STATS["permuted"] += 1
    return prefix, meas, terms, phase, suffix, perm


class _StMirror:
    """Item 82: repeated layers of commuting terms (QAOA's cost layers, an Ising or ZZ Hamiltonian's Trotter steps):
    the term sequence is C_1 L_1 C_2 L_2 ... where every C_k is the same set of pairwise commuting terms on two or more
    qubits (angles may differ) and every L_k holds single-qubit terms. C_1 is scheduled once by STEINER6's mover; C_2
    replays C_1's events (terms and SWAPs) in reverse order, C_3 forwards again, and so on, so every layer costs what
    the first does and every second layer ends where the first began. L_k's terms run where their qubits are. A fixed
    construction, not a search."""

    def __init__(self, n, triples, adj, dist, par, pi0, err, blocks):
        self.n, self.adj, self.dist, self.par, self.err = n, adj, dist, par, err
        self.triples = triples
        self.blocks = blocks  # [("C", [term indices]) | ("L", [term indices])]
        self.pi = list(pi0)
        self.ops, self.gphase = [], 0.0
        self.stats = dict(swaps=0, merged_swaps=0, bridged_terms=0, bridge_excess=0, terms=0)
        self.st_blocks, self._lastb = [], {}  # item 83

    def _term(self, j):
        lb, c, t = self.triples[j]
        first_op = len(self.ops)  # item 83
        pauli = {self.pi[q]: ch for q, ch in enumerate(reversed(lb)) if ch != "I"}
        sup = [k for k, ch in enumerate(reversed(lb)) if ch != "I"]
        phys = [self.pi[k] for k in sup]
        ex = 0
        if len(phys) > 1:
            r, p = _st_steiner_tree(phys, self.dist, self.par)
            ex = _st_tree_cost(r, p, set(phys)) - 2 * (len(phys) - 1)
        dist, par = (self.dist, self.par) if (self.err is None or ex == 0) else (self.err[0], self.err[1])
        for name, q, prm in _st_gadget_ops(lb, c, t, self.pi, dist, par):
            if name == "gphase":
                self.gphase += prm[0]
            else:
                self.ops.append((name, q, prm))
        self.stats["terms"] += 1
        if ex > 0:
            self.stats["bridged_terms"] += 1
        _st_record_term(self, first_op, pauli)  # item 83

    def _swap(self, p, q):
        self.ops.append(("swap", (p, q), ()))
        _st_record_swap(self, len(self.ops) - 1, p, q)  # item 83
        a = self.pi.index(p) if p in self.pi else None
        b = self.pi.index(q) if q in self.pi else None
        if a is not None:
            self.pi[a] = q
        if b is not None:
            self.pi[b] = p
        self.stats["swaps"] += 1

    def build(self, first, c1):
        """first: STEINER6's mover run on C_1's terms alone (c1: their indices here) from the same initial placement;
        its emission gives the events (a term per rz, SWAPs) that every layer replays."""
        events = []
        term_iter = iter(first.order)
        for name, q, prm in first.ops:
            if name == "swap":
                events.append(("s", q[0], q[1]))
            elif name == "rz":
                events.append(("t", self.triples[c1[next(term_iter)]][0]))
        c_index = 0
        for kind, idx in self.blocks:
            if kind == "L":
                for j in idx:
                    self._term(j)
                continue
            by_label = {self.triples[j][0]: j for j in idx}
            for ev in (events if c_index % 2 == 0 else events[::-1]):
                if ev[0] == "s":
                    self._swap(ev[1], ev[2])
                else:
                    self._term(by_label[ev[1]])
            c_index += 1
        return self


def _st_mirror_blocks(triples, limit=4000):
    """Item 82: [("L", idx) | ("C", idx)] when the terms are p >= 2 layers of the same commuting set: the k-th
    occurrence of each multi-qubit label goes to C_k (every multi-qubit label occurs p times; C_1's terms commute),
    and each single-qubit term to the gap right after the last C layer it must follow (that of the latest earlier
    term it anticommutes with; before C_1 if none); the order of the blocks keeps every pair of anticommuting terms in
    its original order, so the reordering is exact. None otherwise."""
    T = len(triples)
    if T > limit:
        return None
    multi = [sum(ch != "I" for ch in lb) >= 2 for lb, _, _ in triples]
    seen, layer = {}, [None] * T
    for j, (lb, _, _) in enumerate(triples):
        if multi[j]:
            layer[j] = seen.get(lb, 0)
            seen[lb] = layer[j] + 1
    if not seen:
        return None
    p = max(seen.values())
    if p < 2 or any(c != p for c in seen.values()):
        return None
    masks = [_st_term_masks(lb) for lb, _, _ in triples]
    slot = [None] * T
    for j in range(T):
        if not multi[j]:
            before = [layer[i] for i in range(j) if multi[i] and not _st_commute(masks[i], masks[j])]
            slot[j] = max(before) if before else -1
    blocks = []
    lead = [j for j in range(T) if not multi[j] and slot[j] == -1]
    if lead:
        blocks.append(("L", lead))
    for k in range(p):
        blocks.append(("C", [j for j in range(T) if multi[j] and layer[j] == k]))
        ls = [j for j in range(T) if not multi[j] and slot[j] == k]
        if ls:
            blocks.append(("L", ls))
    c1 = blocks[1][1] if blocks[0][0] == "L" else blocks[0][1]
    for a in range(len(c1)):
        for b in range(a + 1, len(c1)):
            if not _st_commute(masks[c1[a]], masks[c1[b]]):
                return None
    pos = {}
    for kind, idx in blocks:
        for j in idx:
            pos[j] = len(pos)
    for i in range(T):  # anticommuting pairs keep their order
        for j in range(i + 1, T):
            if pos[j] < pos[i] and not _st_commute(masks[i], masks[j]):
                return None
    return blocks


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


def _st_compile(qc, target, max_error, seed, call_limit, max_trials):
    """Item 68: the circuit compiled by STEINER6 with its layout, or None when the path does not apply (not one
    Pauli evolution, or its terms do not all commute) or its output would use a failed element (then the call goes on
    by the usual path)."""
    global _ST_FAILED
    from qiskit.circuit import QuantumRegister
    from qiskit.transpiler import Layout, TranspileLayout
    ev = _st_parse(qc)  # item 72
    ev = ev + ([], None) if ev is not None else _st_parse_gates(qc)  # items 73, 76
    if ev is None:
        STEINER_STATS["not_applicable"] += 1
        return None
    prefix, meas, triples, extra_phase, suffix, perm = ev
    perm = perm or list(range(qc.num_qubits))
    triples = _st_merge(triples)
    if len(triples) > _ST_MAX_TERMS:  # item 70
        STEINER_STATS["too_large"] += 1
        return None
    if not _st_all_commute(triples):  # item 71: tried all the same; counted
        STEINER_STATS["not_commuting"] += 1
    _ST_FAILED = max_error
    n, N = qc.num_qubits, target.num_qubits
    got = _st_construct(n, triples, target, meas)
    if got is None:
        STEINER_STATS["fallback"] += 1
        return None
    def finish(m, pi0):  # item 82: the circuit of one construction, or None if it uses a failed element
        phys = QuantumCircuit(N, qc.num_clbits)
        for op, q in prefix:  # item 72: the leading single-qubit gates, at the initial placement
            phys.append(op, [pi0[q]])
        at = list(range(N))  # at[p]: the physical qubit whose state is at p (SWAPs move idle qubits' states too)
        ops = m.ops
        if ST_SCHEDULE:  # item 83
            new = _st_reorder(m.ops, getattr(m, "st_blocks", None))
            if new is not None and _st_asap(new) < _st_asap(m.ops) - 1e-9:
                ops = new
                STEINER_STATS["reordered"] += 1
            else:
                STEINER_STATS["reorder_kept_order"] += 1
        for name, q, prm in ops:
            if name == "cx":
                phys.cx(*q)
            elif name == "swap":
                phys.swap(*q)
                at[q[0]], at[q[1]] = at[q[1]], at[q[0]]
            elif name == "rz":
                phys.rz(prm[0], q[0])
            else:
                getattr(phys, name)(q[0])
        phys.global_phase = extra_phase + m.gphase
        for op, q in suffix:  # item 73: the Clifford gates left at the end, at the final places
            phys.append(op, [m.pi[q]])
        for q, cb in meas:
            phys.measure(m.pi[perm[q]], cb)  # item 76: logical q's state is on wire perm[q]'s place
        with _DeterministicVF2(target):  # item 75
            out = transpile(phys, target=target, initial_layout=list(range(N)), routing_method="none",
                            optimization_level=3, seed_transpiler=seed)
        # Item 33's exact re-placement, as a renaming of the physical qubits
        dag = circuit_to_dag(out)
        vf2 = _DeterministicPostLayout(target, call_limit, max_trials, True)  # item 75
        vf2.run(dag)
        rel = list(range(N))
        if vf2.property_set["VF2PostLayout_stop_reason"] == VF2PostLayoutStopReason.SOLUTION_FOUND:
            post = vf2.property_set["post_layout"]
            rel = [post[dag.qubits[i]] for i in range(N)]
        final = QuantumCircuit(N, qc.num_clbits, global_phase=out.global_phase)
        for ins in out.data:
            final.append(ins.operation, [final.qubits[rel[out.find_bit(b).index]] for b in ins.qubits],
                         [final.clbits[out.find_bit(c).index] for c in ins.clbits])
        edges, bad = _failed_elements(target, max_error)
        if _uses_failed(final, edges, bad):
            return None
        pos = [0] * N  # pos[p]: where the state that started at physical p ends
        for p_now, p_start in enumerate(at):
            pos[p_start] = p_now
        if perm != list(range(n)):  # item 76: logical q started at pi0[q] and ends where wire perm[q]'s state ends
            ends = [pos[pi0[w]] for w in range(n)]
            for q in range(n):
                pos[pi0[q]] = ends[perm[q]]
        anc = QuantumRegister(N - n, "ancilla") if N > n else None
        virt = list(qc.qubits) + (list(anc) if anc is not None else [])
        used = set(pi0)
        init_phys = list(pi0) + [p for p in range(N) if p not in used]
        initial_layout = Layout({virt[k]: rel[init_phys[k]] for k in range(N)})
        for reg in qc.qregs:
            initial_layout.add_register(reg)
        if anc is not None:
            initial_layout.add_register(anc)
        final_layout = Layout({final.qubits[rel[p]]: rel[pos[p]] for p in range(N)})
        final._layout = TranspileLayout(initial_layout=initial_layout,
                                        input_qubit_mapping={virt[k]: k for k in range(N)},
                                        final_layout=final_layout, _input_qubit_count=qc.num_qubits,
                                        _output_qubit_list=list(final.qubits))
        return final

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
    if len(outs) == 2 and outs[1] is not None and (final is None or _better(outs[1], final, target)):  # item 84
        final = outs[1]
        STEINER_STATS["mirror_kept"] += 1
    STEINER_STATS["used"] += 1
    return final


def _failed_elements(target, max_error: float, gate_names=("cz", "ecr", "cx"), qubit_gate: str = "sx"):
    """(set of directed failed edges, set of failed qubits) as reported by `target` (changelog item 31)."""
    name = next((g for g in gate_names if g in target.operation_names), None)
    edges = set()
    if name is not None:
        for qargs, props in target[name].items():
            if qargs is not None and props is not None and props.error is not None and props.error >= max_error:
                edges.add(tuple(qargs))
    qubits = {q for q, e in qubit_errors_from_target(target, qubit_gate).items() if e >= max_error}
    qubits |= {q for q, e in readout_errors_from_target(target).items() if e >= max_error}  # item 58
    return edges, qubits


_SYMMETRIC_2Q = frozenset(("cz", "swap", "iswap", "rzz", "rxx", "ryy", "cp", "xx_plus_yy"))


def _uses_failed(circ, edges, qubits) -> bool:
    """Item 31 (direction-aware from item 41): a gate on a failed qubit, a two-qubit gate in a direction reported
    failed, or a symmetric two-qubit gate on a coupler failed in either direction."""
    pos = {q: i for i, q in enumerate(circ.qubits)}  # item 62: the same indices as find_bit, without its cost
    for inst in circ.data:
        idx = tuple(pos[q] for q in inst.qubits)
        if any(i in qubits for i in idx):
            return True
        if len(idx) == 2 and (idx in edges or (inst.operation.name in _SYMMETRIC_2Q and idx[::-1] in edges)):
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
    bad_q |= {q for q, e in readout_errors_from_target(target).items() if e >= max_error}  # item 58
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
        if ABSORB_SYNTH == "qiskit":  # item 62: Qiskit's two-qubit synthesis on CX, in Rust
            dec = TwoQubitBasisDecomposer(CXGate())
            done = {i: (dec(blocked.data[i].operation.to_matrix()),) for i in idx}
        else:
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


def _compile_for_hardware_c49(
    qc: QuantumCircuit,
    coupling_map: CouplingMap | None = None,
    basis_gates: list[str] | None = None,
    block_gate_floor: int = DEFAULT_BLOCK_GATE_FLOOR,
    routing_optimization_level: Union[int, str] = "auto",
    verify: Union[bool, str] = True,
    entangling_basis: str = "canonical",
    seed_transpiler: int | None = None,
    initial_layout: list[int] | None = None,
    on_unsupported: str = "keep",
    tol: float = 1e-5,
    layout_search: bool = False,
    layout_search_time_budget_s: float | None = None,
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
    placement_refine: Union[bool, str] = "auto",
    placement_call_limit: int = 300_000,
    placement_max_trials: int = 2_500,
    steiner: Union[bool, str] = "auto",
    final_resynthesis: Union[bool, str] = False,
    compare_level3: bool = False,
    compare_floor: bool = False,
    candidate_score: str = "excitation",
    _refine_target=None,
    _cancel_done: bool = False,
    backend=None,
    on_failed_elements: str = "raise",
    near_failed_error: float | None = 0.1,
    level3_candidate: Union[bool, str] = "auto",
    _inner: bool = False,
) -> QuantumCircuit:
    """Compress with PSF-Zero, then route (and, if `basis_gates` is given,
    translate) with Qiskit.

    `basis_gates` matters more than it looks: with only a `coupling_map`,
    `transpile()` lays out and routes but never targets a gate set, so the
    RXX/RYY/RZZ this pass emits pass straight through and the result is not
    ISA-submittable.

    `routing_optimization_level` defaults to "auto" (item 64): 3 when the device's Target is given, 1
    otherwise; what follows is about the call without a Target. It defaulted to 1, and earlier to 2, on the
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
    nothing. `candidate_score="hybrid"` (candidate 2026-10-04.c11, item 38) chooses by `hybrid_cost`;
    `candidate_score="kraus"` (candidate 2026-10-06.c15, item 42) by `kraus_cost`.
    `callback` (new, item 13): forwarded verbatim to the internal
    `transpile()` call below, unchanged from what plain `transpile(callback=
    ...)` accepts. `None` by default -- passing nothing here changes nothing
    about this function's behavior or cost. See item 13 in this file's
    changelog for why this exists.
    `backend`, `on_failed_elements` (candidate 2026-10-10.c31, item 58): `backend` supplies the device's Target, and
    from it `coupling_map` and `basis_gates` when they are not given; `coupling_map` may then be left out. With a
    Target, failed couplers and qubits (error >= `prune_max_error`, measurement included) are avoided as in item 31;
    when they cannot be, "raise" (default) raises `FailedElementsError` and "keep" returns the circuit with a warning.
    Without a Target, the first call in a process warns that they cannot be avoided.
    `placement_refine` (candidate 2026-10-10.c32, item 60): "auto" (default) is True when a Target is given and False
    otherwise.
    `steiner` (candidate 2026-10-11.c38, item 68): "auto" (default) is True when the device is given (`backend=` or
    `target=`) and neither `coupling_map` nor `initial_layout` is; a Pauli evolution whose terms all commute is then
    compiled by STEINER6's construction. False compiles every input as the release does.
    `near_failed_error` (prototype 2026-10-11.c42, item 74): with a Target, an output that uses an element with error
    in [`near_failed_error`, `prune_max_error`) is compared with the input compiled with those elements avoided too,
    and the better by ESP is returned. None turns it off.
    """
    if steiner == "auto":  # item 68: decided before the device fills in the coupling map
        steiner = (backend is not None or target is not None) and coupling_map is None and initial_layout is None
    if level3_candidate == "auto":  # item 80: as item 68's "auto"
        level3_candidate = (backend is not None or target is not None) and coupling_map is None and \
            initial_layout is None
    if backend is not None:  # item 58: the device, from which the Target, map and basis follow unless given
        if target is None:
            target = backend.target
    if (CUT_FAILED_QUBITS and not _inner and target is not None and coupling_map is None and initial_layout is None
            and _refine_target is None and layout_edge_errors is None and layout_qubit_errors is None):  # item 85
        out85 = _cut_failed_qubits(dict(locals()))
        if out85 is not None:
            return out85
        del out85  # item 50 copies this call's locals as its arguments
    if basis_gates is None and target is not None:  # item 79: also when the Target is given alone
        basis_gates = [g for g in target.operation_names if g in _BACKEND_BASIS]
    if coupling_map is None:
        if target is None:
            raise ValueError("compile_for_hardware needs coupling_map, target or backend (changelog item 58)")
        coupling_map = target.build_coupling_map()
    if routing_optimization_level == "auto":  # item 64: level 3 with the device's Target, level 1 without
        routing_optimization_level = 3 if target is not None and not MAP_FREE_DEFAULTS else 1  # item 90
    if on_failed_elements not in ("raise", "keep"):
        raise ValueError('on_failed_elements must be "raise" or "keep" (changelog item 58)')
    if target is None and not _inner and WARN_WITHOUT_TARGET and not _WARNED["no_target"]:
        _WARNED["no_target"] = True
        warnings.warn("compile_for_hardware was called without the device's Target (target= or backend=): couplers "
                      "and qubits the device reports as failed cannot be avoided, and a result that uses one is "
                      "noise (changelog item 58). This warning is given once per process.", UserWarning, stacklevel=2)
    call_args = dict(locals())  # item 50: this call's arguments, for the two runs of the pipeline
    if placement_refine == "auto":  # item 60: on with a Target, off without one
        placement_refine = target is not None and not MAP_FREE_DEFAULTS  # item 90
    if placement_refine and target is None:
        raise ValueError("placement_refine=True needs the device `target` (changelog item 33)")
    if steiner and target is None:
        raise ValueError("steiner=True needs the device `target` (changelog item 68)")
    if final_resynthesis not in (False, True, "select"):
        raise ValueError('final_resynthesis must be False, True or "select" (changelog item 35)')
    if final_resynthesis and target is None:
        raise ValueError("final_resynthesis needs the device `target` (changelog item 35)")
    if compare_level3 and target is None:
        raise ValueError("compare_level3=True needs the device `target` (changelog item 36)")
    if compare_floor and target is None:
        raise ValueError("compare_floor=True needs the device `target` (changelog item 37)")
    if candidate_score not in ("excitation", "pauli", "hybrid", "kraus"):
        raise ValueError('candidate_score must be "excitation", "pauli", "hybrid" or "kraus" (changelog items 37-38, 42)')
    if layout_search and initial_layout is not None:
        raise ValueError(
            "layout_search=True and an explicit initial_layout were both "
            "given -- ambiguous which should be used. Pass one or the other, "
            "not both."
        )

    # Candidate 2026-10-02.c3 (item 31): compile as without a target; recompile on the pruned coupling map
    # only if that result uses a failed coupler or qubit.
    if steiner and target is not None and not (final_resynthesis or compare_level3 or compare_floor):  # item 68
        st = _st_compile(qc, target, prune_max_error, seed_transpiler, placement_call_limit, placement_max_trials)
        if st is not None:  # item 69: kept only if its ESP on the device is above the usual path's
            try:
                usual = compile_for_hardware(**dict(call_args, steiner=False))
            except FailedElementsError:
                usual = None
            if usual is None or _better(st, usual, target):  # item 84
                STEINER_STATS["chosen"] += 1
                return st
            STEINER_STATS["kept_usual"] += 1
            return usual
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
                    _refine_target=target if placement_refine else None, _inner=True)
        edges, qubits = _failed_elements(target, prune_max_error)
        PRUNE_STATS["checked"] += 1
        if PRUNE_FIRST and (edges or qubits):  # item 61: one compile, on the map without the failed elements
            PRUNE_STATS["pruned_first"] += 1
            args["coupling_map"] = prune_coupling_map(coupling_map, target, prune_max_error)
            out = _recompile_pruned(args, None, edges, qubits, on_failed_elements)
            if out is None:  # "keep", and no placement on the pruned map: the full map, as item 43 did
                args["coupling_map"] = coupling_map
                out = compile_for_hardware(**args)
        else:
            out = compile_for_hardware(**args)
            if _uses_failed(out, edges, qubits):
                PRUNE_STATS["recompiled"] += 1
                args["coupling_map"] = prune_coupling_map(coupling_map, target, prune_max_error)
                out = _recompile_pruned(args, out, edges, qubits, on_failed_elements)  # item 58
        out = _resynthesise(out, final_resynthesis, target, prune_max_error)
        if not _inner:  # item 74
            out = _near_alternative(out, call_args, target, near_failed_error, prune_max_error)
        if not _inner and level3_candidate:  # item 80
            out = _level3_candidate(qc, out, target, prune_max_error, seed_transpiler)
        if not compare_floor and candidate_score == "excitation":
            if compare_level3 and not _checkable_logical(qc):  # item 45
                SKIP_STATS["level3"] += 1
            elif compare_level3:
                out = _compare_level3(qc, out, target, prune_max_error, seed_transpiler)
            return out
        # Item 37: up to three candidates, chosen by candidate_score.
        cands = [("psf", out)]
        if compare_floor and not _checkable_logical(qc):  # item 45
            SKIP_STATS["floor"] += 1
        elif compare_floor:
            fargs = dict(args, coupling_map=coupling_map, _refine_target=floor_aware_target(target))
            fo = compile_for_hardware(**fargs)
            if _uses_failed(fo, edges, qubits):
                fargs["coupling_map"] = prune_coupling_map(coupling_map, target, prune_max_error)
                fo = _recompile_pruned(fargs, fo)
            fo = _resynthesise(fo, final_resynthesis, target, prune_max_error)
            if not _acceptable(fo, target, prune_max_error):
                COMPARE_STATS["floor_refused"] += 1
            else:
                cands.append(("floor", fo))  # item 47: item 39's check is made in _choose_lazy, if needed
        if compare_level3 and not _checkable_logical(qc):  # item 45
            SKIP_STATS["level3"] += 1
        elif compare_level3:
            l3 = transpile(qc, target=target, optimization_level=3, seed_transpiler=seed_transpiler,
                           approximation_degree=1.0)
            if not _acceptable(l3, target, prune_max_error):
                COMPARE_STATS["level3_refused"] += 1
            else:
                cands.append(("level3", l3))  # item 47: item 39's check is made in _choose_lazy, if needed
        return _choose_lazy(qc, out, cands[1:], target, candidate_score)

    qc = _unroll_wide(qc, basis_gates)  # item 48
    if not _cancel_done:  # item 50
        alt = _cancel_candidate(qc)
        if alt is not None:
            CANCEL_STATS["tried"] += 1
            kw = dict(call_args, qc=qc, _cancel_done=True, _inner=True)
            first = compile_for_hardware(**kw)
            second = compile_for_hardware(**dict(kw, qc=alt))
            if _count_2q(second) < _count_2q(first):
                CANCEL_STATS["cancelled_kept"] += 1
                return second
            CANCEL_STATS["original_kept"] += 1
            return first

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
    qc_compressed = qc if not COMPRESS else compile(  # item 62: COMPRESS=False skips the compression
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
            time_budget_s=1e9 if layout_search_time_budget_s is None else layout_search_time_budget_s,  # 77
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
        with _DeterministicVF2(None):  # item 75: no Target, nothing to re-place by
            return transpile(
                qc_compressed,
                coupling_map=coupling_map,
                basis_gates=basis_gates,
                optimization_level=routing_optimization_level,
                seed_transpiler=seed_transpiler,
                initial_layout=initial_layout,
                callback=callback,
            )
    with _DeterministicVF2(_refine_target):  # item 75
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
        refine = [_DeterministicPostLayout(_refine_target, placement_call_limit, placement_max_trials, True),  # 75
                  _RecordRefine(),
                  ConditionalController(ApplyLayout(), condition=_refine_found)]
        if pm.optimization is None:
            pm.optimization = PassManager(refine)
        else:
            pm.optimization.append(refine)
    return pm.run(qc_compressed, callback=callback)


# ---------------------------------------------------------------------------------------------------------------------
# Prototype c50: item 86 (the device's native fractional gates: rzz and rx); c51: item 86 revised (every rzz folded)
# ---------------------------------------------------------------------------------------------------------------------

FRACTIONAL = "auto"  # item 86: "auto" re-synthesises the output with rzz / rx when the Target offers rzz; False: never
FRAC_STATS = {"tried": 0, "kept": 0, "longer": 0, "not_isa": 0, "not_exact": 0, "unverified": 0, "error": 0,
              "folded": 0, "fold_failed": 0}
_FRAC_DEPTH = [0]


def _errorless(target):
    """Item 86: a copy of `target` with every reported error removed (durations kept), so that Qiskit's synthesis
    passes cannot choose by the device's error table."""
    import copy
    from qiskit.transpiler import InstructionProperties
    t = copy.deepcopy(target)
    for name in t.operation_names:
        for qargs, p in list(t[name].items()):
            if qargs is not None and p is not None and p.error is not None:
                t.update_instruction_properties(name, qargs, InstructionProperties(duration=p.duration, error=None))
    return t


def _isa_ok(circ, target):
    pos = {q: i for i, q in enumerate(circ.qubits)}
    names = set(target.operation_names)
    for ins in circ.data:
        nm = ins.operation.name
        if nm == "barrier":
            continue
        if nm not in names:
            return False
        qargs = tuple(pos[q] for q in ins.qubits)
        if qargs not in target[nm] and None not in target[nm]:
            return False
    return True


def _fractional(out, target):
    """Item 86: `out` re-synthesised on the same physical qubits with the device's native fractional gates, or None.

    Two-qubit blocks are consolidated and re-synthesised (Qiskit's ConsolidateBlocks and UnitarySynthesis on an
    error-free copy of the Target, so no reported error chooses), single-qubit runs are merged (rx is one pulse), rzz
    angles are folded into IBM's executable range 0 < theta <= pi/2 (qiskit-ibm-runtime's FoldRzzAngle). The placement
    and the layout are those of `out`. None when qiskit-ibm-runtime's pass is unavailable or the result is not native
    to the Target."""
    try:
        from qiskit.circuit.library import GlobalPhaseGate  # noqa: F401
        from qiskit.transpiler.passes import (Collect2qBlocks, ConsolidateBlocks, Optimize1qGatesDecomposition,
                                              UnitarySynthesis)
        from qiskit_ibm_runtime.transpiler.passes import FoldRzzAngle
    except Exception:  # noqa: BLE001 - the item needs qiskit-ibm-runtime's fold; without it nothing changes
        FRAC_STATS["error"] += 1
        return None
    nt = _errorless(target)
    pm = PassManager([Collect2qBlocks(), ConsolidateBlocks(target=nt), UnitarySynthesis(target=nt),
                      Optimize1qGatesDecomposition(target=nt), FoldRzzAngle(),
                      Optimize1qGatesDecomposition(target=nt)])
    try:
        new = pm.run(out)
    except Exception:  # noqa: BLE001
        FRAC_STATS["error"] += 1
        return None
    new = _strip_global_phase_gates(new)  # FoldRzzAngle may leave a GlobalPhaseGate: carried into the phase
    new._layout = getattr(out, "_layout", None)
    if not _isa_ok(new, target) or not _rzz_in_range(new):
        FRAC_STATS["not_isa"] += 1
        return None
    return new


def _strip_global_phase_gates(new):
    phase, kept = 0.0, []
    for ins in new.data:
        if ins.operation.name == "global_phase":
            phase += float(ins.operation.params[0])
        else:
            kept.append(ins)
    if len(kept) != len(new.data):
        new.data = kept
        new.global_phase += phase
    return new


def _rzz_in_range(circ):
    return all(0 < float(i.operation.params[0]) <= math.pi / 2 + 1e-12 for i in circ.data if i.operation.name == "rzz")


def _fold(out, target):
    """Item 86 revised (c51): `out` with every rzz angle folded into IBM's executable range 0 < theta <= pi/2
    (qiskit-ibm-runtime's FoldRzzAngle, then single-qubit runs merged on an error-free copy of the Target). Qiskit's
    own stages inside the compiler emit rzz when the Target offers it, with any angle; only a backend object of
    qiskit-ibm-runtime folds them by itself. Raises TranspilerError if an angle stays outside the range."""
    from qiskit.transpiler.passes import Optimize1qGatesDecomposition
    from qiskit_ibm_runtime.transpiler.passes import FoldRzzAngle
    new = PassManager([FoldRzzAngle(), Optimize1qGatesDecomposition(target=_errorless(target))]).run(out)
    new = _strip_global_phase_gates(new)
    new._layout = getattr(out, "_layout", None)
    if not _rzz_in_range(new):
        FRAC_STATS["fold_failed"] += 1
        raise TranspilerError("an rzz angle is outside 0 < theta <= pi/2 after folding (changelog item 86)")
    FRAC_STATS["folded"] += 1
    return new


def _fractional_choice(out, target):
    """Item 86: the re-synthesised circuit when it is shorter (item 84's `_better`) and, where item 39's check can be
    made, acts as `out` does; otherwise `out`."""
    if target is None or "rzz" not in target.operation_names:
        return out
    if not _rzz_in_range(out):  # c51: whatever is returned on a Target with rzz is executable
        out = _fold(out, target)
    if FRACTIONAL is False:
        return out
    FRAC_STATS["tried"] += 1
    new = _fractional(out, target)
    if new is None:
        return out
    if not _better(new, out, target):
        FRAC_STATS["longer"] += 1
        return out
    a = out.remove_final_measurements(inplace=False)
    b = new.remove_final_measurements(inplace=False)
    if len(_touched_qubits(a) | _touched_qubits(b)) <= RESYNTH_MAX_QUBITS:
        if not _same_action(a, b):
            FRAC_STATS["not_exact"] += 1
            return out
    else:
        FRAC_STATS["unverified"] += 1
    FRAC_STATS["kept"] += 1
    return new


def _compile_for_hardware_c51(qc, *args, **kwargs):
    """Item 86 (prototype c50): `_compile_for_hardware_c49` (the compiler as in c49, unchanged), then, on the outermost
    call only, the device's native fractional gates where they shorten the circuit (`_fractional_choice`).

    """
    _FRAC_DEPTH[0] += 1
    try:
        out = _compile_for_hardware_c49(qc, *args, **kwargs)
    finally:
        _FRAC_DEPTH[0] -= 1
    if _FRAC_DEPTH[0] == 0 and not kwargs.get("_inner", False):
        target = kwargs.get("target")
        if target is None and kwargs.get("backend") is not None:
            target = kwargs["backend"].target
        out = _fractional_choice(out, target)
    return out


_compile_for_hardware_c51.__doc__ += _compile_for_hardware_c49.__doc__ or ""


# ---------------------------------------------------------------------------------------------------------------------
# Workplace proposal on prototype c51 (2026-10-14): item 87, a chain Hamiltonian built by formula as a brickwork (N1);
# item 88, on a Target with rzz and rx, its terms emitted as rzz / rx with the signs carried in a Pauli X frame
# ---------------------------------------------------------------------------------------------------------------------

import inspect as _inspect  # noqa: E402  (item 87)

CHAIN_BRICKWORK = True  # item 87 (proposal): chain Pauli evolutions built as a brickwork on a path of the device
CHAIN_FRAME = True  # item 88 (proposal): with rzz and rx in the Target, the chain's terms as rzz / rx, signs in a frame
CHAIN_STATS = {"tried": 0, "not_chain": 0, "not_cz": 0, "no_path": 0, "snake": 0, "vf2_path": 0,
               "common_length": 0, "used": 0, "failed_check": 0, "kept": 0, "kept_default": 0,
               "frame": 0, "frame_x_end": 0, "frame_folded": 0}
_CHAIN_DEPTH = [0]  # item 87: set while c51's own output is compiled for the comparison (no item 87 inside it)
_CHAIN_VF2_CALLS = 5_000_000


def _chain_support(lb):
    n = len(lb)
    return [(n - 1 - i, ch) for i, ch in enumerate(lb) if ch != "I"]


def _chain_commute(a, b):
    sa, sb = dict(_chain_support(a)), dict(_chain_support(b))
    return sum(1 for q in set(sa) & set(sb) if sa[q] != sb[q]) % 2 == 0


def _chain_read(qc):
    """Item 87: (prefix, meas, terms [(label, angle)], phase, suffix, perm, path) when every term (read by items
    72-73, merged where exact by `_st_merge`) acts on one or two qubits and the two-qubit supports form one path
    through all the circuit's qubits; else None."""
    ev = _st_parse(qc)
    ev = ev + ([], None) if ev is not None else _st_parse_gates(qc)
    if ev is None:
        return None
    prefix, meas, triples, phase, suffix, perm = ev
    terms = [(lb, a * t) for lb, a, t in _st_merge(triples)]
    n = qc.num_qubits
    adj = {q: set() for q in range(n)}
    for lb, _ in terms:
        sup = [q for q, _ in _chain_support(lb)]
        if len(sup) > 2:
            return None
        if len(sup) == 2:
            adj[sup[0]].add(sup[1])
            adj[sup[1]].add(sup[0])
    if any(len(v) > 2 for v in adj.values()):
        return None
    ends = [q for q in range(n) if len(adj[q]) == 1]
    if n < 2 or len(ends) != 2:
        return None
    path, prev, cur = [min(ends)], None, min(ends)
    while True:
        nxt = [v for v in adj[cur] if v != prev]
        if not nxt:
            break
        prev, cur = cur, nxt[0]
        path.append(cur)
    if len(path) != n:
        return None
    return prefix, meas, terms, phase, suffix, perm or list(range(n)), path


def _chain_lattice(target, max_error):
    """Item 87: (K, its adjacency, the adjacency restricted to couplers of the device's most common two-qubit gate
    length, the snake path) -- W5's cut: qubits with sx or readout error >= max_error and couplers with error >=
    max_error removed, the largest connected component kept (ties: the smallest qubit)."""
    N = target.num_qubits
    err, dur = {}, {}
    for qa, pr in target["cz"].items():
        if qa is None:
            continue
        k = (min(qa), max(qa))
        err[k] = max(err.get(k, 0.0), pr.error if pr is not None and pr.error is not None else 0.0)
        dur[k] = max(dur.get(k, 0.0), pr.duration if pr is not None and pr.duration is not None else 0.0)
    adj = {q: set() for q in range(N)}
    for a, b in err:
        adj[a].add(b)
        adj[b].add(a)
    bad = set()
    for name in ("sx", "measure"):
        if name in target.operation_names:
            for qa, pr in target[name].items():
                if qa is not None and pr is not None and pr.error is not None and pr.error >= max_error:
                    bad.add(qa[0])
    gadj = {q: set() for q in range(N) if q not in bad}
    for (a, b), e in err.items():
        if e < max_error and a not in bad and b not in bad:
            gadj[a].add(b)
            gadj[b].add(a)
    seen, comps = set(), []
    for q in sorted(gadj):
        if q in seen:
            continue
        comp, fr = {q}, [q]
        while fr:
            u = fr.pop()
            for v in gadj[u]:
                if v not in comp:
                    comp.add(v)
                    fr.append(v)
        seen |= comp
        comps.append(comp)
    if not comps:
        return None
    K = max(comps, key=lambda c: (len(c), -min(c)))
    kadj = {q: {v for v in gadj[q] if v in K} for q in K}
    lens = {}
    for a in K:
        for b in kadj[a]:
            if a < b:
                k = round(dur[(a, b)] * 1e12)
                lens[k] = lens.get(k, 0) + 1
    modal = max(lens, key=lambda x: (lens[x], -x)) if lens else None
    fadj = {q: {v for v in kadj[q] if round(dur[(min(q, v), max(q, v))] * 1e12) == modal} for q in K}
    return K, kadj, fadj, _chain_snake(N, adj)


def _chain_snake(N, adj):
    """Item 87: Heron's snake (W5): rows of consecutive qubits joined by couplers, row 0 left to right, the bridge
    at the far end down, the next row back, and so on. A lattice that is not read as rows gives a shorter path."""
    rows, cur = [], [0]
    for q in range(1, N):
        if q in adj[q - 1]:
            cur.append(q)
        else:
            rows.append(cur)
            cur = [q]
    rows.append(cur)
    rows = [r for r in rows if len(r) > 2]
    inrow = {q: (i, j) for i, r in enumerate(rows) for j, q in enumerate(r)}
    path, col = [], 0
    for i, r in enumerate(rows):
        leftward = i % 2 == 1
        if i + 1 < len(rows):
            br = [(inrow[a][1], b) for b in range(N) if b not in inrow for a in adj[b]
                  if a in inrow and inrow[a][0] == i and any(c in inrow and inrow[c][0] == i + 1 for c in adj[b])]
            if not br:
                return path + list(r)
            exit_col, bq = (min(br) if leftward else max(br))
        else:
            exit_col, bq = (0 if leftward else len(r) - 1), None
        step = -1 if leftward else 1
        start = col if i else 0
        path += [r[j] for j in range(start, exit_col + step, step)]
        if bq is not None:
            path.append(bq)
            below = next(c for c in adj[bq] if c in inrow and inrow[c][0] == i + 1)
            col = inrow[below][1]
    return path


def _chain_place(n, K, kadj, snake):
    """Item 87: n physical qubits forming a path in K: the longest surviving snake piece, centred; else the first
    path rustworkx's VF2 finds with nodes in index order (no score). (path, how) or (None, None)."""
    pieces, cur = [], []
    for q in snake:
        if q in K and (not cur or q in kadj[cur[-1]]):
            cur.append(q)
        else:
            if cur:
                pieces.append(cur)
            cur = [q] if q in K else []
    if cur:
        pieces.append(cur)
    if pieces:
        best = max(pieces, key=lambda p: (len(p), -snake.index(p[0])))
        if len(best) >= n:
            s = (len(best) - n) // 2
            return best[s:s + n], "snake"
    import rustworkx as rx
    kn = sorted(K)
    ki = {q: i for i, q in enumerate(kn)}
    G = rx.PyGraph()
    G.add_nodes_from(kn)
    G.add_edges_from_no_data([(ki[a], ki[b]) for a in kn for b in kadj[a] if a < b])
    P = rx.PyGraph()
    P.add_nodes_from(range(n))
    P.add_edges_from_no_data([(i, i + 1) for i in range(n - 1)])
    m = next(rx.vf2_mapping(G, P, subgraph=True, induced=False, id_order=True, call_limit=_CHAIN_VF2_CALLS), None)
    if m is None:
        return None, None
    inv = {h: kn[g] for g, h in m.items()}
    return [inv[i] for i in range(n)], "vf2_path"





def _chain_rotation(qc, lb, a, frame=False):
    """Item 87: exp(-i a P) on the circuit's own qubits; XX, YY, ZZ as Qiskit's rxx, ryy, rzz. Item 88 (`frame`):
    every two-qubit term as rzz between single-qubit basis changes (X: h; Y: rx(pi/2) before, rx(-pi/2) after), so
    that the device's one-pulse rzz carries it; single-qubit terms as rx / ry / rz."""
    sup = _chain_support(lb)
    if len(sup) == 1:
        q, ch = sup[0]
        getattr(qc, {"X": "rx", "Y": "ry", "Z": "rz"}[ch])(2 * a, q)
        return
    (q1, c1), (q2, c2) = sup
    if c1 == c2 and not frame:
        getattr(qc, {"X": "rxx", "Y": "ryy", "Z": "rzz"}[c1])(2 * a, q1, q2)
        return
    for q, ch in ((q1, c1), (q2, c2)):
        if ch == "X":
            qc.h(q)
        elif ch == "Y":
            qc.rx(math.pi / 2, q)
    if frame:
        qc.rzz(2 * a, q1, q2)
    else:
        qc.cx(q1, q2)
        qc.rz(2 * a, q2)
        qc.cx(q1, q2)
    for q, ch in ((q1, c1), (q2, c2)):
        if ch == "X":
            qc.h(q)
        elif ch == "Y":
            qc.rx(-math.pi / 2, q)


def _chain_frame(log):
    """Item 88: `log` (single-qubit gates, rzz with any angle, measurements) rewritten so that every rzz angle lies in
    IBM's executable range 0 < theta <= pi/2, with the same action (unitary, and outcome distribution when measured).

    Each angle is reduced to (-pi, pi]; an angle of size above pi/2 is moved by pi with Z x Z (rz(pi) on both qubits,
    the phase kept). A negative angle (after the frame's sign, below) is made positive by one x on the rzz's first
    qubit before it, merged into the single-qubit run before it; the x that would follow it is not placed but kept in
    a Pauli X frame: a single-qubit gate G on a qubit in the frame is placed as X G X (merged into one pulse with its
    run), an rzz with exactly one of its qubits in the frame has its sign flipped, and the frame is undone inside the
    circuit -- one x before the qubit's measurement, or at the end of an unmeasured qubit, merged into the last
    single-qubit gate where there is one. No measured bit is flipped afterwards: the circuit itself outputs the
    input's distribution, and acts as the input's unitary when it is not measured."""
    n = log.num_qubits
    out = QuantumCircuit(*log.qregs, *log.cregs, global_phase=log.global_phase)
    f = [0] * n
    for ins in log.data:
        nm = ins.operation.name
        qs = [log.find_bit(q).index for q in ins.qubits]
        if nm == "barrier":
            out.append(ins.operation, qs)
            continue
        if nm == "measure":
            q = qs[0]
            if f[q]:
                out.x(q)
                f[q] = 0
            out.measure(q, log.find_bit(ins.clbits[0]).index)
            continue
        if nm == "rzz":
            a, b = qs
            th = float(ins.operation.params[0])
            th = math.pi - (math.pi - th) % (2 * math.pi)  # (-pi, pi]
            if abs(th) > math.pi / 2:  # rzz(th) = rzz(th - sign(th) pi) (rz(pi) x rz(pi)) e^{i pi/2} for th > 0
                out.rz(math.pi, a)
                out.rz(math.pi, b)
                out.global_phase += math.copysign(math.pi / 2, th) + math.pi * (f[a] + f[b])  # X rz(pi) X = -rz(pi)
                th -= math.copysign(math.pi, th)
                CHAIN_STATS["frame_folded"] += 1
            if abs(th) < 1e-12:
                continue
            eff = -th if f[a] != f[b] else th
            if eff < 0:
                out.x(a)
                f[a] ^= 1
                eff = -eff
            out.rzz(eff, a, b)
            continue
        if len(qs) != 1:
            raise TranspilerError(f"item 88: unexpected {nm} on {len(qs)} qubits")
        q = qs[0]
        if f[q]:
            out.x(q)
        out.append(ins.operation, [q])
        if f[q]:
            out.x(q)
    for q in range(n):
        if f[q]:
            out.x(q)
            CHAIN_STATS["frame_x_end"] += 1
    return out


def _chain_brickwork(qc, target, max_error, seed):
    """Item 87 (workplace proposal, N1): a Pauli evolution whose two-qubit terms form one chain, compiled by formula.
    The chain goes on a path of the device (`_chain_place`: first on couplers of the device's most common two-qubit
    gate length, then on all of the cut device); every term gets a level, 1 + the largest level of an earlier term
    that shares a qubit and does not commute with it; terms are emitted by level, two-qubit terms on even edges of the
    chain, then odd edges, then single-qubit terms (every non-commuting pair keeps its order: exact). Without rzz in
    the Target, Qiskit's level-3 preset then translates on that fixed layout without routing, given the coupling map
    and basis only (no error value is read). Item 88: with rzz and rx in the Target (`CHAIN_FRAME`), the terms are
    emitted as rzz between basis changes, diagonal gates directly before measurements are dropped (Qiskit's
    RemoveDiagonalGatesBeforeMeasure), the signs are carried in a Pauli X frame (`_chain_frame`), Qiskit's level-0
    preset places it on the fixed layout, and every single-qubit run is merged into rz rx rz (one pulse); an rzz angle
    outside 0 < theta <= pi/2 would go through item 86's `_fold`. None when the input is not such a chain, the
    device's two-qubit gate is not cz, no path of n qubits exists, or the output would use a failed element."""
    CHAIN_STATS["tried"] += 1
    if "cz" not in target.operation_names:
        CHAIN_STATS["not_cz"] += 1
        return None
    got = _chain_read(qc)
    if got is None:
        CHAIN_STATS["not_chain"] += 1
        return None
    prefix, meas, terms, phase, suffix, perm, path = got
    n, N = qc.num_qubits, target.num_qubits
    lat = _chain_lattice(target, max_error)
    if lat is None:
        CHAIN_STATS["no_path"] += 1
        return None
    K, kadj, fadj, snake = lat
    phys, how = _chain_place(n, K, fadj, snake)
    if phys is not None:
        CHAIN_STATS["common_length"] += 1
    else:
        phys, how = _chain_place(n, K, kadj, snake)
    if phys is None:
        CHAIN_STATS["no_path"] += 1
        return None
    CHAIN_STATS[how] += 1
    rank = {path[i]: i for i in range(n)}
    level, last = [], {}
    for k, (lb, _) in enumerate(terms):
        lv = 0
        for q, _ in _chain_support(lb):
            for j, lb2 in last.get(q, ()):
                if not _chain_commute(lb, lb2):
                    lv = max(lv, level[j] + 1)
        level.append(lv)
        for q, _ in _chain_support(lb):
            last.setdefault(q, []).append((k, lb))

    def key(k):
        sup = [q for q, _ in _chain_support(terms[k][0])]
        if len(sup) == 2:
            lo = min(rank[q] for q in sup)
            return (level[k], 0, lo % 2, lo, k)
        return (level[k], 1, 0, rank[sup[0]], k)
    frame = CHAIN_FRAME and "rzz" in target.operation_names and "rx" in target.operation_names
    log = QuantumCircuit(*qc.qregs, *qc.cregs, global_phase=phase)
    for op, q in prefix:
        log.append(op, [q])
    for k in sorted(range(len(terms)), key=key):
        _chain_rotation(log, *terms[k], frame=frame)
    for op, q in suffix:
        log.append(op, [q])
    for q, cb in meas:
        log.measure(perm[q], cb)  # item 76: logical q's state is on wire perm[q]
    cm = CouplingMap()
    for q in range(N):
        cm.add_physical_qubit(q)
    for a in sorted(K):
        for b in sorted(kadj[a]):
            cm.add_edge(a, b)
    layout = [phys[rank[q]] for q in range(n)]
    if frame:
        from qiskit.transpiler.passes import Optimize1qGatesDecomposition, RemoveDiagonalGatesBeforeMeasure
        log = PassManager([RemoveDiagonalGatesBeforeMeasure()]).run(log)
        low = _chain_frame(log)
        pm = generate_preset_pass_manager(0, coupling_map=cm, basis_gates=["rzz", "rx", "rz", "sx", "x", "cz"],
                                          initial_layout=layout, routing_method="none", seed_transpiler=seed)
        placed = pm.run(low)
        out = PassManager([Optimize1qGatesDecomposition(basis=["rz", "rx"])]).run(placed)
        out._layout = placed._layout
        if not _rzz_in_range(out):
            out = _fold(out, target)
        CHAIN_STATS["frame"] += 1
    else:
        pm = generate_preset_pass_manager(3, coupling_map=cm, basis_gates=["cz", "rz", "sx", "x"],
                                          initial_layout=layout, routing_method="none", seed_transpiler=seed)
        out = pm.run(log)
    edges, bad = _failed_elements(target, max_error)
    if _uses_failed(out, edges, bad):
        CHAIN_STATS["failed_check"] += 1
        return None
    CHAIN_STATS["used"] += 1
    return out


_C49_SIGNATURE = _inspect.signature(_compile_for_hardware_c49)


def compile_for_hardware(qc, *args, **kwargs):
    """Items 87-88 (workplace proposal on prototype c51): with the device given (`steiner` "auto" true) and no
    `coupling_map`, `initial_layout`, `final_resynthesis`, `compare_level3` or `compare_floor` of the caller's, a chain
    Pauli evolution is built by formula (`_chain_brickwork`) and compared with c51's own output for the same call
    (`_compile_for_hardware_c51`: c49, then item 86) by item 84's rule (`_better`; a tie keeps the brickwork). Any
    other input, and every nested call, is compiled exactly as by c51.

    """
    if _CHAIN_DEPTH[0] or _FRAC_DEPTH[0] or not CHAIN_BRICKWORK:
        return _compile_for_hardware_c51(qc, *args, **kwargs)
    try:
        bound = _C49_SIGNATURE.bind(qc, *args, **kwargs)
    except TypeError:
        return _compile_for_hardware_c51(qc, *args, **kwargs)
    bound.apply_defaults()
    a = bound.arguments
    target = a["target"] if a["target"] is not None else getattr(a["backend"], "target", None)
    steiner = a["steiner"]
    if steiner == "auto":  # item 68's "auto"
        steiner = (a["backend"] is not None or a["target"] is not None) and a["coupling_map"] is None and \
            a["initial_layout"] is None
    out87 = None
    if (steiner and not a["_inner"] and target is not None and a["coupling_map"] is None
            and a["initial_layout"] is None
            and not (a["final_resynthesis"] or a["compare_level3"] or a["compare_floor"])):
        out87 = _chain_brickwork(qc, target, a["prune_max_error"], a["seed_transpiler"])
        if out87 is None and GRAPH_BRICKWORK:  # item 91
            out87 = _graph_brickwork(qc, target, a["prune_max_error"], a["seed_transpiler"])
    if out87 is None:
        return _compile_for_hardware_c51(qc, *args, **kwargs)
    _CHAIN_DEPTH[0] += 1
    try:
        own = _compile_for_hardware_c51(qc, *args, **kwargs)
    except FailedElementsError:
        own = None
    finally:
        _CHAIN_DEPTH[0] -= 1
    if own is not None and _better(own, out87, target):
        CHAIN_STATS["kept_default"] += 1
        return own
    CHAIN_STATS["kept"] += 1
    return out87


compile_for_hardware.__doc__ += _compile_for_hardware_c49.__doc__ or ""


# ---------------------------------------------------------------------------------------------------------------------
# Workplace proposal on c53 (2026-10-14): item 91, a Pauli evolution whose interaction graph embeds in the device,
# built by formula: first VF2 embedding (no score), edges coloured into at most 3 matchings (Koenig), one layer per
# colour
# ---------------------------------------------------------------------------------------------------------------------

GRAPH_BRICKWORK = True  # item 91 (proposal): embeddable interaction graphs built by edge colouring
GRAPH_STATS = {"tried": 0, "not_read": 0, "degree": 0, "not_cz": 0, "no_embedding": 0, "common_length": 0,
               "used": 0, "colours": {}, "failed_check": 0, "frame": 0}
_GRAPH_VF2_CALLS = 2_000_000


def _graph_read(qc):
    """Item 91: (prefix, meas, terms [(label, angle)], phase, suffix, perm, edges) when every term (items 72-73's
    reader, merged where exact) acts on one or two qubits; edges: the sorted two-qubit supports. Else None."""
    ev = _st_parse(qc)
    ev = ev + ([], None) if ev is not None else _st_parse_gates(qc)
    if ev is None:
        return None
    prefix, meas, triples, phase, suffix, perm = ev
    terms = [(lb, a * t) for lb, a, t in _st_merge(triples)]
    edges = set()
    for lb, _ in terms:
        sup = sorted(q for q, _ in _chain_support(lb))
        if len(sup) > 2:
            return None
        if len(sup) == 2:
            edges.add((sup[0], sup[1]))
    if not edges:
        return None
    return prefix, meas, terms, phase, suffix, perm or list(range(qc.num_qubits)), sorted(edges)


def _graph_colours(edges):
    """Item 91: a proper edge colouring of a bipartite graph with max-degree colours (Koenig), deterministic: edges in
    sorted order; an edge (u, v) takes the smallest colour a free at u; if a is used at v, the a/b path from v (b the
    smallest colour free at v) is swapped first. {edge: colour} or None when the graph is not bipartite."""
    deg = {}
    for u, v in edges:
        deg[u] = deg.get(u, 0) + 1
        deg[v] = deg.get(v, 0) + 1
    D = max(deg.values())
    at = {x: {} for x in deg}  # node -> {colour: neighbour}
    for u, v in edges:
        a = min(c for c in range(D) if c not in at[u])
        if a in at[v]:
            b = min(c for c in range(D) if c not in at[v])
            path, x, c, seen = [], v, a, {v}
            while c in at[x]:
                y = at[x][c]
                path.append((x, y, c))
                if y in seen or y == u:
                    return None  # an odd cycle: not bipartite
                seen.add(y)
                x, c = y, (b if c == a else a)
            for x, y, c in path:
                del at[x][c]
                del at[y][c]
            for x, y, c in path:
                d = b if c == a else a
                at[x][d] = y
                at[y][d] = x
        at[u][a] = v
        at[v][a] = u
    col = {}
    for x, m in at.items():
        for c, y in m.items():
            col[(min(x, y), max(x, y))] = c
    return col


def _graph_place(n, edges, K, kadj):
    """Item 91: the first VF2 embedding (rustworkx, nodes in index order, no score) of the interaction graph into the
    cut device K: {logical qubit: physical qubit} or None."""
    import rustworkx as rx
    kn = sorted(K)
    ki = {q: i for i, q in enumerate(kn)}
    G = rx.PyGraph()
    G.add_nodes_from(kn)
    G.add_edges_from_no_data([(ki[a], ki[b]) for a in kn for b in kadj[a] if a < b])
    P = rx.PyGraph()
    P.add_nodes_from(range(n))
    P.add_edges_from_no_data(edges)
    if n > len(kn):
        return None
    m = next(rx.vf2_mapping(G, P, subgraph=True, induced=False, id_order=True, call_limit=_GRAPH_VF2_CALLS), None)
    if m is None:
        return None
    return {h: kn[g] for g, h in m.items()}


def _graph_brickwork(qc, target, max_error, seed):
    """Item 91 (workplace proposal on c53): a Pauli evolution whose terms act on one or two qubits and whose
    interaction graph embeds in the cut device (max degree <= 3, bipartite), compiled by formula. Placement: the
    first VF2 embedding, tried first on the couplers of the device's most common two-qubit gate length, then on all
    of the cut device (item 87's lattice). Order: every term at level 1 + the largest level of an earlier term that
    shares a qubit and does not commute with it (exact); within a level the two-qubit terms by colour of a proper
    edge colouring with max-degree colours (each colour a matching: one layer), then the single-qubit terms.
    Emission as item 87 (Qiskit level 3 on the fixed layout, CZ) or, with rzz and rx in the Target, item 88 (rzz,
    rx, Pauli frame undone inside the circuit). None when not applicable."""
    GRAPH_STATS["tried"] += 1
    if "cz" not in target.operation_names:
        GRAPH_STATS["not_cz"] += 1
        return None
    got = _graph_read(qc)
    if got is None:
        GRAPH_STATS["not_read"] += 1
        return None
    prefix, meas, terms, phase, suffix, perm, edges = got
    n, N = qc.num_qubits, target.num_qubits
    deg = {}
    for u, v in edges:
        deg[u] = deg.get(u, 0) + 1
        deg[v] = deg.get(v, 0) + 1
    if max(deg.values()) > 3:
        GRAPH_STATS["degree"] += 1
        return None
    col = _graph_colours(edges)
    if col is None:
        GRAPH_STATS["no_embedding"] += 1
        return None
    lat = _chain_lattice(target, max_error)
    if lat is None:
        GRAPH_STATS["no_embedding"] += 1
        return None
    K, kadj, fadj, _ = lat
    place = _graph_place(n, edges, K, fadj)
    if place is not None:
        GRAPH_STATS["common_length"] += 1
    else:
        place = _graph_place(n, edges, K, kadj)
    if place is None:
        GRAPH_STATS["no_embedding"] += 1
        return None
    ncol = max(col.values()) + 1
    GRAPH_STATS["colours"][ncol] = GRAPH_STATS["colours"].get(ncol, 0) + 1
    level, last = [], {}
    for k, (lb, _) in enumerate(terms):
        lv = 0
        for q, _ in _chain_support(lb):
            for j, lb2 in last.get(q, ()):
                if not _chain_commute(lb, lb2):
                    lv = max(lv, level[j] + 1)
        level.append(lv)
        for q, _ in _chain_support(lb):
            last.setdefault(q, []).append((k, lb))

    def key(k):
        sup = sorted(q for q, _ in _chain_support(terms[k][0]))
        if len(sup) == 2:
            return (level[k], 0, col[(sup[0], sup[1])], k)
        return (level[k], 1, 0, k)
    frame = CHAIN_FRAME and "rzz" in target.operation_names and "rx" in target.operation_names
    log = QuantumCircuit(*qc.qregs, *qc.cregs, global_phase=phase)
    for op, q in prefix:
        log.append(op, [q])
    for k in sorted(range(len(terms)), key=key):
        _chain_rotation(log, *terms[k], frame=frame)
    for op, q in suffix:
        log.append(op, [q])
    for q, cb in meas:
        log.measure(perm[q], cb)  # item 76
    cm = CouplingMap()
    for q in range(N):
        cm.add_physical_qubit(q)
    for a in sorted(K):
        for b in sorted(kadj[a]):
            cm.add_edge(a, b)
    used = set(place.values())
    free = [q for q in sorted(K) if q not in used]
    layout = [place[q] if q in place else free.pop(0) for q in range(n)]  # qubits without a term: unused places
    if frame:
        from qiskit.transpiler.passes import Optimize1qGatesDecomposition, RemoveDiagonalGatesBeforeMeasure
        log = PassManager([RemoveDiagonalGatesBeforeMeasure()]).run(log)
        low = _chain_frame(log)
        pm = generate_preset_pass_manager(0, coupling_map=cm, basis_gates=["rzz", "rx", "rz", "sx", "x", "cz"],
                                          initial_layout=layout, routing_method="none", seed_transpiler=seed)
        placed = pm.run(low)
        out = PassManager([Optimize1qGatesDecomposition(basis=["rz", "rx"])]).run(placed)
        out._layout = placed._layout
        if not _rzz_in_range(out):
            out = _fold(out, target)
        GRAPH_STATS["frame"] += 1
    else:
        pm = generate_preset_pass_manager(3, coupling_map=cm, basis_gates=["cz", "rz", "sx", "x"],
                                          initial_layout=layout, routing_method="none", seed_transpiler=seed)
        out = pm.run(log)
    edges_f, bad = _failed_elements(target, max_error)
    if _uses_failed(out, edges_f, bad):
        GRAPH_STATS["failed_check"] += 1
        return None
    GRAPH_STATS["used"] += 1
    return out
