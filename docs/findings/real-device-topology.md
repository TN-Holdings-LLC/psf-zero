> **Archived record — part 4 of 6.**
> This file is one slice of the original PSF-Zero README, kept **verbatim**. It is a
> lab notebook: it contains claims that were later retracted, hypotheses that were
> later refuted, and corrections written directly underneath the text they correct.
> Nothing has been deleted. For what the project currently claims, read the
> [README](../../README.md); for a settled account of one topic, read
> [`docs/findings/`](../findings/).
>
> Sections 5-6 - coupling-map-constrained gate count and depth, the `routing_optimization_level` 2 -> 1 decision, the spare-qubit cliff from first correlation through controlled experiment and hardware reproduction, and the Benchpress sanity check.
>
>
> **One edit was made to this file when it was archived:** file-relative image paths
> were rewritten from `./docs/...` to `../../docs/...` for the new directory depth.
> Root-relative paths (`/docs/...`) were left untouched — they resolve correctly at
> any depth. No wording, number, claim or correction was altered. The 13 substitutions
> are listed in [`README.md`](README.md#the-one-edit).
> Source: README.md lines 1470-1952, as of 2026-09-11.

---

### 5. Real-device topology (coupling-map-constrained)

We ran a coupling-map-constrained comparison (50–500 qubits, grid topology)
across all three `routing_optimization_level` settings (0, 1, 2), twice each
in independent sweeps run in opposite order (0→1→2 and 2→1→0) to make sure the
setting — not run order — was what determined the outcome. In every run, the
number of blocks PSF-Zero processed matched the number of qubit pairs in the
circuit exactly: it correctly found and synthesized every qualifying block
once real hardware connectivity constraints were introduced, not just in the
unconstrained case above.

| Qubits | Qiskit gates / depth | PSF-Zero, level=0 | PSF-Zero, level=1 | PSF-Zero, level=2 |
| :---: | :---: | :---: | :---: | :---: |
| 50 | 500 / 20 | 75 / 9 | 75 / 5 | 75 / 5 |
| 100 | 1000 / 20 | 150 / 9 | 150 / 5 | 150 / 5 |
| 156 | 1562 / ~39 | 306 / ~24 | 236 / ~9 | 237 / 10 |
| 300 | 3000 / 20 | 450 / 9 | 450 / 5 | 450 / 5 |
| 500 | 5003 / 41 | 992 / ~30 | 753 / 10 | 753 / 10 |

![Real-device topology: Qiskit vs. PSF-Zero at routing_optimization_level 0, 1, and 2](../../docs/090307.png)

(Each cell above is a mean over 6 seeds — 2 sweeps × 3 seeds — except Qiskit,
pooled across all 18 runs per scale.) `routing_optimization_level=0` gives
Qiskit's router noticeably less work to do, and PSF-Zero's post-synthesis
routing pass inherits that: consistently more 2Q gates and higher depth than
levels 1 or 2. Levels 1 and 2 were statistically indistinguishable from each
other at every scale we tested — for this circuit family, the extra search
budget of level 2 bought nothing over level 1. (An earlier draft of this
benchmark mislabeled which run was which level, based on a single sweep; the
numbers above come from two independent, oppositely-ordered sweeps and we're
confident in this mapping.)

Code: [`benchmarks/test1.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/test1.py)

**Compile time under the same constraint.** The table above only measured
the *output* circuit (gate count, depth) — not how long either engine took
to produce it. Measuring that needed its own round of confound-hunting,
similar in spirit to section 4's.

First pass (coupling-map-constrained `compile_for_hardware()` fed
`random_circuit()`-generated circuits): PSF-Zero's own debug output showed
`0/N blocks` processed at every scale. `random_circuit()`'s gate mix almost
never produces more than `block_gate_floor` (12) consecutive same-pair
gates — the same root cause section 4 hit and fixed — so PSF-Zero's actual
synthesis path never ran; `compile_for_hardware()` was silently falling
through to a near-no-op `compile()` followed by nothing more than Qiskit's
own routing. Switching to the same dense-pair-blocks circuit generator as
section 4 fixed that.

With real blocks flowing through, the first `verify=False`-enabled run of
`compile_for_hardware()` still showed PSF-Zero 1.7x–2.9x *slower* than
Qiskit's own `optimization_level=3` at every scale — the opposite of
section 4's finding. Isolating each hypothesis in turn: running Qiskit's
`transpile()` inside a `multiprocessing.Process` (as every worker in this
project's benchmarks does) does *not* suppress its internal parallel search
— a direct main-process-vs-subprocess comparison on identical input came
back at 0.96x, i.e. no meaningful difference. The actual cause was simpler:
`transpile(optimization_level=3)` doesn't pin `seed_transpiler`, so its
internal randomized layout/routing search returns a different solution —
and takes a different amount of time — on every call, even for the
identical circuit. A single-seed measurement could land almost anywhere in
a wide range; we saw the same 500-qubit circuit measured at both 0.06s and
0.23s across separate runs of otherwise-identical code.

Pinning `seed_transpiler=<circuit seed>` on the Qiskit side and expanding
to 10 seeds resolved it. Run twice independently (20 measurements per scale
in total):

| Qubits | Qiskit (mean, opt L3, seed-pinned) | PSF-Zero (mean, `compile_for_hardware`, verify=False) | Ratio (Qiskit ÷ PSF) |
| :---: | :---: | :---: | :---: |
| 50 | 0.0189s | 0.0155s | 1.23x |
| 100 | 0.0332s | 0.0260s | 1.28x |
| 156 | 0.0432s | 0.0421s | 1.03x (essentially tied) |
| 300 | 0.1606s | 0.1172s | 1.37x |
| 500 | 0.2220s | 0.1834s\* | 1.21x\* |

\* One of the 20 measurements at 500 qubits returned 0.825s — a ~4x outlier
against every other point at that scale. Re-running the entire script did
not reproduce it (that same seed came back at 0.173s the second time), so
we're treating it as transient system noise rather than a real effect and
excluding it from the mean above; including it drops the ratio to ~1.03x.
Worth a further check if it recurs.

So: once `compile_for_hardware()`'s own confounds are controlled for the
same way section 4's were, plus the additional `seed_transpiler` fix this
section needed, PSF-Zero is faster than Qiskit's own routed compilation at
every scale tested here too — by a smaller, more scale-dependent margin
(1.0x–1.4x) than section 4's `compile()`-only comparison (2.4x–5.2x), which
makes sense: `compile_for_hardware()` pays for both PSF-Zero's own block
synthesis *and* a full separate Qiskit routing pass on top of it, whereas
section 4 measured synthesis alone.

**A limitation worth stating plainly, given this section's title:** the
dense-pair-blocks circuit used here (and, it appears, in the gate-count/
depth benchmark above, given the matching numbers) only places blocks on
adjacent logical pairs — (0,1), (2,3), (4,5), … — which land on adjacent
physical qubits under `CouplingMap.from_grid()`'s row-major layout. Neither
engine ever needed to insert a single SWAP gate in this comparison
(coupling violations were 0 throughout, with zero extra gates from
routing). So what's measured above is compile time for block synthesis
plus a routing pass that had nothing to route — not the cost of genuine
SWAP-insertion under real connectivity pressure, which is what "real-device
topology" benchmarks are usually meant to stress. A version using
non-adjacent logical pairs (so routing has real work to do) would be needed
to test that specifically — see Roadmap.

One residual asymmetry we haven't closed: `compile_for_hardware()` doesn't
yet expose a `seed_transpiler` parameter of its own, so its internal
routing call stays unpinned. We saw no sign of instability from this on the
PSF-Zero side (no repeat of anything like the 500-qubit outlier), but the
comparison isn't perfectly symmetric yet.

This complements, rather than replaces, the gate-count/depth table above,
which doesn't depend on `verify` or `seed_transpiler` and still stands
unchanged.

Code:
[`benchmarks/phase3_v4_dense_pair_blocks.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/phase3_v4_dense_pair_blocks.py)
(fixed the `0/N blocks` circuit-generation problem),
[`benchmarks/compile_for_hardware_verify_passthrough.patch`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/compile_for_hardware_verify_passthrough.patch)
(threaded `verify` through `compile_for_hardware()`),
[`benchmarks/profile_compile_for_hardware_breakdown.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/profile_compile_for_hardware_breakdown.py)
and
[`benchmarks/profile_warmup_depth.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/profile_warmup_depth.py)
(ruled out insufficient warm-up as the cause of the initial 1.7x–2.9x
slowdown),
[`benchmarks/profile_qiskit_multiprocess_vs_mainprocess.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/profile_qiskit_multiprocess_vs_mainprocess.py)
(ruled out the multiprocessing-suppresses-Qiskit's-own-parallelism
hypothesis), and
[`benchmarks/phase3_v5_seeded.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/phase3_v5_seeded.py)
(the `seed_transpiler` fix and 10-seed expansion that produced the table
above).

#### Update (2026-09-08): a second, independent finding narrows `routing_optimization_level` further — 2 → 1

A separate investigation, run against a different, deliberately wide dense-block
sweep (`phase3_v4.py` — a second, independently-built script converging on the
same "use dense pair blocks, not `random_circuit`" fix as
`phase3_v4_dense_pair_blocks.py` above, kept distinct here rather than
silently merged into it), found a mechanism this section's own tables above
don't isolate: neither table above states which `routing_optimization_level`
`compile_for_hardware()` was using internally, and it turns out to matter more
than it looks.

Directly diffing `compile_for_hardware()`'s output against a bare
`transpile(qc, ..., optimization_level=2)` call on the *uncompressed* input
circuit showed they are **bit-identical** — same gates, same qubits, same
parameters, verified at 4, 6, and 7 qubits — whenever
`routing_optimization_level=2` is used. The reason: Qiskit's
`optimization_level=2` preset re-runs `ConsolidateBlocks` and
`UnitarySynthesis` in its own `init` stage on whatever it's handed, so it
re-derives its own decomposition from scratch rather than trusting
PSF-Zero's. At that level, `compile()`'s own synthesis work is real, but it
is computed and then thrown away — every millisecond `compile_for_hardware()`
spends synthesizing before handing off to `transpile(..., optimization_level=2)`
is pure overhead on top of what calling `transpile()` directly would have
done anyway. (This does not make section 8's earlier fix below wrong — before
`basis_gates` was threaded through, level 2 was the only way to get a
target-basis translation to run at all. It means that now that `basis_gates`
is always passed, level 2's only remaining effect on top of that is this
wasted re-synthesis.)

This reframes, without contradicting, this section's own 1.0x–1.4x number
above: if `compile_for_hardware()` was already defaulting to level 2 when
that table was produced, its advantage over plain Qiskit `optimization_level=3`
most plausibly came from internally using a cheaper Qiskit preset (2 is
faster than 3) rather than from PSF-Zero's own synthesis contributing
anything at that level. We have not gone back and re-run that exact table
with the level pinned and logged to confirm this reading with certainty —
flagging the relationship here rather than leaving the two findings looking
like they disagree.

A custom `PassManager` that strips the redundant re-synthesis stages out of
Qiskit's preset pipeline — so a target basis is still reached, but nothing
gets re-derived from scratch — was prototyped and benchmarked at 50/100/156
qubits on this workload, and rejected: it landed within 2–4% of simply using
`routing_optimization_level=1`, and was worse on depth at the larger sizes.
`routing_optimization_level=1` already gets the same effect for free, with
no extra pass manager to maintain.

On this workload (grid coupling map, `basis_gates=["rz","sx","x","cx"]`, all
output verified ISA-submittable and unitarily equivalent to the input),
`routing_optimization_level=1` gives the same 2-qubit gate count as Qiskit's
`optimization_level` 2 and 3 (150 gates at 100 qubits, 240 at 156 qubits) for
1/20th to 1/59th of their compile time, at roughly 30–40% more depth (23 vs.
16 at 100 qubits, 44 vs. 35 at 156 qubits). Against Qiskit
`optimization_level=1` it wins outright: 1.8x faster, 20x fewer 2-qubit
gates, 10x shallower.

Full per-scale timing numbers, the bit-identical-output verification, the
rejected custom-`PassManager` benchmark, and a `random_circuit` passthrough
control (confirming PSF-Zero correctly reports `0/0 blocks` and contributes
nothing on workloads it isn't designed for) are in this project's
`phase3-hardware-routing-regression.md` note rather than duplicated here.

**The change made:** `compile_for_hardware()`'s default
`routing_optimization_level` is now **1**, not 2 (down from the value set by
section 8's own fix below). Its docstring now states directly that level 2
reproduces plain `transpile(optimization_level=2)` exactly and charges
PSF-Zero's synthesis on top of it for nothing in return — call
`transpile(..., optimization_level=2)` directly (skipping `compile_for_hardware()`
entirely) if minimum depth matters more than compile time; level 1 is the
setting where PSF-Zero's own synthesis is actually the thing producing the
output.

##### Confirmed on the project's slower machine (2026-09-09)

The default change above was decided from cloud-sandbox measurements. It has
since been re-run end to end on real hardware with the real `psf_zero_core`
— specifically the project's *slower*, noisier machine, not the faster one
used for section 4's cumulative-loop and methodology-corrected updates
above — at 3 seeds per point, both workloads,
out to 300 qubits, one scale further than the sandbox run reached. Median
compile time (ms):

| Qubits | qiskit opt=1 | qiskit opt=2 | qiskit opt=3 | **psf rl=1 (new default)** | psf rl=2 (old default) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 50 | 13.7 | 15.0 | 33.4 | **9.6** | 12.3 |
| 100 | 23.7 | 879.3 | **10,134** | **16.1** | 866.2 |
| 156 | 67.3 | 1,121.6 | **12,858–31,269** | **49.4** | 1,108.1 |
| 300 | 70.1 | 56.8 | 96.8 | **39.9** | 42.7 |

2-qubit gate count: `opt=1` emits 1500/3000/4686/9000; every other arm emits
the same 75/150/240/450. Depth: `rl=1` is 23/23/44/23, everything else
16/16/35/16 — the ~30–40% depth cost stated above, reproduced exactly.
Equivalence checked at 6 qubits, all arms < 4e-15. `psf rl=2` tracks
`qiskit opt=2` to within a couple of percent at every scale (866.2 vs.
879.3ms at 100 qubits), which is the bit-identical-output finding above
showing up in the timings.

**An unexplained instability in Qiskit's own higher optimization levels,
worth flagging because it is not ours and it is large.** At 100 and 156
qubits `opt=3` takes 10–31 *seconds* on this workload, against tens of
milliseconds at 50 and 300 qubits. The same non-monotonic blow-up appeared
independently in the sandbox run (14.3s at 100q, 19.5s at 156q), so it
reproduces across environments. It correlates exactly with whether the grid
coupling map has spare qubits: 50→56 and 300→306 have 6 unused physical
qubits and are fast; 100→100 and 156→156 are exactly saturated and are
catastrophically slow. **The 300-qubit `passthrough` control rules out the
simplest version of that story**, though — there `opt=3` took 75.2s against
`opt=2`'s 57.1s, a monotonic increase rather than a blow-up, on a grid that
also has 6 spare qubits but a completely different circuit structure; and
at 50/100/156 qubits the `passthrough` arms never showed the inversion
either. So the trigger appears to need *both* a saturated coupling map *and*
the dense adjacent-pair structure, not either alone. This is an observation
with a correlation and no confirmed mechanism — but it is a second,
independent reason to prefer `routing_optimization_level=1`, which never
enters that regime at all (9.6–49.4ms across every scale tested). Worth
noting given this ran on the noisier of the project's two machines: `rl=1`
stayed tight and predictable here despite that, while both Qiskit and
`rl=2` show their worst variance on exactly this machine — the opposite of
what "just a slower PC" would predict if it affected every arm equally.

The `passthrough` control behaved exactly as designed at every scale
including 300 qubits: `0/0 blocks` reported every time, and `rl=1` and
`rl=2` matching `opt=1` and `opt=2` respectively on 2-qubit gate count and
depth to the digit (300q: `rl=1` 624,556 gates / depth 76,863, identical to
`opt=1`). On circuits PSF-Zero is not designed for it is neither help nor
harm, which is the behaviour `block_gate_floor` exists to produce.

##### Update (2026-09-10): a third machine reproduces the blow-up, and a controlled experiment turns the spare-qubit correlation into a cause

Two separate things happened here and they should not be read as one. The
first is another reproduction, which raises confidence and contributes
nothing about cause. The second is an actual experiment, which settles the
cause — and corrects the shape of the claim above while doing so.

**1. `phase3_v4.py` re-run unchanged on a third machine.** Same script,
same 3 seeds × 3 reps, both workloads, out to 300 qubits, on a machine
distinct from the one that produced the 2026-09-09 table above (CPU
signature `Intel64 Family 6 Model 181` vs. that run's
`AMD64 Family 25 Model 80`; note also Python 3.11.9 here against 3.10.11
there — two variables moved, not one). Median-of-min compile time, dense
workload (ms):

| Qubits | qiskit opt=1 | qiskit opt=2 | qiskit opt=3 | **psf rl=1** | psf rl=2 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 50 | 13.4 | 15.6 | 33.2 | **9.6** | 12.4 |
| 100 | 23.8 | 893.6 | **12,985** | **15.7** | 888.5 |
| 156 | 71.7 | 1,135.0 | **13,189–28,018** | **50.3** | 1,144.7 |
| 300 | 70.3 | 59.1 | 96.7 | **43.4** | 42.3 |

Three things are worth recording from it.

*The circuit outputs are bit-identical between the two machines.* Every
2-qubit gate count and every depth value, across all 120 rows of both
runs — dense and passthrough, all four scales, all three seeds, all five
arms — matches exactly. So the two runs are unambiguously the same
workload, and timing is the only variable that moved. That is a stronger
statement than the usual "reproduced" and it is worth having.

*The non-monotonic blow-up reproduces exactly where it did before*, in a
third independent environment (Linux sandbox → AMD machine → this one).

*`opt=3`'s seed-to-seed spread is much worse on this machine than on the
other one.* At 156 qubits its three per-seed minima were 13.2s / 28.0s /
27.3s — a 2.1x spread, against 1.03x for the same arm on the AMD machine.
Any single number quoted for `opt=3` at these sizes is therefore a draw
from a wide distribution, and the range is the honest way to report it.

*A note on this README's "faster machine"/"slower machine" labels:* they
do not survive this comparison. On the dense workload this machine is
uniformly faster (0.65x–0.89x the other's time on every arm and scale);
on the large `passthrough` workload it is uniformly *slower* (1.4x–1.9x at
300 qubits). Which machine is "the fast one" depends on the workload, and
the Python-version difference above is confounded with the hardware
difference anyway. Read those labels, wherever they appear in this README,
as identifying *which run* a number came from — not as a claim about
hardware speed.

**2. The controlled experiment: it really is the spare qubits, and the
threshold is not zero.** The paragraph above ("It correlates exactly with
whether the grid coupling map has spare qubits") was a correlation across
four points, and reproducing those same four points on more machines could
never improve it: `get_grid_cmap()` produces a saturated grid at exactly
n=100 and n=156 and a 6-spare grid at exactly n=50 and n=300, so "has no
spare qubits" and "is one of those two sizes" were perfectly confounded in
every run this project had done. Replication is not a test.

`benchmarks/phase3_v5_spare_qubits.py` breaks the confound by holding the
coupling map fixed and varying only how much of it the circuit occupies
(and, separately, holding the circuit fixed and varying the map). It reuses
this project's own `get_grid_cmap()` and `build_dense_pair_blocks_circuit()`
verbatim, and re-derives the n=100 saturated point as an anchor to prove
the fixture matches: `opt=2` 1,139ms and `opt=3` 13,060ms here, against
1,244ms / 14,343ms for the same point in the sandbox run quoted above.
Run on the Linux sandbox, Qiskit 2.5.2, min-of-reps, median over seeds:

| Grid | Spare | Circuit qubits | qiskit opt=2 | qiskit opt=3 |
| :---: | :---: | :---: | :---: | :---: |
| 6×7 = 42 | 4 | 38 | 34.1 ms | — |
| 6×7 = 42 | **0** | 42 | **621.1 ms** | — |
| 7×8 = 56 | 6 | 50 | 16.1 ms | — |
| 7×8 = 56 | **0** | 56 | **767.7 ms** | — |
| 8×8 = 64 | 4 | 60 | 20.0 ms | — |
| 8×8 = 64 | **0** | 64 | **840.8 ms** | — |
| 8×9 = 72 | 6 | 66 | 20.5 ms | 69.9 ms |
| 8×9 = 72 | 4 | 68 | 20.6 ms | 84.7 ms |
| 8×9 = 72 | 2 | 70 | 19.9 ms | 35.6 ms |
| 8×9 = 72 | **0** | 72 | **871.8 ms** | **10,115 ms** |
| 10×10 = 100 | 8 | 92 | 21.4 ms | 61.3 ms |
| 10×10 = 100 | 6 | 94 | 21.6 ms | 93.5 ms |
| 10×10 = 100 | 4 | 96 | 24.4 ms | 53.4 ms |
| 10×10 = 100 | **2** | 98 | **1,127.9 ms** | **13,033 ms** |
| 10×10 = 100 | **0** | 100 | **1,139.3 ms** | **13,060 ms** |
| 10×11 = 110 | 10 | 100 | 25.0 ms | — |
| 11×11 = 121 | 21 | 100 | 34.7 ms | — |

**The size explanation is dead.** A 42-qubit circuit on a saturated 42-qubit
grid takes 621ms, while a *larger* 50-qubit circuit with 6 spare qubits
takes 16ms — the smaller circuit is 39x slower. At `opt=3` the reversal is
190x (72 qubits / 108 two-qubit gates on a saturated grid: 10.1 seconds;
96 qubits / 144 two-qubit gates with 4 spare: 53ms). No amount of
"bigger circuits are harder" produces that.

**It is a cliff, not a slope.** On the 10×10 grid, `opt=3` goes from
13,033ms at 2 spare qubits to 53ms at 4 — a 244x change from removing two
qubits from the circuit, with the coupling map untouched. On the 8×9 grid
the same cliff sits between 0 and 2 spare (10,115ms → 36ms, 284x).

**And the threshold is not zero, which the four-point data could not have
shown.** On the 100-qubit grid, 2 spare qubits is still fully in the slow
regime; on the 72-qubit grid, 2 spare is already fully out of it. So it is
not a fixed number of spare qubits — the boundary sits higher on the
larger map. Two grids is not enough to say whether it tracks area,
perimeter, or something else, and we have not tried to find out.

**What the slow runs are *not* doing is extra work.** Every configuration
above emitted exactly 3 two-qubit gates per logical pair (63/84/96/108/150
for 21/28/32/36/50 pairs) with zero coupling violations — the router found
a SWAP-free solution in every single case, including the slow ones. The
1000x is spent searching for a solution it eventually finds, not producing
a bigger circuit.

> **Correction (2026-09-10): "in every single case" is wrong — see the
> correction in the reproduction subsection below.** At n=72 on the 8×9
> grid the router sometimes does insert SWAPs. The reading the sentence
> supports is unaffected; the absolute is not true.

**What is still not known: the mechanism.** This is an intervention result
— vary one variable, hold the rest identical — so the causal direction is
established, but nothing here identifies *which* pass burns the time or
why a nearly-full coupling map is pathological for it. We did not
instrument Qiskit's pass timings, and this is one Qiskit version (2.5.2)
on one topology family. Nor does this touch the other half of the
2026-09-09 observation above: the `passthrough` control was not re-run
here, so "the trigger needs the dense adjacent-pair structure as well"
remains as stated — untested by this experiment, not confirmed by it.

**The practical consequence is new, though, and cheap.** If you are running
`optimization_level` 2 or 3 against a coupling map your circuit almost
fills, padding the map by a few spare qubits removes the blow-up entirely
(25.0ms on a 110-qubit grid against 1,139ms on the 100-qubit one, for the
identical 100-qubit circuit). And for this project specifically, it
promotes the earlier "second, independent reason to prefer
`routing_optimization_level=1`" from a correlation to a measured one:
`rl=1` never enters the regime at all.

Data: [`psf-zero/data/phase3_v4_intel_machine_2026-09-10.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/phase3_v4_intel_machine_2026-09-10.csv) (the third-machine
run) and [`psf-zero/data/phase3_v5_spare_qubits_linux_2026-09-10.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/phase3_v5_spare_qubits_linux_2026-09-10.csv) (the
controlled experiment). Full reasoning, including the pre-registered
predictions written before the experiment was run, is in
`phase3-hardware-routing-regression.md`.

##### Update (2026-09-10): the controlled experiment reproduces on real hardware, and one claim in it is corrected

The experiment above was run in a Linux sandbox. The identical script was
then run unmodified on the Intel machine (Windows, Python 3.11.9, Qiskit
2.5.2 — the same Qiskit version, so this is a genuine second environment
rather than a second Qiskit), axis C, 2 seeds × 2 reps:

| Grid | Spare | Circuit qubits | `opt=2` Intel | `opt=2` sandbox | `opt=3` Intel |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 6×7 = 42 | 4 | 38 | 11.6 ms | 34.1 ms | 23.4 ms |
| 6×7 = 42 | **0** | 42 | **549.6 ms** | **621.1 ms** | **7,725 ms** |
| 7×8 = 56 | 6 | 50 | 12.6 ms | 16.1 ms | 29.2 ms |
| 7×8 = 56 | **0** | 56 | **633.0 ms** | **767.7 ms** | **9,770 ms** |
| 8×8 = 64 | 4 | 60 | 12.2 ms | 20.0 ms | 29.8 ms |
| 8×8 = 64 | **0** | 64 | **706.6 ms** | **840.8 ms** | **8,184 ms** |
| 8×9 = 72 | 6 | 66 | 15.8 ms | 20.5 ms | 36.3 ms |
| 8×9 = 72 | **0** | 72 | **720.0 ms** | **877.8 ms** | **8,295 ms** |

Same cliff, same place, on hardware: 45x–58x at `opt=2` and 228x–335x at
`opt=3`, between two circuits on the *same* coupling map differing only in
how many qubits they leave spare. The size reversal reproduces too — on
this machine a 42-qubit circuit on a saturated 42-qubit grid takes 549.6ms
against 12.6ms for a *larger* 50-qubit circuit with 6 spare (44x at
`opt=2`, 265x at `opt=3`). The controlled result is now two environments
deep, and `opt=3`, which the sandbox run only covered on two grids, shows
the effect on all four.

> **Correction (2026-09-10): the sentence "the router found a SWAP-free
> solution in every single case, including the slow ones" in the update
> above is wrong, and this run is what caught it.** Checking both datasets
> against the expected 3-gates-per-pair: at n=72 on the 8×9 grid the output
> is sometimes 108 gates / depth 16 (SWAP-free) and sometimes 114 gates /
> depth 35 (six SWAPs inserted) — in 4 of the sandbox measurements and 1 of
> the Intel ones. That grid is 9 columns wide, so consecutive logical pairs
> straddle row boundaries unless the layout pass happens to find a mapping
> that avoids it; `transpile()` is not seed-pinned here, so it finds one
> some runs and not others. Every *other* configuration in both runs was
> SWAP-free as stated, and coupling violations were zero everywhere.
> **The conclusion the sentence was supporting is unaffected**: the slow
> cases are not slow because they emit more gates. At n=72/spare 0 the
> SWAP-free and six-SWAP outcomes took 0.807s and 0.913s respectively —
> both about 45x the spare-6 point on the same map, which produced its
> 99-gate output in 15.8–20.5ms. The correction is to the "every single
> case" absolute, not to the reading.

Raw data: [`psf-zero/data/phase3_v5_spare_qubits_intel_2026-09-10.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/phase3_v5_spare_qubits_intel_2026-09-10.csv).

### 6. Sanity check against Benchpress

We don't have our own results in [Benchpress](https://github.com/Qiskit/benchpress) — IBM's open-source SDK benchmarking
suite (Nation et al., *Benchmarking the performance of quantum computing
software for quantum circuit creation, manipulation and compilation*,
[Nat. Comput. Sci. 5, 427–435 (2025)](https://doi.org/10.1038/s43588-025-00792-y)) — but two of our numbers above line up
with what that paper independently reports for TKET against Qiskit, which is
worth stating plainly rather than leaving unmentioned:

- Benchpress reports TKET's transpilation is "over an order of magnitude
  slower than Qiskit" across its 1,066-test suite. Our own N=300 result
  (TKET median 153.5ms vs. Qiskit 6.0ms, ~26x) and native-scale comparison
  (TKET 150–270x slower than PSF-Zero, itself faster than Qiskit) point the
  same direction — our absolute TKET-vs-Qiskit timing gap isn't an artifact
  of our narrow test construction.
- Benchpress specifically calls out Hamiltonian-simulation circuits as the
  case where TKET's synthesis step yields "substantial 2Q depth reduction
  relative to Qiskit," and that this synthesis advantage matters most on
  well-connected topologies and fades as routing starts to dominate on
  sparser ones. That is exactly the pattern in our own Hamiltonian
  benchmark (section 3) and in our coupling-map-constrained result (section
  5), where PSF-Zero's and TKET's edge over Qiskit shrinks once routing
  becomes the bottleneck rather than synthesis.

This is corroboration of the general trend, not a substitute for the real
test: Benchpress's suite is far broader than ours (1,066 tests, up to 930
qubits and O(10⁶) 2Q gates, real device coupling maps, and circuit families
we haven't touched — quantum volume, QAOA, HamLib, Feynman, QASMBench —
versus our own narrowly-constructed dense-pair circuits). Running PSF-Zero
through Benchpress's own harness is the obvious, credible next step, and
we haven't done it yet — see Roadmap.
