# W3 (2026-10-15): item 88's frame in a training loop. Option A removes every shape change; B only saves memory

Workplace, 2026-10-15. Measurement only; home decides. Offline.

- The machine was kept free for C63-VAL part R, as its pre-registration requires. The three runs were made after part
  R was stopped (10:01-10:08 CEST).
- A and B ran at the same time on this 2-CPU machine, and C overlapped B's second half. Bind times therefore carry
  some contention.

## Options

**A: frame fixed in the template** (`make_c63a.py` + `c63a_frame_code.py` give `c63a/psf_compile.py`; measurement
variant, not a proposal by itself).

- In the template, a negative rzz angle is written as X RZZ(|theta|) X on its first qubit, locally. The Pauli frame is
  not flipped.
- The items on both sides of every rzz are pinned (c63's `_t63_emit`). A pinned run is always emitted in full form
  (rz rx rz with fractional gates).
- A sign change therefore alters angles only, never the shape.
- A fold (|theta| crossing pi/2) is put into the same pinned items: Rz(pi) on both qubits. Only an angle of exactly 0
  still changes the shape.
- Expected cost: an extra rx pulse wherever a pinned run would otherwise have been z-only or empty. This is what the
  duration column must show.
- Fixed gates are never X-conjugated in A, so c63's constant-run baking stays valid. The sign-dependent items are new
  tuples on every bind, so they are never baked.
- c63's default call is unchanged. `TEMPLATE_FRAME_FIXED = False` gives back c63's template.

**B: the current output with a smaller cache.** c63 with `TEMPLATE_CACHE = 32` (set at run time; no file change).

**C: reference.** c63 as validated (cache 256).

## Measurement (`w3_walk.py`; one variant per process, so that peak RSS is per variant)

The conditions are C61-VAL part K's:

- tfim_n12, FakeFez F;
- 100,000 binds on a random walk: every value plus N(0, 0.05) per lap, with the same seed "C61K|F".

Recorded:

- shape changes (rebuilt + switched);
- bind time per lap: median, p99, max;
- RSS every 1,000 laps (peak);
- every 1,000th lap: exactness against the reader's terms (Aer state vector); native, rzz range and failed elements;
  duration and two-qubit count of the output, and of c62's default call on the same input (c62 F = c60 F).

```
python <dir>/w3/w3_walk.py A <out>/w3_A.json
python <dir>/w3/w3_walk.py B <out>/w3_B.json
python <dir>/w3/w3_walk.py C <out>/w3_C.json
```

Run them from the repository root (bundle q), one at a time, with `OMP_NUM_THREADS=1`. `W3_LAPS` shortens a trial run.

**Reference from C61-VAL (c61, F):** 64,766 shape changes in 100k laps; peak RSS 307 MB at lap 10k and 1,142 MB at
the end.

**No pass criterion is set.** Note 8 asks for measurement only, and home decides.

## Results (100,000 laps each; `run/w3_{A,B,C}.json`, logs in `run/`)

| | A: frame fixed | B: cache 32 | C: c63, cache 256 |
|---|---|---|---|
| shape changes | **0** | 76,885 (67,458 rebuilt, 9,427 from cache) | 76,885 (67,458 rebuilt, 9,427 from cache) |
| bind median | **0.75 ms** | 2.86 ms | 2.86 ms |
| bind p99 | 3.4 ms | 10.2 ms | 9.9 ms |
| bind max | 344 ms | 405 ms | 627 ms |
| RSS at lap 10k / peak | **211 / 221 MB** | 381 / 418 MB | 720 / 734 MB |
| exact (100 checks, largest difference) | 2.9e-16 | 3.0e-16 | 3.0e-16 |
| native, rzz in range, no failed element | all | all | all |
| duration against c62 (= c60 on F) | median 1.000; above c62 in 42 of 100 checks; largest **1.044** | equal in all 100 | equal in all 100 |
| two-qubit count against c62 | equal in all | equal in all | equal in all |

**Reading (for home's decision).**

- **A** removes every shape change on this walk. Its bind is about 4 x faster than B or C, and its memory stays flat
  near the start level.
- **A's cost:** in 42 of 100 checked laps the circuit is longer than c62's, by up to 4.4 %. This is the extra rx
  pulse where a pinned run would have been z-only or empty. Two-qubit counts are unchanged.
- **B** keeps c62's duration and two-qubit count in every check and lowers peak memory (418 MB against 734 MB). Its speed is the same as C's: the
  32-shape cache hits exactly as often as 256, 9,427 times, because the walk rarely returns to an old shape.
- Compared with c61 in C61-VAL (64,766 changes; peak 1,142 MB), c63 (C) changes shape more often on this walk:
  76,885. C61-VAL's figure came from c61's frame code on the same seed. Its memory is lower (734 MB).
- The 100,000 laps include walks far outside the initial angles: the step is N(0, 0.05) per value, so the spread is
  about 16 rad by lap 100,000. Folds and sign changes are therefore frequent, as in a long training loop.
