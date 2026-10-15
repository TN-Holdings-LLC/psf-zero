# Workplace delivery 2026-10-15c: W1-W3 of 「次の指示8」 (base: bundle q, c60)

Handed over at the user's request before the end of the day. Offline only: no IBM connection, no commit, no push. Check
the delivery with `python check_intake.py`. Run everything from the repository root (bundle q).

**Folder names.** The scripts find each other by these names, and every file still matches the SHA-256 fixed in its
pre-registration.

| folder | contents |
|---|---|
| `w33/` | W1, item 101, c62 |
| `w34/` | W2, c63 |
| `w34/w3/` | W3 |
| `w31_probes/` | the missing probe2-4 |

## W1, item 101, prototype c62 (from c60): done

Items 87/91's CZ path is emitted by PSF-Zero itself:

- a single interaction becomes a local Clifford around RZZ;
- other blocks use the Weyl closed form;
- `CZ_OWN_EMIT = False` restores c60.

C62-VAL (pre-registered 08:21:36 CEST), 490 cells:

| ID | prediction | result |
|---|---|---|
| P1 | exact, native, rzz in range, no failed element | CONFIRMED (330 outputs; largest difference 4.2e-15) |
| P2 | Ising T <= c60, geometric mean <= 0.85 | CONFIRMED: **0.7995** (0.740-0.874); two-qubit counts lower in all 45 cells |
| P3 | Heisenberg T <= 1.01 x c60 | CONFIRMED (1.000; one cell 0.9975) |
| P4 | identical to c60 on F and item 99; same refusals | CONFIRMED (275 / 275; 160 refusals) |
| P5 | deterministic | CONFIRMED (330 / 330) |

- Compile median on the CZ-path cells: 32.8 ms (c60: 59.0 ms).
- Report: `w33/w1-item101-c62-2026-10-15.md`. It also discloses that the document header's time ("08:25") differs
  from the fixed time.

## W2, faster template binding, prototype c63 (c62 + item 100): interim

**What changed:**

- planned binding;
- constant runs baked into the circuit;
- one numpy batch per bind for items 87/91 (item 99 keeps CPython arithmetic, for bit identity);
- pinned full runs at general Weyl blocks: hhtree_heis_n16 T no longer changes shape (c61: 15.5 ms; c63: about
  3.5 ms isolated);
- batched Weyl checks.

C63-VAL was pre-registered at 08:52:23 CEST. **Part R was stopped at 09:59:18 CEST, after 20 of 38 cells** (the ten
items-87/91 tests, T and F, 1,101 inputs each). In those 20 cells:

- V2 (T2) holds: 11,010 F outputs, duration and 2q equal to c62 in every one.
- V3 holds: 11,010 T outputs, all exact.
- S2 (bind back to back) is within 2 ms in all 20 cells.
- S1 (bind interleaved with compilation and simulation, C61-VAL's measure) misses in 4 cells (2.05-2.67 ms).
- **No verdicts yet.**
  - Not run: hhtree_heis_n16, ghz_n12 and the 7 native tests in part R, and all of part D.
  - hhtree_heis_n16 T is expected to miss S1 and S2.
  - `w34/w2-c63-template-speed-2026-10-15.md` §4 gives the two commands to finish the run at home.

## W3, item 88's frame in a training loop: measured (C61-VAL part K's conditions, tfim_n12 F, 100k laps, seed C61K|F)

See `w34/w3/w3-frame-options-2026-10-15.md`. Correctness held in all three: exact in 100 checks each (largest
difference 3.0e-16).

| | shape changes | bind median / p99 | peak RSS | duration vs c62 |
|---|---|---|---|---|
| A | **0** | **0.75 / 3.4 ms** | **221 MB** | up to **+4.4 %** (above c62 in 42 of 100 checks) |
| B | 76,885 | 2.86 / 10.2 ms | 418 MB | equal |
| C | 76,885 | 2.86 / 9.9 ms | 734 MB | equal |

- Two-qubit counts are equal in all three.
- B's 32-shape cache hits as often as C's 256 (9,427 times), so B only saves memory.
- A and B ran at the same time on 2 CPUs, so their bind times carry some contention.

- **A:** frame fixed in the template. Sign changes and folds are absorbed by pinned runs. This is `c63a`, a
  measurement variant.
- **B:** c63 with a cache of 32 shapes.
- **C:** c63 as validated, for reference.

## Missing files from 2026-10-15b

`w31_probes/probe2.py`-`probe4.py` are included. Only their absolute workplace paths were replaced, by the 2026-10-15b
delivery's location in bundle q. The probes `w33/probe5.py`, `w34/probe6.py` and `w34/dbg9.py` were treated the same
way.
