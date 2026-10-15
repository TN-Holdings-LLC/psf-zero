# Workplace delivery 2026-10-15b: W1 and W2 of 「次の指示7」 (base: bundle o, c59)

Finished 08:06 CEST (`TZ=Europe/Zurich date`). Offline only: no IBM connection, no commit, no push. Check the delivery
with `python check_intake.py`.

**Folder names.** `w30/` is W1 (item 99, c60) and `w31/` is W2 (item 100, c61). The two scripts find each other by
these names: c61's test and C61-VAL load `../w30`. They are kept so that every file still matches the SHA-256 fixed in
its pre-registration. Run everything from the repository root (bundle o).

## Contents

**`w30/`: item 99, prototype c60 (from c59).** CZ / CX chains and lattices, `real_amplitudes`, `efficient_su2` and
CZ-ring classifiers, which c59 refused, are built by formula as written:

- item 91's first VF2 placement;
- commuting diagonal gates ordered by item 91's edge colouring;
- fixed native forms, each multiplied out and checked;
- ZYZ single-qubit runs written by PSF-Zero itself.

C60-VAL (pre-registered 04:29:52 CEST), 210 cells:

| ID | prediction | result |
|---|---|---|
| V1 | every output exact, native, rzz in range | CONFIRMED (largest difference 6.7e-16) |
| V2 | deterministic | CONFIRMED |
| V4 | identical to c59 where c59 builds; same refusals | CONFIRMED (190 / 190, 160 / 160) |
| V5 | c59's former refusals are exact | CONFIRMED (30 / 30) |
| V3 | geometric mean C60 / C57 <= 1.00 and no cell > 1.05 | REFUTED: geometric mean **0.803**, but FakeTorino hh_cz_n16 is 1.25 / 1.28 (R4's coupler lengths) |

- Against Qiskit level 3: 0.707. Compile median 29 ms (c57: 773 ms).
- One scoring fix was made before reading the results (the match on c59's refusal message); it is described in the
  report.

**`w31/`: item 100, prototype c61 (from c60): `compile_template` / `CompileTemplate`.** Built once; angles bound in
closed form; a shape signature with a cache.

C61-VAL (pre-registered 05:50:12 CEST):

| ID | prediction | result |
|---|---|---|
| T1 | item 99 templates identical to c60 | CONFIRMED (17,616 / 17,616) |
| T2 | frame path: duration and 2q equal to c60 | CONFIRMED (12,111) |
| T5 | build <= 2 x c60's compile | CONFIRMED |
| T6 | correct on the other 4 devices | CONFIRMED (7,752) |
| T7 | 100k binds exact, no memory growth after lap 10k | CONFIRMED (worst 4.4e-16) |
| T3 | CZ path not longer than c60 | REFUTED: Heisenberg tests up to 1.6% / 2.7% longer in part of the draws |
| T4 | bind <= 2 ms | REFUTED: 25 of 38 cells meet it; the rest are 2.0-6.9 ms |

- Not one of 49,590 bound outputs was wrong.
- PSF-Zero's own CZ-path emission is 0.76-0.83 x c60 on Ising models. This is a proposal for items 87/91's default
  call.
- **Memory:** in the F random-walk loop, the shape cache took peak RSS from 307 MB to 1,142 MB, then flat. This is in
  the report.
