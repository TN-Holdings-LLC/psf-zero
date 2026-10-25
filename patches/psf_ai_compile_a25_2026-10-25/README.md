# AI front end a25 (2026-10-25): the zipper also for odd n

`psf_ai_compile.py` here is a24 (`patches/psf_ai_compile_a24_2026-10-22/`) with one item, made by `make_a25.py`
(edits to a24's text). With psf_compile 2026-10-18.1 (unchanged) and `rust/psfbind` ring r3.

- **Item 49, the zipper for odd n.** With `PSF_RING_ZIP=1`, a ring of odd n >= 5 also gets the zipper schedule, by
  formula: the same start order as for even n (0, 1, 2, n-1, n-2, 3, 4, ...); one CZ + SWAP block at the left end
  only; (n - 3)/2 interior SWAPs; **(5n - 7)/2 CZ per layer** (9, 14, 19 at n = 5, 7, 9). Item 84's rule chooses, as
  for even n. Even n: the rounds and every output are a24's. `rust/psfbind` ring r3 builds the same schedule in Rust;
  a25 needs r3 or later (item 46's check), and a24 still runs on r3 (it never asks for an odd zipper).
- **Why this schedule.** TOQB `results/zipper-odd-2026-10-25`: one layer equals the ring's CZ layer and a permutation,
  and forward then backward is the identity, for n = 4 to 41; (5n - 7)/2 is the least found by exhaustive search
  (every state for n = 5, 7, 9; every minimum-length start for n = 5-11), reached only from the zipper's start up to
  symmetry. A search result, not a proof for every n. Not run on a device.

**Checks.** `test_a25.py` (13 tests; a24's 5 tests also pass on ring r3). `check_a25.py` with `check_a25_result.txt`:
check_a24's 272 ring cases (four fake devices, n = 4-20, L = 1, 2, 4, 8), two calls each, with
`PSF_RUST_TEMPLATE=check` (every Rust template equal to the Python's): without the switch a25 equals a24 on all 544
calls; with `PSF_RING_ZIP=1` it equals a24 on every even-n call (288) and builds every odd-n call (256), choosing the
zipper for every odd ring size on every device, with (5n - 7)/2 CZ per layer. RESULT PASS.

Files: `make_a25.py`, `psf_ai_compile.py` (generated), `test_a25.py`, `check_a25.py`, `check_a25_result.txt`.
