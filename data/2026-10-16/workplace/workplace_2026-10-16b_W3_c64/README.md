# Workplace delivery 2026-10-16b: W3 of 「次の指示10」 — c64 (c63a without unreachable code)

Offline only: no IBM connection, no commit, no push. Check the delivery with `python check_intake.py`. Base: the
2026-10-15c delivery's `w34/w3/c63a/psf_compile.py` (SHA-256 `9d0fecbd...`) and bundle q. Run the scripts from the
repository root (bundle q); `c64_check.py` expects the 2026-10-15c delivery's `w34/` next to this folder's parent, or
`C63A_FILE` pointing to c63a.

**Pre-registration:** `C64-PREREG.md`, fixed 2026-10-16 03:50:14 CEST, following home's two answers.

| ID | result |
|---|---|
| D1 | CONFIRMED: 49,590 inputs (190 cells, 5 devices, T and F) and 58 refusal checks; every `compile_for_hardware` and `compile_template().bind()` output `==` c63a's, layouts equal, same exception types; value_sig equal too. No removal restored |
| D2 | CONFIRMED: `make_c64.py` reproduces c64 byte for byte |
| D3 | CONFIRMED: pyflakes reports nothing |
| D4 | CONFIRMED: the 6 tests pass on both; as predicted, test_rebuild_on_sign_change fails on both and test_version on c64 only |
| D5 | CONFIRMED: median c64 / c63a compile time 0.9958 (range 0.60-1.27) |

**Size and timing:**

- 9,240 lines become 3,045.
- 136 definitions removed (4,306 lines), plus the overwritten duplicate.
- Every switch is fixed at c63a's default. `NATIVE_BRICKWORK` stays as a constant True for a23.

**Restart during D1:** a machine restart killed D1's first start after 4 cells (kept in `run_d1_killed/`). D1 was rerun
in full, in restart-safe segments (`run_d1.sh`), 04:07-05:52 CEST.

The report is `w3-c64-2026-10-16.md`.
