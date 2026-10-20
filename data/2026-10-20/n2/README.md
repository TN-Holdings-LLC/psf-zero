# N2: the adversarial set for milestone M0

**Status: FIXED** on 2026-10-20 at 06:35 CEST, with the owner's OK (06:35 CEST) to the home additions below. Prepared
at about 04:00 CEST; the owner decided that N2 is fixed before any gate-1 code is written, and none exists. Fixing means: the owner's OK, then the SHA-256 of `n2.json` and the CEST time written below
in one commit. After that, `n2.json` is never edited; a change is a new file with a new name.

- **Source:** the workplace's W3 (`docs/design/W3_N2_adversarial_set_2026-10-20.md`) with the owner's decisions of
  2026-10-20 applied: W3 §0 adopted (IR r2, `docs/design/IR_v0_spec_r2_2026-10-20.md`), the decline record adopted.
- **Pins:** `n2.json` carries the SHA-256 of the IR r2 text, the r2 schema and the four fixtures, so its own SHA-256
  fixes all of them.
- **Records are raw text** (an overflowing literal such as 1e999 cannot be written by a JSON library).
- **Check numbers follow IR r2** (parse 0, version 1, finite 2, schema 3, device 4, gate set 5); W3 used the draft's
  numbers.

**Contents:** 23 refusal records (R01-R18 with R04b, R08b, R09b; R20 and R21), one fault injection (R19), six controls
(K1-K6), 34 requests with the expected stop and what must never happen.

**Home additions to W3, following the decline decision** (confirmed by the owner, 06:35 CEST):

- R20: a decline record with another field → `SCHEMA_INVALID`.
- R21: a decline record with an unknown reason → `SCHEMA_INVALID`.
- K6: the plain decline record → declined.
- Requests 1, 7 and 8 now expect D (the decline record) instead of E; W3 had marked them "(D)".

**Checked before fixing** (2026-10-20, Python's `json` and `jsonschema` 4.26 on the r2 schema, the r2 check order;
this is a check of the set, not gate-1 code): every record that W3 expects to stop at checks 0-3 stops there with the
expected code; every other record and every control passes those checks; K6 is a decline. 29 of 29.

**Depends on open points** (IR r2 §4): K4 assumes TFIM `circuit` output is in v0. `chi_max` 256 here is a test value
(the owner's 128 is for the noisy twin), as are `CZ_max` 500 and `SX_max` 2,000.

**Files:** `make_fixtures.py` (writes `fixtures/F1-F4.json`, reproducible: two runs, same SHA-256), `make_n2.py`
(writes `n2.json`, reproducible), `n2.json`.

Fixed: 2026-10-20 06:35 CEST. SHA-256 of `n2.json`:
`458e53df0d3df9f869a1beb4958994ecf802ee30b7719104322c30f991d781ef`. All pins were rechecked against the files at
fixing (IR r2 text, r2 schema, F1-F4: all equal). From here `n2.json` is never edited; a change is a new file.
