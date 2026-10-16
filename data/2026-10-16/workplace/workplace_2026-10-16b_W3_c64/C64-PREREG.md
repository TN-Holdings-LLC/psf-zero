# C64: pre-registration (workplace; fixed before the work, at the CEST time in PREREG_SHA256_C64.txt)

**Task.** Note 10, W3, as amended by home's two answers of 2026-10-16. `c64/make_c64.py` builds `c64/psf_compile.py`
(`VERSION = "2026-10-16.c64"`) from c63a. c64 has the same outputs as c63a and none of the code that its default calls
cannot reach. Behaviour is not changed.

## 1. Inputs, fixed by SHA-256

    9d0fecbd09c4745d5384324fc66fea644deca95b2b07a8e06df74aad5b209492  w34/w3/c63a/psf_compile.py (2026-10-15.c63a, the 2026-10-15c delivery)
    13771bd67ba5f74fa20efb13d37e936211ec096d93460e87fbc98df8e0fe1d1f  w34/c63/test_c63.py (D4)
    28de2c381a8e9e748eb35a4291d374a8826e56749c923999adc9f86f31001c8a  w34/c62/psf_compile.py (test_c63.py's reference)
    ee9e4cc5bb98830b4ba7e7df44d2b6814931a2a05bb7f38d244f7efb7a4c783e  w34/c63_val.py (D1's inputs are its inputs)
    f05e952e2a11ec3c7576d32bc59524346a588248412057a9db7c4e1a554ebf3d  search_survey_2026-10-16.md (home's survey; context only)

**Environment.** Python 3.11.15, Qiskit 2.5.2, pyflakes 4.0.3, bundle q, the same 2-CPU machine as C63-VAL.

**Written after fixing** (their SHA-256 values go into the report):

- `c64/make_c64.py`, `c64/psf_compile.py`, `c64/CHANGELOG_c63a.md`;
- the D1/D5 harness `c64_check.py` and its scoring.

## 2. What is removed and what is kept (home's answers, unchanged)

1. **Switches.** Every module-level switch takes c63a's default value, and the branches that this value cannot reach
   are removed. Examples: `CHAIN_BRICKWORK`, `CHAIN_FRAME`, `GRAPH_BRICKWORK`, `EXACT_CHECK`, `ZZ_RELAYER`,
   `FRACTIONAL`, `SELECT_BY`, `PLACE_SCORE`, `COMPRESS`, `ABSORB_SYNTH`, `PRUNE_FIRST`, `CUT_FAILED_QUBITS`,
   `MAP_FREE_DEFAULTS`, `DROP_DIAGONAL_BEFORE_MEASURE`, `ST_SCHEDULE`, `USE_*`, `TEMPLATE_FRAME_FIXED`, `GATE_CLOSED`,
   `FORMULA_ONLY`, `FORMULA_CHECK`, `SCOPE_GATE`, `CZ_OWN_EMIT`, `TEMPLATE_BATCH`.
   - **Recursion guards.** A guard such as `_S96_DEPTH` is treated as constant 0 only where every statement that
     raises it has been removed.
2. **Kept code.** The default calls are `compile_for_hardware` and `compile_template` (with `CompileTemplate`'s
   methods). After the switches are fixed, everything referenced by name from kept code is kept, and so on
   transitively. This is decided by reading the code and following the references. The survey's "Never run" table is
   **not** used. Among others, `_fold` (with `_errorless`, `_strip_global_phase_gates`), `_duration_cx`, `_T100Mat`,
   `_T100NotHere`, `OutOfScopeError` and `_st_cliff1q` are kept; any further names kept this way are listed in the
   report.
3. **Names used by c63-line files.** These are kept even when they start with `_`, together with everything they
   reference. "Other files" means only the files that load the c63 line: the workplace's `c63_val.py`,
   `test_c63.py` and `w3_walk.py`, and home's a23. Files that load older versions do not count.
   - From the workplace files: `compile_for_hardware`, `compile_template`, `VERSION`, `WARN_WITHOUT_TARGET`,
     `TEMPLATE_CACHE`, `TEMPLATE100_STATS`, `OutOfScopeError`, `_failed_elements`, `_uses_failed`, `_X93_BASIS`,
     `_n99_same`, `_n99_kron_steps`, `_t100_canonical`.
   - From a23 (home's list): `compile_for_hardware`, `compile_template`, `TranspilerError` (an import), `_chain_lattice`,
     `_chain_place`, `_failed_elements`, `_uses_failed`, `_fold`, `_scope96_edges`, `_scope96_embeds`,
     `_native_brickwork`, `NATIVE_BRICKWORK`.
   - **`NATIVE_BRICKWORK`** stays as a constant fixed at `True`, because a23 reads it with getattr. Nothing inside c64
     branches on it.
   - `WARN_WITHOUT_TARGET` and `TEMPLATE_CACHE` stay as settable module variables, because those files set them.
4. **Also removed:**
   - the first, overwritten definition of `_n99_kron_steps`;
   - pyflakes' 5 existing findings in c63a's kept code (3 unused imports, the unused local `N`, the duplicate
     definition);
   - module variables (such as `*_STATS`) and imports that are left unused.
5. **Not removed:**
   - VF2's first match (`_chain_place`, `_graph_place`, `_scope96_embeds`);
   - Qiskit's level-0 pass manager on the F path of `_chain_brickwork` / `_graph_brickwork`;
   - `_better`.
6. **Docstring.** The module docstring (the changelog of items 1-101) is moved unchanged to `c64/CHANGELOG_c63a.md`.
   c64's docstring is a few lines.
7. **Hand edits.** None to the output: `make_c64.py` applies exact substitutions and line-range deletions to c63a.

## 3. Criteria

| ID | criterion |
|---|---|
| D1 | **Outputs identical to c63a** on every input of C63-VAL: part R (FakeFez) and part D (FakeMarrakesh, FakeKingston, FakeTorino, the Marrakesh snapshot); T and F; the build input plus the same-sign and free draws, in C63-VAL's order and seeds. That is 49,590 inputs. For each input, compare `compile_for_hardware(qc, target=tg)` and `compile_template(qc, tg).bind(qc)` between the two: the circuits are equal under `==`, and the initial and final layouts are equal (`initial_index_layout(filter_ancillas=False)` and `final_index_layout(filter_ancillas=False)`, or both None). Where c63a raises, c64 raises an exception of the same type. **Refusals:** for each of the 19 tests, a call without a Target (`compile_for_hardware(qc)` and `compile_template(qc, None)`); for each of the 5 devices, T and F, a circuit with a three-qubit gate (ccx) through both calls. The same exception type is required. |
| D2 | Running `make_c64.py` again gives c64 byte for byte. |
| D3 | pyflakes reports 0 unused imports, 0 unused variables and 0 undefined names on c64. |
| D4 | `test_c63.py` (8 tests), run with `C63` pointing to c63a and then to c64 (`C62` = w34's c62): the 6 tests other than `test_version` and `test_rebuild_on_sign_change_and_still_exact` give the same result on both. |
| D5 | Compile time: in each of D1's 190 cells (38 cells x 5 devices), on the build input, `compile_for_hardware` is timed alternately (c63a, c64) 3 times each. The median over the cells of the ratio median(c64) / median(c63a) is <= 1.00. |

**D4: predicted outside the criterion** (stated now):

- `test_rebuild_on_sign_change_and_still_exact` fails on c63a and on c64 alike. Option A does not rebuild on a sign
  change. Seen before fixing: c63a gives 7 passed, 1 failed.
- `test_version` fails on c64 only, because `VERSION` changes to `2026-10-16.c64`.

**Reported, not criteria:**

- line counts (c63a to c64);
- the number and lines of removed functions and classes;
- the public names and switches kept;
- the names kept by rule 2 beyond the survey's "run" list;
- import time (c63a and c64, median of 5, each in a fresh process);
- D5's largest and smallest ratio;
- whether `value_sig` (repr-exact floats) also agrees in D1.

**If D1 fails in any cell:** that cell is reported. The removal that caused it is restored in c64 through
`make_c64.py`, and D1 is run again in full. Every restoration is reported. The criteria are not changed.

**Order of work.** make_c64.py and c64, then D2, D3, D4, then D1 and D5 with nothing else on the machine.
