# W2 (2026-10-15, interim): faster template binding (prototype c63 = c62 + item 100 with planned binding); C63-VAL part R stopped after 20 of 38 cells at hand-over

Workplace, 2026-10-15. Base: c62 (W1). Offline only.

- **Pre-registration:** fixed at 08:52:23 CEST (`C63-VAL_preregistration.md`; SHA-256 and time in
  `PREREG_SHA256_C63.txt`).
- **Part R:** started 08:52 CEST and stopped on purpose at 09:59:18 CEST (`TZ=Europe/Zurich date`), when the user
  left. 20 cells are complete: the 10 items-87/91 tests on FakeFez, T and F, 1,101 inputs each.
- **Not run:**
  - part R: hhtree_heis_n16, ghz_n12 and the 7 W1 native tests (18 cells);
  - part D (the other 4 devices).

## 1. What c63 changes (in `c63_100_code.py`; `make_c63.py` on c62)

See the pre-registration for the details. In short:

- **Planned binding:** each shape keeps its emission plan, so a bind multiplies runs and does not re-emit.
- **Constant runs:** their angles are baked into the circuit.
- **Items 87/91:** one numpy batch per bind for all runs (products, ZYZ, class check). Item 99 keeps CPython's
  arithmetic, which is needed for bit identity.
- **Pinned runs:** a run containing a factor of a general Weyl block is always full form, so Weyl branches do not
  change the shape. hhtree_heis_n16 T was 15.5 ms in c61 because it changed shape on almost every draw; it is now about
  3-3.5 ms (isolated probe).
- **Weyl checks:** batched once per bind.
- **Term supports:** cached.

## 2. C63-VAL part R, the 20 completed cells (`run63/c63_val_score.txt`)

| ID | result on the 20 cells |
|---|---|
| V2 (T2) | **holds:** 11,010 F outputs; exact (largest difference 2.3e-15; 3,081 not checkable or not scheduled at 17-40 qubits), native, rzz in range, no failed element; duration and 2q equal to c62's in every output |
| V3 | **holds:** 11,010 T outputs; exact (largest 8.8e-15), native, no failed element |
| V1, V4 | not reached (the native tests and part D were not run) |
| S1 (interleaved) | **misses in 4 of 20:** heis_n12_r2 T 2.41, hhI_n16 T 2.14, hhI_n20 T 2.67, hhI_n20 F 2.05 ms; tfim_n40_r2 T 2.45 / F 1.96 (limit 5) |
| S2 (back to back) | **20 of 20 within the limit;** largest tfim_n40_r2 T 1.62, hhI_n20 T 1.45 ms |

**Notes on the table.**

- No verdict is drawn: the predictions are over 38 cells and part D.
- From the isolated probe, hhtree_heis_n16 T is expected to miss both S1 and S2 (about 3.5 ms).
- C61-VAL's interleaved medians, for comparison: tfim_n12 T 1.15 ms, ring12_tfim T 2.11 ms. c63: 1.08 and 1.76 ms.

**Reported.**

- **Durations against c62:**
  - every Ising T and every F output equals c62's duration and two-qubit count;
  - heis_n12_r2 T: median 1.0029 x, largest 1.0236 x, at or below c62 in 366 of 1,101 inputs. This is c61's XX + YY + ZZ
    closed form against item 101's Weyl form.
- **Shape changes:** 0 in every T cell. Exactly 101 in every F cell: the 100 free draws, which change signs, plus one.
  None on the same-sign draws.
- **`values_of` is now often slower than the bind:** 1.4-5.4 ms (for example tfim_n40_r2 5.3 ms, hhI_n20 3.4 ms). It
  reads Pauli evolutions from a circuit. A loop that already has the values should call `bind_values` directly.
- **The bind's own assign_parameters:** 0.15-0.53 ms of the bind.
- **Build against c62's compile:** 16-100 ms against 16-110 ms. F builds are faster than c62's compile.

## 3. Where the remaining time goes (hhtree_heis_n16 T, isolated, `prof6.py`)

| part | time |
|---|---|
| item generation | 1.3-1.4 ms (11 general Weyl blocks about 0.54 ms, 38 XX + YY + ZZ blocks, Python yields) |
| numpy batch | 0.4-0.5 ms |
| assign_parameters | 0.6-1.0 ms |

- **Next step if 2 ms is wanted:** generate the value-dependent factors of the closed forms as numpy arrays directly from
  the values (no Python item tuples). The Weyl decompositions would remain per block.

## 4. To finish C63-VAL at home (same script, same criteria)

```
OMP_NUM_THREADS=1 C63_ONLY=hhtree_heis_n16,ghz_n12,chain_cz_n12,chain_cz_n16,ring_cz_n12,hh_cz_n16,hh_cz_n20,cx_chain_n12,cx_chain_n16 \
    python <dir>/c63_val.py <out>/R2 R
OMP_NUM_THREADS=1 python <dir>/c63_val.py <out>/D D
```

Then put `R2`'s rows file into `run63/R/` (under its own name) and the D output in `run63/D/`, and run
`python c63_score.py run63`.

**Speed numbers from home are not comparable.** The home machine differs, so S1 and S2 measured there must be reported
as from the other machine. The correctness predictions V1-V4 do not depend on the machine.
