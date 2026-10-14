# N5 (2026-10-14): c49's default call with each remaining estimate-based part removed, one at a time

**Status: offline, done (a list only; removal is home's decision).** Script `n5_ablate.py`, rows
`n5/run/n5_rows.jsonl`. C48-VAL's 16 tests, FakeFez and FakeTorino, C49-VAL's call (`backend=`,
`entangling_basis="cx"`, `layout_search=True`, seed 0).

| removed | how | outputs changed (of 32) | duration vs c49 | read |
|---|---|---|---|---|
| items 33/60 (re-placement by the Target's errors) | `placement_refine=False` | 3 | 0.96-1.00, never longer | removable at no cost here |
| item 64 (routing at level 3 with a device) | `routing_optimization_level=1` | 2 | 1.00 (same CZ, same duration) | removable at no cost here |
| item 75's score (deterministic re-placement scored by -ln(1-e)) | re-placement applies nothing (the current placement kept, as if every placement scored the same) | 29 | 0.86-1.11 (geometric mean 1.00); 11 shorter, 8 longer | decides most outputs, in both directions; on FakeTorino TFIM it is longer without the score (1.05-1.11) |
| item 78 (re-placement at Qiskit's routing stage) | the routing stage gets no re-placement (as before c45) | 2 | 1.00-1.003 | removable at no cost here |

**Notes.**
- **Determinism:** "item 75's score removed" keeps the output deterministic, because nothing is chosen.
- **What the score carries on FakeTorino:** FakeTorino's gate lengths differ from coupler to coupler. The
  error-scored re-placement there also lands on shorter gates more often than the current placement does. That is the
  only place in this list where an error value plausibly carries duration information.
- **Possible replacement (not implemented):** a re-placement scored by the device's gate lengths (the summed duration
  of the placed operations) is the duration-only counterpart. It would keep what item 75 gains on FakeTorino without
  reading an error value.
