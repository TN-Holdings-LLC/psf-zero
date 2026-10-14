# Workplace -> home, 2026-10-14 (c): N3, N4, N5, N6, and the quick rx / rzz / Pauli-frame check

Offline only; no IBM connection, no push. Common rule followed. Base: bundle psf-zero_2026-10-14a (5ae4947); c49
sha256 5544d7c9...; scripts run from the repository root.

| folder | result |
|---|---|
| `n3/` | W3's parity is item 73's reader: at odd L one CZ ring is left at the end (entangling), so the circuit is not read (32/32 odd cells `not_applicable`, c48 and c49) and goes by the usual path; at even L the rings cancel and STEINER6 applies. c48's 6-qubit step at L = 10 was item 69's ESP choice ("kept_usual" up to L = 8); c49 takes STEINER6 from L = 2 (FakeFez n=6 L=8: 10.00 vs 15.05 us) but is 2-4% longer than c48 from L = 10. Fix idea: drop a diagonal Clifford left before measurements. |
| `n4/` | Variant B vs c49: equal or shorter only where an exact embedding exists (ghz; SCALE1 L and R2: up to 10% shorter at n=96); everywhere routing is needed it loses (median 2.7x on C49-VAL). No refusals on either side. |
| `n5/` | Removing 33/60, 64 or 78 changes 2-3 of 32 outputs and never lengthens duration by more than 0.3%; removing item 75's score changes 29 of 32 (0.86-1.11x, geometric mean 1.00). |
| `n6/` | Pre-registered compiler comparison on real hardware: still not found (52 citing works of Mills et al. 2021 scanned, ~16 searches). Two near misses (frozen protocol but no hardware); one of them is TOQB, this project's own work. |
| `frac/` | rx as one pulse 0.85-0.90x on chains; native rzz 0.34-0.46x on chains (Heisenberg too, when built as a brickwork); Pauli frame recovers negative angles (0.44 -> 0.37); classifier: rx only (0.93-0.97). |

`frac_quick.py`, `n4_variantB.py`: the import paths were set to the repository's committed W2/W5 folders (`n4` also
needs the item 86 build from the N1 kit for its lattice helpers, path as its second argument).

Check: `python check_intake.py`.
