# Prototype B (2026-10-16): formula against VF2 in c63a's placement sites

`protoB.py` patched c63a in place (its file unchanged) for the measurement. Calls: C63-VAL's 19 tests (5 devices, T
and F, `compile_for_hardware` and `compile_template` + `bind`; 380) and TOQB's ring circuits through a23 (544).

**Time in the VF2 sites, c63a as is:**

| site | C63-VAL | TOQB rings |
|---|---|---|
| `_chain_place` (snake first; VF2 never reached on C63-VAL) | 0.02% | 0.1% |
| `_graph_place` | 5.4% | 0.2% |
| `_scope96_embeds` | 6.8% | 19.3% |

**B-a** (adopted as item 102, c65): `_scope96_embeds` decides by formula first -- an odd cycle, or a cycle shorter
than the device's shortest, cannot embed in heavy-hex; a path on the snake's longest working piece embeds. Output:
924 of 924 identical, bit for bit. Time (alternating, 5 rounds, medians): C63-VAL +0.7% (noise), TOQB rings -10.4%
(up to -72% per call: a 19-qubit ring's template 32 -> 10 ms).

**B-b** (not adopted): a path placed on the snake instead of `_graph_place`'s first VF2 match. No time gained (about
3 ms per call before); placement changed in 92 calls (GHZ, CZ and CX chains), with the same two-qubit count and
duration in all. By the owner's rule, the cheaper one stays: VF2.

**Left:** a 40-qubit ring (tfim_n40_r2) still costs VF2 about 35 ms twice (test and placement), 30-60% of that call;
no formula for long rings on heavy-hex was built. `_chain_lattice` is rebuilt on every call (about 5%).
