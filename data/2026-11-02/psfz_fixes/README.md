# psfz fixes of 2 November 2026: before/after (exploratory, not declared)

`ab.py` compiles chain circuits (n = 8, FakeFez's device file) with psfz at eb864ead ("before") and with the fixes
("after"), checks each output's outcome distribution against the input (1 - total variation distance) and counts cz,
sx, x and the as-soon-as-possible duration. Families: `gen` (rxx, rz, ryy per pair: a general block), `pi` (rzz(pi),
rxx, ryy: a canonical block with one angle at pi), `mix` (both), `gen2` (general blocks with varied single-qubit parts).

| family (circuits) | cz | sx | duration (ns) | | file |
|---|---|---|---|---|---|
| pi (8) | 18 -> 14 | 68 -> 60 | 684 -> 552 | fix 1 (lower_can mod pi) | `final_ac157fad.json` |
| mix (8) | 16 -> 14 | 68 -> 64 | 684 -> 576 | fix 1 | `final_ac157fad.json` |
| gen (8) | 14 -> 14 | 68 -> 68 | 576 -> 576 | unchanged | `final_ac157fad.json` |
| gen2 (40), with d810f8ed | 21 -> 21 | 81.5 -> 81.7 (8 changed: 6 more, 2 fewer) | 840 -> 840 | fix 2, reverted in ac157fad | `with_d810f8ed.json` |
| gen2 (40), final | 21 -> 21 | 81.5 -> 81.5 | 840 -> 840 | unchanged | `final_ac157fad.json` |

Every output is exact (minimum 1 - TVD 0.999999999998, rounding).
