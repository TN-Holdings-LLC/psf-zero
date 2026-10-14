# TIME1: design draft (not fixed; nothing runs without the owner's go-ahead), 2026-10-14

**Question.** On a third IBM device, does the as-soon-as-possible duration of a compiled circuit pick which of two
compilations of the same input the device runs better? Is it better at this than the two-qubit gate count and ESP?

- On HW2 (ibm_marrakesh) duration picked the winner on 7 of 7 pairs, but only after the fact.
- On HW1 (ibm_kingston) the clear wins were also the much shorter circuits.
- Part S (HW2): the same gates in a shorter order raised F by 1.30 and 1.17 times.

The new compiler c49 chooses by duration (item 84). TIME1 tests the assumption that choice rests on.

**Device.** A Heron r2 device other than ibm_kingston and ibm_marrakesh: ibm_fez if it is operational, otherwise
the one with the fewest pending jobs. The rule is fixed before the run; the calibration time is recorded and used
for nothing else.

**Pairs.** One logical input, two compilations, scored by normalised linear XEB F, as in HW2. Every input is read
from a fixed random product state.

| class | pairs | what differs |
|---|---|---|
| T1 (order only) | TFIM chains n = 12, 16, 24 with 4 and 8 Trotter steps: c49 with item 83 on and off | Only the order; same gates and count. Durations differ by 3-10 times. |
| T2 (duration against gates) | Pairs where the shorter circuit has more two-qubit gates or a lower ESP. Chosen offline, before the run, from c49 / c48o / QK3 outputs of HW2's inputs on the device's Target. At least 4 pairs. | Duration and gate count / ESP point in opposite directions. |
| T3 (HW2's seven again) | HW2's seven reading tests: c49 (default call) against QK3 | Replication of "the shorter wins" on a third device. |

**Fixed predictions (draft).**

- **TA:** in T1 the shorter order has the higher F on at least 5 of 6 pairs.
- **TB:** in T2 the shorter circuit has the higher F on at least 3 of 4 (or 75% if there are more).
- **TC:** over all pairs, the rank correlation of 1 / duration ratio with F ratio is at least +0.5, and higher than
  that of the two-qubit ratio and of the ESP ratio.
- **TD:** in T3, c49 at or above 0.95 x QK3's F on at least 6 of 7.

If TA, TB and TC hold, "choose by duration" is recorded as confirmed on a third device and can be stated in the
README. If they do not, item 84 is re-examined before any release claim.

**Size.** About 30-40 circuits, 2 repeats, 3,000 shots each: about 60-90 QPU seconds (HW2: 60 circuits, 102 s).
About 5.5 minutes of QPU are left.

**Open before fixing.**

- The exact T2 pairs. They are selected offline by a rule written first (opposite directions; duration ratio < 0.8 or
  > 1.25).
- XEB needs exact output distributions. n = 24 is 2^24 amplitudes, which is fine offline.
- The kit (prepare / submit / fetch / score scripts, rehearsal on a fake device), as for HW2.
