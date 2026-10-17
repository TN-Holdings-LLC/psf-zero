# Prototype c66 (home, 2026-10-16): item 103, item 88's frame in a template by segments

`make_c66.py` = release 2026-10-16.1 (`psf_compile.py`, c65) + `item103_frame_code.py` -> `psf_compile.py`.
**Validated in C66-VAL (Addendum 513) and released as 2026-10-16.2 (Addendum 514).** `make_release.py` makes the
release from this prototype (docstring, version line, and the prototype switch `TEMPLATE_FRAME_SEGMENTS` removed: here,
`False` gives back option A, c63a's template); `check_release.py` compares the installed release with this prototype,
bit for bit, on C63-VAL's inputs (`check_release_result.txt`: 49,590 template binds and 2,090 default calls, 0 different).

**Why.** Option A (kept by the owner, Addendum 508) fixes a template's shape by writing every negative rzz as
X RZZ(|theta|) X with both runs pinned to the full form; on fractional targets, same-sign inputs then cost 1.15-1.35 x
c62's duration (C63A-VAL). Counting pulses on tfim_n12 (FakeFez, F): c62 24 rx, option A 50; 20 of the extra 26 sat
between two rzz, 6 after the last.

**Item 103.** The non-diagonal single-qubit runs ("anchors") cut each qubit's timeline into segments. Conjugating a
whole inner segment by X costs nothing (the two X's merge into the anchors' runs, whose class does not change) and
flips the sign of every rzz in it on that qubit. Which segments to conjugate so that every rzz angle is positive is a
GF(2) system, one equation per rzz, solved by a spanning forest found at build; an rzz that closes a cycle, or joins
two fixed segments, keeps option A's pinned local form. An rzz on two qubits that are still basis states, or (in a
measured circuit) after their last anchor, needs no sign. Folds are a Z frame carried to the next anchor.

**First measurements (exploratory, before any pre-registration; FakeFez, FakeMarrakesh, FakeKingston, FakeTorino and
the Marrakesh snapshot; F targets; C63-VAL's 11 F-path tests; per cell the build input, 10 same-sign and 10 free
draws from a label C63-VAL does not use):**

- duration over c62's, same-sign inputs: option A 1.000-1.350 (median per cell); c66 0.806-1.000, never above c62;
- shape changes: 0 in every cell (the first version changed shape 5 and 9 times on the Heisenberg cells; anchors are
  now chosen by whole runs, not single gates);
- exact (outcome distribution within 1e-6) on every input that could be checked: all inputs of 12 qubits, and 3 inputs
  each of the 16- and 20-qubit cells (largest difference 9.7e-15); the 40-qubit cell was not checked;
- native and rzz in range everywhere;
- C61-VAL part K's 100,000-bind walk (tfim_n12, FakeFez F): 0 shape changes, exact at every 1,000th lap, duration
  0.83-0.87 x c62's, the same two-qubit count, bind median 0.64 ms (option A 0.39 ms here), peak memory 213 MB.
