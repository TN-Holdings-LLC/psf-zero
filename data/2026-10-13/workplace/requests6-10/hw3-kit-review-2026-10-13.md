# Workplace review of data/2026-10-13/hw3_kit (request 10 of 2026-10-13), from bundle psf-zero_2026-10-13d (Addendum 463): DD placement and the pad/dd schedules are right; D1 and D2 have a selection loophole (regression to the mean); and a correction to the earlier ns_sim review that also affects HW2-PLUS's P1 and P2

**Status: independent reading by a session that did not write the kit, with small checks on FakeFez.** Times CEST.

## 1. Is decoupling inserted only on the active qubits?

**Yes.** `variant(..., "dd")` passes `qubits=active`, the qubits that carry any instruction other than a barrier, to
`PadDynamicalDecoupling`. Checked on a three-qubit circuit on FakeFez:
- a qubit idle in |0> before its first gate gets a delay, not X-X (Qiskit's default `skip_reset_qubits=True`), so
  decoupling does not flip qubits that are still in their initial state;
- a qubit that waits after its last gate until the common measurement gets X-X pulses in that window.

## 2. Are "pad" and "dd" on the same schedule?

**Yes.** Both run `ALAPScheduleAnalysis(target)` on the same compiled circuit; "pad" fills idle windows with delays,
"dd" with X-delay-X-delay sequences inside the same windows. Gate start times and the total duration are the same (the
FakeFez check gave equal durations). Two points to state with the results:
- "dd" adds X gates, each with its own error; the longer circuit receives more of them. D2 (decoupling helps the
  longer circuit more) can therefore fail even if waiting error is real, simply because the pulses cost.
- ALAP does not remove waiting after a qubit's last gate when that gate is entangled with a qubit that continues: in
  the check, qubit 0 waited from its CZ to the measurement. "pad" leaves that window undecoupled, "dd" decouples it,
  which is the intended contrast.

## 3. Are D1 and D2 conditioned without a loophole?

**No: the selection rule uses the same "pad" measurements that D1 and D2 then compare, so regression to the mean can
confirm both without any effect of decoupling.**

- A pair enters D1/D2 if F_short(pad) - F_long(pad) exceeds the largest repeat difference. Pairs whose pad ratio was
  pushed up by noise pass more easily. Their dd ratio does not share that noise, so |ln ratio(dd)| < |ln ratio(pad)|
  (D1) is favoured even if decoupling does nothing.
- D2 compares F_long(dd)/F_long(pad) with F_short(dd)/F_short(pad). The selection favours pairs where F_long(pad) came
  out low by chance, which inflates F_long(dd)/F_long(pad). D2 is favoured the same way.
- The threshold (largest repeat difference, over all six circuits of the pair) reduces this but does not remove it:
  with three pairs and two repeats, one noisy value can decide.

**Fixes that keep the predictions' meaning (any one, fixed before the run):**
1. **Split the repeats:** select pairs on repeat 1 (pad) and evaluate D1/D2 on repeat 2 only (or the reverse, fixed
   in advance).
2. **Select on something measured independently:** for example on "plain" (separate circuits), or on the duration
   ratio known at `prepare` (no outcome involved).
3. **Signed D1:** use ln ratio(dd) < ln ratio(pad) instead of the absolute values, so that a dd result where the
   shorter circuit loses by more than it won under pad is not counted as "shrinking" by accident; or report both.

## 4. Correction to the workplace's ns_sim review (and what it means for HW2-PLUS P1/P2)

The earlier review accepted the docstring's statement "with ALAP no qubit waits after its last gate". That is not true
in general (section 2), and the code never charges any waiting after a qubit's last gate: in `noisy_circuit` (ns_sim.py
and ns_sim2.py) relaxation is applied only to gaps between two gates on the same qubit, and nothing is applied from a
qubit's last gate to the measurement at `total`.

- Under **ASAP** (HW2-PLUS's P1 and P2), early-finishing qubits wait until the common measurement, and that window is
  not charged. On HW1's circuits it is 7% to 52% of all idle time (ASAP, Heron durations), and it differs between arms
  of the same test (tfim_n12: PSF 31%, QK3 13%; ba_n18: PSF 17%, QK3 9%), so it can move the ratios.
- Under **ALAP** the uncharged window is smaller but not zero (section 2's example).
- P1 and P2 are fixed with their code hashes and should stay as they are. Suggestion, before HW2's results are read:
  declare a corrected variant (P1c, P2c: relaxation from each measured qubit's last gate to `total` added) as reported
  predictors, and record that it was added after this review and before the results.
- This was missed in the first review; it is recorded here rather than corrected silently.

## 5. Smaller points

- D4 (plain against pad) is useful: it shows how the device times circuits that carry no explicit delays.
- The rough QPU estimate (144 s for 36 submissions) is consistent with HW1's ratio; fine within the allowance.
- The kit's own SHA256SUMS were not re-verified here against the owner's copy; `95_hw3_prep.sh` does that.
