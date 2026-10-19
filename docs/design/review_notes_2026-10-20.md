# Home review of the workplace drafts of 2026-10-20 (morning CEST)

**`psfz_improvements_proposal_2026-10-20.md`**

- §1: the identity M = G · T_m ··· T_1 with T_k = A_k^† S_k A_k was checked by hand for m = 1 and 2, and the table of
  forms for t_G = 0, π, π/2 matches r4's `zsx` branches with l → l + φ. The rule reads only gate names and fixed
  factors, so S1-S3 hold as claimed. Its change to `pick` (pulses by class) can move basis changes against today's r5
  output; the proposal already lists that as a check.
- §2: the pattern exists in the compiled output (psfz merges the H of the ZZ form into the neighbouring runs, so the
  middle of CZ · run · CZ is one run on b and nothing on a).
- §3: correct as an upper bound at block boundaries. Note for the IR draft (below): gate 1 and the twin must use the
  same bound.
- Nothing here is built yet; building it is the owner's call.

**`IR_v0_spec_draft_2026-10-20.md`**

- R4's counts match the template check of 2026-10-19 (n 12: L 8 96 CZ / 216 sx, L 16 192 / 408).
- Z5's folds hold: rzz(θ + 2π) = −rzz(θ); (X ⊗ I) rzz(θ) (X ⊗ I) = rzz(−θ); rzz(θ ± π) = ∓i rzz(θ) (Z ⊗ Z).
- T3 uses the input-gate bound (rank 2 per rzz). Today's `psfz twin` refuses by the CZ-count bound (2^(2 · steps)), so
  until improvement §3 is built, gate 1 would admit records that the twin then refuses. One formula, shared by both, is
  needed (or T3 cites today's bound until then).
- `ring_classifier` requires n >= 12 (heavy-hex), so TOQB's design C circuit (n = 10, folded onto a line by the AI
  front end's ring formula) is not an IR v0 record. That is consistent with psfz's scope; it means the first device
  stamp does not exercise IR v0.
