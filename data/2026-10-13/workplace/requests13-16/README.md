# Workplace -> home, requests 13-16 (2026-10-13)

Packed 2026-10-13 09:48:33 CEST (`TZ=Europe/Zurich date`). Nothing here submits a job; no real-device job was run.

| request | folder / file | status |
|---|---|---|
| 13 T2*/ZZ v2 | `t2zz_kit/` (design v2, `t2zz_real.py` prepare/submit/fetch/run/score/rehearse, 4 tests, power table, FakeFez+Aer rehearsal) | done; fixed with this pack |
| 14 c49 harness | `t2p_kit/` (`t2p_eval.py`, fixed circuit list `circuits/t2p_circuits.json` with the 20 new BA seeds 4,840,000 + 1000 n + i, design note) | done; the full stand-in rehearsal (31 circuits x 4 devices) was still running at packing: `rehearsal_output_partial/` has FakeFez, 26 circuits (H0 26/26, V0 53/53, no errors) |
| 15 HW2 sheet | `hw2_sheet/hw2_sheet.py`; `sample_sheet_from_hw2_rehearsal_partial.md` | script done and run on the HW2 rehearsal (FakeFez, noiseless) with part R/S and partial NS2; NS2 score, P1-P4, P1c/P2c and the plan-v2 analysis had not finished on the rehearsal at packing. `verify_on_rehearsal.sh` reruns the whole chain (edit the two placeholders) |
| 16 literature | `heron-zz-and-scheduling-2026-10-13.md` | done: no Heron-specific ZZ value published (bound < 10 kHz on the precursor architecture); no official ASAP/ALAP statement, but `scheduler_timing` returns the executed schedule |

Main points:
- 13: v1 was underpowered (ZZ SE 0.4-0.9 kHz); v2 = 2,000 shots + tau 30 us + weighted fit up to 40 us: SE 0.16 kHz at
  T2* 60 us; rough QPU 100 s (about 40 s by HW1's ratio). Rule (a) changed to T2 >= 25 us and a d(0) rule added,
  both disclosed in the design (home may revert).
- 14: contract with c49: `placement_score` = "esp" (must equal c48: check H0) / "t2p" / "t2p_half".
- 15: the sheet only reads the tools' outputs; any number that disagrees with a tool's own is flagged.

Check: `python check_intake.py`.
