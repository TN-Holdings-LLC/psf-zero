# NS2 -- fixed before HW2 runs (local; committed before HW2's results exist)

Fixed: 2026-10-13 04:58:06 CEST (Europe/Zurich).

**What it is.** The same noise simulation as NS1 (`ns_sim.py`, sha256
0d02400d713e18cdc6076d1c8d6bfc0337993335c54e617c6eb054467135e379, unchanged since NS1) on HW2's submitted circuits,
with HW2's device's calibration saved right after HW2's `prepare` by `ns_snapshot.py` (sha256
1c4a4cc5b8ad11499aee44ca3eb1295310add3ce9b0814dae2a197bada55b288; run by `92s_ns_snapshot.sh`, sha256
c96d5fef34194696cf6c9ef74602ea2d78147f026ea6c85531433238ab621c54, VPN off, read-only). Runner: `ns_hw2.py`
(sha256 7ee7457d8867e4e208d177d65a8a48cf3ca379a5dfb3ab03404febc158c96930). The snapshot has no initialisation
errors (not in the backend's properties): they count as 0, as the model allows.

**Seen before this file:** NS1's results (Addendum 462): on HW1, the simulation picked the device's winner on 3 of 7
reading tests (NS1a refuted), with rank correlation 0.00 (NS1b refuted); its magnitudes were within 0.5-2 times on 14
of 16 circuits (NS1c), and the arms fell short of it alike (NS1d). Nothing from HW2's device.

**Circuits.** HW2's part R circuits on the seven reading tests, GHZ and tfimP_n12/16 (repeat 1's), and part S's pairs.

**Predictions** (the same thresholds as NS1; NS1 refuted NS1a and NS1b, so these are a second, prospective trial of
the same model, not an expectation of success):

| ID | prediction |
|---|---|
| NS2a | the simulation picks the device's winner (PSF or QK3) on >= 6 of every 7 reading tests scored |
| NS2b | rank correlation of the simulated and measured F ratios on the reading tests >= 0.6 |
| NS2c | simulated F within 0.5-2 times the measured F on >= 80% of part R's circuits (tfim_n16 left out) |
| NS2d | F_real / F_sim per arm: \|ln(PSF / QK3)\| < 0.153 |
| NS2e | reported: part S, the simulation's S1 / S0 ratio against the device's |

**Reading, fixed now.** If NS2a and NS2b are refuted again, a Markovian model of the reported error rates and T1/T2
does not rank compilers on these devices (two devices, two days), and the next test is of an error the model leaves
out, by hardware (dynamical decoupling on against off on the same circuits), not by adding fitted terms to the model.
