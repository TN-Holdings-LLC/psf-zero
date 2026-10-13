# C48-VAL -- prestatement (fixed before the run)

Fixed: 2026-10-12 15:40:10 CEST (Europe/Zurich). Offline only: fake devices, no real device, no payment.

**What is tested.** Prototype c48 = c47 + item 83: STEINER6's blocks (one gadget per term, one per SWAP; a SWAP that
joins the last gadget on its pair stays fused to it) re-emitted by list scheduling, only commuting blocks changing
places, kept when the as-soon-as-possible schedule in CX units is shorter. Motivation: Addendum 455 (HW1 on
ibm_kingston): PSF-Zero won on hardware where its circuits were shorter in time, not where its ESP was higher.

**Files.**
- prototype `psf_compile.py` sha256 0400100f416421dba36c921fa3801f75c37e9e15ea39c22a6639cbc9db976d39
- tests `test_c48.py` sha256 3d54d659cff4c6c107b98b206ef7bd1f5c6b8701ff236cf8f27ea2789d6ab2c9 (36 passed before this prestatement, 38.8 s)
- run script `c48_val.py` sha256 3c37dd3615683d59a5d7ce1a3072155c0208cfdc63354362958d90841396320c

**Design.** Devices FakeFez, FakeMarrakesh, FakeKingston, FakeTorino. 16 tests: HW1's nine (qaoa_n12_p2, qaoa_n20_p2,
ba_n14/16/18/20, tfim_n12, tfim_n16, ghz_n12) and seven of c47's breakdown (tfim_n20_r3, heis_n12_r2, heis2d_n16_r1,
chem_n12_r1, qaoa3reg_n20_r3, qaoa3reg_n40_r3, tfim_n40_r2). Arms: QK3 (Qiskit level 3, seed 0), C47 (c48 with
`ST_SCHEDULE = False`), C48; both PSF arms with `level3_candidate=False`, as HW1's PSF arm. Duration: each circuit
scheduled as soon as possible on the Target's gate durations (measurements excluded). Exactness: the measured
distribution against the input's own synthesis, on outputs with at most 20 active qubits.

**Predictions.**

| ID | prediction |
|---|---|
| V1 | gates: on every test and device, C48's two-qubit count is within 5% of C47's and its ESP at least 0.97 x C47's |
| V2 | C48's duration <= 1.02 x C47's on at least 90% of the 64 test-device pairs |
| V3 | on every Ising test (tfim_n12, tfim_n16, tfim_n20_r3, tfim_n40_r2) and device, C48's duration <= 0.5 x C47's |
| V4 | every C48 output with <= 20 active qubits reproduces the input's distribution within 1e-6 |
| V5 | reported: median, geometric mean, min and max of the duration ratio C48 / C47 |
| V6 | reported: on how many pairs C47's and C48's durations are <= QK3's; on how many C48's ESP >= QK3's |
| V7 | C48's compile time <= 1.5 x C47's + 2 s on every pair |
| V8 | C48 gives the same output on a second call (qaoa_n20_p2, tfim_n16, every device) |

**Reading, fixed now.** Item 83 is worth carrying if V1, V3 and V4 are confirmed. V2 refuted would mean the
CX-unit schedule misleads on the devices' durations (then the guard needs the Target's durations). No claim about
hardware fidelity follows from this run: duration is the quantity HW1 pointed to, and whether a shorter schedule with
the same gates gives higher fidelity on a real device is a separate test (not run without the owner's go-ahead).
