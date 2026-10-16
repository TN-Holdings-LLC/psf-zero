# Search in PSF-Zero: what the default calls run, and what can be deleted (2026-10-16)

Read-only survey. Nothing in PSF-Zero was changed.

**Method.**

- Every call into the latest prototype (c63a) and into the AI front end (a22, with c63a under it) was recorded while
  they compiled:
  - C63-VAL's 19 tests, T and F targets, on FakeFez, FakeMarrakesh, FakeKingston and FakeTorino, with
    `compile_for_hardware(qc, target=)` and `compile_template(qc, target).bind(qc)` (304 calls, all built), plus the
    refusals (no Target; a three-qubit gate: 8 OutOfScopeError);
  - TOQB's model circuits (rings of 4, 6, 8, 12 qubits; 2 and 8 layers) on FakeFez and FakeTorino, through a22's
    `model_template` and `compile_for_model_circuit` (32 calls).
- Every top-level function was then marked "run" or "never run".
- A function is marked as search when it contains Qiskit transpile or a preset pass manager, Sabre, VF2PostLayout,
  compare-and-choose (`_better`), seeds or trials, or rustworkx VF2.

## Summary

**c63a: 9,240 lines.**

| part | functions | lines |
|---|---|---|
| run by the default calls (kept) | 69 | 2,709 |
| never run, search | 17 | 1,205 |
| never run, other (the old pipelines' helpers) | 125 | 3,078 |
| never run, exactness checks with random test states (not search) | 2 | 83 |
| module docstring (changelog of items 1-101) | - | about 1,240 |

**Search still on the default path of c63a.**

- **VF2 first match, no score:** `_chain_place`, `_graph_place`, `_scope96_embeds`. This is the owner's placement
  rule, and prototype B decides whether it stays.
- **Qiskit's preset pass manager at level 0** in `_chain_brickwork` / `_graph_brickwork`, on F targets only (44 calls
  in the survey). This is translation into rzz / rx / rz / sx / x on a fixed placement, not a search. Qiskit still
  writes those gates (as it did on the CZ path before item 101).

**a22: 1,200 lines.** 23 functions run (813 lines), 11 never run (135 lines, no search).

Search or compare-and-choose on its default path:

- **`_ring_candidate`:** Qiskit's preset pass manager at level 3 with the placement fixed and routing off (24 calls).
  This is optimisation and translation by Qiskit, not a layout search.
- **`_formula_only`, `compile_for_model_circuit`, `ring_template`:** they build two ring formulas and keep the shorter
  (`_better`). This is compare-and-choose.
- **`_hand_off`:** a call without a Target is refused (item 36). With a Target it goes to `compile_for_hardware`.
  Its `transpile(optimization_level=3)` branch is not reached.

The tables below list every function with its lines.

## c63a

### Run by the default calls and carrying a search marker

| function | lines | markers |
|---|---|---|
| `_better` | 4648-4676 (29) | compare & choose (_better) |
| `_chain_place` | 6512-6543 (32) | VF2 search (rustworkx) |
| `_chain_brickwork` | 6642-6742 (101) | preset pass manager |
| `_graph_place` | 6866-6883 (18) | VF2 search (rustworkx) |
| `_graph_brickwork` | 6886-6994 (109) | preset pass manager |
| `_scope96_embeds` | 7282-7313 (32) | VF2 search (rustworkx) |

### Never run by the default calls, with a search marker (to delete)

| function | lines | markers |
|---|---|---|
| `_same_action` | 2801-2832 (32) | seeds / trials |
| `_implements` | 2835-2885 (51) | seeds / trials |
| `_compare_level3` | 3261-3284 (24) | transpile( |
| `_st_embeddings` | 3578-3622 (45) | VF2 search (rustworkx) |
| `_DeterministicPostLayout` | 3656-3859 (204) | VF2PostLayout, seeds / trials, VF2 search (rustworkx) |
| `_NoPostLayout` | 3862-3867 (6) | VF2PostLayout |
| `_DeterministicVF2` | 3870-3902 (33) | VF2PostLayout, seeds / trials |
| `_level3_candidate` | 3911-3938 (28) | preset pass manager, compare & choose (_better) |
| `_near_alternative` | 3971-4003 (33) | compare & choose (_better) |
| `_st_construct` | 4836-4896 (61) | VF2 search (rustworkx) |
| `_st_compile` | 5457-5567 (111) | transpile(, compare & choose (_better), seeds / trials, VF2 search (rustworkx) |
| `_layout_map_to_list` | 5672-5690 (19) | transpile(, VF2 search (rustworkx) |
| `_compile_for_hardware_c49` | 5757-6204 (448) | transpile(, preset pass manager, VF2PostLayout, Sabre, compare & choose (_better), seeds / trials, VF2 search (rustworkx) |
| `_fractional_choice` | 6312-6337 (26) | compare & choose (_better) |
| `_compile_for_hardware_c53g` | 6748-6789 (42) | compare & choose (_better) |
| `_exact93` | 7056-7082 (27) | transpile( |
| `_compile_for_hardware_c54` | 7085-7129 (45) | preset pass manager |
| `_compile_for_hardware_c55` | 7206-7243 (38) | compare & choose (_better) |
| `_hand_off96` | 7316-7330 (15) | preset pass manager |

### Never run by the default calls, no search marker (dead code; to delete with them)

| function | lines |
|---|---|
| `_cx_cost_as_written` | 1335-1340 (6) |
| `_block_unitary_4x4` | 1347-1379 (33) |
| `_block_saves_cx` | 1382-1409 (28) |
| `_aligned_errors` | 1434-1444 (11) |
| `_aligned_distance` | 1447-1450 (4) |
| `_weyl_exact` | 1453-1460 (8) |
| `_su2_zyz` | 1463-1472 (10) |
| `_exact_rebuild` | 1475-1500 (26) |
| `_accept` | 1503-1514 (12) |
| `_guarded_cx_synthesis` | 1517-1562 (46) |
| `_validate_verify` | 1581-1596 (16) |
| `GeodesicPSFHyper` | 1599-1618 (20) |
| `unitary_fidelity` | 1621-1633 (13) |
| `_zyz_matrix` | 1636-1641 (6) |
| `_reconstruct` | 1644-1652 (9) |
| `_infidelity` | 1655-1658 (4) |
| `_pack` | 1667-1668 (2) |
| `_unpack` | 1671-1675 (5) |
| `_zyz_and_derivatives` | 1678-1686 (9) |
| `_reconstruct_with_jacobian` | 1689-1716 (28) |
| `_refine_decomposition` | 1719-1753 (35) |
| `_zyz_batch` | 1760-1771 (12) |
| `_zyz_derivatives_batch` | 1774-1786 (13) |
| `_kron_batch` | 1789-1791 (3) |
| `_reconstruct_batch` | 1794-1822 (29) |
| `_lstsq_batch` | 1825-1834 (10) |
| `_refine_batch` | 1837-1869 (33) |
| `SU4GeodesicPSFSynthesizer` | 1872-2110 (239) |
| `_cx_core_cached` | 2113-2155 (43) |
| `_append_cx_core_closed_form` | 2171-2217 (47) |
| `compile` | 2220-2327 (108) |
| `edge_errors_from_target` | 2330-2344 (15) |
| `FailedElementsError` | 2361-2362 (2) |
| `_recompile_pruned` | 2389-2412 (24) |
| `_RecordRefine` | 2416-2422 (7) |
| `_core57_buffer` | 2435-2451 (17) |
| `_core57_estimate` | 2454-2487 (34) |
| `_core57_apply` | 2490-2502 (13) |
| `excitation_cost` | 2505-2561 (57) |
| `_product_state` | 2572-2575 (4) |
| `_apply_ops` | 2578-2592 (15) |
| `_touched_qubits` | 2599-2601 (3) |
| `_narrow` | 2607-2609 (3) |
| `_n_narrow` | 2612-2616 (5) |
| `_narrow_touched` | 2619-2622 (4) |
| `_cannot_implement` | 2625-2640 (16) |
| `_ops_of` | 2643-2674 (32) |
| `_embed_1q` | 2681-2695 (15) |
| `_gate_matrix` | 2703-2730 (28) |
| `_fuse_ops` | 2733-2751 (19) |
| `_lower` | 2757-2760 (4) |
| `_rho1` | 2766-2771 (6) |
| `_p1` | 2774-2777 (4) |
| `_populations` | 2780-2784 (5) |
| `_free_diagonal` | 2787-2792 (6) |
| `_seen` | 2795-2798 (4) |
| `_final_resynthesis` | 2888-2900 (13) |
| `_resynthesis_candidate` | 2903-2921 (19) |
| `decoherence_floor` | 2928-2948 (21) |
| `floor_aware_target` | 2951-2966 (16) |
| `pauli_cost` | 2969-3015 (47) |
| `readout_cost` | 3018-3033 (16) |
| `hybrid_cost` | 3036-3112 (77) |
| `kraus_cost` | 3115-3177 (63) |
| `_choose` | 3180-3192 (13) |
| `_cost_fn` | 3195-3196 (2) |
| `_choose_lazy` | 3199-3243 (45) |
| `_acceptable` | 3246-3258 (13) |
| `_cancel_candidate` | 3297-3308 (12) |
| `_has_wide` | 3311-3313 (3) |
| `_unroll_wide` | 3316-3330 (15) |
| `_checkable_logical` | 3333-3341 (9) |
| `_resynthesise` | 3344-3358 (15) |
| `_select_resynthesis` | 3361-3381 (21) |
| `_refine_found` | 3384-3385 (2) |
| `_st_working_graph` | 3396-3414 (19) |
| `_st_bfs_all` | 3417-3429 (13) |
| `_st_steiner_tree` | 3432-3473 (42) |
| `_st_tree_cost` | 3476-3478 (3) |
| `_st_placement` | 3481-3502 (22) |
| `_st_gadget_ops` | 3505-3537 (33) |
| `_st_error_distances` | 3540-3564 (25) |
| `_st_interaction` | 3567-3575 (9) |
| `_st_placement_cost` | 3625-3640 (16) |
| `_near_used` | 3941-3968 (28) |
| `_near_used_elements` | 4006-4042 (37) |
| `_near_alternative_c42` | 4045-4065 (21) |
| `_st_commute` | 4083-4084 (2) |
| `_StMover` | 4087-4275 (189) |
| `_StMover2` | 4278-4421 (144) |
| `_st_record_term` | 4432-4439 (8) |
| `_st_record_swap` | 4442-4454 (13) |
| `_st_blocks_commute` | 4457-4475 (19) |
| `_st_op_dur` | 4478-4479 (2) |
| `_st_asap` | 4482-4489 (8) |
| `_st_reorder` | 4492-4546 (55) |
| `_duration_cx` | 4628-4645 (18) |
| `_working_target` | 4679-4710 (32) |
| `_embed_working` | 4713-4752 (40) |
| `_cut_failed_qubits` | 4755-4784 (30) |
| `_st_evolution` | 4787-4820 (34) |
| `_st_all_commute` | 4823-4833 (11) |
| `_st_cliff1q` | 5084-5105 (22) |
| `_StMirror` | 5316-5388 (73) |
| `_st_mirror_blocks` | 5391-5439 (49) |
| `_st_mirror_construct` | 5442-5454 (13) |
| `prune_coupling_map` | 5599-5630 (32) |
| `_edge_weights_from_errors` | 5633-5656 (24) |
| `_layout_weights` | 5659-5669 (11) |
| `_CarryPermutation` | 5693-5702 (10) |
| `_resolve_auto` | 5705-5710 (6) |
| `_AbsorbRoutingSwaps` | 5713-5754 (42) |
| `_errorless` | 6217-6227 (11) |
| `_isa_ok` | 6230-6242 (13) |
| `_fractional` | 6245-6275 (31) |
| `_strip_global_phase_gates` | 6278-6288 (11) |
| `_fold` | 6295-6309 (15) |
| `_compile_for_hardware_c51` | 6340-6355 (16) |
| `_dist93` | 7014-7053 (40) |
| `_zz_relayered` | 7146-7203 (58) |
| `_compile_for_hardware_c57` | 7333-7399 (67) |
| `OutOfScopeError` | 7418-7419 (2) |
| `_n99_kron_steps` | 7680-7693 (14) |
| `_T100Mat` | 8078-8088 (11) |
| `_T100NotHere` | 8413-8414 (2) |

## a22

### Run by the default calls and carrying a search marker

| function | lines | markers |
|---|---|---|
| `_ring_candidate` | 447-494 (48) | preset pass manager |
| `_hand_off` | 606-639 (34) | transpile( |
| `_formula_only` | 642-663 (22) | compare & choose (_better) |
| `compile_for_model_circuit` | 666-778 (113) | compare & choose (_better) |
| `ring_template` | 1126-1143 (18) | compare & choose (_better) |

### Never run by the default calls, with a search marker (to delete)

| function | lines | markers |
|---|---|---|

### Never run by the default calls, no search marker (dead code; to delete with them)

| function | lines |
|---|---|
| `_load_a15` | 130-138 (9) |
| `_outcomes` | 507-538 (32) |
| `_exact` | 541-566 (26) |
| `_exact_filter` | 569-587 (19) |
| `OutOfScopeError` | 791-792 (2) |
| `_m_rz` | 795-797 (3) |
| `_m_ry` | 800-803 (4) |
| `_m_rx` | 806-809 (4) |
| `_m_op` | 815-821 (7) |
| `_zyz` | 824-834 (11) |
| `_t_op` | 845-862 (18) |
