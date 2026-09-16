# Qiskit Sabre layout/routing source (`crates/accelerate/src/sabre/`)

Two files from Qiskit's own Rust core, pasted into this project's chat on
2026-09-16 (source: Qiskit, Apache License 2.0 — each file carries its own
IBM copyright header and license notice, retained below unmodified):

- `layout.rs` -- `sabre_layout_and_routing` / `layout_trial` / `add_heuristic_layouts`.
  Runs after `VF2Layout` gives up (or is skipped) and `SabreLayout` takes over.
  Includes the hardcoded starting rings for 127/133/156-qubit IBM Eagle/Heron
  devices (`add_heuristic_layouts`).
- `route.rs` -- `sabre_routing` / `swap_map` / `swap_map_trial` / the `State`
  swap-insertion loop. The actual SWAP-insertion search Sabre runs once a
  layout is chosen.

## Why these were pulled in, and what they do and don't explain

**Not the ~145-iteration compile-time period (Addenda 19-22).** Checked by
direct grep: neither file contains a literal `145` or `187` constant (the one
`145` hit in `layout.rs` is just an index inside the hardcoded 156-qubit ring
array, `[...150, 149, 148, 147, 146, 145, 144, 143]` -- not a period, cache
size or limit). More fundamentally, the Addendum 19-22 experiment
(`test_cumulative_compile_scale.py`) runs `transpile(optimization_level=3)`
with **no coupling_map**, and `sabre_layout_and_routing`'s own first branch
returns immediately on `TargetCouplingError::AllToAll` without running any of
the search logic in either file:

```rust
let coupling = match target.coupling_graph() {
    Ok(coupling) => coupling,
    Err(TargetCouplingError::AllToAll) => {
        let mut out = dag.clone();
        out.make_physical(num_physical_qubits);
        let trivial = NLayout::generate_trivial_layout(num_physical_qubits as u32);
        return Ok((out, trivial.clone(), trivial));
    }
    ...
};
```

So this module cannot be responsible for the 145/187-iteration period found
in Addenda 19-22 on two separate machines -- that experiment's circuits never
meaningfully execute this code. The cause of that period remains
unidentified; it is not in this module.

**Is directly relevant to the spare-qubit-cliff finding (addenda 5-22,
README "A finding that is not about PSF-Zero").** That investigation *does*
use coupling maps, so once `VF2Layout` reports `NO_SOLUTION_FOUND`, this is
the code that actually runs: `sabre_layout_and_routing` picks a starting
layout (`layout_trial`, with the hardcoded IBM rings as one of several
starting candidates via `add_heuristic_layouts`), and `route.rs`'s
`swap_map_trial` does the greedy swap-insertion search
(`choose_best_swap`, the decay heuristic, `force_enable_closest_node` as a
release valve when the heuristic search stalls). This is downstream of
`VF2Layout` itself (a separate file, not included here), so it does not show
the `seed=-1` VF2 detail the addenda describe -- only what SabreLayout does
once VF2 has already failed or been skipped.

## Provenance

Pasted directly by the user from what they described as Qiskit's own
repository source (not fetched by this session -- WebFetch's provenance
restriction blocks unprompted GitHub fetches). Not independently verified
against a live Qiskit checkout in this session; treat as user-supplied
source text, licensed Apache 2.0 per its own header.
</content>
