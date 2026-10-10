"""Builds candidate 2026-10-10.c34 from candidate 2026-10-10.c33, and psf_smart_layout c34 from the release's
benchmarks/psf_smart_layout.py, by exact substitutions.
Usage: python make_c34.py <c33 psf_compile.py> <release psf_smart_layout.py> <out psf_compile.py> <out psf_smart_layout.py>"""
import sys

src, lay_src, dst, lay_dst = sys.argv[1:5]


def subs(s, pairs):
    for old, new in pairs:
        assert s.count(old) == 1, (old[:80], s.count(old))
        s = s.replace(old, new)
    return s


s = subs(open(src, encoding="utf-8").read(), [
    ("VERSION: 2026-10-10.c33 -- candidate (from candidate 2026-10-10.c32): items 58, 59a, 60 and 61",
     "VERSION: 2026-10-10.c34 -- candidate (from candidate 2026-10-10.c33): items 58, 59a, 60, 61 and 62"),
    ('VERSION = "2026-10-10.c33"  # candidate: c32 (release 2026-10-10.2 + items 58, 59a, 60) + item 61 (with a Target, one compile on the map without the failed elements)',
     'VERSION = "2026-10-10.c34"  # candidate: c33 (release 2026-10-10.2 + items 58, 59a, 60, 61) + item 62 (speed: _uses_failed; switches to measure what the compression and the SWAP absorption are worth)'),
    ("""    used no failed element (the router then saw the whole map).
\"\"\"""", """    used no failed element (the router then saw the whole map).
62. **SPEED, and switches to measure the compiler's own steps (candidate 2026-10-10.c34).** Timers around the stages
    of c33's default call (STAGETIME, 2026-10-10) found the time in PSF-Zero's own steps, not in Qiskit's: the SWAP
    absorption after routing (item 30) took 30% of it, the compression 16-22%, and on small circuits the first call's
    layout search 23% (networkx's import). Here:
    - `_uses_failed` reads each qubit's index from a table instead of `find_bit` (the same answer, faster);
    - `ABSORB_SYNTH = "psf"` (default) synthesises the absorbed blocks as before; `"qiskit"` uses Qiskit's
      TwoQubitBasisDecomposer on CX (Rust; exact; the same CX count for a two-qubit unitary);
    - `COMPRESS = True` (default) compresses as before; False routes the input uncompressed.
    The defaults change no output. psf_smart_layout's two matching-size checks use rustworkx instead of networkx
    (the size of a maximum matching does not depend on the algorithm), in patches/psf_compile_c34_2026-10-10.
\"\"\""""),
    ("""    for inst in circ.data:
        idx = tuple(circ.find_bit(q).index for q in inst.qubits)
        if any(i in qubits for i in idx):""", """    pos = {q: i for i, q in enumerate(circ.qubits)}  # item 62: the same indices as find_bit, without its cost
    for inst in circ.data:
        idx = tuple(pos[q] for q in inst.qubits)
        if any(i in qubits for i in idx):"""),
    ("""        synth = SU4GeodesicPSFSynthesizer(
            GeodesicPSFHyper(tol=self._tol, on_unsupported=self._on_unsupported, entangling_basis="cx"),
            verify=self._verify)
        done = dict(zip(idx, synth.synthesize_many([blocked.data[i].operation.to_matrix() for i in idx])))""",
     """        if ABSORB_SYNTH == "qiskit":  # item 62: Qiskit's two-qubit synthesis on CX, in Rust
            dec = TwoQubitBasisDecomposer(CXGate())
            done = {i: (dec(blocked.data[i].operation.to_matrix()),) for i in idx}
        else:
            synth = SU4GeodesicPSFSynthesizer(
                GeodesicPSFHyper(tol=self._tol, on_unsupported=self._on_unsupported, entangling_basis="cx"),
                verify=self._verify)
            done = dict(zip(idx, synth.synthesize_many([blocked.data[i].operation.to_matrix() for i in idx])))"""),
    ("""PRUNE_FIRST = True  # item 61: with failed elements in the Target, compile once on the pruned map""",
     """PRUNE_FIRST = True  # item 61: with failed elements in the Target, compile once on the pruned map
ABSORB_SYNTH = "psf"  # item 62: "psf" (default) or "qiskit": how item 30's absorbed blocks are synthesised
COMPRESS = True  # item 62: False routes the input without PSF-Zero's compression (a switch for measurement)"""),
    ("""    qc_compressed = compile(
        qc,""", """    qc_compressed = qc if not COMPRESS else compile(  # item 62: COMPRESS=False skips the compression
        qc,"""),
])
open(dst, "w", encoding="utf-8", newline="\n").write(s)

lay = subs(open(lay_src, encoding="utf-8").read(), [
    ("""    import networkx as nx
    g = nx.Graph()
    g.add_edges_from(interaction_pairs)
    return len(nx.max_weight_matching(g, maxcardinality=True))""",
     """    import rustworkx as rx  # c34: rustworkx (loaded with Qiskit) instead of networkx; the size is the same
    g = rx.PyGraph()
    index = {}
    for a, b in interaction_pairs:
        for v in (a, b):
            if v not in index:
                index[v] = g.add_node(v)
        g.add_edge(index[a], index[b], None)
    return len(rx.max_weight_matching(g, max_cardinality=True))"""),
    ("""    import networkx as nx
    g = nx.Graph()
    g.add_nodes_from(range(cmap.size()))
    g.add_edges_from([tuple(e) for e in cmap.get_edges()])
    m = nx.max_weight_matching(g, maxcardinality=True)
    return len(m) >= num_logical_pairs""",
     """    import rustworkx as rx  # c34: rustworkx (loaded with Qiskit) instead of networkx; the size is the same
    g = rx.PyGraph()
    g.add_nodes_from(range(cmap.size()))
    for a, b in {(min(a, b), max(a, b)) for a, b in cmap.get_edges() if a != b}:
        g.add_edge(a, b, None)
    m = rx.max_weight_matching(g, max_cardinality=True)
    return len(m) >= num_logical_pairs"""),
])
open(lay_dst, "w", encoding="utf-8", newline="\n").write(lay)
print("written", dst, lay_dst)
