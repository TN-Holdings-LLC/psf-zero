"""Generates c38's STEINER block (st_core.py) from STEINER6's code (benchmarks/steiner6m.py, run v6 of Addendum 444) and
its STEINER3/5 helpers, by extracting the sources and renaming (prefix _st_), so nothing is transcribed by hand; then
keeps only the "chain3" ordering of run v6.
    cd <psf-zero repository>; python patches/psf_compile_c38_2026-10-11/gen_block.py <output st_core.py>"""
import inspect, os, re, sys
REPO = os.getcwd()
OUT = sys.argv[1]
sys.path[:0] = [os.path.join(REPO, "data", "2026-10-05", "workplace", "depth1"), os.path.join(REPO, "benchmarks"), REPO]
sys.argv = ["x"]
import steiner3 as S3, steiner5 as S5
import importlib.util
spec = importlib.util.spec_from_file_location("s6v6", os.path.join(REPO, "benchmarks", "steiner6m.py"))
S6 = importlib.util.module_from_spec(spec); sys.modules["s6v6"] = S6; spec.loader.exec_module(S6)
funcs = [(S3, "working_graph"), (S3, "bfs_all"), (S3, "steiner_tree"), (S3, "tree_cost"), (S3, "placement"),
         (S3, "gadget_ops"), (S5, "error_distances"), (S5, "interaction"), (S5, "embeddings"), (S5, "placement_cost"),
         (S6, "term_masks"), (S6, "commute")]
classes = [(S6, "Mover"), (S6, "Mover2")]
names = [n for _, n in funcs]
out = []
for mod, n in funcs:
    out.append(inspect.getsource(getattr(mod, n)))
for mod, n in classes:
    out.append(inspect.getsource(getattr(mod, n)))
src = "\n\n".join(out)
# references to the helper modules and between helpers
src = re.sub(r"\bS3\.|\bS5\.", "", src)
for n in names:
    src = re.sub(r"(?<![\w.])" + n + r"\b", "_st_" + n, src)
src = re.sub(r"\bclass Mover2\(Mover\)", "class _StMover2(_StMover)", src)
src = re.sub(r"\bclass Mover\b", "class _StMover", src)
# module constants used inside
src = src.replace("w=W, f=F, e=E", "w=64, f=12, e=24")
src = src.replace("FAILED", "_ST_FAILED")
src = src.replace('float(os.environ.get("S6_LAMBDA", "0.1"))', "0.1").replace('float(os.environ.get("S6_MU", "0.03"))', "0.03")
src = src.replace("VF2_CHECK_LIMIT", "_ST_VF2_CHECK").replace("VF2_MAPS", "_ST_VF2_MAPS").replace("KEEP", "_ST_KEEP")
assert "os.environ" not in src, [l for l in src.splitlines() if "os.environ" in l]
# ORDER is "chain3" in run v6: keep only that branch
i = src.index('                if ORDER == "chain3":')
j = src.index('                stuck = 0\n                continue\n            # the blocked terms looked at')
src = src[:i] + '''                small = [k for k in free if len(self.masks[k][2]) <= 2]
                if small:  # terms on one or two qubits: in the given order, all at once
                    for k in small:
                        self._run(k, unrun, elig, ex, succ, blocked)
                        left -= 1
                else:  # terms on 3 or more qubits commute pairwise: the most similar to the last one next
                    j = max(free, key=lambda k: (self.similar(self.last_term, k), -k))
                    self._run(j, unrun, elig, ex, succ, blocked)
                    left -= 1
''' + src[j:]
src = src.replace("_st_commute pairwise", "commute pairwise").replace("terms _st_commute", "terms commute")
assert "ORDER" not in src
open(OUT, "w").write(src)
print(len(src.splitlines()), "lines written to", OUT)
