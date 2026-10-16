"""C64 inventory (reported, not criteria): line counts, removed and kept definitions, kept public names and switches,
names kept beyond the survey's 'run' list.   python c64_inventory.py <c63a> <c64> <survey.md>"""
import ast
import re
import sys

a_src, b_src, survey = (open(p, encoding="utf-8").read() for p in sys.argv[1:4])
ta, tb = ast.parse(a_src), ast.parse(b_src)


def defs(t):
    out = {}
    for n in t.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            out.setdefault(n.name, []).append(n.end_lineno - n.lineno + 1)
    return out


def assigned(t):
    out = []
    for n in t.body:
        if isinstance(n, (ast.Assign, ast.AnnAssign)):
            for tg in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                for x in ast.walk(tg):
                    if isinstance(x, ast.Name):
                        out.append(x.id)
    return out


da, db = defs(ta), defs(tb)
sec = None
c63a_part = survey.split("## a22")[0]
never = set()
for line in c63a_part.split("\n"):
    if line.startswith("### "):
        sec = line
    m = re.match(r"\| `([^`]+)` \|", line)
    if m and sec and "Never run" in sec:
        never.add(m.group(1))
removed = {k: v for k, v in da.items() if k not in db}
dup_removed = [k for k in db if len(da.get(k, [])) > len(db[k])]
print(f"lines: c63a {a_src.count(chr(10))} -> c64 {b_src.count(chr(10))}")
print(f"top-level definitions: c63a {sum(len(v) for v in da.values())} -> c64 {sum(len(v) for v in db.values())}")
print(f"removed: {len(removed)} names ({sum(sum(v) for v in removed.values())} lines) plus overwritten duplicates of "
      f"{dup_removed} ({sum(da[k][0] for k in dup_removed)} lines)")
print("kept although in the survey's 'Never run' tables:", sorted(k for k in db if k in never))
print("kept public definitions:", sorted(k for k in db if not k.startswith("_")))
va, vb = assigned(ta), assigned(tb)
print("kept public module variables:", sorted({v for v in vb if not v.startswith("_")}))
print("removed public module variables:", sorted({v for v in va if not v.startswith("_")} - set(vb)))
print("removed definitions:")
for k in sorted(removed):
    print(f"  {k} ({sum(removed[k])} lines)")
