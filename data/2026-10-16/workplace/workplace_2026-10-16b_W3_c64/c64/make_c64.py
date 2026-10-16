"""make_c64.py -- workplace, 2026-10-16 (note 10, W3; C64-PREREG.md): prototype 2026-10-16.c64 from c63a.

c64 = c63a with every module switch fixed at c63a's default and the branches that value cannot reach removed, then
every top-level definition, variable and import that the kept code no longer references removed (reference closure
from the default calls and from the names used by the c63-line files). The module docstring (items 1-101) is moved
unchanged to CHANGELOG_c63a.md. Nothing is edited by hand; the output is the same byte for byte on every run.

    python make_c64.py <c63a psf_compile.py> <out dir>      (writes <out dir>/psf_compile.py, CHANGELOG_c63a.md)"""
import ast
import os
import sys

src_path, out_dir = sys.argv[1:3]
s = open(src_path, encoding="utf-8").read()

# ---------------------------------------------------------------------------------------------------------------------
# 1. switches fixed at c63a's defaults: exact substitutions (each must occur exactly `count` times)
# ---------------------------------------------------------------------------------------------------------------------
SUBS = []


def sub(old, new, count=1):
    SUBS.append((old, new, count))


exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "c64_edits.py"), encoding="utf-8").read())

for old, new, count in SUBS:
    k = s.count(old)
    if k != count:
        raise SystemExit(f"expected {count} occurrence(s), found {k}: {old[:100]!r}")
    s = s.replace(old, new)

# ---------------------------------------------------------------------------------------------------------------------
# 2. whole definitions replaced (REPLACE: name -> new source of that top-level def; the last def of the name)
# ---------------------------------------------------------------------------------------------------------------------
for name, new in REPLACE.items():
    tree = ast.parse(s)
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name]
    if not nodes:
        raise SystemExit(f"no definition {name}")
    n = nodes[-1]
    first = n.lineno if not n.decorator_list else n.decorator_list[0].lineno
    lines = s.split("\n")
    s = "\n".join(lines[:first - 1] + new.rstrip("\n").split("\n") + lines[n.end_lineno:])

# ---------------------------------------------------------------------------------------------------------------------
# 3. reference closure: keep what the roots reach; remove every other top-level statement
# ---------------------------------------------------------------------------------------------------------------------
ROOTS = [
    # the default calls
    "compile_for_hardware", "compile_template", "CompileTemplate",
    # not removed (C64-PREREG.md section 2.5; run by the default calls in home's survey)
    "_better",
    # names used by the c63-line files (C64-PREREG.md section 2.3): c63_val.py, test_c63.py, w3_walk.py, a23
    "VERSION", "WARN_WITHOUT_TARGET", "TEMPLATE_CACHE", "TEMPLATE100_STATS", "OutOfScopeError", "_failed_elements",
    "_uses_failed", "_X93_BASIS", "_n99_same", "_n99_kron_steps", "_t100_canonical", "TranspilerError",
    "_chain_lattice", "_chain_place", "_fold", "_scope96_edges", "_scope96_embeds", "_native_brickwork",
    "NATIVE_BRICKWORK",
]


def bound_names(n):
    """Names a top-level statement binds (functions, classes, assignments, imports, and those inside if / try)."""
    out = set()
    if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
        out.add(n.name)
    elif isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
            if isinstance(t, ast.Name):
                out.add(t.id)
            elif isinstance(t, ast.Tuple):
                out |= {x.id for x in t.elts if isinstance(x, ast.Name)}
    elif isinstance(n, (ast.Import, ast.ImportFrom)):
        out |= {(a.asname or a.name).split(".")[0] for a in n.names}
    elif isinstance(n, (ast.Try, ast.If)):
        for x in ast.walk(n):
            if isinstance(x, (ast.Import, ast.ImportFrom)):
                out |= {(a.asname or a.name).split(".")[0] for a in x.names}
            elif isinstance(x, ast.Assign):
                for t in x.targets:
                    if isinstance(t, ast.Name):
                        out.add(t.id)
    out.discard("_")  # the throwaway name binds nothing that kept code reads
    return out


def attr_base(n):
    """For `X.attr ... = value` / `X.attr += value` at top level: the base name X (else None)."""
    if isinstance(n, (ast.Assign, ast.AugAssign)):
        t = n.targets[0] if isinstance(n, ast.Assign) else n.target
        while isinstance(t, ast.Attribute):
            t = t.value
        if isinstance(t, ast.Name) and t is not (n.targets[0] if isinstance(n, ast.Assign) else n.target):
            return t.id
    return None


def is_doc_concat(n):
    return isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Attribute) and n.target.attr == "__doc__"


def names_in(n):
    return {x.id for x in ast.walk(n) if isinstance(x, ast.Name)}


tree = ast.parse(s)
body = tree.body
binders, attached = {}, {}
for i, n in enumerate(body):
    for nm in bound_names(n):
        binders.setdefault(nm, []).append(i)
    b = attr_base(n)
    if b is not None and not is_doc_concat(n):
        attached.setdefault(b, []).append(i)
seen, stack = set(), list(ROOTS)
while stack:
    k = stack.pop()
    if k in seen:
        continue
    seen.add(k)
    for i in binders.get(k, []) + attached.get(k, []):
        for nm in names_in(body[i]):
            if nm in binders and nm not in seen:
                stack.append(nm)
missing = [r for r in ROOTS if r not in binders]
if missing:
    raise SystemExit(f"roots without a definition: {missing}")

keep = []
for i, n in enumerate(body):
    if i == 0 and isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant):
        keep.append((i, "doc"))
        continue
    names = bound_names(n)
    if names:
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            used = [a for a in n.names if (a.asname or a.name).split(".")[0] in seen - LOCAL_ONLY]
            if used:
                keep.append((i, used))
        elif names & seen:
            keep.append((i, None))
        continue
    b = attr_base(n)
    if b is not None:
        if b in seen and names_in(n) <= seen | {"__doc__"} and not (is_doc_concat(n) and not names_in(n) <= seen):
            keep.append((i, None))
        continue
    if names_in(n) <= seen:  # any other statement whose names are all kept
        keep.append((i, None))

lines = s.split("\n")
kept_idx = {i: how for i, how in keep}
removed_defs = [(n.name, n.end_lineno - n.lineno + 1) for i, n in enumerate(body)
                if i not in kept_idx and isinstance(n, (ast.FunctionDef, ast.ClassDef))]


def seg(n):
    first = n.lineno if not getattr(n, "decorator_list", None) else n.decorator_list[0].lineno
    return first, n.end_lineno


repl = {}  # first line -> (last line, replacement lines)
doc_text = None
for i, n in enumerate(body):
    a, b = seg(n)
    how = kept_idx.get(i, "drop")
    if how == "doc":  # the docstring's source text, exactly as written between its quotes
        raw = "\n".join(lines[a - 1:b])
        q = raw[:3]
        assert q in ('"""', "'''") and raw.endswith(q)
        doc_text = raw[3:-3]
        repl[a] = (b, ["@@DOC@@"])
    elif how == "drop":
        while a > 1 and lines[a - 2].lstrip().startswith("#") and not lines[a - 2].startswith(" "):
            a -= 1  # the comment block directly above a removed statement goes with it
        repl[a] = (b, [])
    elif isinstance(how, list) and len(how) != len(n.names):  # an import with some names unused: rewritten
        names = ", ".join(x.name + (f" as {x.asname}" if x.asname else "") for x in how)
        repl[a] = (b, [f"from {n.module} import {names}" if isinstance(n, ast.ImportFrom) else f"import {names}"])
out, k = [], 1
while k <= len(lines):
    if k in repl:
        b, new = repl[k]
        out += new
        k = b + 1
    else:
        out.append(lines[k - 1])
        k += 1
# a section header (a block of top-level comment lines framed by "# ----" rules) whose section lost all its code
i = 0
while i < len(out):
    if out[i].startswith("# ---") and out[i].strip("#- ") == "":
        j = i + 1
        while j < len(out) and out[j].startswith("#") and not (out[j].startswith("# ---") and out[j].strip("#- ") == ""):
            j += 1
        if j < len(out) and out[j].startswith("# ---"):  # j: the closing rule
            k = j + 1
            while k < len(out) and not out[k].strip():
                k += 1
            if k == len(out) or (out[k].startswith("# ---") and out[k].strip("#- ") == ""):
                del out[i:k]
                continue
    i += 1
text, blank = [], 0  # at most two blank lines in a row
for line in out:
    blank = blank + 1 if not line.strip() else 0
    if blank <= 2:
        text.append(line.rstrip())
# ---------------------------------------------------------------------------------------------------------------------
# 4. docstring moved to CHANGELOG_c63a.md; a short one written; version
# ---------------------------------------------------------------------------------------------------------------------
os.makedirs(out_dir, exist_ok=True)
open(os.path.join(out_dir, "CHANGELOG_c63a.md"), "w", encoding="utf-8", newline="\n").write(
    "# Changelog of psf_compile.py up to 2026-10-15.c63a (items 1-101), moved unchanged from its module docstring\n\n"
    "```text\n" + doc_text + "\n```\n")
head = '''"""PSF-Zero compiler -- VERSION: 2026-10-16.c64 (workplace proposal, from 2026-10-15.c63a by make_c64.py).

Two entry points: `compile_for_hardware(qc, target=...)` and `compile_template(qc, target)`. c64 has c63a's outputs
and none of the code its default calls cannot reach: every switch is fixed at c63a's default. The changelog of items
1-101 (c63a's module docstring) is in CHANGELOG_c63a.md next to this file."""'''
result = "\n".join(text).replace("@@DOC@@", head).rstrip("\n") + "\n"
open(os.path.join(out_dir, "psf_compile.py"), "w", encoding="utf-8", newline="\n").write(result)
print(f"written {out_dir}/psf_compile.py: {result.count(chr(10))} lines; removed {len(removed_defs)} definitions "
      f"({sum(k for _, k in removed_defs)} lines)")
