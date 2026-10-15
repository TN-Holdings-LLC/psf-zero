import hashlib, os, sys
here = os.path.dirname(os.path.abspath(__file__))
rows = [l.rstrip("\n").split("\t") for l in open(os.path.join(here, "MANIFEST.tsv"), encoding="utf-8")][1:]
bad = [p for s, h, p in rows if not os.path.isfile(os.path.join(here, p))
       or hashlib.sha256(open(os.path.join(here, p), "rb").read()).hexdigest() != h]
for p in bad: print("MISMATCH", p)
print(f"{len(rows) - len(bad)}/{len(rows)} OK"); sys.exit(1 if bad else 0)
