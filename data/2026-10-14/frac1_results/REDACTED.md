# FRAC1 results: what was changed before committing

These are the files the kit wrote to its `results/` on the owner's PC (sent as a zip on 2026-10-14; every file
matched `FRAC1_SHA256SUMS.txt` on arrival). They are stored here rather than in the kit's `results/` so that the
owner's local copy does not collide with them.

One change: in the three logs `frac1_logs/frac1_prepare.txt`, `frac1_submit.txt` and `frac1_fetch.txt`, line 5
(`SCRIPT <path> <sha256>`) had the home folder's absolute path replaced by `~`. Those three files therefore no
longer match `FRAC1_SHA256SUMS.txt`; every other file does. Nothing else was changed.
