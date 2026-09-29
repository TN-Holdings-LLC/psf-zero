B=<sandbox>/repo929/benchmarks/pl_heavyhex_chain.py
PYTHONPATH=<sandbox>/deg/pkg_rel python3 -u $B run --arm Q3 > pl_chain_Q3.txt 2>&1
PYTHONPATH=<sandbox>/deg/pkg_rel python3 -u $B run --arm P > pl_chain_P.txt 2>&1
PYTHONPATH=<sandbox>/deg/pkg_cand python3 -u $B run --arm PN --layout-dir <sandbox>/hh/cand_layout > pl_chain_PN.txt 2>&1
python3 -u $B score > pl_chain_score.txt 2>&1
echo ALLDONE > done.flag
