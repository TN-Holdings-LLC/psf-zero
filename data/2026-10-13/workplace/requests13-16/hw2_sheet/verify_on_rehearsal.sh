set -x
cd <psf-zero repo>
# (Rust core on PYTHONPATH as usual)
W=<rehearsal dir>
python benchmarks/ns_snapshot.py $W/hw2 --fake FakeFez
time python -u benchmarks/ns_hw2.py $W/ns2 $W/hw2 > $W/ns2.log 2>&1
time python -u benchmarks/hw2_predictors.py $W/pred $W/hw2 > $W/pred.log 2>&1
time python -u benchmarks/hw2_predictors_c.py $W/predc $W/hw2 --with $W/pred/hw2_predictors.jsonl > $W/predc.log 2>&1
python <plan v2 dir>/hw2_within_test_v2.py $W/hw2 $W/pred/hw2_predictors.jsonl $W/ns2/ns_hw2.jsonl $W/predc/hw2_predictors_c.jsonl --out $W/hw2_within_test_v2.json > $W/within.log 2>&1
echo ALLDONE
