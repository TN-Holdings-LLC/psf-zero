"""ring1_make_model.py (workplace tool; not run at home): trains RING1's classifiers with DEPTH1's model and recipe
(data re-uploading, CZ ring, BC data, PCA to n, Adam 300 steps on exact gradients) and freezes the test points and
weights into ring1_model.json, so that the kit needs no scikit-learn. Test points: the first 12 of each class in the
test split's order (DEPTH1's split seed)."""
import importlib.util, json, sys, numpy as np
spec = importlib.util.spec_from_file_location("depth_eval", "data/2026-10-05/workplace/depth1/depth_eval.py")
D = importlib.util.module_from_spec(spec); spec.loader.exec_module(D)
out = {"dataset": "BC (scikit-learn breast_cancer)", "recipe": "DEPTH1 (data/2026-10-05/workplace/depth1/depth_eval.py): "
       "80/20 stratified split (split seed %d), standardised, PCA to n, scaled to [-1, 1]; Adam lr 0.05, batch 64, 300 steps, "
       "init seed %d + 1000 L" % (D.SPLIT_SEED, D.INIT_SEED), "configs": []}
for n in (4, 6):
    Xtr, ytr, Xte, yte = D.data("BC", n, D.SPLIT_SEED)
    pos = [i for i in range(len(yte)) if yte[i] > 0][:12]; neg = [i for i in range(len(yte)) if yte[i] < 0][:12]
    idx = sorted(pos + neg)
    for L in (4, 8):
        th = D.train(Xtr, ytr, n, L, seed=D.INIT_SEED + 1000 * L)
        zte = D.forward(th, Xte, n, L); zs = D.forward(th, Xte[idx], n, L)
        cfg = dict(n=n, L=L, theta=[float(t) for t in th], X=[[float(v) for v in Xte[i]] for i in idx],
                   y=[float(yte[i]) for i in idx], z_ideal=[float(z) for z in zs],
                   test_split_acc=float(np.mean(np.sign(zte) == yte)), test_split_margin=float(np.mean(yte * zte)),
                   points_acc=float(np.mean(np.sign(zs) == yte[idx])), points_margin=float(np.mean(yte[idx] * zs)))
        out["configs"].append(cfg)
        print(n, L, "split acc %.3f margin %.3f | 24 points acc %.3f margin %.3f" % (cfg["test_split_acc"], cfg["test_split_margin"], cfg["points_acc"], cfg["points_margin"]), flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
