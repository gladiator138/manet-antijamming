#!/usr/bin/env python3
"""Build a small *development* forest + quant.json + held-out test vectors.

    python3 make_dev_model.py [--out dev] [--slots 600] [--simdir ..] [--procs 2]

Trains RandomForestClassifier(n_estimators=10, max_depth=12, random_state=0) on
make_data(TRAIN_SEEDS[:3], slots) from run_experiments.py, computes the
lo/hi quantisation range exactly as run_experiments.fixedpoint() does
(0.01th / 99.99th percentile of the training features) and draws test
vectors from make_data(TEST_SEEDS, slots).  Only reads the simulator; it
never modifies it.  Uses at most --procs worker processes (make_data itself
hard-codes Pool(4), so we call its _collect() helper through our own pool).
"""
import argparse, json, os, pickle, sys
import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "dev"))
    ap.add_argument("--simdir", default=os.path.join(HERE, ".."))
    ap.add_argument("--slots", type=int, default=600)
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--bits", type=int, default=16)
    a = ap.parse_args()
    sys.dont_write_bytecode = True          # do not touch ../__pycache__ of the shared simulator
    sys.path.insert(0, os.path.abspath(a.simdir))
    import run_experiments as R
    from sklearn.ensemble import RandomForestClassifier

    def data(seeds):
        jobs = [(s, R.POWERS[i % len(R.POWERS)], 6, a.slots) for i, s in enumerate(seeds)]
        with Pool(a.procs) as p:
            parts = p.map(R._collect, jobs)
        return np.concatenate([x for x, _ in parts]), np.concatenate([y for _, y in parts])

    Xtr, ytr = data(R.TRAIN_SEEDS[:3])
    Xte, yte = data(R.TEST_SEEDS)   # all 7 test seeds -> every jammer power
    rf = RandomForestClassifier(n_estimators=10, max_depth=12, random_state=0, n_jobs=a.procs).fit(Xtr, ytr)
    rf.n_jobs = 1
    lo, hi = np.percentile(Xtr, 0.01, axis=0), np.percentile(Xtr, 99.99, axis=0)
    os.makedirs(a.out, exist_ok=True)
    pickle.dump(rf, open(os.path.join(a.out, "forest.pkl"), "wb"))
    json.dump(dict(lo=lo.tolist(), hi=hi.tolist(), bits=a.bits, features=list(getattr(R, "FEATURES", [])),
                   thresh=R.THRESH, source="make_dev_model.py: make_data(TRAIN_SEEDS[:3], slots=%d)" % a.slots),
              open(os.path.join(a.out, "quant.json"), "w"), indent=1)
    np.savez(os.path.join(a.out, "test_vectors.npz"), X=Xte, y=yte)
    print("train", Xtr.shape, "pos", ytr.mean(), "test", Xte.shape, "nodes",
          [e.tree_.node_count for e in rf.estimators_], "depth", [e.get_depth() for e in rf.estimators_])


if __name__ == "__main__":
    main()
