#!/usr/bin/env python3
"""Write quant.json (and optionally test vectors) for the PRODUCTION model, using the same
range definition as run_experiments.fixedpoint(): lo/hi = 0.01th/99.99th percentile of the
features of make_data(TRAIN_SEEDS) (W=6, slots=1500 by default).

    python3 make_quant.py --out quant.json [--bits 16] [--slots 1500] [--procs 2]
                          [--vectors test_vectors.npz --test-seeds 1]

Reads the simulator only (imports run_experiments from --simdir); uses --procs workers.
"""
import argparse, json, os, sys
import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--simdir", default=os.path.join(HERE, ".."))
    ap.add_argument("--bits", type=int, default=16)
    ap.add_argument("--slots", type=int, default=1500)
    ap.add_argument("--window", type=int, default=6)
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--vectors", default=None, help="also save held-out vectors (npz with X, y)")
    ap.add_argument("--test-seeds", type=int, default=7, help="how many TEST_SEEDS to use for --vectors")
    a = ap.parse_args()
    sys.dont_write_bytecode = True          # do not touch ../__pycache__ of the shared simulator
    sys.path.insert(0, os.path.abspath(a.simdir))
    import run_experiments as R

    def data(seeds):
        jobs = [(s, R.POWERS[i % len(R.POWERS)], a.window, a.slots) for i, s in enumerate(seeds)]
        with Pool(a.procs) as p:
            parts = p.map(R._collect, jobs)
        return np.concatenate([x for x, _ in parts]), np.concatenate([y for _, y in parts])

    Xtr, _ = data(R.TRAIN_SEEDS)
    lo, hi = np.percentile(Xtr, 0.01, axis=0), np.percentile(Xtr, 99.99, axis=0)
    json.dump(dict(lo=lo.tolist(), hi=hi.tolist(), bits=a.bits, thresh=R.THRESH,
                   features=list(getattr(R, "FEATURES", [])),
                   source=f"make_quant.py: percentiles 0.01/99.99 of make_data(TRAIN_SEEDS, W={a.window}, slots={a.slots})"),
              open(a.out, "w"), indent=1)
    print("lo", lo, "hi", hi)
    if a.vectors:
        Xte, yte = data(R.TEST_SEEDS[:a.test_seeds])
        np.savez(a.vectors, X=Xte, y=yte)
        print("vectors", Xte.shape)


if __name__ == "__main__":
    main()
