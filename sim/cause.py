"""Jammer-type (cause) classifier on the network's alarm-consensus history.

Classes: none, spot, pulsed, sweep, two (two spot jammers). The input is the per-slot
largest alarm count of any connected component (the quantity the trigger already
computes), over the last up to 200 listening slots.
"""
import numpy as np
from scipy.sparse.csgraph import connected_components

CLASSES = ["none", "spot", "pulsed", "sweep", "two"]
CAUSE_FEATURES = ["consensus share", "mean alarms / k", "alarm share", "mean run", "max run", "runs per 100",
                  "mean gap", "peak autocorr", "peak lag", "autocorr 5", "autocorr 40"]


def alarm_series(tr, trigger, W):
    """Largest per-component alarm count in every slot, without any countermeasure."""
    T, N = tr.usable.shape[:2]
    iu = np.triu(np.ones((N, N), bool), 1)
    out = np.zeros(T, int)
    for t in range(W, T):
        a = trigger[t]
        a = (a | a.T) & iu & tr.usable_nj[t]
        ncomp, comp = connected_components(tr.usable[t], directed=False)
        ai, aj = np.nonzero(a)
        cnt = np.zeros(ncomp, int)
        np.add.at(cnt, comp[ai], 1)
        np.add.at(cnt, comp[aj], (comp[aj] != comp[ai]).astype(int))
        out[t] = cnt.max() if ncomp else 0
    return out


def _runs(x):
    """Lengths of consecutive True runs and of False gaps."""
    if len(x) == 0:
        return np.array([]), np.array([])
    d = np.diff(np.r_[0, x.astype(int), 0])
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    runs = e - s
    gaps = s[1:] - e[:-1]
    return runs, gaps


def cause_features(h, k):
    h = np.asarray(h, float)
    raw = h >= k
    runs, gaps = _runs(raw)
    x = raw - raw.mean()
    v = (x * x).sum()
    ac = np.array([(x[:-L] * x[L:]).sum() / v if v > 0 and L < len(x) else 0.0 for L in range(1, 61)])
    return np.array([raw.mean(), h.mean() / k, (h > 0).mean(),
                     runs.mean() if len(runs) else 0.0, runs.max() if len(runs) else 0.0,
                     100.0 * len(runs) / len(h), gaps.mean() if len(gaps) else len(h),
                     ac[4:].max(), 5 + int(np.argmax(ac[4:])), ac[4], ac[39]])


def samples(series, k, cls, W, hist=200, stride=10):
    X = [cause_features(series[t - hist:t], k) for t in range(W + hist, len(series), stride)]
    return np.array(X), np.full(len(X), cls)
