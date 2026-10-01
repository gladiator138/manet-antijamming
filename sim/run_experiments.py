"""Reproduce every number and figure in the paper.

python3 run_experiments.py detector   -> results/detector.json, results/rf.pkl
python3 run_experiments.py routing    -> results/routing.json
python3 run_experiments.py follow     -> results/follow.json
"""
import json, pickle, sys, time, os
import numpy as np
from multiprocessing import Pool
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix,
                             average_precision_score, balanced_accuracy_score)
from manet_sim import Params, simulate_trace, dataset_from_trace, FEATURES
from countermeasure import replay_cm, replay_netswitch

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
os.makedirs(RES, exist_ok=True)
POWERS = [None, 4, 8, 12, 16, 20, 24]
TRAIN_SEEDS = list(range(1000, 1014))
TEST_SEEDS = list(range(2000, 2007))
ROUTE_SEEDS = list(range(101, 111))   # disjoint from seeds used while designing the gate
RF_KW = dict(n_estimators=120, max_depth=12, n_jobs=-1, random_state=0)
THRESH = 0.80          # chosen with PERSIST on design seeds 1-3 (sim/../README)


def _collect(args):
    seed, pj, W, slots = args
    p = Params(jam_dbm=pj, window=W)
    tr = simulate_trace(p, slots, seed)
    return dataset_from_trace(tr, stride=5)


def make_data(seeds, W=6, slots=1500):
    jobs = [(s, POWERS[i % len(POWERS)], W, slots) for i, s in enumerate(seeds)]
    with Pool(4) as pool:
        parts = pool.map(_collect, jobs)
    return np.concatenate([a for a, _ in parts]), np.concatenate([b for _, b in parts])


def metrics(y, prob, thr=0.5):
    yh = prob >= thr
    return dict(acc=accuracy_score(y, yh), bacc=balanced_accuracy_score(y, yh), prec=precision_score(y, yh),
                rec=recall_score(y, yh), f1=f1_score(y, yh), auc=roc_auc_score(y, prob),
                prauc=average_precision_score(y, prob))


def detector():
    out = {}
    Xtr, ytr = make_data(TRAIN_SEEDS)
    Xte, yte = make_data(TEST_SEEDS)
    out["n_train"], out["n_test"] = int(len(ytr)), int(len(yte))
    out["pos_train"], out["pos_test"] = float(ytr.mean()), float(yte.mean())
    rf = RandomForestClassifier(**RF_KW).fit(Xtr, ytr)
    prob = rf.predict_proba(Xte)[:, 1]
    out["rf"] = metrics(yte, prob, THRESH)
    out["rf_cm"] = confusion_matrix(yte, prob >= THRESH).tolist()
    fpr, tpr, _ = roc_curve(yte, prob)
    idx = np.linspace(0, len(fpr) - 1, 300).astype(int)
    out["rf_roc"] = dict(fpr=fpr[idx].tolist(), tpr=tpr[idx].tolist())
    out["importance"] = dict(zip(FEATURES, rf.feature_importances_.tolist()))
    # logistic regression baseline
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr, ytr)
    out["lr"] = metrics(yte, lr.predict_proba(Xte)[:, 1], 0.5)
    # fixed-threshold baseline: best single-feature threshold on the TRAIN set (max F1)
    best = None
    for j, name in enumerate(FEATURES):
        for sign in (1, -1):
            for q in np.quantile(Xtr[:, j], np.linspace(0.02, 0.98, 49)):
                yh = sign * Xtr[:, j] >= sign * q
                f = f1_score(ytr, yh)
                if best is None or f > best[0]:
                    best = (f, j, sign, q)
    _, j, sign, q = best
    score = sign * (Xte[:, j] - q)
    m = metrics(yte, score, 0.0)
    m["auc"] = roc_auc_score(yte, sign * Xte[:, j])
    m["prauc"] = average_precision_score(yte, sign * Xte[:, j])
    out["thr"] = dict(m, feature=FEATURES[j], sign=int(sign), threshold=float(q))
    # consistency check in the spirit of Xu et al. [5]: low delivery despite high signal strength
    cc = fit_consistency(Xtr, ytr)
    m = metrics(yte, consistency_flag(Xte, cc).astype(float), 0.5)
    m["auc"] = m["prauc"] = float("nan")
    out["cc"] = dict(m, **cc)
    # window-length sensitivity
    out["window"] = {}
    for W in (3, 6, 10):
        if W == 6:
            out["window"][W] = out["rf"]
            continue
        a, b = make_data(TRAIN_SEEDS, W=W)
        c, d = make_data(TEST_SEEDS, W=W)
        r = RandomForestClassifier(**RF_KW).fit(a, b)
        out["window"][W] = metrics(d, r.predict_proba(c)[:, 1], THRESH)
    # on-node cost
    nodes = sum(e.tree_.node_count for e in rf.estimators_)
    depth = [e.tree_.max_depth for e in rf.estimators_]
    rf.set_params(n_jobs=1)
    x1 = Xte[:1]
    t = time.perf_counter()
    for _ in range(200):
        rf.predict_proba(x1)
    single_ms = (time.perf_counter() - t) / 200 * 1e3
    t = time.perf_counter()
    rf.predict_proba(Xte[:20000])
    batch_us = (time.perf_counter() - t) / 20000 * 1e6
    out["cost"] = dict(total_nodes=int(nodes), mean_depth=float(np.mean(depth)), max_depth=int(max(depth)),
                       max_comparisons=int(sum(depth)), bytes_16b_node=int(nodes * 16),
                       sklearn_single_ms=single_ms, sklearn_batch_us=batch_us)
    json.dump(out, open(os.path.join(RES, "detector.json"), "w"), indent=1)
    pickle.dump(rf, open(os.path.join(RES, "rf.pkl"), "wb"))
    print(json.dumps({k: v for k, v in out.items() if k not in ("rf_roc",)}, indent=1))


def fit_consistency(X, y, n=300000):
    """Flag a window when PLR >= a and mean RSSI >= b; (a, b) maximise F1 on the training set."""
    i = np.random.default_rng(0).choice(len(y), min(n, len(y)), replace=False)
    X, y = X[i], y[i]
    best = None
    for a in np.arange(0.1, 0.95, 0.05):
        for b in np.quantile(X[:, 0], np.linspace(0.02, 0.98, 49)):
            f = f1_score(y, (X[:, 2] >= a) & (X[:, 0] >= b))
            if best is None or f > best[0]:
                best = (f, a, b)
    return dict(plr_min=float(best[1]), rssi_min=float(best[2]))


def consistency_flag(X, cc):
    return (X[..., 2] >= cc["plr_min"]) & (X[..., 0] >= cc["rssi_min"])


def _cc_flags(tr):
    cc = json.load(open(os.path.join(RES, "detector.json")))["cc"]
    return tr.full & consistency_flag(tr.feats, cc)


def _flags(rf, tr):
    T, N = tr.usable.shape[:2]
    m = tr.full
    prob = np.zeros((T, N, N), np.float32)
    prob[m] = rf.predict_proba(tr.feats[m])[:, 1]
    return prob >= THRESH


def load_quantised():
    """10-tree forest with 16-bit inputs and split thresholds, as implemented on the FPGA."""
    rf = pickle.load(open(os.path.join(RES, "rf_compact.pkl"), "rb"))
    qj = json.load(open(os.path.join(RES, "quant.json")))
    lo, hi, levels = np.array(qj["lo"]), np.array(qj["hi"]), 2 ** qj["bits"] - 1
    scale = (hi - lo) / levels
    for e in rf.estimators_:
        t = e.tree_
        inner = t.feature >= 0
        f = t.feature[inner]
        t.threshold[inner] = np.clip(np.round((t.threshold[inner] - lo[f]) / scale[f]), 0, levels) * scale[f] + lo[f] + scale[f] / 2
    rf.set_params(n_jobs=1)

    def q(x):
        return np.clip(np.round((x - lo) / scale), 0, levels) * scale + lo
    return rf, q


def _flags_q(rfq, tr):
    rf, q = rfq
    T, N = tr.usable.shape[:2]
    m = tr.full
    prob = np.zeros((T, N, N), np.float32)
    prob[m] = rf.predict_proba(q(tr.feats[m]))[:, 1]
    return prob >= THRESH


SCHEMES = {
    "HOP": dict(metric="hop"),
    "HOP+ML-reroute": dict(metric="hop", reroute="ml"),
    "ETX": dict(metric="etx"),
    "ETX+ML-reroute": dict(metric="etx", reroute="ml"),
    "HOP+Label-reroute": dict(metric="hop", reroute="label"),
    "ETX+Label-reroute": dict(metric="etx", reroute="label"),
}
NET = {  # network-wide frequency change: name -> (trigger, loss threshold, gate, extra kwargs)
    "ETX+Loss-FC(0.7)": ("naive", 0.7, None, {}),
    "ETX+Loss-FC(0.9)": ("naive", 0.9, None, {}),
    "ETX+Consistency-FC": ("cc", None, None, {}),
    "ETX+ML-FC": ("ml", None, None, {}),
    "ETX+Loss-FC(0.7)+gate": ("naive", 0.7, "episode", {}),
    "ETX+Loss-FC(0.9)+gate": ("naive", 0.9, "episode", {}),
    "ETX+Consistency-FC+gate": ("cc", None, "episode", {}),
    "ETX+ML-FC+fixed gate": ("ml", None, "fixed", {}),
    "ETX+ML-FC+gate (proposed)": ("ml", None, "episode", {}),
    "ETX+ML-FC+gate, compact": ("mlq", None, "episode", {}),
    "ETX+ML-FC+gate, ideal ctrl": ("ml", None, "episode", dict(cplane="ideal")),
    "ETX+ML-FC, ideal ctrl": ("ml", None, None, dict(cplane="ideal")),
    "ETX+Label-FC": ("oracle", None, None, {}),
    "ETX+ML-FC+cause": ("ml", None, None, dict(cause="rf")),
    "ETX+ML-FC+cause+gate": ("ml", None, "episode", dict(cause="rf")),
    "ETX+ML-FC+gate, flood ctrl": ("ml", None, "episode", dict(cplane="flood")),
    "ETX+ML-FC+gate, mean test": ("ml", None, "episode", dict(gate_test="mean", gate_win=20, learn_new=False, ep_prior=100.0)),
}


_CAUSE = None


def load_cause():
    global _CAUSE
    if _CAUSE is None:
        from cause import cause_features
        m = pickle.load(open(os.path.join(RES, "cause.pkl"), "rb"))
        m.set_params(n_jobs=1)
        _CAUSE = (m, cause_features)
    return _CAUSE


def run_net(tr, p, trig, names=None, **kw):
    out = {}
    for name, (kind, th, gate, extra) in NET.items():
        if names is not None and name not in names:
            continue
        a = dict(k_links=K_LINKS, persist=PERSIST, naive_plr=th or 0.5, gate=gate is not None,
                 gate_mode=gate or "episode")
        a.update(extra)
        if a.get("cause") == "rf":
            a["cause"] = load_cause()
        a.update(kw)
        out[name] = replay_netswitch(tr, p, trig[kind], **a)
    return out
K_LINKS = 8
PERSIST = 3            # consensus must hold for this many consecutive slots


def _route_job(args):
    seed, pj, slots = args
    rf = pickle.load(open(os.path.join(RES, "rf.pkl"), "rb"))
    rf.set_params(n_jobs=1)
    p = Params(jam_dbm=pj)
    tr = simulate_trace(p, slots, seed)
    ml = _flags(rf, tr)
    trig = dict(naive="naive", ml=ml, mlq=_flags_q(load_quantised(), tr), cc=_cc_flags(tr), oracle=tr.label)
    r = replay_cm(tr, p, SCHEMES, {"ml": ml, "label": tr.label})
    r["ETX-net"] = replay_netswitch(tr, p, None)          # PDR while the jammer is on (p_J) and off (p0)
    r.update(run_net(tr, p, trig))
    # diversity: flows whose hop-count path crosses a jammed link and a jam-free alternative exists
    from routing import replay as rp
    div = rp(tr, {"b": None}, p)["_diversity"]
    return dict(seed=seed, pj=pj, res=r, div=div,
                det=dict(tp=int((ml & tr.label & tr.usable).sum()), fp=int((ml & ~tr.label & tr.usable).sum())))


def routing(slots=1500):
    jobs = [(s, pj, slots) for pj in POWERS for s in ROUTE_SEEDS]
    with Pool(4) as pool:
        out = pool.map(_route_job, jobs)
    json.dump(out, open(os.path.join(RES, "routing.json"), "w"), indent=1)


def _follow_job(args):
    seed, fs, slots = args
    rf = pickle.load(open(os.path.join(RES, "rf.pkl"), "rb"))
    rf.set_params(n_jobs=1)
    p = Params(jam_dbm=16.0)
    tr = simulate_trace(p, slots, seed)
    ml = _flags(rf, tr)
    return dict(seed=seed, follow=fs,
                none=replay_netswitch(tr, p, None)["pdr"],
                ml=replay_netswitch(tr, p, ml, k_links=K_LINKS, persist=PERSIST, follow_slots=fs)["pdr"],
                mlgate=replay_netswitch(tr, p, ml, k_links=K_LINKS, persist=PERSIST, follow_slots=fs, gate=True, gate_follow=fs)["pdr"],
                mlgate50=replay_netswitch(tr, p, ml, k_links=K_LINKS, persist=PERSIST, follow_slots=fs, gate=True, gate_follow=50)["pdr"],
                label=replay_netswitch(tr, p, tr.label, k_links=K_LINKS, persist=PERSIST, follow_slots=fs)["pdr"])


def _tau_job(args):
    seed, tau, slots = args
    rf = pickle.load(open(os.path.join(RES, "rf.pkl"), "rb"))
    rf.set_params(n_jobs=1)
    p = Params(jam_dbm=16.0)
    tr = simulate_trace(p, slots, seed)
    ml = _flags(rf, tr)
    return dict(seed=seed, tau=tau, none=replay_netswitch(tr, p, None)["pdr"],
                ml=replay_netswitch(tr, p, ml, k_links=K_LINKS, persist=PERSIST, net_outage=tau)["pdr"],
                mlgate=replay_netswitch(tr, p, ml, k_links=K_LINKS, persist=PERSIST, net_outage=tau, gate=True)["pdr"])


def tau(slots=1500):
    """Sensitivity to the retune-and-resynchronise time tau (Proposition 2 depends on F / tau)."""
    jobs = [(s, t, slots) for t in (5, 10, 20, 40) for s in ROUTE_SEEDS]
    with Pool(4) as pool:
        out = pool.map(_tau_job, jobs)
    json.dump(out, open(os.path.join(RES, "tau.json"), "w"), indent=1)


def follow(slots=1500):
    jobs = [(s, fs, slots) for fs in (10, 25, 50, 100, 200) for s in ROUTE_SEEDS]
    with Pool(4) as pool:
        out = pool.map(_follow_job, jobs)
    json.dump(out, open(os.path.join(RES, "follow.json"), "w"), indent=1)


def compact():
    """Accuracy versus model size for edge/FPGA deployment."""
    rng = np.random.default_rng(0)
    Xtr, ytr = make_data(TRAIN_SEEDS)
    Xte, yte = make_data(TEST_SEEDS)
    i = rng.choice(len(ytr), 400000, replace=False)
    j = rng.choice(len(yte), 200000, replace=False)
    out = []
    for trees in (10, 20, 50, 120):
        for depth in (6, 8, 10, 12):
            rf = RandomForestClassifier(n_estimators=trees, max_depth=depth, n_jobs=-1, random_state=0).fit(Xtr[i], ytr[i])
            m = metrics(yte[j], rf.predict_proba(Xte[j])[:, 1], THRESH)
            nodes = sum(e.tree_.node_count for e in rf.estimators_)
            out.append(dict(trees=trees, depth=depth, nodes=int(nodes), kib=nodes * 16 / 1024, **m))
            print(out[-1], flush=True)
    json.dump(out, open(os.path.join(RES, "compact.json"), "w"), indent=1)


VARIANTS = {
    "Baseline (design)": {},
    "Dense, N = 60": dict(n_nodes=60),
    "Sparse, N = 25": dict(n_nodes=25),
    "Strong links, Pt = 20 dBm": dict(pt_dbm=20.0),
    "Weak links, Pt = 10 dBm": dict(pt_dbm=10.0),
    "Heavy traffic, 20 flows": dict(n_flows=20),
    "Carrier sense 450 m": dict(cs_range=450.0),
    "Steeper PER, k = 1.5": dict(per_k=1.5),
}
ROBUST_NET = ["ETX+Loss-FC(0.7)", "ETX+Loss-FC(0.9)", "ETX+Consistency-FC", "ETX+ML-FC", "ETX+Loss-FC(0.9)+gate",
              "ETX+Consistency-FC+gate", "ETX+ML-FC+gate (proposed)", "ETX+ML-FC+gate, compact", "ETX+Label-FC",
              "ETX+ML-FC+cause+gate", "ETX+ML-FC+gate, mean test"]
ROBUST_SEEDS = list(range(301, 306))


def _robust_job(args):
    vname, seed, pj, slots = args
    rf = pickle.load(open(os.path.join(RES, "rf.pkl"), "rb"))
    rf.set_params(n_jobs=1)
    p = Params(jam_dbm=pj, **VARIANTS[vname])
    tr = simulate_trace(p, slots, seed)
    ml = _flags(rf, tr)
    out = dict(variant=vname, seed=seed, pj=pj)
    out["ETX"] = replay_netswitch(tr, p, None)
    trig = dict(naive="naive", ml=ml, mlq=_flags_q(load_quantised(), tr), cc=_cc_flags(tr), oracle=tr.label)
    for k, v in run_net(tr, p, trig, ROBUST_NET).items():
        out[k.replace("ETX+", "")] = v
    m = tr.full
    y, yh = tr.label[m], ml[m]
    out["det"] = dict(acc=float((y == yh).mean()), prec=float((y & yh).sum() / max(yh.sum(), 1)),
                      rec=float((y & yh).sum() / max(y.sum(), 1)))
    return out


def robust(slots=1200):
    jobs = [(v, s, pj, slots) for v in VARIANTS for pj in (None, 8, 16, 24) for s in ROBUST_SEEDS]
    with Pool(4) as pool:
        out = pool.map(_robust_job, jobs)
    json.dump(out, open(os.path.join(RES, "robust.json"), "w"), indent=1)


def detpoints():
    """Random-Forest test metrics at several decision thresholds (the ROC operating points)."""
    rf = pickle.load(open(os.path.join(RES, "rf.pkl"), "rb"))
    Xte, yte = make_data(TEST_SEEDS)
    prob = rf.predict_proba(Xte)[:, 1]
    out = {f"{t:.1f}": metrics(yte, prob, t) for t in (0.5, 0.6, 0.7, 0.8)}
    json.dump(out, open(os.path.join(RES, "detpoints.json"), "w"), indent=1)
    print(out)


JAMMERS = {
    "Spot (training model)": dict(),
    "Pulsed, 20% duty": dict(jam_mode="pulsed"),
    "Sweep, 8 channels": dict(jam_mode="sweep"),
    "Two spot jammers": dict(n_jammers=2),
}


def _jammer_job(args):
    jname, seed, slots = args
    rf = pickle.load(open(os.path.join(RES, "rf.pkl"), "rb"))
    rf.set_params(n_jobs=1)
    p = Params(jam_dbm=16.0, **JAMMERS[jname])
    tr = simulate_trace(p, slots, seed)
    ml = _flags(rf, tr)
    cc = _cc_flags(tr)
    F = 0 if p.jam_mode == "sweep" else 50      # a sweeper is already on every channel
    out = dict(jammer=jname, seed=seed)
    out["ETX"] = replay_netswitch(tr, p, None)
    trig = dict(naive="naive", ml=ml, mlq=_flags_q(load_quantised(), tr), cc=cc, oracle=tr.label)
    names = ["ETX+Loss-FC(0.9)", "ETX+Consistency-FC", "ETX+ML-FC", "ETX+Loss-FC(0.9)+gate", "ETX+Consistency-FC+gate",
             "ETX+ML-FC+fixed gate", "ETX+ML-FC+gate (proposed)", "ETX+ML-FC+cause", "ETX+ML-FC+cause+gate",
             "ETX+ML-FC+gate, mean test", "ETX+Label-FC"]
    for k, v in run_net(tr, p, trig, names, follow_slots=F).items():
        out[k.replace("ETX+", "")] = v
    m = tr.full
    y, yh = tr.label[m], ml[m]
    tp = int((y & yh).sum())
    prec, rec = tp / max(yh.sum(), 1), tp / max(y.sum(), 1)
    out["det"] = dict(acc=float((y == yh).mean()), prec=float(prec), rec=float(rec),
                      f1=float(2 * prec * rec / max(prec + rec, 1e-9)), pos=float(y.mean()))
    return out


def jammers(slots=1500):
    jobs = [(j, s, slots) for j in JAMMERS for s in ROBUST_SEEDS]
    with Pool(4) as pool:
        out = pool.map(_jammer_job, jobs)
    json.dump(out, open(os.path.join(RES, "jammers.json"), "w"), indent=1)


def fixedpoint():
    """16-bit fixed-point inputs and split thresholds (as on an FPGA) for the compact and full forests."""
    rng = np.random.default_rng(0)
    Xtr, ytr = make_data(TRAIN_SEEDS)
    Xte, yte = make_data(TEST_SEEDS)
    i = rng.choice(len(ytr), 400000, replace=False)
    j = rng.choice(len(yte), 200000, replace=False)
    lo, hi = np.percentile(Xtr, 0.01, axis=0), np.percentile(Xtr, 99.99, axis=0)
    out = []
    for bits in (16, 12, 8):
        levels = 2 ** bits - 1
        scale = (hi - lo) / levels

        def q(x):
            return np.clip(np.round((x - lo) / scale), 0, levels) * scale + lo
        for trees, depth in ((10, 12), (120, 12)):
            rf = RandomForestClassifier(n_estimators=trees, max_depth=depth, n_jobs=-1, random_state=0).fit(Xtr[i], ytr[i])
            base = metrics(yte[j], rf.predict_proba(Xte[j])[:, 1], THRESH)
            for e in rf.estimators_:
                t = e.tree_
                inner = t.feature >= 0
                f = t.feature[inner]
                t.threshold[inner] = np.clip(np.round((t.threshold[inner] - lo[f]) / scale[f]), 0, levels) * scale[f] + lo[f] + scale[f] / 2
            qm = metrics(yte[j], rf.predict_proba(q(Xte[j]))[:, 1], THRESH)
            if bits == 16 and trees == 10:
                # float model and quantisation range for the FPGA flow (sim/fpga/run_fpga.sh)
                rf_f = RandomForestClassifier(n_estimators=trees, max_depth=depth, n_jobs=-1, random_state=0).fit(Xtr[i], ytr[i])
                pickle.dump(rf_f, open(os.path.join(RES, "rf_compact.pkl"), "wb"))
                json.dump(dict(lo=lo.tolist(), hi=hi.tolist(), bits=bits, thresh=THRESH),
                          open(os.path.join(RES, "quant.json"), "w"), indent=1)
                np.save(os.path.join(RES, "fpga_vectors.npy"), Xte[j][:5000])
            out.append(dict(bits=bits, trees=trees, depth=depth, f1_float=base["f1"], f1_fixed=qm["f1"],
                            acc_float=base["acc"], acc_fixed=qm["acc"], auc_fixed=qm["auc"]))
            print(out[-1], flush=True)
    json.dump(out, open(os.path.join(RES, "fixedpoint.json"), "w"), indent=1)


CAUSE_JAM = {"none": dict(jam_dbm=None), "spot": dict(), "pulsed": dict(jam_mode="pulsed"),
             "sweep": dict(jam_mode="sweep"), "two": dict(n_jammers=2)}


def _cause_job(args):
    cls, pj, seed, slots = args
    from cause import alarm_series, samples
    rf = pickle.load(open(os.path.join(RES, "rf.pkl"), "rb"))
    rf.set_params(n_jobs=1)
    kw = dict(CAUSE_JAM[cls])
    kw.setdefault("jam_dbm", pj)
    p = Params(**kw)
    tr = simulate_trace(p, slots, seed)
    ser = alarm_series(tr, _flags(rf, tr), p.window)
    return samples(ser, K_LINKS, cls, p.window)


def cause(slots=1500):
    """Jammer-type classifier on the alarm-consensus history (train seeds / held-out test seeds)."""
    from cause import CLASSES, CAUSE_FEATURES

    def jobs(seeds):
        return [(c, pj, s, slots) for c in CLASSES for pj in ((None,) if c == "none" else (8, 16, 24))
                for s in seeds]
    with Pool(4) as pool:
        tr_ = pool.map(_cause_job, jobs(TRAIN_SEEDS[:4]))
        te_ = pool.map(_cause_job, jobs(TEST_SEEDS[:3]))
    Xtr, ytr = np.concatenate([a for a, _ in tr_]), np.concatenate([b for _, b in tr_])
    Xte, yte = np.concatenate([a for a, _ in te_]), np.concatenate([b for _, b in te_])
    m = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=0).fit(Xtr, ytr)
    yh = m.predict(Xte)
    out = dict(classes=CLASSES, n_train=int(len(ytr)), n_test=int(len(yte)), acc=accuracy_score(yte, yh),
               bacc=balanced_accuracy_score(yte, yh),
               cm=confusion_matrix(yte, yh, labels=CLASSES).tolist(),
               importance=dict(zip(CAUSE_FEATURES, m.feature_importances_.tolist())))
    json.dump(out, open(os.path.join(RES, "cause.json"), "w"), indent=1)
    pickle.dump(m, open(os.path.join(RES, "cause.pkl"), "wb"))
    print(json.dumps(out, indent=1))


def _probe_job(args):
    seed, P, slots = args
    rf = pickle.load(open(os.path.join(RES, "rf.pkl"), "rb"))
    rf.set_params(n_jobs=1)
    p = Params(jam_dbm=16.0, probe_every=P)
    tr = simulate_trace(p, slots, seed)
    ml = _flags(rf, tr)
    m = tr.full
    y, yh = tr.label[m], ml[m]
    tp = int((y & yh).sum())
    return dict(seed=seed, P=P, none=replay_netswitch(tr, p, None)["pdr"],
                ml=replay_netswitch(tr, p, ml, k_links=K_LINKS, persist=PERSIST, cplane="lossy"),
                mlgate=replay_netswitch(tr, p, ml, k_links=K_LINKS, persist=PERSIST, gate=True, cplane="lossy"),
                prec=tp / max(yh.sum(), 1), rec=tp / max(y.sum(), 1))


def probe(slots=1500):
    """Probe rate: one probe per node every P slots (100/P per second); the window spans W * P slots."""
    jobs = [(s, P, slots) for P in (1, 2, 5, 10) for s in ROUTE_SEEDS]
    with Pool(4) as pool:
        out = pool.map(_probe_job, jobs)
    json.dump(out, open(os.path.join(RES, "probe.json"), "w"), indent=1)


if __name__ == "__main__":
    {"detector": detector, "routing": routing, "follow": follow, "compact": compact, "robust": robust,
     "fixedpoint": fixedpoint, "jammers": jammers, "detpoints": detpoints, "tau": tau,
     "cause": cause, "probe": probe}[sys.argv[1]]()
