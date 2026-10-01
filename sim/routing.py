"""Per-slot routing replay on a precomputed trace (paired across schemes)."""
import numpy as np
from scipy.sparse.csgraph import dijkstra
from manet_sim import Params, Trace, per_from_sinr, active_flows


def route_paths(weight, usable, srcs):
    """Shortest paths from each source; weight is (N,N) cost, 0 where not usable."""
    w = np.where(usable, weight, 0.0)
    dist, pred = dijkstra(w, directed=True, indices=srcs, return_predecessors=True)
    return dist, pred


def extract(pred_row, src, dst):
    if pred_row[dst] < 0 and dst != src:
        return None
    path = [dst]
    while path[-1] != src:
        path.append(pred_row[path[-1]])
    return path[::-1]


def etx_weight(tr, t):
    plr = np.clip(tr.feats[t, ..., 2], 0.0, 0.9)
    plr = np.maximum(plr, plr.T)
    return 1.0 / (1.0 - plr) ** 2          # ETX = 1/(df*dr) with symmetric loss


def replay(tr: Trace, flags: dict, p: Params, penalty: float = 10.0, soft: dict | None = None,
           metric: dict | None = None):
    """flags: scheme -> (T,N,N) bool (None for baseline). Returns per-scheme stats.

    soft: scheme -> (T,N,N) float probability, cost = 1 + penalty * prob.
    """
    T, N = tr.usable.shape[0], tr.usable.shape[1]
    W = p.window
    srcs, dsts = tr.flows[:, 0], tr.flows[:, 1]
    bits = (p.payload_bytes + p.aes_overhead_bytes + p.header_bytes) * 8
    t_tx = bits / p.rate_bps
    schemes = list(flags) + (list(soft) if soft else [])
    out = {s: dict(sent=0, deliv=0, delay=[], hops=[], noroute=0, jam_links=0) for s in schemes}
    diversity = dict(flowslots=0, base_jammed=0, alt_exists=0)
    for t in range(W, T):
        use = tr.usable[t]
        lab = tr.label[t] & use
        act = active_flows(t, p, len(tr.flows))
        if len(act) == 0:
            continue
        for s in schemes:
            if soft and s in soft:
                wgt = 1.0 + penalty * soft[s][t]
            else:
                fl = flags[s]
                base_w = etx_weight(tr, t) if (metric and metric.get(s) == "etx") else np.ones((N, N))
                wgt = base_w if fl is None else base_w + np.where(fl[t], penalty - 1.0, 0.0)
            dist, pred = route_paths(wgt, use, srcs[act])
            for ai, fi in enumerate(act):
                o = out[s]
                o["sent"] += 1
                path = extract(pred[ai], srcs[fi], dsts[fi])
                if path is None:
                    o["noroute"] += 1
                    continue
                ok, delay = True, p.aes_enc_s + p.aes_dec_s
                for a, b in zip(path[:-1], path[1:]):
                    o["jam_links"] += int(lab[a, b])
                    hop_ok = False
                    for att in range(p.max_attempts):
                        delay += t_tx + tr.backoff[t, att, a, b] * 1e-3 * (2 ** att)
                        if tr.delivery_u[t, att, a, b] >= per_from_sinr(tr.sinr_attempt_db[t, att, a, b], p):
                            hop_ok = True
                            break
                    if not hop_ok:
                        ok = False
                        break
                if ok:
                    o["deliv"] += 1
                    o["delay"].append(delay)
                    o["hops"].append(len(path) - 1)
        # diversity: does baseline path cross a truly jammed link, and does a jam-free path exist?
        dist_b, pred_b = route_paths(np.ones((N, N)), use, srcs[act])
        dist_c, _ = route_paths(np.ones((N, N)), use & ~lab, srcs[act])
        for ai, fi in enumerate(act):
            path = extract(pred_b[ai], srcs[fi], dsts[fi])
            if path is None:
                continue
            diversity["flowslots"] += 1
            if any(lab[a, b] for a, b in zip(path[:-1], path[1:])):
                diversity["base_jammed"] += 1
                if np.isfinite(dist_c[ai, dsts[fi]]):
                    diversity["alt_exists"] += 1
    res = {}
    for s, o in out.items():
        res[s] = dict(
            pdr=o["deliv"] / max(o["sent"], 1),
            thr_kbps=o["deliv"] * p.payload_bytes * 8 / ((T - W) * p.slot_s) / 1e3,
            delay_ms=float(np.mean(o["delay"]) * 1e3) if o["delay"] else float("nan"),
            hops=float(np.mean(o["hops"])) if o["hops"] else float("nan"),
            noroute=o["noroute"] / max(o["sent"], 1),
            jam_links_per_pkt=o["jam_links"] / max(o["sent"], 1),
        )
    fs = max(diversity["flowslots"], 1)
    res["_diversity"] = dict(base_jammed=diversity["base_jammed"] / fs,
                             alt_given_jammed=diversity["alt_exists"] / max(diversity["base_jammed"], 1),
                             alt_frac=diversity["alt_exists"] / fs)
    return res
