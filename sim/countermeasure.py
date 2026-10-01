"""Stateful replay: rerouting and/or per-link channel switching as jamming countermeasures.

A spot jammer occupies the primary channel. A link that switches to a backup
channel suffers an outage of `switch_slots`, then stays on the backup channel
for `dwell_slots`. A follower jammer re-tunes to the backup channel after
`follow_slots` (np.inf = never). Routing cost is hop count or ETX.
"""
import numpy as np
from manet_sim import Params, Trace, per_from_sinr, active_flows
from routing import route_paths, extract


def _etx(plr):
    plr = np.clip(plr, 0.0, 0.9)
    plr = np.maximum(plr, plr.T)
    return 1.0 / (1.0 - plr) ** 2


def replay_cm(tr: Trace, p: Params, schemes: dict, flags: dict, penalty=10.0,
              switch_slots=2, dwell_slots=100, follow_slots=50.0, naive_plr=0.5):
    """schemes: name -> dict(metric='hop'|'etx', reroute=flag_key|None, switch=flag_key|'naive'|None).
    flags: flag_key -> (T,N,N) bool."""
    T, N = tr.usable.shape[:2]
    W = p.window
    srcs, dsts = tr.flows[:, 0], tr.flows[:, 1]
    bits = (p.payload_bytes + p.aes_overhead_bytes + p.header_bytes) * 8
    t_tx = bits / p.rate_bps
    iu = np.triu(np.ones((N, N), bool), 1)
    res = {}
    for name, cfg in schemes.items():
        sw_left = np.zeros((N, N), np.int32)    # outage slots remaining
        dwell = np.zeros((N, N), np.int32)      # slots remaining on backup
        bage = np.zeros((N, N), np.int32)       # slots since switch
        sent = deliv = nswitch = ntx = 0
        delays, hops = [], []
        for t in range(W, T):
            # state update
            on_backup = dwell > 0
            dwell = np.maximum(dwell - 1, 0)
            sw_left = np.maximum(sw_left - 1, 0)
            bage = np.where(on_backup, bage + 1, 0)
            sw = cfg.get("switch")
            if sw is not None:
                trig = (tr.full[t] & (tr.feats[t, ..., 2] >= naive_plr)) if sw == "naive" else flags[sw][t]
                trig = (trig | trig.T) & tr.usable_nj[t] & ~on_backup
                new = trig & iu
                new = new | new.T
                nswitch += int(new.sum() // 2)
                sw_left = np.where(new, switch_slots, sw_left)
                dwell = np.where(new, dwell_slots, dwell)
                bage = np.where(new, 0, bage)
                on_backup = dwell > 0
            clean = on_backup & (bage < follow_slots)    # jammer has not followed yet
            use = np.where(on_backup, tr.usable_nj[t], tr.usable[t]) & (sw_left == 0)
            if cfg["metric"] == "etx":
                w = _etx(np.where(clean, tr.plr_nj[t], tr.feats[t, ..., 2]))
            else:
                w = np.ones((N, N))
            rr = cfg.get("reroute")
            if rr is not None:
                w = w + np.where(flags[rr][t] & ~clean, penalty - 1.0, 0.0)
            act = active_flows(t, p, len(srcs))
            if len(act) == 0:
                continue
            dist, pred = route_paths(w, use, srcs[act])
            for ai, fi in enumerate(act):
                sent += 1
                path = extract(pred[ai], srcs[fi], dsts[fi])
                if path is None:
                    continue
                ok, delay = True, p.aes_enc_s + p.aes_dec_s
                for a, b in zip(path[:-1], path[1:]):
                    sinr = tr.sinr_attempt_nj_db[t, :, a, b] if clean[a, b] else tr.sinr_attempt_db[t, :, a, b]
                    hop_ok = False
                    for att in range(p.max_attempts):
                        delay += t_tx + tr.backoff[t, att, a, b] * 1e-3 * (2 ** att)
                        ntx += 1
                        if tr.delivery_u[t, att, a, b] >= per_from_sinr(sinr[att], p):
                            hop_ok = True
                            break
                    if not hop_ok:
                        ok = False
                        break
                if ok:
                    deliv += 1
                    delays.append(delay)
                    hops.append(len(path) - 1)
        dur = (T - W) * p.slot_s
        res[name] = dict(pdr=deliv / max(sent, 1),
                         thr_kbps=deliv * p.payload_bytes * 8 / dur / 1e3,
                         delay_ms=float(np.mean(delays) * 1e3) if delays else float("nan"),
                         hops=float(np.mean(hops)) if hops else float("nan"),
                         switches_per_s=nswitch / dur, airtime=ntx * t_tx / dur)
    return res


def onoff_phi(H, L_on, L_off):
    """Mean probability that a two-state Markov jammer, ON now, is ON during the next H slots."""
    pi = L_on / (L_on + L_off)
    lam = max(1.0 - 1.0 / L_on - 1.0 / L_off, 0.0)
    if lam >= 1.0:
        return 1.0
    return pi + (1.0 - pi) * lam * (1.0 - lam ** H) / (H * (1.0 - lam))


def gate_threshold(p0, F, tau, L_on=np.inf, L_off=np.inf, p_new=None):
    """Change only if the PDR while jammed is below this (Proposition 1; persistent jammer when L = inf).
    p_new: PDR on the new channel after a change, if it differs from p0 (e.g. nodes that miss the order)."""
    H = F + tau
    phi = 1.0 if (np.isinf(L_on) or np.isinf(L_off)) else onoff_phi(H, L_on, L_off)
    pn = p0 if p_new is None else p_new
    return (F * pn - H * (1.0 - phi) * p0) / (H * phi)


def replay_netswitch(tr: Trace, p: Params, trigger, k_links=3, net_outage=10,
                     follow_slots=50.0, naive_plr=0.5, holdoff=10, gate=False, gate_win=40, p0_alpha=0.01,
                     gate_follow=50.0, persist=1, gate_mode="episode", ep_prior=40.0, ep_gap=5,
                     cplane="lossy", rejoin_slots=5, gate_test="bayes", gate_conf=0.95,
                     ord_rounds=3, ord_deadline=3, cause=None, cause_allow=("spot", "two"), cause_hist=200,
                     learn_new=True, new_prior=20.0):
    """Network-wide frequency change (single-channel tactical net) with ETX routing.

    trigger: (T,N,N) bool alarm per link, 'naive' (window PLR >= naive_plr), or None.
    A change is ordered when >= k_links links alarm for `persist` consecutive slots; the whole
    net is silent for `net_outage` slots (tau). After the change, the jammer needs `follow_slots`
    (F) after the net resumes on the new channel to find it (as in Proposition 1). Alarms are ignored
    for `holdoff` slots after a change.

    gate: severity gate of Proposition 2. gate_mode 'fixed' assumes a persistent jammer;
    'episode' estimates the jammer's mean ON/OFF episode lengths online from the alarm history
    (censored maximum likelihood, one pseudo-episode of `ep_prior` slots as prior).

    gate_test: 'mean' compares the mean delivery of the last `gate_win` slots with the threshold;
    'bayes' is a sequential test: with a uniform prior on the PDR p_J and the s delivered and f lost
    packets of the last `gate_win` slots, change when P(p_J < threshold | s, f) >= gate_conf.

    cplane: 'ideal' = alarms and the order reach every node instantly; 'flood' = alarms travel
    and the order is flooded only over the jammed channel, so each connected component decides
    alone (it changes when it hears >= k_links alarms). Nodes in components that do not change
    miss the order; they notice that their neighbours went silent and step to the next epoch
    channel on their own after `rejoin_slots` (isolated meanwhile). 'lossy' additionally floods the
    order hop by hop over the jammed links (`ord_rounds` relay rounds per slot, each relay
    broadcasting twice, reception drawn from the faded SINR), with alarms heard one slot late; the
    order names a switch time `ord_deadline` slots ahead and nodes it has not reached by then are
    stranded.

    cause: optional (model, feature_fn) jammer-type classifier applied to the alarm-consensus
    history; a change is allowed only while the predicted type is in `cause_allow`.
    """
    from scipy.sparse.csgraph import connected_components
    T, N = tr.usable.shape[:2]
    W = p.window
    srcs, dsts = tr.flows[:, 0], tr.flows[:, 1]
    bits = (p.payload_bytes + p.aes_overhead_bytes + p.header_bytes) * 8
    t_tx = bits / p.rate_bps
    outage = 0
    since = 10 ** 9           # slots since last change (large = jammer already on our channel)
    sent = deliv = nsw = ntx = 0
    sent_on = deliv_on = 0    # slots in which the jammer is active on our channel
    delays = []
    exposed_prev, exp_start = False, 0     # jammer active on our channel (not yet escaped)
    n_exposures = n_escaped = 0
    escaped_this = False
    trig_delay = []
    iu = np.triu(np.ones((N, N), bool), 1)
    run = 0                  # consecutive slots with an alarm consensus
    recent = []              # per-slot delivery ratio on the current channel (end-to-end ACKs)
    p0_est = None            # running clean-channel PDR estimate (slots without an alarm consensus)
    stranded = np.zeros(N, np.int32)       # slots until a node that missed the order rejoins
    strand_frac = []
    # online episode statistics (seen through the alarm consensus)
    ep_on, ep_off, ep_state, ep_len, ep_quiet = 0.0, 0.0, None, 0, 0
    n_on_done = n_off_done = 0
    ep_start = 0
    pn_s = pn_n = 0          # delivered / offered packets in the F slots after past changes
    rec_s, rec_n = [], []    # delivered and offered packets per listening slot (for the sequential test)
    pending, pend_miss = 0, None           # order in flight (lossy control plane)
    prev_alarm = None
    hist = []                              # alarm-consensus history while listening (cause classifier)
    ord_frac = []
    cause_ok = True
    for t in range(W, T):
        clean = since < net_outage + follow_slots
        exposed = bool(tr.jam_active[t]) and not clean
        if exposed and not exposed_prev:
            exp_start, n_exposures, escaped_this = t, n_exposures + 1, False
        exposed_prev = exposed
        act = active_flows(t, p, len(srcs))
        stranded = np.maximum(stranded - 1, 0)
        alive = stranded == 0
        if trigger is None:
            alarm = np.zeros((N, N), bool)
        elif isinstance(trigger, str):
            plr = tr.plr_nj[t] if clean else tr.feats[t, ..., 2]
            alarm = tr.full[t] & (plr >= naive_plr)
        else:
            alarm = trigger[t] & (not clean)   # on the new clean channel the detector sees no jamming
        alarm = (alarm | alarm.T) & iu & tr.usable_nj[t] & alive[:, None] & alive[None, :]
        use_now = (tr.usable_nj[t] if clean else tr.usable[t]) & alive[:, None] & alive[None, :]
        if cplane == "lossy":                       # alarm reports arrive one slot late
            alarm, prev_alarm = (prev_alarm if prev_alarm is not None else np.zeros_like(alarm)), alarm
            alarm = alarm & alive[:, None] & alive[None, :]
        if cplane in ("flood", "lossy"):
            ncomp, comp = connected_components(use_now, directed=False)
            ai, aj = np.nonzero(alarm)
            # an alarm is heard in the component of either endpoint (each component decides alone)
            cnt = np.zeros(ncomp, int)
            np.add.at(cnt, comp[ai], 1)
            np.add.at(cnt, comp[aj], (comp[aj] != comp[ai]).astype(int))
            decide = cnt >= k_links
            n_alarm = int(cnt.max()) if ncomp else 0
        else:
            n_alarm = int(alarm.sum())
        raw = n_alarm >= k_links
        # episode statistics (only while the jammer could be on our channel and we are listening)
        if outage == 0 and not clean:
            hist.append(n_alarm)
            if len(hist) > cause_hist:
                hist.pop(0)
            if raw:
                if ep_state != "on":
                    if ep_state == "off":
                        n_off_done += 1
                    ep_state, ep_len = "on", 0
                    ep_start = len(rec_s)
                ep_len += 1
                ep_on += 1
                ep_quiet = 0
            else:
                if ep_state == "on":
                    ep_quiet += 1
                    ep_on += 1
                    if ep_quiet >= ep_gap:           # episode over (hysteresis against single drops)
                        ep_on -= ep_gap
                        ep_off += ep_gap
                        n_on_done += 1
                        ep_state = "off"
                elif ep_state == "off":
                    ep_off += 1
                else:
                    ep_state = "off"
                    ep_off += 1
        if pending > 0:                             # an order is in flight: the switch happens at its deadline
            pending -= 1
            if pending == 0:
                recent, rec_s, rec_n = [], [], []
                nsw += 1
                outage = net_outage
                since = 0
                ep_state, ep_quiet = None, 0
                stranded = np.where(pend_miss, net_outage + rejoin_slots, stranded)
                strand_frac.append(float(pend_miss.mean()))
        elif trigger is not None and outage == 0 and since >= holdoff:
            run = run + 1 if raw else 0
            consensus = run >= persist
            if cause is not None and consensus:
                model, ffn = cause
                consensus = len(hist) < 20 or model.predict(ffn(np.array(hist), k_links)[None])[0] in cause_allow
            if gate and consensus:
                if gate_test == "bayes":
                    from scipy.special import betainc
                    k0 = max(len(rec_s) - gate_win, ep_start if ep_state == "on" else len(rec_s))
                    s_, n_ = sum(rec_s[k0:]), sum(rec_n[k0:])     # packets of the current alarm episode
                    pj = None
                else:
                    pj = np.mean(recent[-gate_win:]) if len(recent) >= gate_win else 1.0
                p0 = p0_est if p0_est is not None else 1.0
                p_new = (pn_s + new_prior * p0) / (pn_n + new_prior) if learn_new else None
                if gate_mode == "fixed":
                    thr = gate_threshold(p0, gate_follow, net_outage, p_new=p_new)
                else:
                    L_on = (ep_on + ep_prior) / (n_on_done + 1)
                    L_off = (ep_off + ep_prior) / (n_off_done + 1)
                    thr = gate_threshold(p0, gate_follow, net_outage, L_on, L_off, p_new=p_new)
                if pj is None:
                    consensus = betainc(1.0 + s_, 1.0 + n_ - s_, max(thr, 0.0)) >= gate_conf
                else:
                    consensus = pj < thr
            if consensus:
                run = 0
                if exposed and not escaped_this:
                    trig_delay.append(t - exp_start)
                    n_escaped += 1
                    escaped_this = True
                if cplane == "lossy" and ncomp:
                    # flood the order from the node with the most alarms in each deciding component
                    rng = np.random.default_rng(t)
                    deg = np.zeros(N, int)
                    ai, aj = np.nonzero(alarm)
                    np.add.at(deg, ai, 1)
                    np.add.at(deg, aj, 1)
                    informed = np.zeros(N, bool)
                    for c in np.flatnonzero(decide):
                        mem = comp == c
                        informed[np.flatnonzero(mem)[np.argmax(deg[mem])]] = True
                    frontier = informed.copy()
                    for r in range(ord_rounds * ord_deadline):
                        ts = min(t + r // ord_rounds, T - 1)
                        sinr = tr.sinr_attempt_db[ts] if not clean else tr.sinr_attempt_nj_db[ts]
                        ok = (rng.random(sinr.shape) >= per_from_sinr(sinr, p)).any(0) & use_now
                        new_ = ok[frontier].any(0) & ~informed & alive
                        informed |= new_
                        frontier = new_
                        if not frontier.any():
                            break
                    pend_miss = ~informed & alive
                    ord_frac.append(float(informed[alive].mean()) if alive.any() else 1.0)
                    pending = ord_deadline
                else:
                    recent, rec_s, rec_n = [], [], []
                    nsw += 1
                    outage = net_outage
                    since = 0
                    ep_state, ep_quiet = None, 0          # current episode is censored by the change
                    if cplane == "flood" and ncomp:
                        miss = ~decide[comp] & alive
                        stranded = np.where(miss, net_outage + rejoin_slots, stranded)
                        strand_frac.append(float(miss.mean()))
        since += 1
        if outage > 0:
            outage -= 1
            sent += len(act)
            if exposed:
                sent_on += len(act)
            continue
        clean = since < net_outage + follow_slots
        alive = stranded == 0
        use = (tr.usable_nj[t] if clean else tr.usable[t]) & alive[:, None] & alive[None, :]
        w = _etx(tr.plr_nj[t] if clean else tr.feats[t, ..., 2])
        if len(act) == 0:
            continue
        dist, pred = route_paths(w, use, srcs[act])
        sinr_all = tr.sinr_attempt_nj_db[t] if clean else tr.sinr_attempt_db[t]
        d0 = deliv
        for ai_, fi in enumerate(act):
            sent += 1
            path = extract(pred[ai_], srcs[fi], dsts[fi])
            if path is None:
                continue
            ok, delay = True, p.aes_enc_s + p.aes_dec_s
            for a, b in zip(path[:-1], path[1:]):
                hop_ok = False
                for att in range(p.max_attempts):
                    delay += t_tx + tr.backoff[t, att, a, b] * 1e-3 * (2 ** att)
                    ntx += 1
                    if tr.delivery_u[t, att, a, b] >= per_from_sinr(sinr_all[att, a, b], p):
                        hop_ok = True
                        break
                if not hop_ok:
                    ok = False
                    break
            if ok:
                deliv += 1
                delays.append(delay)
        if exposed:
            sent_on += len(act)
            deliv_on += deliv - d0
        r_slot = (deliv - d0) / len(act)
        recent.append(r_slot)
        rec_s.append(deliv - d0)
        rec_n.append(len(act))
        if nsw and since <= net_outage + gate_follow:
            pn_s += deliv - d0
            pn_n += len(act)
        if not raw:
            p0_est = r_slot if p0_est is None else (1 - p0_alpha) * p0_est + p0_alpha * r_slot
    dur = (T - W) * p.slot_s
    return dict(pdr=deliv / max(sent, 1), thr_kbps=deliv * p.payload_bytes * 8 / dur / 1e3,
                delay_ms=float(np.mean(delays) * 1e3) if delays else float("nan"),
                switches_per_s=nsw / dur, airtime=ntx * t_tx / dur,
                exposures=n_exposures, escaped=n_escaped,
                trig_delay_slots=float(np.mean(trig_delay)) if trig_delay else float("nan"),
                pdr_on=deliv_on / max(sent_on, 1), sent_on=sent_on,
                pdr_off=(deliv - deliv_on) / max(sent - sent_on, 1),
                stranded=float(np.mean(strand_frac)) if strand_frac else 0.0,
                order_reach=float(np.mean(ord_frac)) if ord_frac else float("nan"),
                L_on=(ep_on + ep_prior) / (n_on_done + 1), L_off=(ep_off + ep_prior) / (n_off_done + 1))
