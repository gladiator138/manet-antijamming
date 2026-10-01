"""Discrete-time tactical MANET simulator with a mobile intermittent jammer.

Channel trace and per-link features are generated first; because the routing
decision does not change the channel (contention is abstracted), detector
outputs can be computed in batch and routing replayed per slot on the same
trace for every scheme (paired comparison with common random numbers).
"""
import numpy as np
from dataclasses import dataclass, field

FEATURES = ["rssi_mean", "sinr_mean", "plr", "ber_mean", "sinr_std"]


@dataclass
class Params:
    n_nodes: int = 40
    area: float = 1000.0
    v_min: float = 2.0
    v_max: float = 15.0
    comm_range: float = 300.0
    pt_dbm: float = 14.0
    pl0_db: float = 40.0
    pl_exp: float = 2.5
    noise_dbm: float = -95.0
    sinr_floor_db: float = 0.0       # mean SINR below this removes the link
    sinr50_db: float = 4.0
    per_k: float = 0.5
    slot_s: float = 0.01
    window: int = 6
    # jammer
    jam_dbm: float | None = 12.0     # None = no jammer
    jam_region: tuple = (250.0, 750.0)  # jammer roams the central region
    jam_v: tuple = (1.0, 5.0)
    jam_on_mean: float = 40.0        # mean episode length (slots)
    jam_off_mean: float = 40.0
    jam_mode: str = "spot"           # "spot": coherent on/off episodes; "pulsed": short low-duty bursts;
                                     # "sweep": sweeps n_sweep_ch channels, dwelling sweep_dwell slots on each
    n_jammers: int = 1
    n_sweep_ch: int = 8
    sweep_dwell: int = 5
    pulsed_on_mean: float = 4.0
    pulsed_off_mean: float = 16.0
    label_margin_db: float = 3.0
    label_mode: str = "loss"         # "presence": INR >= margin at an endpoint
                                     # "impairment": presence AND jammed mean SINR < impair_sinr_db
    impair_sinr_db: float = 10.0     # (legacy threshold, unused by "loss" mode)
    impair_dper: float = 0.10        # "loss" mode: jamming raises expected PER by >= this
    # measurement noise
    rssi_noise_db: float = 1.0
    sinr_noise_db: float = 1.5
    # traffic / MAC
    n_flows: int = 10
    pkt_interval: int = 10           # each flow offers one packet every pkt_interval slots (staggered)
    own_act: float = 0.02            # per-slot probability that a node transmits concurrently (hidden terminal),
                                     # about twice the per-node airtime share at the default load
    cs_range: float = 300.0          # nodes within this distance of a transmitter defer (carrier sense)
    probe_bytes: int = 32
    payload_bytes: int = 1024
    header_bytes: int = 40
    aes_overhead_bytes: int = 28     # AES-GCM: 12-byte nonce + 16-byte tag
    rate_bps: float = 11e6
    max_attempts: int = 2
    aes_enc_s: float = 62e-6         # measured AES-128-GCM seal per 1 KB packet (PyCryptodome)
    aes_dec_s: float = 62e-6
    warmup_s: float = 300.0          # random-waypoint warm-up discarded before t = 0
    probe_every: int = 1             # each node probes every probe_every slots; a window spans W * probe_every slots


def db2lin(x):
    return 10.0 ** (np.asarray(x) / 10.0)


def lin2db(x):
    return 10.0 * np.log10(np.maximum(x, 1e-30))


def pathloss_db(d, p):
    return p.pl0_db + 10 * p.pl_exp * np.log10(np.maximum(d, 1.0))


def per_from_sinr(sinr_db, p):
    return 1.0 / (1.0 + np.exp(p.per_k * (sinr_db - p.sinr50_db)))


def ber_from_sinr(sinr_lin):
    # BPSK-equivalent uncoded BER, used only as an observable feature
    from scipy.special import erfc
    return 0.5 * erfc(np.sqrt(np.maximum(sinr_lin, 0.0)))


_EPER_GRID = np.linspace(-20, 60, 801)
_EPER_CACHE = {}


def expected_per(mean_sinr_db, p):
    """E[PER] under Rayleigh fading, via a Monte Carlo lookup table."""
    key = (p.sinr50_db, p.per_k)
    if key not in _EPER_CACHE:
        g = np.random.default_rng(0).exponential(size=20000)
        tab = [per_from_sinr(m + lin2db(g), p).mean() for m in _EPER_GRID]
        _EPER_CACHE[key] = np.array(tab)
    return np.interp(mean_sinr_db, _EPER_GRID, _EPER_CACHE[key])


_EPJ_X = np.arange(-20.0, 60.01, 0.5)    # mean SNR (dB)
_EPJ_Y = np.arange(-30.0, 50.01, 0.5)    # mean INR of the jammer (dB)
_EPJ_CACHE = {}


def expected_per_jam(snr_db, inr_db, p):
    """E[PER] when both the signal and the jammer undergo independent Rayleigh fading
    (bilinear lookup on a Monte Carlo table)."""
    key = (p.sinr50_db, p.per_k)
    if key not in _EPJ_CACHE:
        r = np.random.default_rng(1)
        g, gj = r.exponential(size=4000), r.exponential(size=4000)
        sx = db2lin(_EPJ_X)[:, None, None] * g[None, None, :]
        tab = np.empty((len(_EPJ_X), len(_EPJ_Y)))
        for k, y in enumerate(_EPJ_Y):
            sinr = sx[:, 0, :] / (1.0 + db2lin(y) * gj[None, :])
            tab[:, k] = per_from_sinr(lin2db(sinr), p).mean(1)
        tab[:, 0] = expected_per(_EPJ_X, p)        # INR at the bottom of the grid = no jammer
        _EPJ_CACHE[key] = tab
    tab = _EPJ_CACHE[key]
    x = np.clip((np.asarray(snr_db) - _EPJ_X[0]) / 0.5, 0, len(_EPJ_X) - 1.001)
    y = np.clip((np.asarray(inr_db) - _EPJ_Y[0]) / 0.5, 0, len(_EPJ_Y) - 1.001)
    i, j = x.astype(int), y.astype(int)
    fx, fy = x - i, y - j
    return ((1 - fx) * (1 - fy) * tab[i, j] + fx * (1 - fy) * tab[i + 1, j]
            + (1 - fx) * fy * tab[i, j + 1] + fx * fy * tab[i + 1, j + 1])


class RWP:
    """Random waypoint mobility (zero pause) in a rectangular region."""

    def __init__(self, n, lo, hi, vmin, vmax, rng):
        self.lo, self.hi, self.vmin, self.vmax, self.rng = lo, hi, vmin, vmax, rng
        self.pos = rng.uniform(lo, hi, size=(n, 2))
        self.dst = rng.uniform(lo, hi, size=(n, 2))
        self.v = rng.uniform(vmin, vmax, size=n)

    def step(self, dt):
        vec = self.dst - self.pos
        dist = np.linalg.norm(vec, axis=1)
        move = self.v * dt
        arrive = dist <= move
        with np.errstate(invalid="ignore", divide="ignore"):
            self.pos = np.where(arrive[:, None], self.dst,
                                self.pos + vec / np.maximum(dist, 1e-9)[:, None] * move[:, None])
        k = int(arrive.sum())
        if k:
            self.dst[arrive] = self.rng.uniform(self.lo, self.hi, size=(k, 2))
            self.v[arrive] = self.rng.uniform(self.vmin, self.vmax, size=k)
        return self.pos


@dataclass
class Trace:
    """Everything the routing replay needs, per slot."""
    usable: np.ndarray          # (T, N, N) bool, symmetric: in range and mean SINR >= floor
    label: np.ndarray           # (T, N, N) bool ground truth "jammed" (window majority)
    full: np.ndarray            # (T, N, N) bool window is full (link observed W slots)
    feats: np.ndarray           # (T, N, N, 5) float32 window features
    sinr_attempt_db: np.ndarray  # (T, A, N, N) float32 per-attempt faded SINR (dB)
    sinr_attempt_nj_db: np.ndarray  # same fading draws, jammer removed (backup channel)
    plr_nj: np.ndarray          # (T, N, N) float32 window loss rate on a jam-free channel
    jam_active: np.ndarray      # (T,) bool
    usable_nj: np.ndarray       # (T, N, N) bool usable on a jam-free channel
    backoff: np.ndarray         # (T, A, N, N) float32 uniform draws for MAC backoff
    flows: np.ndarray           # (F, 2) int
    delivery_u: np.ndarray      # (T, A, N, N) float32 uniform draws for packet success


def simulate_trace(p: Params, n_slots: int, seed: int) -> Trace:
    rng = np.random.default_rng(seed)
    N, W, A = p.n_nodes, p.window, p.max_attempts
    nodes = RWP(N, 0.0, p.area, p.v_min, p.v_max, rng)
    jam = RWP(p.n_jammers, p.jam_region[0], p.jam_region[1], p.jam_v[0], p.jam_v[1], rng)
    for _ in range(int(p.warmup_s)):      # discard the random-waypoint transient (speed decay, density)
        nodes.step(1.0)
        jam.step(1.0)
    n0 = db2lin(p.noise_dbm)
    on_m, off_m = (p.pulsed_on_mean, p.pulsed_off_mean) if p.jam_mode == "pulsed" else (p.jam_on_mean, p.jam_off_mean)
    jam_on = rng.random(p.n_jammers) < on_m / (on_m + off_m)
    jam_timer = np.array([rng.geometric(1.0 / (on_m if o else off_m)) for o in jam_on])
    sweep_phase = int(rng.integers(p.n_sweep_ch * p.sweep_dwell))

    flows = np.array([rng.choice(N, 2, replace=False) for _ in range(p.n_flows)])

    # rolling window buffers
    buf_rssi = np.zeros((W, N, N), np.float32)
    buf_sinr = np.zeros((W, N, N), np.float32)
    buf_loss = np.zeros((W, N, N), np.float32)
    buf_ber = np.zeros((W, N, N), np.float32)
    buf_lab = np.zeros((W, N, N), bool)
    age = np.zeros((N, N), np.int32)

    usable = np.zeros((n_slots, N, N), bool)
    label = np.zeros((n_slots, N, N), bool)
    full = np.zeros((n_slots, N, N), bool)
    feats = np.zeros((n_slots, N, N, 5), np.float32)
    sinr_att = np.zeros((n_slots, A, N, N), np.float32)
    sinr_att_nj = np.zeros((n_slots, A, N, N), np.float32)
    plr_nj = np.zeros((n_slots, N, N), np.float32)
    jam_act = np.zeros(n_slots, bool)
    usable_nj = np.zeros((n_slots, N, N), bool)
    buf_loss_nj = np.zeros((W, N, N), np.float32)
    backoff = rng.random((n_slots, A, N, N), dtype=np.float32)
    deliv_u = rng.random((n_slots, A, N, N), dtype=np.float32)
    eye = np.eye(N, dtype=bool)

    for t in range(n_slots):
        pos = nodes.step(p.slot_s)
        jpos = jam.step(p.slot_s)
        # jammer activity: coherent ON/OFF episodes (spot, pulsed) or a periodic sweep
        jam_timer -= 1
        for q in np.flatnonzero(jam_timer <= 0):
            jam_on[q] = not jam_on[q]
            jam_timer[q] = rng.geometric(1.0 / (on_m if jam_on[q] else off_m))
        if p.jam_mode == "sweep":
            on_now = np.full(p.n_jammers, ((t + sweep_phase) // p.sweep_dwell) % p.n_sweep_ch == 0)
        else:
            on_now = jam_on
        on_now = on_now & (p.jam_dbm is not None)
        active = bool(on_now.any())

        d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
        s_mean = db2lin(p.pt_dbm - pathloss_db(d, p))            # (N,N) signal at j from i
        j_mean = np.zeros(N)
        for q in np.flatnonzero(on_now):
            dj = np.linalg.norm(pos - jpos[q], axis=1)
            j_mean = j_mean + db2lin(p.jam_dbm - pathloss_db(dj, p))  # (N,) interference at node
        # co-channel interference from the network's own hidden terminals: node k can hit link (i, j)
        # only if it is outside carrier-sense range of transmitter i
        hidden = (d > p.cs_range)                                  # hidden[i, k]
        s_k = s_mean.copy(); np.fill_diagonal(s_k, 0.0)            # s_k[k, j] power of k at j
        i_mean = p.own_act * (hidden @ s_k)                        # (N,N) mean own interference on link i->j
        inr_db = lin2db(j_mean / n0)                              # interference-to-noise
        rx_jammed = inr_db >= p.label_margin_db
        lab_now = rx_jammed[:, None] | rx_jammed[None, :]         # either endpoint
        in_range = (d <= p.comm_range) & ~eye

        mean_sinr = s_mean / (n0 + i_mean + j_mean[None, :])
        mean_sinr_sym = np.minimum(mean_sinr, mean_sinr.T)
        if p.label_mode == "impairment":
            lab_now = lab_now & (lin2db(mean_sinr_sym) < p.impair_sinr_db)
        elif p.label_mode == "loss":
            npr = n0 + i_mean                                      # noise plus mean own interference
            snr_db = lin2db(s_mean / npr)
            inr_db = lin2db(j_mean[None, :] / npr)
            eper_j = expected_per_jam(snr_db, inr_db, p)          # (N,N) at receiver j
            eper_0 = expected_per(snr_db, p)
            dper = np.maximum(eper_j - eper_0, (eper_j - eper_0).T)   # either direction of the link
            lab_now = (j_mean.max() > 0) & (dper >= p.impair_dper)
        use = in_range & (lin2db(mean_sinr_sym) >= p.sinr_floor_db)
        usable[t] = use
        snij = s_mean / (n0 + i_mean)
        snr_sym = np.minimum(snij, snij.T)
        usable_nj[t] = in_range & (lin2db(snr_sym) >= p.sinr_floor_db)

        # per-attempt faded SINR (Rayleigh on signal and on jammer)
        g = rng.exponential(size=(A, N, N))
        gj = rng.exponential(size=(A, 1, N))
        act = (rng.random((A, N)) < p.own_act).astype(np.float64)
        gk = rng.exponential(size=(A, N, N))
        own = np.stack([(hidden * act[a][None, :]) @ (s_k * gk[a]) for a in range(A)])   # (A,N,N)
        sinr_lin = s_mean[None] * g / (n0 + own + j_mean[None, None, :] * gj)
        sinr_att[t] = lin2db(sinr_lin).astype(np.float32)
        sinr_nj = s_mean[None] * g / (n0 + own)
        sinr_att_nj[t] = lin2db(sinr_nj).astype(np.float32)
        jam_act[t] = active

        # observations from a hello/probe packet on each link (first attempt draw)
        s1 = sinr_lin[0]
        rssi = lin2db(s_mean * g[0] + own[0] + j_mean[None, :] * gj[0] + n0) + rng.normal(0, p.rssi_noise_db, (N, N))
        sinr_obs = lin2db(s1) + rng.normal(0, p.sinr_noise_db, (N, N))
        u_probe = rng.random((N, N))
        loss = u_probe < per_from_sinr(lin2db(s1), p)
        loss_nj = u_probe < per_from_sinr(lin2db(sinr_nj[0]), p)
        ber = ber_from_sinr(db2lin(sinr_obs))      # estimated from the measured (noisy) SINR

        P = p.probe_every
        if t % P == 0:                        # a probe round: the window advances by one observation
            k = (t // P) % W
            buf_rssi[k], buf_sinr[k], buf_loss[k], buf_ber[k], buf_lab[k] = rssi, sinr_obs, loss, ber, lab_now
            buf_loss_nj[k] = loss_nj
        plr_nj[t] = buf_loss_nj.mean(0)
        age = np.where(in_range, age + 1, 0)
        f = age >= W * P
        full[t] = f
        label[t] = buf_lab.sum(0) * 2 > W
        feats[t, ..., 0] = buf_rssi.mean(0)
        feats[t, ..., 1] = buf_sinr.mean(0)
        feats[t, ..., 2] = buf_loss.mean(0)
        feats[t, ..., 3] = buf_ber.mean(0)
        feats[t, ..., 4] = buf_sinr.std(0)

    return Trace(usable, label, full, feats, sinr_att, sinr_att_nj, plr_nj, jam_act, usable_nj, backoff, flows, deliv_u)


def active_flows(t, p, n_flows):
    """Flows that offer a packet in slot t (one packet per flow every pkt_interval slots, staggered)."""
    return np.flatnonzero((t + np.arange(n_flows)) % p.pkt_interval == 0)


def dataset_from_trace(tr: Trace, stride: int = 5, rng=None):
    """Labelled windows (every link observed for a full window) for detector training."""
    X, y = [], []
    for t in range(0, tr.usable.shape[0], stride):
        m = tr.full[t]                      # every link observed for a full window
        X.append(tr.feats[t][m])
        y.append(tr.label[t][m])
    return np.concatenate(X), np.concatenate(y)
