"""Build the Springer LNNS (llncs) manuscript from results/*.json. Every number in the text is computed here.

python3 build_paper.py   -> paper/manet_antijamming.tex and .pdf (needs pdflatex)
"""
import json, os, re, shutil, subprocess
import numpy as np
from manet_sim import Params, expected_per
from countermeasure import onoff_phi

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
RES, FIG, OUT = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures"), os.path.join(ROOT, "paper")
os.makedirs(OUT, exist_ok=True)
L = lambda n: json.load(open(os.path.join(RES, n)))
dpts = L("detpoints.json")
det, rows, rob, fol, jam, taus = (L("detector.json"), L("routing.json"), L("robust.json"), L("follow.json"),
                                  L("jammers.json"), L("tau.json"))
compact, fixed = L("compact.json"), L("fixedpoint.json")
cause_r, probe_r = L("cause.json"), L("probe.json")
fpga = L("fpga.json") if os.path.exists(os.path.join(RES, "fpga.json")) else None
from run_experiments import THRESH, K_LINKS, PERSIST, VARIANTS
P = Params()
POW = [None, 4, 8, 12, 16, 20, 24]
FOLLOW, TAU, GATE_WIN, EP_PRIOR, EP_GAP, REJOIN = 50, 10, 40, 40, 5, 5
GATE_CONF, ORD_ROUNDS, ORD_DEADLINE = 0.95, 3, 3
REPO_URL = "https://github.com/gladiator138/manet-antijamming"          # public repository, once created
H = FOLLOW + TAU

ML, MLU, LAB, NONE = "ETX+ML-FC+gate (proposed)", "ETX+ML-FC", "ETX+Label-FC", "ETX"
L7, L9, CC = "ETX+Loss-FC(0.7)", "ETX+Loss-FC(0.9)", "ETX+Consistency-FC"
L9G, CCG, MLF, MLQ = "ETX+Loss-FC(0.9)+gate", "ETX+Consistency-FC+gate", "ETX+ML-FC+fixed gate", "ETX+ML-FC+gate, compact"
MLI, MLUI = "ETX+ML-FC+gate, ideal ctrl", "ETX+ML-FC, ideal ctrl"
MLFL, MLMT, MLC, MLCG = "ETX+ML-FC+gate, flood ctrl", "ETX+ML-FC+gate, mean test", "ETX+ML-FC+cause", "ETX+ML-FC+cause+gate"


def m(s, pj, k="pdr"):
    return float(np.mean([r["res"][s][k] for r in rows if r["pj"] == pj]))


def pd(a, b, pj, k="pdr", src=None):
    """Paired difference a - b: mean and 95% half-width (t-based is close to 1.96 sigma/sqrt(n) for n = 10)."""
    src = rows if src is None else src
    v = np.array([r["res"][a][k] - r["res"][b][k] for r in src if r["pj"] == pj])
    return float(v.mean()), float(2.262 * v.std(ddof=1) / np.sqrt(len(v)))    # t(0.975, 9)


def pm(d, digits=1):
    return (f"{100 * d[0]:+.{digits}f}".replace("-", "$-$").replace("+", "$+$")
            + f"\\,$\\pm$\\,{100 * d[1]:.{digits}f}")


def sig(d):
    return abs(d[0]) > d[1]


def nanmean(v):
    v = [x for x in v if x == x]
    return float(np.mean(v)) if v else float("nan")


def rb(v, s, pj, k="pdr"):
    return float(np.mean([r[s][k] for r in rob if r["variant"] == v and r["pj"] == pj]))


def rbd(v, a, b, pj):
    x = np.array([r[a]["pdr"] - r[b]["pdr"] for r in rob if r["variant"] == v and r["pj"] == pj])
    return float(x.mean()), float(2.776 * x.std(ddof=1) / np.sqrt(len(x)))     # t(0.975, 4)


def rdet(v, k):
    return float(np.mean([r["det"][k] for r in rob if r["variant"] == v and r["pj"] is not None]))


def jm(j, s, k="pdr"):
    return float(np.mean([r[s][k] for r in jam if r["jammer"] == j]))


def jd(j, k):
    return float(np.mean([r["det"][k] for r in jam if r["jammer"] == j]))


def jpd(j, a, b):
    x = np.array([r[a]["pdr"] - r[b]["pdr"] for r in jam if r["jammer"] == j])
    return float(x.mean()), float(2.776 * x.std(ddof=1) / np.sqrt(len(x)))


pc = lambda x, d=1: f"{100 * x:.{d}f}\\%"
pt = lambda x, d=1: f"{100 * x:.{d}f}"
sg = lambda x, d=1: f"{100 * x:+.{d}f}".replace("-", "$-$").replace("+", "$+$")
V = {}
rf, lr, th, cc, cost = det["rf"], det["lr"], det["thr"], det["cc"], det["cost"]
p0 = m(NONE, None)

# ---------------------------------------------------------------- Proposition 2 validation (on/off jammer)
phi = onoff_phi(H, P.jam_on_mean, P.jam_off_mean)
pstar = p0 * (1 - TAU / (H * phi))
pstar_pers = p0 * FOLLOW / H
p_on = {pj: m("ETX-net", pj, "pdr_on") for pj in POW[1:]}
g_ideal = {pj: pd(MLUI, NONE, pj) for pj in POW[1:]}
pred = {pj: p_on[pj] < pstar for pj in POW[1:]}
pred_pers = {pj: p_on[pj] < pstar_pers for pj in POW[1:]}
agree = sum((g_ideal[pj][0] > 0) == pred[pj] for pj in POW[1:])
agree_pers = sum((g_ideal[pj][0] > 0) == pred_pers[pj] for pj in POW[1:])
amb = [pj for pj in POW[1:] if not sig(g_ideal[pj])]

# rerouting
rr = {}
for base, meth in (("HOP", "HOP+ML-reroute"), ("HOP", "HOP+Label-reroute"), ("ETX", "ETX+ML-reroute"), ("ETX", "ETX+Label-reroute")):
    rr[meth] = {pj: pd(meth, base, pj) for pj in POW}
rr_max = max(abs(d[0]) for v in rr.values() for d in v.values())
rr_sig_pos = [(k, pj) for k, v in rr.items() for pj, d in v.items() if d[0] - d[1] > 0]
rr_sig_neg = [(k, pj) for k, v in rr.items() for pj, d in v.items() if d[0] + d[1] < 0]
etx_gain = [(m("ETX", pj) - m("HOP", pj)) / m("HOP", pj) for pj in POW]
div = {pj: np.mean([r["div"]["alt_given_jammed"] for r in rows if r["pj"] == pj]) for pj in POW[1:]}
basej = {pj: np.mean([r["div"]["base_jammed"] for r in rows if r["pj"] == pj]) for pj in POW[1:]}
q = {s: 1 - expected_per(s, P) ** P.max_attempts for s in (8, 10)}
air = m(NONE, None, "airtime")
probe_air = P.n_nodes * P.probe_bytes * 8 / P.rate_bps / P.slot_s
offered = P.n_flows * P.payload_bytes * 8 / (P.pkt_interval * P.slot_s) / 1e6
hi = [pj for pj in POW if pj and pj >= 16]
trig_delay = nanmean([r["res"][ML]["trig_delay_slots"] for r in rows if r["pj"] in hi])
esc = sum(r["res"][ML]["escaped"] for r in rows if r["pj"] in hi) / max(1, sum(r["res"][ML]["exposures"] for r in rows if r["pj"] in hi))
fs = sorted({r["follow"] for r in fol})
fo = {k: {f: float(np.mean([r[k] for r in fol if r["follow"] == f])) for f in fs}
      for k in ("none", "ml", "mlgate", "mlgate50", "label")}
ms = lambda f: int(f * P.slot_s * 1e3)
VARS = list(dict.fromkeys(r["variant"] for r in rob))
JAMS = list(dict.fromkeys(r["jammer"] for r in jam))
full = [r for r in compact if r["trees"] == 120 and r["depth"] == 12][0]
c10 = [r for r in compact if r["trees"] == 10 and r["depth"] == 12][0]
f16 = [r for r in fixed if r["bits"] == 16 and r["trees"] == 10][0]
f8 = [r for r in fixed if r["bits"] == 8 and r["trees"] == 10][0]
TS = sorted({r["tau"] for r in taus})
to = {k: {t: float(np.mean([r[k] for r in taus if r["tau"] == t])) for t in TS} for k in ("none", "ml", "mlgate")}

# ---------------------------------------------------------------- text
T = r"""\documentclass[runningheads]{llncs}
\usepackage{graphicx,amsmath,amssymb,booktabs,subcaption}
\usepackage[hidelinks]{hyperref}
\captionsetup[subfigure]{font=footnotesize}
\emergencystretch=1.5em
\renewcommand{\topfraction}{0.9}\renewcommand{\textfraction}{0.08}\renewcommand{\floatpagefraction}{0.75}
\setcounter{topnumber}{3}\setcounter{totalnumber}{4}
\begin{document}
\title{Cause- and Severity-Aware Frequency Change against Jamming in Tactical MANETs}
\titlerunning{Cause- and Severity-Aware Anti-Jamming for Tactical MANETs}
\author{Sahil Singh\inst{1} \and Abhishek Madhukar\inst{1} \and Siddharth Sangwan\inst{1}\thanks{Corresponding author.}}
\authorrunning{S. Singh et al.}
\institute{Faculty of Communication Engineering, Military College of Telecommunication Engineering (MCTE), Mhow, India\\
\email{siddharthsangwan99@gmail.com}}
\maketitle
\begin{abstract}
Jamming detectors for mobile ad hoc networks (MANETs) are usually evaluated by accuracy, not by what the network does
with their output. In a paired multi-seed simulation of a 40-node AES-protected tactical MANET with hidden-terminal
interference, we show that routing around detected jamming, even with perfect knowledge of the jammed links, changes the
packet-delivery ratio (PDR) by at most <<rr_max>> percentage points relative to hop-count or ETX routing. Frequency change
does escape the jammer, but it silences every radio while it retunes and it pays only when jamming is both the cause of
the loss and severe enough. We derive the break-even condition for an on/off jammer from its mean on and off times, the
retune time and the jammer's follow time, and turn it into a detector-agnostic gate that estimates the jammer's episode
statistics online and tests the severity sequentially. With orders flooded hop by hop over the jammed network, a
Random-Forest detector (ROC-AUC <<auc>>, PR-AUC <<prauc>>) wrapped in the gate raises PDR by <<gain24_ml>> points over ETX
under strong jamming and never loses more than <<worst_any>> points, in seven unseen deployments and against pulsed and
sweeping jammers, where ungated triggers lose up to <<ungated_sweep>> points; a plain loss threshold in the same gate does
almost as well. Orders are authenticated with AES-GCM, and the 16-bit detector synthesised for an Artix-7 FPGA uses
<<fpga_short>>.
\keywords{Tactical MANET \and Jamming \and Frequency agility \and Decision rule \and Random Forest \and FPGA}
\end{abstract}

\section{Introduction}
Tactical mobile ad hoc networks (MANETs) carry command-and-control traffic over multi-hop radio links under high
mobility and against an adversary who can jam \cite{pirayesh}. Classical defences spread the signal \cite{pickholtz};
adaptive defences react to detected jamming by routing around it \cite{lou,lee,sheikh},
changing channel \cite{surfing,navda} or learning a channel policy \cite{liu18,xudrl}. Machine-learning (ML) detectors
of jamming are now accurate \cite{xu05,punal,jamshield,bessouf,viana,ghelani}, but they are usually judged by detection
accuracy rather than by the decision they drive.

We ask what a detector should drive. Routing around flagged links does not help, as the detours are weak and a
link-quality metric (ETX \cite{etx}) already avoids bad links. Frequency change escapes the jammer but silences every
radio while they retune, so a change against a weak, short or sweeping jammer loses more than it gains.
The contributions are:
\begin{itemize}
\item a break-even condition for network-wide frequency change against an on/off jammer, in terms of its mean on and off
times, the retune time and the jammer's follow time, which reduces to a simple ratio for a persistent jammer
(Proposition~\ref{prop:fc});
\item a detector-agnostic severity gate that evaluates this condition online, as a sequential Bayesian test on the
delivered traffic with jammer statistics estimated from the alarm history; ablations show that it, rather than the
detector, provides the gain and the robustness;
\item a paired multi-seed evaluation with hidden-terminal interference and a control plane that floods orders hop by hop
over the jammed network, against loss-threshold, signal-consistency \cite{xu05} and ML triggers and an ML jammer-type
classifier, including a negative result for detection-driven rerouting;
\item authenticated orders and a 16-bit FPGA detector with the same network-level results as the float model.
\end{itemize}

\section{Related Work}
\textbf{Jamming detection.} Xu et al.\ showed that neither signal strength nor delivery ratio alone separates jamming
from ordinary degradation and proposed checking their consistency \cite{xu05}. Supervised classifiers \cite{punal},
over-the-air detection \cite{jamshield}, UAV and FANET detectors \cite{viana,ghelani} and lightweight Random Forests on
an edge--fog--cloud hierarchy \cite{bessouf} followed; these works stop at detection.

\textbf{Countermeasures.} Spread spectrum \cite{pickholtz} and uncoordinated frequency hopping, which avoids a shared
secret hopping pattern \cite{strasser}, resist jamming without detection, at a cost in bandwidth. Jamming-aware routing
weights links by received jamming power \cite{sheikh} or adds route recovery \cite{lee,lou}. Channel surfing
\cite{surfing} and channel hopping \cite{navda} escape the jammer in frequency, and deep reinforcement learning learns
anti-jamming channel and rate selection \cite{liu18,xudrl,song}, usually for a single link and without the cost of
coordinating a whole network, which our gate models explicitly.

\section{System Model}
\textbf{Network and traffic.} $N=<<N>>$ nodes move in a <<area>>\,m$\times$<<area>>\,m area by the random-waypoint
model \cite{camp} at <<vmin>>--<<vmax>>\,m/s with zero pause; the first <<warm>>\,s are discarded so that the speed and
density transient of the model \cite{yoon} has passed. Time is slotted ($\Delta t=<<slot_ms>>$\,ms). <<flows>> unicast
flows between random node pairs each offer one <<payload>>-byte packet every <<pkt_int>> slots, staggered in time
(<<offered>>\,Mbit/s over an <<rate>>\,Mbit/s radio). The MAC makes up to <<att>> attempts per hop with binary-exponential
backoff. Data occupy <<air>> of the channel time and per-slot probes of <<probe_b>> bytes per node a further
<<probe_air>>, before MAC overheads that could roughly double both; the high probe rate (which in a radio would ride on
data or beacons) lets a window fill within tens of milliseconds; Sect.~\ref{sec:fol} varies it.

\textbf{Channel.} With transmit power $P_t=<<pt>>$\,dBm, path loss $PL(d)=PL_0+10\alpha\log_{10}\max(d,1\,\mathrm{m})$
($PL_0=<<pl0>>$\,dB, $\alpha=<<alpha>>$) and independent Rayleigh gains $g\sim\mathrm{Exp}(1)$, drawn afresh for every
MAC attempt, the SINR of link $(i,j)$ is
\begin{equation}
\gamma_{ij}=\frac{P_t\,10^{-PL(d_{ij})/10}\,g_{ij}}{N_0+\sum_{k\in\mathcal{H}_i\setminus\{j\}}a_k P_t\,10^{-PL(d_{kj})/10}g_{kj}
+\sum_{J}P_J\,10^{-PL(d_{Jj})/10}g_{Jj}},
\end{equation}
with powers in mW, $N_0=<<n0>>$\,dBm, $\mathcal{H}_i$ the nodes outside the carrier-sense range (<<cs>>\,m) of
transmitter $i$ (hidden terminals) and $a_k\sim\mathrm{Bernoulli}(\rho)$, $\rho=<<rho>>$, marking a concurrent
transmission. A carrier-sense range equal to the communication range is pessimistic, as it maximises hidden terminals;
Sect.~\ref{sec:rob} also tests <<cs2>>\,m. A packet is lost with probability
$\mathrm{PER}(\gamma)=1/(1+\exp[k(\gamma_{\mathrm{dB}}-\gamma_{50})])$ of the instantaneous SINR ($\gamma_{50}=<<g50>>$\,dB,
$k=<<pk>>$\,dB$^{-1}$). This smooth waterfall is a modelling assumption; Sect.~\ref{sec:rob} also tests $k=<<pk2>>$, closer
to a coded PHY. A hop succeeds with probability $q=1-\mathrm{PER}_1\mathrm{PER}_2$ over two independently faded attempts.
A link is usable within <<range>>\,m if its mean SINR is at least <<floor>>\,dB.

\textbf{Jammer.} The default adversary is a spot jammer on the network's channel that roams the central
<<jreg>>\,m$\times$<<jreg>>\,m square at <<jv0>>--<<jv1>>\,m/s and radiates in on/off episodes of geometric length (mean
<<jon>> slots each), i.e.\ a two-state Markov chain. After a frequency change it needs a follow time $F$ (default <<F>>
slots, counted from when the network resumes) to find the new channel. Pulsed, sweeping and multiple jammers are used
only for testing (Sect.~\ref{sec:jam}) and to train the optional cause classifier.

\textbf{Security model.} Payloads are protected with AES-128-GCM \cite{gcm} under a pre-shared group key (nonce and tag
add <<aes_over>> of airtime). The adversary can transmit arbitrary frames but does not hold the key; it could otherwise
forge orders or alarms (Sect.~\ref{sec:auth}).

\section{Cause- and Severity-Aware Frequency Change}
\begin{figure}[t]
\centering\includegraphics[width=\textwidth]{fig1_architecture.pdf}
\caption{Decision loop; any detector can supply the alarms.}\label{fig:arch}
\end{figure}
Fig.~\ref{fig:arch} shows the decision loop. \textbf{Features and label.} Each node keeps the last $W=<<W>>$ probe
observations per neighbour link: mean RSSI (total received power), mean measured SINR, packet-loss rate (PLR), mean bit
error rate estimated from the measured SINR, and SINR standard deviation, with <<rssi_n>>\,dB and <<sinr_n>>\,dB
measurement noise on RSSI and SINR. A slot is labelled jammed when the jammer raises the expected packet-error rate of
the link, averaged over the Rayleigh fading of both the signal and the jammer, by at least $\delta=<<delta>>$ in either
direction, and a window is labelled by majority:
\begin{equation}
y_{ij}(t)=\mathbf{1}\Big\{\sum_{s=t-W+1}^{t}\mathbf{1}\{\mathrm{E}_{g,g_J}[\mathrm{PER}(\gamma^{J}_{ij}(s))]-\mathrm{E}_{g}[\mathrm{PER}(\gamma^{0}_{ij}(s))]\ge\delta\}>\tfrac{W}{2}\Big\},
\end{equation}
where $\gamma^{J}$ and $\gamma^{0}$ are the SINR with and without the jammer, with own interference at its mean.

\textbf{Detectors and trigger.} The ML detector is a Random Forest \cite{breiman} of 120 trees of depth 12, trained offline
and deployed unchanged; a link alarms when its jamming probability is at least $\theta=<<theta>>$. We also use a loss
threshold (window PLR $\ge 0.7$ or $0.9$) and the consistency check of Xu et al.\ \cite{xu05} (PLR $\ge<<cc_plr>>$ while
RSSI $\ge<<cc_rssi>>$\,dBm, fitted for maximum F1 on the training set). With any detector, a change is considered when at
least $k=<<k>>$ links alarm in <<persist>> consecutive slots; $\theta$ and the persistence were chosen on separate design
seeds. A change silences the net for $\tau=<<tau>>$ slots, during which alarms are ignored. All other loss is left to
ETX routing \cite{etx}, computed from the window PLR.

\textbf{Severity gate.} A change is ordered only if the network's PDR while jammed, $p_J$, is below the break-even
point $p^\ast$ of Proposition~\ref{prop:fc}. Because only about one packet per slot is offered, $p_J$ is estimated from
few packets, and a plain average misfires near $p^\ast$. The gate therefore runs a sequential Bayesian test: with a uniform
prior and the $s$ delivered and $f$ lost packets since the current alarm episode began (at most <<gw>> slots), it changes
when $P(p_J<p^\ast\mid s,f)=I_{p^\ast}(1+s,1+f)\ge<<conf>>$, where $I$ is the regularised incomplete beta function. Severe
jamming passes after a few lost packets; jamming near the break-even point is left to ETX. $p^\ast$ is evaluated with a
running average $\hat p_0$ of the delivery ratio in slots without an alarm consensus, the PDR $\hat p_1$ measured in the
$F$ slots after past changes (prior $\hat p_0$, weight 20 packets), a planning value of $F$, and the jammer's
mean on and off times $\hat L_{\mathrm{on}},\hat L_{\mathrm{off}}$. These are estimated online from the alarm-consensus
history (an episode ends after <<gap>> quiet slots) by censored maximum likelihood, total on (off) time divided by the
number of completed on (off) episodes, where an episode cut short by a change is censored; one pseudo-episode of
<<prior>>\,slots each is the prior. Confidence, window and prior were chosen on the design seeds.

\textbf{Cause classifier (optional).} To test whether ML can also recognise the \emph{type} of jammer, a second Random
Forest classifies the last 200 slots of the largest per-component alarm count (consensus share, run and gap lengths,
autocorrelation; 11 features) as no jammer, spot, pulsed, sweeping or two spot jammers, and a change is allowed only for
the spot types. Unlike the gate, it must be trained on every jammer type.

\textbf{Control plane.} Alarms and orders travel over the jammed channel. Each connected component of the currently
usable graph aggregates the alarms of its own links, heard one slot late, and the node with most alarms orders a change
when it hears $k$ of them. The order names a switch slot <<od>> slots ahead and is flooded hop by hop over the jammed
links, <<orr>> relay rounds per slot, each relay broadcasting twice, with reception drawn from the faded SINR. Nodes that
the flood has not reached by the switch slot, including all nodes of other components, stay behind, notice that their
neighbours have gone silent, and step to the next epoch's channel on their own after <<rejoin>> further slots, isolated
meanwhile.

\textbf{Authenticated orders.}\label{sec:auth} An order carries the net identifier, an increasing epoch $e$, the sender
and the switch slot, sealed with AES-GCM (header as associated data, nonce = node identifier and local counter so that
nonces never repeat under the shared key \cite{gcm}). The channel is never sent on air: each radio derives
$c_e=\mathrm{AES}_{K_c}(e)\bmod C$ from a channel key obtained by HKDF \cite{hkdf}, which also lets a stranded node find
the next channel. Radios accept only orders whose tag verifies and whose epoch is new, which rejects replays. Alarms are
sealed the same way; a group key authenticates membership, not the sender, so a compromised insider needs per-node
signatures and a rate limit on orders.

\textbf{Analysis.} A detour of $m$ clean hops of success $q_c$ beats one jammed hop of success $q_J$ only if
$q_c^{\,m}>q_J$; with our PHY two 10\,dB hops (<<q10sq>>) are no better than one 8\,dB hop (<<q8>>).

\begin{proposition}\label{prop:fc}
Let the jammer be a two-state Markov chain with mean on and off times $L_{\mathrm{on}},L_{\mathrm{off}}$, so that
$\pi=L_{\mathrm{on}}/(L_{\mathrm{on}}+L_{\mathrm{off}})$ and $\lambda=1-1/L_{\mathrm{on}}-1/L_{\mathrm{off}}$. Let $p_0$ be the
PDR when the jammer is off or escaped and $p_J$ the PDR while it is on. A change costs $\tau$ silent slots, whose offered
packets are lost, and the jammer finds the new channel after $F$ slots. A change ordered while the jammer is on raises the
expected number of delivered packets if and only if
\begin{equation}
p_J<p^\ast=p_0\Big(1-\frac{\tau}{H\,\varphi(H)}\Big),\qquad H=F+\tau,\quad
\varphi(H)=\pi+(1-\pi)\frac{\lambda(1-\lambda^{H})}{H(1-\lambda)} .
\end{equation}
For a persistent jammer ($\varphi=1$) this is $p_J<p_0F/(F+\tau)$, i.e.\ $F/\tau>p_J/(p_0-p_J)$.
\end{proposition}
\begin{proof}
A jammer that is on now is on $s$ slots later with probability $\pi+(1-\pi)\lambda^{s}$, so a fraction
$\varphi(H)$ of the next $H$ slots is jammed in expectation. Without a change the network delivers
$H[p_0-(p_0-p_J)\varphi(H)]$ per offered packet-slot over these slots, with a change $Fp_0$. Afterwards the jammer is on
the network's channel in either case, in the same state distribution since its on/off process does not depend on the
network. The detection delay and the hold-off precede the decision and are common to both cases.\qed
\end{proof}
If the PDR after a change is $p_1\ne p_0$, e.g.\ because some nodes miss the order, $Fp_0$ becomes $Fp_1$ and the
condition becomes $p_J<[Fp_1-H(1-\varphi)p_0]/(H\varphi)$; the gate uses this form with $\hat p_1$.
For short bursts ($H\varphi\le\tau$) no change can pay, however severe the jamming. Likewise, $r$ false changes per
second, forged or not, remove a fraction $r\tau\Delta t$ of airtime.

\section{Simulation Setup}
The Python simulator (NumPy, SciPy, scikit-learn \cite{sklearn}) replays every scheme on the same channel trace per
seed, so differences between schemes are paired. The detector is trained on <<ntrain>> windows from 14
scenarios and tested on <<ntest>> windows from 7 scenarios with held-out seeds of the same generator, each scenario with
one of the jammer powers in Table~\ref{tab:param}. Trigger parameters were chosen on three further design seeds. Network
results use 10 other seeds of 1500 slots per jammer power; differences between schemes are given as paired 95\%
$t$-intervals across seeds and called significant when the interval excludes zero.

\begin{table}[t]
\caption{Simulation parameters.}\label{tab:param}
\centering\scriptsize\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}ll@{\hspace{1em}}ll@{}}
\toprule
Nodes, area & <<N>>, <<area>>\,m $\times$ <<area>>\,m & Radio rate, slot & <<rate>>\,Mbit/s, <<slot_ms>>\,ms \\
Mobility & RWP <<vmin>>--<<vmax>>\,m/s, <<warm>>\,s warm-up & Traffic & <<flows>> flows, <<payload>>\,B/<<pkt_int>> slots \\
$P_t$, range, $N_0$ & <<pt>>\,dBm, <<range>>\,m, $<<n0>>$\,dBm & Hidden term. & $\rho=<<rho>>$, CS <<cs>>\,m \\
Path loss & <<pl0>>\,dB at 1\,m, $\alpha=<<alpha>>$, Rayleigh & PER, MAC & $\gamma_{50}=<<g50>>$\,dB, $k=<<pk>>$; <<att>> attempts \\
Jammer & 4--24\,dBm, on/off <<jon>>/<<jon>> slots & Follow time & $F=<<F>>$ slots \\
Detector & RF 120 trees, depth 12, $W=<<W>>$ & Trigger & $\theta=<<theta>>$, $k=<<k>>$, <<persist>> slots, $\tau=<<tau>>$ \\
Encryption & AES-128-GCM, +<<aes_b>>\,B & Gate & conf.\ <<conf>>, $\le$<<gw>>, prior <<prior_slots>> slots \\
\bottomrule
\end{tabular}
\end{table}

\section{Results}
\subsection{Detection}
In Table~\ref{tab:det}, <<prior>> of the held-out windows are jammed, so a detector that never alarms would reach
<<allneg>> accuracy; ROC-AUC, PR-AUC and balanced accuracy are the meaningful measures. The Random Forest has the highest ROC-AUC (<<auc>>) and PR-AUC (<<prauc>>), ahead of logistic regression
(<<auc_lr>>, <<prauc_lr>>) and far ahead of the best single-feature threshold (<<thr_feat>>, <<auc_thr>>). At the
operating point $\theta=<<theta>>$ it trades recall (<<rec>>) for precision (<<prec>>), because a false alarm consensus
silences every radio while a missed window costs little as long as $k$ jammed links are flagged. At 16--24\,dBm the proposed scheme changes on average <<trig_delay>> slots after the jammer reaches the net and
escapes <<esc>> of the jamming episodes.

\begin{table}[t]
\caption{Detectors on held-out test windows (<<prior>> jammed). All values in \% except F1 and the AUCs. The consistency
check is a binary rule and has no score to rank, hence no AUC.}\label{tab:det}
\centering\scriptsize\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{@{}lccccccc@{}}
\toprule
Detector & Acc. & Bal.\ acc. & Prec. & Recall & F1 & ROC-AUC & PR-AUC \\
\midrule
<<det_rows>>
\bottomrule
\end{tabular}
\end{table}

\subsection{Rerouting Around Detected Jamming}\label{sec:reroute}
Without a jammer, min-hop routing delivers <<hop0>> of packets and ETX <<etx0>>. Fig.~\ref{fig:pdr}(a) shows the paired
PDR change when flagged links are penalised (cost 10) in either metric, with the Random-Forest flags or with the true
labels. No variant changes PDR by more than <<rr_max>> points at any power; <<rr_sig_text>> At 8\,dBm a jam-free alternative exists for only <<div8>> of the flow-slots whose min-hop path is
jammed, and at 16\,dBm and above <<div16>>.

\begin{figure}[!t]
\centering
\begin{subfigure}[t]{0.49\textwidth}\centering\includegraphics[width=\linewidth]{fig5_pdr_reroute.pdf}\caption{PDR gain of rerouting over its base metric}\end{subfigure}\hfill
\begin{subfigure}[t]{0.49\textwidth}\centering\includegraphics[width=\linewidth]{fig6_pdr_fc.pdf}\caption{PDR with frequency change on top of ETX}\end{subfigure}\\[2pt]
\begin{subfigure}[t]{0.49\textwidth}\centering\includegraphics[width=\linewidth]{fig7_switch_rate.pdf}\caption{Frequency changes per second (log scale)}\end{subfigure}\hfill
\begin{subfigure}[t]{0.49\textwidth}\centering\includegraphics[width=\linewidth]{fig8_follow.pdf}\caption{PDR versus follow time $F$ at 16\,dBm}\end{subfigure}
\caption{Results versus jammer power and follow time (10 seeds; bars: 95\% intervals, paired in (a)). The y-axes of (b)
and (d) do not start at zero.}\label{fig:pdr}
\end{figure}

\subsection{Frequency Change and the Break-Even Condition}\label{sec:fc}
Table~\ref{tab:theory} tests Proposition~\ref{prop:fc}. With $F=<<F>>$, $\tau=<<tau>>$ and the jammer's
<<jon>>/<<jon>>-slot episodes, $\varphi(H)=<<phi>>$ and the measured $p_0=<<p0>>$ give $p^\ast=<<pstar>>$. We measured
$p_J$ as the ETX PDR in slots with the jammer on and compared the prediction with the paired gain of the ungated ML
trigger under an ideal control plane, as the proposition assumes. The prediction agrees
in sign at <<agree>> of 6 powers<<amb_text>>. The persistent-jammer condition $p_J<p_0F/(F+\tau)=<<pstar_pers>>$ agrees at
only <<agree_pers>> of 6, because it ignores that the jammer may switch off by itself.

\begin{table}[t]
\caption{Break-even test of Proposition~\ref{prop:fc}: measured PDR while the jammer is on, predicted sign of the gain,
and simulated paired gain of an ungated change (ideal control plane) over ETX, in points.}\label{tab:theory}
\centering\scriptsize\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{@{}lcccccc@{}}
\toprule
Jammer power (dBm) & 4 & 8 & 12 & 16 & 20 & 24 \\
\midrule
<<theory_rows>>
\bottomrule
\end{tabular}
\end{table}

Fig.~\ref{fig:pdr}(b) and (c) compare the triggers with the lossy control plane. Without a jammer the proposed scheme
delivers <<ml0>> (ETX: <<etx0>>) and orders <<ml0_sw>> changes per second, while the ungated loss trigger at 0.7 orders
<<l70_sw>> per second and loses <<l70_loss>> points. Under strong jamming the proposed scheme raises PDR from <<etx24>> to
<<ml24>> at 24\,dBm (paired gain <<g24>>) and from <<etx16>> to <<ml16>> at 16\,dBm (<<g16>>). Its largest loss at any
power is <<worst_gate>> points (ungated: <<worst_ungated>>; ungated with true labels: <<worst_label>>), at a cost against
the ungated trigger of <<gate_cost>> points at 24\,dBm: the sequential test waits for evidence. A plain average of the last
20 slots instead (as in a fixed-window gate) loses up to <<worst_mean>> points. The gate is what matters, not the
detector: the loss threshold at 0.9 with the same gate reaches <<l9g24>> at 24\,dBm and <<l9g16>> at 16\,dBm (difference to
the proposed scheme <<l9g_vs_ml24>> and <<l9g_vs_ml16>> points). The flood reaches <<reach16>> of the nodes before the
switch slot at 16\,dBm, so <<strand16>> of the nodes are stranded per change (<<strand24>> at 24\,dBm). The control plane
costs <<cp_cost24>> points at 24\,dBm: with flooding that is instant but limited to connected nodes the proposed scheme
would reach <<flood24>>, and with an ideal control plane <<ideal24>>. The Random Forest quantised to 10 trees and 16 bits
(Sect.~\ref{sec:hw}) gives PDRs within <<q_maxdiff>> points of the full model at every power.

\subsection{Robustness to Other Deployments}\label{sec:rob}
Table~\ref{tab:rob} reuses the detector and all thresholds without retraining in seven deployments not used in design
(5 new seeds of 1200 slots each). <<rob_text>>

\begin{table}[t]
\caption{Deployments not used in design (5 seeds each). FC/s: changes per second without a jammer. L0.9+G: loss trigger
0.9 with the gate. Prop.: ML with the gate. PDR in \%.}\label{tab:rob}
\centering\scriptsize\setlength{\tabcolsep}{2.4pt}
\begin{tabular}{@{}lccccccccccc@{}}
\toprule
 & Det. & \multicolumn{5}{c}{FC/s without jammer} & \multicolumn{2}{c}{PDR, no jammer} & \multicolumn{3}{c}{PDR at 16\,dBm} \\
\cmidrule(lr){3-7}\cmidrule(lr){8-9}\cmidrule(l){10-12}
Condition & acc.\,\% & L0.7 & Cons. & ML & L0.9+G & Prop. & ETX & Prop. & ETX & L0.9+G & Prop. \\
\midrule
<<rob_rows>>
\bottomrule
\end{tabular}
\end{table}

\subsection{Other Jammer Types}\label{sec:jam}
Table~\ref{tab:jam} uses jammers not seen by the detector or the gate, at 16\,dBm: pulsed (mean <<pon>>-slot bursts, <<pduty>>\% duty),
sweeping <<nsw>> channels with a <<sdw>>-slot dwell, so that it also hits every new channel ($F=0$), and two spot jammers. <<jam_text>>

\begin{table}[t]
\caption{Jammer types at 16\,dBm (5 seeds each). Recall and precision of the Random Forest (\%; recall is lower than in
Table~\ref{tab:det}, which pools all powers), estimated mean on time $\hat L_{\mathrm{on}}$ (slots), and PDR (\%). Fixed G:
persistent-jammer gate. The cause classifier was trained on all four types (other seeds).}\label{tab:jam}
\centering\scriptsize\setlength{\tabcolsep}{2.3pt}
\begin{tabular}{@{}lcccccccccc@{}}
\toprule
 & & & & & No gate & \multicolumn{3}{c}{With gate} & \multicolumn{2}{c}{Cause classifier} \\
\cmidrule(lr){7-9}\cmidrule(l){10-11}
Jammer & Rec. & Prec. & $\hat L_{\mathrm{on}}$ & ETX & ML & Loss 0.9 & ML, fixed G & ML (prop.) & ML & ML + G \\
\midrule
<<jam_rows>>
\bottomrule
\end{tabular}
\end{table}

\subsection{Follow Time, Retune Time and Probe Rate}\label{sec:fol}
A faster jammer shortens the clean interval after each change (Fig.~\ref{fig:pdr}(d)). Without the gate, the ML-triggered
change beats no change (<<fnone>>) only for $F\ge<<fbreak>>$\,ms, and at $F=<<fmin>>$\,ms it falls to <<fml_min>>. With the
true $F$ the gate limits the loss at $F=<<fmin>>$\,ms to <<fg_loss>> points. Varying the retune time $\tau$ from <<tmin>> to <<tmax>> slots at 16\,dBm, <<tau_text>> <<probe_text>>

\subsection{Hardware Cost}\label{sec:hw}
Random forests map well to FPGAs \cite{vanessen}. The full forest has <<nodes_full>> nodes; 10 trees of depth 12 reach F1 = <<f1_c10>> (120 trees:
<<f1_full>>), and 16-bit features and thresholds leave F1 at <<f16_fixed>> (8 bits: <<f8_fixed>>). <<fpga_text>> Feature extraction (window sums, a square root and a dB conversion) is not included.


\subsection{Limitations}
Hidden-terminal interference is drawn independently of the routed traffic, collisions are not simulated per packet,
routing uses instantaneous link state, and alarm reports are assumed to reach the deciding node within a slot. Labels,
features and jammers come from parametric models of one generator, the gate needs a planning value of $F$, and its
confidence trades the gain under strong jamming against safety near the break-even point.

\section{Conclusion}
Rerouting around flagged links, even with perfect labels, changed PDR by less than <<rr_max_int>> points. For frequency
change, the decisive component is not the detector but a severity gate derived from an on/off break-even condition and
applied as a sequential test. Wrapped in the gate, an ML detector and a plain loss threshold both raised PDR by about
<<gain24_int>> points under strong jamming while never losing more than about a point in any tested deployment or against
pulsed and sweeping jammers, even with orders flooded over the jammed network. Recognising the jammer type with ML adds
little once the gate is in place. Future work: reactive jammers and over-the-air validation.

\begin{credits}
\subsubsection{\discintname} The authors have no competing interests to declare that are relevant to the content of
this article.

\subsubsection{Code availability.} The simulator and the scripts that generate every number, table and figure are
<<repo>>.
\end{credits}

<<bibliography>>
\end{document}
"""

REFS = {
    "pirayesh": r"Pirayesh, H., Zeng, H.: Jamming attacks and anti-jamming strategies in wireless networks: a comprehensive survey. IEEE Commun. Surv. Tutor. \textbf{24}(2), 767--809 (2022). \doi{10.1109/COMST.2022.3159185}",
    "pickholtz": r"Pickholtz, R., Schilling, D., Milstein, L.: Theory of spread-spectrum communications -- a tutorial. IEEE Trans. Commun. \textbf{30}(5), 855--884 (1982). \doi{10.1109/TCOM.1982.1095533}",
    "strasser": r"Strasser, M., P\"opper, C., \v{C}apkun, S., \v{C}agalj, M.: Jamming-resistant key establishment using uncoordinated frequency hopping. In: 2008 IEEE Symposium on Security and Privacy, pp. 64--78 (2008). \doi{10.1109/SP.2008.9}",
    "liu18": r"Liu, X., Xu, Y., Jia, L., Wu, Q., Anpalagan, A.: Anti-jamming communications using spectrum waterfall: a deep reinforcement learning approach. IEEE Commun. Lett. \textbf{22}(5), 998--1001 (2018). \doi{10.1109/LCOMM.2018.2815018}",
    "yoon": r"Yoon, J., Liu, M., Noble, B.: Random waypoint considered harmful. In: Proc. IEEE INFOCOM 2003, vol.~2, pp. 1312--1321 (2003). \doi{10.1109/INFCOM.2003.1208967}",
    "lou": r"Lou, L., Fan, J.H.: A new anti-jamming reliable routing protocol for tactical MANETs. In: 2017 First International Conference on Electronics Instrumentation \& Information Systems (EIIS). IEEE (2017). \doi{10.1109/EIIS.2017.8298766}",
    "xu05": r"Xu, W., Trappe, W., Zhang, Y., Wood, T.: The feasibility of launching and detecting jamming attacks in wireless networks. In: Proc. 6th ACM MobiHoc, pp. 46--57 (2005). \doi{10.1145/1062689.1062697}",
    "punal": r"Pu\~nal, O., Akta\c{s}, I., Schnelke, C.J., Abidin, G., Wehrle, K., Gross, J.: Machine learning-based jamming detection for IEEE 802.11: design and experimental evaluation. In: Proc. IEEE WoWMoM 2014, pp. 1--10 (2014). \doi{10.1109/WoWMoM.2014.6918964}",
    "jamshield": r"Panitsas, I., Yigit, Y., Tassiulas, L., Maglaras, L., Canberk, B.: JamShield: a machine learning detection system for over-the-air jamming attacks. In: Proc. IEEE ICC 2025, pp. 1067--1072 (2025). \doi{10.1109/ICC52391.2025.11161395}",
    "bessouf": r"Bessouf, H., Baadache, A., Semchedine, F.: Leveraging lightweight machine learning for RF jamming detection in mobile ad-hoc networks: a three-tier edge-fog-cloud computing approach. Cybern. Inf. Technol. \textbf{26}(1), 72--92 (2026). \doi{10.2478/cait-2026-0005}",
    "viana": r"Viana, J., Farkhari, H., Campos, L.M., Sebasti\~ao, P., Cercas, F., Bernardo, L., Dinis, R.: Two methods for jamming identification in UAV networks using new synthetic dataset. In: 2022 IEEE 95th Vehicular Technology Conference (VTC2022-Spring), pp. 1--6. IEEE (2022). \doi{10.1109/VTC2022-Spring54318.2022.9860816}",
    "ghelani": r"Ghelani, J., Gharia, P., El-Ocla, H.: Gradient monitored reinforcement learning for jamming attack detection in FANETs. IEEE Access \textbf{12}, 23081--23095 (2024). \doi{10.1109/ACCESS.2024.3361945}",
    "lee": r"Lee, J.J., Lee, J., Lim, J.: Jamming-aware routing in ad hoc networks. IEICE Trans. Commun. \textbf{E95-B}(1), 293--295 (2012). \doi{10.1587/transcom.E95.B.293}",
    "sheikh": r"Sheikholeslami, A., Ghaderi, M., Pishro-Nik, H., Goeckel, D.: Energy-efficient routing in wireless networks in the presence of jamming. IEEE Trans. Wirel. Commun. \textbf{15}(10), 6828--6842 (2016). \doi{10.1109/TWC.2016.2591016}",
    "etx": r"De Couto, D.S.J., Aguayo, D., Bicket, J., Morris, R.: A high-throughput path metric for multi-hop wireless routing. In: Proc. 9th ACM MobiCom, pp. 134--146 (2003). \doi{10.1145/938985.939000}",
    "surfing": r"Xu, W., Trappe, W., Zhang, Y.: Channel surfing: defending wireless sensor networks from interference. In: Proc. 6th IPSN, pp. 499--508 (2007). \doi{10.1109/IPSN.2007.4379710}",
    "navda": r"Navda, V., Bohra, A., Ganguly, S., Rubenstein, D.: Using channel hopping to increase 802.11 resilience to jamming attacks. In: Proc. IEEE INFOCOM 2007, pp. 2526--2530 (2007). \doi{10.1109/INFCOM.2007.314}",
    "xudrl": r"Xu, Y., Lei, M., Li, M., Zhao, M., Hu, B.: A new anti-jamming strategy based on deep reinforcement learning for MANET. In: Proc. IEEE 89th VTC2019-Spring, pp. 1--5 (2019). \doi{10.1109/VTCSpring.2019.8746494}",
    "song": r"Song, L., Guo, D., Hu, T., Yang, J.: Deep reinforcement learning-based anti-jamming rate adaptation and frequency selection algorithm for UAV ad hoc networks. IEEE Trans. Consum. Electron. \textbf{71}(4), 11782--11790 (2025). \doi{10.1109/TCE.2025.3627267}",
    "vanessen": r"Van Essen, B., Macaraeg, C., Gokhale, M., Prenger, R.: Accelerating a random forest classifier: multi-core, GP-GPU, or FPGA? In: Proc. IEEE 20th FCCM, pp. 232--239 (2012). \doi{10.1109/FCCM.2012.47}",
    "camp": r"Camp, T., Boleng, J., Davies, V.: A survey of mobility models for ad hoc network research. Wirel. Commun. Mob. Comput. \textbf{2}(5), 483--502 (2002). \doi{10.1002/wcm.72}",
    "gcm": r"Dworkin, M.: Recommendation for block cipher modes of operation: Galois/Counter Mode (GCM) and GMAC. NIST Special Publication 800-38D (2007). \doi{10.6028/NIST.SP.800-38D}",
    "hkdf": r"Krawczyk, H., Eronen, P.: HMAC-based Extract-and-Expand Key Derivation Function (HKDF). RFC 5869, IETF (2010). \doi{10.17487/RFC5869}",
    "breiman": r"Breiman, L.: Random forests. Mach. Learn. \textbf{45}(1), 5--32 (2001). \doi{10.1023/A:1010933404324}",
    "sklearn": r"Pedregosa, F., et al.: Scikit-learn: machine learning in Python. J. Mach. Learn. Res. \textbf{12}, 2825--2830 (2011)",
}

# ---------------------------------------------------------------- values
V.update(N=P.n_nodes, area=f"{P.area:.0f}", vmin=f"{P.v_min:.0f}", vmax=f"{P.v_max:.0f}", slot_ms=f"{P.slot_s * 1e3:.0f}",
         warm=f"{P.warmup_s:.0f}", flows=P.n_flows, payload=P.payload_bytes, pkt_int=P.pkt_interval, offered=f"{offered:.2f}",
         rate=f"{P.rate_bps / 1e6:.0f}", att=P.max_attempts, air=pc(air), probe_b=P.probe_bytes, probe_air=pc(probe_air),
         pt=f"{P.pt_dbm:.0f}", pl0=f"{P.pl0_db:.0f}", alpha=P.pl_exp, n0=f"{P.noise_dbm:.0f}", cs=f"{P.cs_range:.0f}",
         cs2=f"{VARIANTS['Carrier sense 450 m']['cs_range']:.0f}", pk2=VARIANTS["Steeper PER, k = 1.5"]["per_k"],
         rho=P.own_act, g50=f"{P.sinr50_db:.0f}", pk=P.per_k, range=f"{P.comm_range:.0f}", floor=f"{P.sinr_floor_db:.0f}",
         jreg=f"{P.jam_region[1] - P.jam_region[0]:.0f}", jv0=f"{P.jam_v[0]:.0f}", jv1=f"{P.jam_v[1]:.0f}",
         jon=f"{P.jam_on_mean:.0f}", F=FOLLOW, aes_over=pc(P.aes_overhead_bytes / P.payload_bytes),
         aes_b=P.aes_overhead_bytes, W=P.window, delta=f"{P.impair_dper:.2f}", rssi_n=f"{P.rssi_noise_db:g}",
         sinr_n=f"{P.sinr_noise_db:g}", theta=f"{THRESH:.2f}", k=K_LINKS, persist=PERSIST, tau=TAU,
         tau_ms=f"{TAU * P.slot_s * 1e3:.0f}", gw=GATE_WIN, conf=f"{GATE_CONF:.2f}", od=ORD_DEADLINE, orr=ORD_ROUNDS, gap=EP_GAP, prior_slots=EP_PRIOR, rejoin=REJOIN,
         q8=f"{q[8]:.3f}", q10=f"{q[10]:.3f}", q10sq=f"{q[10] ** 2:.3f}",
         ntrain=f"{det['n_train']:,}", ntest=f"{det['n_test']:,}",
         cc_plr=f"{cc['plr_min']:.2f}", cc_rssi=f"{cc['rssi_min']:.1f}",
         prec=pc(rf["prec"]), rec=pc(rf["rec"]), auc=f"{rf['auc']:.3f}", prauc=f"{rf['prauc']:.2f}",
         auc_lr=f"{lr['auc']:.3f}", prauc_lr=f"{lr['prauc']:.2f}", auc_thr=f"{th['auc']:.3f}",
         prior=pc(det["pos_test"]), allneg=pc(1 - det["pos_test"]),
         thr_feat=f"{th['feature'].replace('_mean', '').upper()} {'$<$' if th['sign'] < 0 else '$>$'} ${th['threshold']:.1f}$\\,dB",
         imp_rssi=f"{det['importance']['rssi_mean']:.2f}", imp_sinr=f"{det['importance']['sinr_mean']:.2f}",
         trig_delay=f"{trig_delay:.1f}", esc=pc(esc, 0), f1_05=f"{dpts['0.5']['f1']:.3f}")
T = T.replace("prior <<prior>> slots", "prior <<prior_slots>> slots").replace("<<prior>>\\,slots", "<<prior_slots>>\\,slots")


def drow(name, d, auc=True):
    a = f"{d['auc']:.3f} & {d['prauc']:.3f}" if auc else "-- & --"
    return (f"{name} & {pt(d['acc'])} & {pt(d['bacc'])} & {pt(d['prec'])} & {pt(d['rec'])} & {d['f1']:.3f} & {a} \\\\")


V["det_rows"] = "\n".join([
    drow(f"Random Forest, $\\theta={THRESH:.1f}$ (used)", rf), drow("Random Forest, $\\theta=0.5$", dpts["0.5"]),
    drow("Logistic regression, 0.5", lr), drow("Consistency check \\cite{xu05}", cc, auc=False),
    drow("Best single threshold", th)])

# rerouting text
V.update(hop0=pc(m("HOP", None)), etx0=pc(p0), etxg_lo=f"{100 * min(etx_gain):.0f}", etxg_hi=f"{100 * max(etx_gain):.0f}",
         rr_max=f"{100 * rr_max:.1f}", rr_max_int=f"{np.ceil(100 * rr_max):.0f}", basej8=pc(basej[8], 0), div8=pc(div[8], 0),
         div16=("for none" if max(div[pj] for pj in (16, 20, 24)) < 0.005 else
                "for at most " + pc(max(div[pj] for pj in (16, 20, 24)), 0)))
nmr = {"HOP+ML-reroute": "min-hop with ML flags", "HOP+Label-reroute": "min-hop with true labels",
       "ETX+ML-reroute": "ETX with ML flags", "ETX+Label-reroute": "ETX with true labels"}
plab = lambda pj: "no jammer" if pj is None else f"{pj}\\,dBm"
if not rr_sig_pos and not rr_sig_neg:
    V["rr_sig_text"] = "none of the paired differences is significant."
else:
    parts = []
    if rr_sig_pos:
        parts.append(f"{len(rr_sig_pos)} of the 28 paired differences are significantly positive (largest: "
                     f"{max(100 * rr[k][pj][0] for k, pj in rr_sig_pos):.1f} points)")
    if rr_sig_neg:
        parts.append(f"{len(rr_sig_neg)} {'are' if rr_sig_pos else 'of the 28 paired differences are'} significantly negative")
    V["rr_sig_text"] = " and ".join(parts) + ", so rerouting never yields a gain of practical size."

# theory
V.update(phi=f"{phi:.3f}", p0=pc(p0), pstar=pc(pstar), pstar_pers=pc(pstar_pers), agree=agree, agree_pers=agree_pers)
V["amb_text"] = ("" if not amb else
                 f"; the simulated gain at {', '.join(str(pj) for pj in amb)}\\,dBm, next to the break-even point, is not significantly different from zero")
V["theory_rows"] = "\n".join([
    "PDR while on, $p_J$ (\\%) & " + " & ".join(pt(p_on[pj]) for pj in POW[1:]) + " \\\\",
    f"Predicted ($p_J<{pt(pstar)}$) & " + " & ".join("gain" if pred[pj] else "loss" for pj in POW[1:]) + " \\\\",
    "Simulated gain & " + " & ".join(pm(g_ideal[pj]) for pj in POW[1:]) + " \\\\"])

# frequency change
fcu = {pj: m(MLU, pj) - m(NONE, pj) for pj in POW}
fcg = {pj: m(ML, pj) - m(NONE, pj) for pj in POW}
fcl = {pj: m(L9G, pj) - m(NONE, pj) for pj in POW}
V.update(ml0=pc(m(ML, None)), ml0_sw=f"{m(ML, None, 'switches_per_s'):.2f}", l70_sw=f"{m(L7, None, 'switches_per_s'):.2f}",
         l70_loss=f"{100 * (p0 - m(L7, None)):.1f}", etx24=pc(m(NONE, 24)), ml24=pc(m(ML, 24)), g24=pm(pd(ML, NONE, 24)),
         etx16=pc(m(NONE, 16)), ml16=pc(m(ML, 16)), g16=pm(pd(ML, NONE, 16)),
         worst_gate=f"{max(0, -100 * min(fcg[pj] for pj in POW)):.1f}", worst_ungated=f"{max(0, -100 * min(fcu[pj] for pj in POW)):.1f}",
         gate_cost=f"{100 * (m(MLU, 24) - m(ML, 24)):.1f}", l9g24=pc(m(L9G, 24)), l9g16=pc(m(L9G, 16)),
         l9g_vs_ml24=pm(pd(L9G, ML, 24)), l9g_vs_ml16=pm(pd(L9G, ML, 16)),
         gain24_int=f"{100 * min(fcg[24], fcl[24]):.0f}", gain24_ml=f"{100 * fcg[24]:.0f}",
         strand16=pc(nanmean([r["res"][ML]["stranded"] for r in rows if r["pj"] == 16]), 0),
         strand24=pc(nanmean([r["res"][ML]["stranded"] for r in rows if r["pj"] == 24]), 0),
         ideal16=pc(m(MLI, 16)), ideal24=pc(m(MLI, 24)), flood24=pc(m(MLFL, 24)),
         cp_cost24=f"{100 * (m(MLI, 24) - m(ML, 24)):.1f}",
         reach16=pc(nanmean([r["res"][ML]["order_reach"] for r in rows if r["pj"] == 16]), 0),
         worst_label=f"{max(0, -100 * min(m(LAB, pj) - m(NONE, pj) for pj in POW)):.1f}",
         worst_mean=f"{max(0, -100 * min(m(MLMT, pj) - m(NONE, pj) for pj in POW)):.1f}",
         q_maxdiff=f"{100 * max(abs(m(MLQ, pj) - m(ML, pj)) for pj in POW):.1f}")
dfe = {pj: pd(ML, MLF, pj) for pj in POW}
worst_fe = min(POW, key=lambda pj: dfe[pj][0])
best_fe = max(POW, key=lambda pj: dfe[pj][0])
V["fixed_vs_ep"] = (f"changes PDR by between {sg(dfe[worst_fe][0])} ({plab(worst_fe)}) and {sg(dfe[best_fe][0])} points "
                    f"({plab(best_fe)}) against the spot jammer; its benefit lies with other jammer types "
                    "(Sect.~\\ref{sec:jam})")

# robustness
SHORT = {"Baseline (design)": "Design", "Dense, N = 60": "Dense, $N=60$", "Sparse, N = 25": "Sparse, $N=25$",
         "Strong links, Pt = 20 dBm": "Strong, $P_t=20$", "Weak links, Pt = 10 dBm": "Weak, $P_t=10$",
         "Heavy traffic, 20 flows": "Heavy, 20 flows", "Carrier sense 450 m": "CS range 450\\,m",
         "Steeper PER, k = 1.5": "PER $k=1.5$"}
fc0 = lambda v, s: rb(v, s, None, "switches_per_s")
PR, L9GR, MLR = "ML-FC+gate (proposed)", "Loss-FC(0.9)+gate", "ML-FC"
V["rob_rows"] = "\n".join(
    f"{SHORT[v]} & {pt(rdet(v, 'acc'), 0)} & {fc0(v, 'Loss-FC(0.7)'):.2f} & {fc0(v, 'Consistency-FC'):.2f} & "
    f"{fc0(v, MLR):.2f} & {fc0(v, L9GR):.2f} & {fc0(v, PR):.2f} & {pt(rb(v, 'ETX', None))} & {pt(rb(v, PR, None))} & "
    f"{pt(rb(v, 'ETX', 16))} & {pt(rb(v, L9GR, 16))} & {pt(rb(v, PR, 16))} \\\\" for v in VARS)
OTH = [v for v in VARS if v != "Baseline (design)"]
worst_fc = max(fc0(v, PR) for v in OTH)
worst_fcl = max(fc0(v, L9GR) for v in OTH)
loss0 = {v: rb(v, "ETX", None) - rb(v, PR, None) for v in OTH}
loss0l = {v: rb(v, "ETX", None) - rb(v, L9GR, None) for v in OTH}
loss8 = {v: rb(v, "ETX", 8) - rb(v, PR, 8) for v in OTH}
gain16 = {v: rb(v, PR, 16) - rb(v, "ETX", 16) for v in OTH}
gain16l = {v: rb(v, L9GR, 16) - rb(v, "ETX", 16) for v in OTH}
ung_v = max(OTH, key=lambda v: fc0(v, MLR))
wv, wvl, w8 = max(OTH, key=loss0.get), max(OTH, key=loss0l.get), max(OTH, key=loss8.get)
PROSE = {"Dense, N = 60": "dense", "Sparse, N = 25": "sparse", "Strong links, Pt = 20 dBm": "strong-link",
         "Weak links, Pt = 10 dBm": "weak-link", "Heavy traffic, 20 flows": "heavy-traffic",
         "Carrier sense 450 m": "450\\,m carrier-sense", "Steeper PER, k = 1.5": "steep-PER"}
nm = lambda v: PROSE[v] + " deployment"
V["rob_text"] = (
    f"Detection accuracy (jammed runs) stays between {pt(min(rdet(v, 'acc') for v in VARS), 0)}\\% and "
    f"{pt(max(rdet(v, 'acc') for v in VARS), 0)}\\%. Without a jammer the proposed scheme orders at most {worst_fc:.2f} "
    f"changes per second and loses at most {100 * max(0, max(loss0.values())):.1f} points ({nm(wv)}); the gated loss trigger "
    f"orders at most {worst_fcl:.2f} and loses at most {100 * max(0, max(loss0l.values())):.1f} ({nm(wvl)}). The ungated ML "
    f"trigger is not robust: in the {nm(ung_v)} it orders {fc0(ung_v, MLR):.2f} false changes per second, and the 0.7 loss "
    f"trigger up to {max(fc0(v, 'Loss-FC(0.7)') for v in OTH):.1f}. At 16\\,dBm the "
    f"proposed scheme changes PDR relative to ETX by {sg(min(gain16.values()))} to {sg(max(gain16.values()))} points and the "
    f"gated loss trigger by {sg(min(gain16l.values()))} to {sg(max(gain16l.values()))}. At 8\\,dBm, where a change rarely "
    f"pays, the proposed scheme loses at most {100 * max(0, max(loss8.values())):.1f} points ({nm(w8)}); the ungated ML "
    f"trigger loses up to {100 * max(rb(v, 'ETX', 8) - rb(v, MLR, 8) for v in OTH):.1f}.")

worst_any = max([-(m(ML, pj) - m(NONE, pj)) for pj in POW] +
                [rb(v, "ETX", pj) - rb(v, PR, pj) for v in VARS for pj in (None, 8, 16, 24)] +
                [jm(j, "ETX") - jm(j, "ML-FC+gate (proposed)") for j in JAMS])
V["worst_any"] = f"{100 * worst_any:.1f}"
V["repo"] = (f"available at \\url{{{REPO_URL}}}" if REPO_URL else "available from the corresponding author on request")

# jammers
V["pon"] = f"{P.pulsed_on_mean:.0f}"
V["pduty"] = f"{100 * P.pulsed_on_mean / (P.pulsed_on_mean + P.pulsed_off_mean):.0f}"
V["nsw"], V["sdw"] = P.n_sweep_ch, P.sweep_dwell
JSHORT = {"Spot (training model)": "Spot (design)", "Pulsed, 20% duty": "Pulsed, 20\\%", "Sweep, 8 channels": "Sweep, 8 ch.",
          "Two spot jammers": "Two spot"}
JP = "ML-FC+gate (proposed)"
V["jam_rows"] = "\n".join(
    f"{JSHORT[j]} & {pt(jd(j, 'rec'))} & {pt(jd(j, 'prec'))} & {jm(j, JP, 'L_on'):.0f} & {pt(jm(j, 'ETX'))} & "
    f"{pt(jm(j, 'ML-FC'))} & {pt(jm(j, 'Loss-FC(0.9)+gate'))} & "
    f"{pt(jm(j, 'ML-FC+fixed gate'))} & {pt(jm(j, JP))} & {pt(jm(j, 'ML-FC+cause'))} & {pt(jm(j, 'ML-FC+cause+gate'))} \\\\" for j in JAMS)
sw = [j for j in JAMS if j.startswith("Sweep")][0]
two = [j for j in JAMS if j.startswith("Two")][0]
pul = [j for j in JAMS if j.startswith("Pulsed")][0]
spot = JAMS[0]
ung_loss = max(jm(j, "ETX") - jm(j, s) for j in (sw, pul) for s in ("ML-FC", "Loss-FC(0.9)"))
gated_loss = max(jm(j, "ETX") - jm(j, s) for j in (sw, pul) for s in (JP, "Loss-FC(0.9)+gate"))
V["ungated_sweep"] = f"{100 * ung_loss:.0f}"
V["sweep_loss"] = f"{100 * gated_loss:.1f}"
V["jam_text"] = (
    f"Recall is low at this power ({pt(min(jd(j, 'rec') for j in JAMS), 0)}--{pt(max(jd(j, 'rec') for j in JAMS), 0)}\\%) but "
    f"precision stays at {pt(min(jd(j, 'prec') for j in JAMS), 0)}--{pt(max(jd(j, 'prec') for j in JAMS), 0)}\\%, and $k$ "
    f"flagged links still suffice: against two jammers the proposed scheme raises PDR from {pc(jm(two, 'ETX'))} to "
    f"{pc(jm(two, JP))}. The episode estimates follow the jammer: $\\hat L_{{\\mathrm{{on}}}}$ is "
    f"{jm(spot, JP, 'L_on'):.0f} slots for the spot jammer but {jm(pul, JP, 'L_on'):.0f} for the pulsed and "
    f"{jm(sw, JP, 'L_on'):.0f} for the sweeping one, so the gate sees that a change cannot pay against short bursts. "
    f"Against the sweep it cuts the loss against ETX from {100 * (jm(sw, 'ETX') - jm(sw, 'ML-FC')):.1f} (ungated) and "
    f"{100 * (jm(sw, 'ETX') - jm(sw, 'ML-FC+fixed gate')):.1f} (fixed gate) to {pm(jpd(sw, JP, 'ETX'))} points; against the "
    f"pulsed jammer the proposed scheme is {pm(jpd(pul, JP, 'ETX'))} points from ETX. The gate's cost shows with two jammers, where "
    f"it delivers {pt(jm(two, 'ML-FC') - jm(two, JP))} points less than the ungated trigger. The cause classifier "
    f"identifies the jammer type in {pc(cause_r['bacc'], 0)} of held-out windows (balanced accuracy; it confuses mainly one "
    f"and two spot jammers, which call for the same action). Used alone, it avoids the losses of the ungated trigger "
    f"(pulsed {sg(jm(pul, 'ML-FC+cause') - jm(pul, 'ETX'))}, sweep {sg(jm(sw, 'ML-FC+cause') - jm(sw, 'ETX'))} points against "
    f"ETX) but, unlike the gate, needs every jammer type in training and cannot tell weak jamming from strong: at 4\\,dBm "
    f"it changes PDR by {pm(pd(MLC, NONE, 4))} points. Combined with the gate it trims the remaining losses against pulsed and "
    f"sweeping jammers to {sg(jm(pul, 'ML-FC+cause+gate') - jm(pul, 'ETX'))} and {sg(jm(sw, 'ML-FC+cause+gate') - jm(sw, 'ETX'))} "
    f"points, at a cost of {pt(m(ML, 24) - m(MLCG, 24))} points at 24\\,dBm.")

# follow
fbreak = [f for f in fs if fo["ml"][f] > fo["none"][f]]
V.update(fnone=pc(fo["none"][fs[0]]), fbreak=ms(min(fbreak)) if fbreak else "n/a", fmin=ms(fs[0]),
         fml_min=pc(fo["ml"][fs[0]]), fg_loss=f"{100 * max(0, fo['none'][fs[0]] - fo['mlgate'][fs[0]]):.1f}",
         F_ms=ms(FOLLOW))
d50 = {f: fo["mlgate"][f] - fo["mlgate50"][f] for f in fs}
V["f50_text"] = (f"it loses {100 * d50[fs[0]]:.1f} points more than the true-$F$ gate at $F={ms(fs[0])}$\\,ms and at most "
                 f"{100 * max(abs(d50[f]) for f in fs[1:]):.1f} points elsewhere.")
V.update(tmin=TS[0], tmax=TS[-1])
PS = sorted({r["P"] for r in probe_r})
pg = {P_: float(np.mean([r["mlgate"]["pdr"] - r["none"] for r in probe_r if r["P"] == P_])) for P_ in PS}
pdl = {P_: nanmean([r["mlgate"]["trig_delay_slots"] for r in probe_r if r["P"] == P_]) for P_ in PS}
V["probe_text"] = (f"Probing every {PS[-1]} slots instead of every slot ({100 // PS[-1]} instead of 100 probes per second, "
                   f"probe airtime {pc(probe_air / PS[-1])}) lengthens the window to {P.window * PS[-1]} slots and the change "
                   f"delay from {pdl[PS[0]]:.0f} to {pdl[PS[-1]]:.0f} slots, and the gain over ETX at 16\\,dBm falls from "
                   f"{100 * pg[PS[0]]:.1f} to {100 * pg[PS[-1]]:.1f} points, without retraining the detector.")
dt = {k: {t: to[k][t] - to["none"][t] for t in TS} for k in ("ml", "mlgate")}
V["tau_text"] = (f"the ungated trigger's change in PDR relative to no change goes from {sg(dt['ml'][TS[0]])} to "
                 f"{sg(dt['ml'][TS[-1]])} points and the gated scheme's from {sg(dt['mlgate'][TS[0]])} to "
                 f"{sg(dt['mlgate'][TS[-1]])}: the gate, which knows $\\tau$, "
                 + ("removes the loss when retuning is slow." if dt['mlgate'][TS[-1]] > -0.005 else
                    "removes most but not all of the loss when retuning is slow."))
# hardware
V.update(nodes_full=f"{cost['total_nodes']:,}", mib_full=f"{cost['bytes_16b_node'] / 2 ** 20:.1f}", f1_c10=f"{c10['f1']:.3f}",
         f1_full=f"{full['f1']:.3f}", nodes_c10=f"{c10['nodes']:,}", f16_fixed=f"{f16['f1_fixed']:.3f}",
         f8_fixed=f"{f8['f1_fixed']:.3f}")
if fpga:
    u, dv = fpga["utilization"], fpga["device"]
    V["fpga_short"] = f"{u['luts_pct']:.1f}\\% of its LUTs and {u['bram_pct']:.0f}\\% of its block RAM"
    tm = fpga.get("timing_estimate", {})
    fmax = (f" A longest-path estimate with Artix-7 cell delays and 450\\,ps per net gives about "
            f"{tm['with_routing']['fmax_mhz']:.0f}\\,MHz.") if tm.get("available") else ""
    V["fpga_text"] = (
        f"We generated Verilog for this forest ({fpga['nodes']:,} reachable nodes, {fpga['bits']}-bit features and thresholds, "
        f"{fpga['leaf_bits']}-bit leaf probabilities) as a pipeline with one stage per tree level, node tables in ROM and all "
        f"trees in parallel. On {fpga['n_vectors']:,} test vectors the RTL matched a bit-exact integer model on "
        f"{100 * fpga['rtl_vs_int']['iverilog']['match_fraction']:.0f}\\%, and the integer model agreed with the floating-point "
        "forest " + ("on " if fpga['int_vs_float_agreement_real'] == 1.0 else "except for ") +
        (("all vectors. " if fpga['int_vs_float_agreement_stress'] == 1.0 else
          f"all held-out windows and on {100 * fpga['int_vs_float_agreement_stress']:.1f}\\% of the threshold-stress vectors. ")
         if fpga['int_vs_float_agreement_real'] == 1.0 else
         f"{round((1 - fpga['int_vs_float_agreement_real']) * fpga['n_vectors_real'])} of the {fpga['n_vectors_real']:,} held-out "
         f"windows, which differs through threshold rounding, and agreed on {100 * fpga['int_vs_float_agreement_stress']:.1f}\\% of the "
         "threshold-stress vectors. ") +
        f"Synthesis with Yosys (synth\\_xilinx, no vendor place-and-route) for an Artix-7 XC7A200T uses {fpga['luts']:,} LUTs "
        f"({u['luts_pct']:.1f}\\%), {fpga['ffs']:,} flip-flops ({u['ffs_pct']:.1f}\\%), {fpga['bram36_equiv']:.0f} 36-Kb block RAMs "
        f"({u['bram_pct']:.0f}\\%) and no DSP blocks, with a latency of {fpga['latency_cycles']} cycles and one decision per "
        "cycle (node tables could also use LUT RAM)." + fmax)
    V["fpga_table"] = (
        "\\begin{table}[t]\n\\caption{Yosys synthesis (synth\\_xilinx, Xilinx 7 series) of the 10-tree 16-bit forest; not "
        "vendor place-and-route.}\\label{tab:fpga}\n"
        "\\centering\\scriptsize\n\\begin{tabular}{@{}lrrrrr@{}}\n\\toprule\n"
        "Resource & LUTs & Flip-flops & BRAM (36\\,Kb) & DSP & Latency \\\\\n\\midrule\n"
        f"Used & {fpga['luts']:,} & {fpga['ffs']:,} & {fpga['bram36_equiv']:.0f} & {fpga['dsps']} & {fpga['latency_cycles']} cycles \\\\\n"
        f"XC7A200T & {dv['luts']:,} & {dv['ffs']:,} & {dv['bram36']} & {dv['dsps']} & \\\\\n"
        f"Share & {u['luts_pct']:.1f}\\% & {u['ffs_pct']:.1f}\\% & {u['bram_pct']:.1f}\\% & {u['dsps_pct']:.0f}\\% & \\\\\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}")
else:
    V["fpga_short"] = V["fpga_text"] = V["fpga_table"] = "[FPGA synthesis pending]"

# bibliography in order of first citation
order = []
for keys in re.findall(r"\\cite\{([^}]*)\}", T.replace("<<det_rows>>", V["det_rows"])):
    for k in keys.split(","):
        if k not in order:
            order.append(k)
missing = [k for k in order if k not in REFS]
assert not missing, missing
V["bibliography"] = ("\\begin{thebibliography}{99}\n" +
                     "\n".join(f"\\bibitem{{{k}}} {REFS[k]}" for k in order) + "\n\\end{thebibliography}")

tex = T
for k in sorted(V, key=len, reverse=True):
    tex = tex.replace(f"<<{k}>>", str(V[k]))
left = re.findall(r"<<\w+>>", tex)
assert not left, left
path = os.path.join(OUT, "manet_antijamming.tex")
open(path, "w").write(tex)
for f in ("fig1_architecture.pdf", "fig5_pdr_reroute.pdf", "fig6_pdr_fc.pdf", "fig7_switch_rate.pdf", "fig8_follow.pdf"):
    shutil.copy(os.path.join(FIG, f), OUT)
shutil.copy(os.path.join(ROOT, "springer_template", "llncs", "llncs.cls"), OUT)
for _ in range(2):
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "manet_antijamming.tex"], cwd=OUT, capture_output=True, text=True)
log = open(os.path.join(OUT, "manet_antijamming.log")).read()
pages = re.search(r"Output written on .*?\((\d+) pages", log)
print("pages:", pages.group(1) if pages else "?", "| warnings:", len(re.findall(r"Warning", log)),
      "| overfull:", len(re.findall(r"Overfull", log)), "| unused refs:", sorted(set(REFS) - set(order)))
