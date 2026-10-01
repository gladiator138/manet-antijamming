"""Draw every figure in the paper from results/*.json (300+ dpi PNG and vector PDF)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(FIG, exist_ok=True)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MK = ["o", "s", "^", "D", "v", "P"]
LS = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]
INK, MUTED = "#222222", "#6b6b6b"
plt.rcParams.update({"font.family": "serif", "font.size": 9.4, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.grid": True, "grid.color": "#e4e4e4", "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "mathtext.fontset": "dejavuserif", "lines.linewidth": 1.6, "lines.markersize": 5})
PLAB = ["off", "4", "8", "12", "16", "20", "24"]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=600, bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_arch():
    fig = plt.figure(figsize=(6.3, 1.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off(); ax.set_xlim(0, 128.5); ax.set_ylim(0, 30)

    def box(x, y, w, h, t, fc="#eef4fc"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0",
                                    fc=fc, ec="#1c5cab", lw=0.9))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=9, color=INK)

    def arr(x1, y1, x2, y2):
        ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color="#1c5cab", lw=0.9))
    y, h, w, g = 17, 11, 21.6, 4.5
    xs = [1 + k * (w + g) for k in range(5)]
    for x, t, fc in zip(xs, ["Probes on\nchannel $c_e$", "Window\nfeatures", "Detector\n(RF, loss, ...)",
                             "$\\geq k$ alarms\nfor 3 slots?", "Gate:\n$\\hat p_J<p^*$?"],
                        ["#eef4fc"] * 3 + ["#fff4e6"] * 2):
        box(x, y, w, h, t, fc)
    for x in xs[:-1]:
        arr(x + w + 0.6, y + h / 2, x + w + g - 0.6, y + h / 2)
    box(xs[3], 1, xs[4] + w - xs[3], 9, "Flood sealed order; retune to $c_{e+1}$", fc="#fff4e6")
    box(xs[0], 1, xs[2] + w - xs[0], 9, "No change: ETX routing handles the loss")
    arr(xs[4] + w / 2, y - 0.6, xs[4] + w / 2, 10.6)
    ax.text(xs[4] + w / 2 + 1.5, 13, "yes", fontsize=9, style="italic", color=MUTED)
    ax.plot([xs[3] + w / 2, xs[3] + w / 2], [y - 0.6, 13.5], color="#1c5cab", lw=0.9)
    ax.plot([xs[4] + 4, xs[4] + 4, xs[3] + w / 2], [y - 0.6, 13.5, 13.5], color="#1c5cab", lw=0.9)
    ax.plot([xs[3] + w / 2, xs[2] + w / 2], [13.5, 13.5], color="#1c5cab", lw=0.9)
    arr(xs[2] + w / 2, 13.5, xs[2] + w / 2, 10.6)
    ax.text(xs[2] + w / 2 + 1.5, 12.2, "no", fontsize=9, style="italic", color=MUTED, va="center")
    fig.savefig(os.path.join(FIG, "fig1_architecture.png"), dpi=600)
    fig.savefig(os.path.join(FIG, "fig1_architecture.pdf"))
    plt.close(fig)


def fig_detector():
    d = json.load(open(os.path.join(RES, "detector.json")))
    cm = np.array(d["rf_cm"])
    fig, ax = plt.subplots(figsize=(3.0, 2.6))
    ax.grid(False)
    ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            v = cm[i, j] / cm[i].sum()
            ax.text(j, i, f"{cm[i, j]:,}\n({v:.1%})", ha="center", va="center", fontsize=8,
                    color="white" if v > 0.6 else INK)
    ax.set_xticks([0, 1], ["clean", "jammed"]); ax.set_yticks([0, 1], ["clean", "jammed"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    save(fig, "fig2_confusion")

    fig, ax = plt.subplots(figsize=(3.2, 2.7))
    ax.plot(d["rf_roc"]["fpr"], d["rf_roc"]["tpr"], color=C[0], label=f"Random Forest (AUC {d['rf']['auc']:.3f})")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=":")
    ax.text(0.35, 0.3, f"Logistic regression AUC {d['lr']['auc']:.3f}\nBest single threshold AUC {d['thr']['auc']:.3f}",
            fontsize=7, color=MUTED)
    ax.set_xlabel("False-positive rate"); ax.set_ylabel("True-positive rate")
    ax.legend(loc="lower right", fontsize=7)
    save(fig, "fig3_roc")

    imp = d["importance"]
    names = {"rssi_mean": "Mean RSSI", "sinr_mean": "Mean SINR", "plr": "Packet-loss rate",
             "ber_mean": "Mean BER", "sinr_std": "SINR jitter"}
    ks = sorted(imp, key=imp.get)
    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    ax.barh([names[k] for k in ks], [imp[k] for k in ks], color=C[0], height=0.6)
    ax.set_xlabel("Mean decrease in impurity")
    ax.grid(axis="y", visible=False)
    save(fig, "fig4_importance")


def _agg(rows, key, metric, base=None):
    out = []
    for pl in [None, 4, 8, 12, 16, 20, 24]:
        v = np.array([r["res"][key][metric] - (r["res"][base][metric] if base else 0) for r in rows if r["pj"] == pl])
        out.append((v.mean(), 1.96 * v.std(ddof=1) / np.sqrt(len(v))))
    return np.array(out)


LEG = dict(fontsize=8.6, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, handlelength=1.7,
           columnspacing=0.6, labelspacing=0.25, handletextpad=0.4, borderaxespad=0.1)


def panel_fig():
    fig, ax = plt.subplots(figsize=(2.75, 2.3))
    fig.subplots_adjust(left=0.2, right=0.98, top=0.97, bottom=0.42)
    return fig, ax


def save_panel(fig, ax, name):
    ax.legend(bbox_transform=fig.transFigure, **LEG)
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


STY = {"ETX": (0, "o", "-"), "ETX+Loss-FC(0.7)": (1, "s", "--"), "ETX+Loss-FC(0.9)": (1, "s", ":"),
       "ETX+Consistency-FC": (2, "^", "-."), "ETX+ML-FC": (3, "D", ":"), "ETX+Loss-FC(0.9)+gate": (2, "v", (0, (5, 1))),
       "ETX+ML-FC+gate (proposed)": (4, "P", "-"), "ETX+Label-FC": (5, "X", (0, (3, 1, 1, 1)))}


def fig_routing():
    rows = json.load(open(os.path.join(RES, "routing.json")))
    x = np.arange(7)

    def panel(keys, labels, metric, ylabel, name, scale=1.0, base=None, logy=False, sty=None):
        fig, ax = panel_fig()
        for i, (k, lab) in enumerate(zip(keys, labels)):
            b = base[i] if isinstance(base, list) else base
            a = _agg(rows, k, metric, b) * scale
            c, mk, ls = sty[k] if sty else (i, MK[i], LS[i])
            ax.errorbar(x + (i - len(keys) / 2) * 0.06, a[:, 0], yerr=a[:, 1], color=C[c], marker=mk, ls=ls,
                        capsize=2, label=lab, lw=1.4, ms=4)
        if base is not None:
            ax.axhline(0, color=MUTED, lw=0.8)
        if logy:
            ax.set_yscale("symlog", linthresh=0.01)
            ax.set_yticks([0, 0.01, 0.1, 1, 10], ["0", "0.01", "0.1", "1", "10"])
            ax.set_ylim(-0.002, 12)
        ax.set_xticks(x, PLAB); ax.set_xlabel("Jammer transmit power (dBm)"); ax.set_ylabel(ylabel)
        save_panel(fig, ax, name)
    panel(["HOP+ML-reroute", "HOP+Label-reroute", "ETX+ML-reroute", "ETX+Label-reroute"],
          ["Min-hop + ML", "Min-hop + labels", "ETX + ML", "ETX + labels"],
          "pdr", "Rerouting gain (points)", "fig5_pdr_reroute", 100,
          base=["HOP", "HOP", "ETX", "ETX"])
    panel(["ETX", "ETX+Loss-FC(0.9)", "ETX+Loss-FC(0.9)+gate", "ETX+ML-FC", "ETX+ML-FC+gate (proposed)", "ETX+Label-FC"],
          ["ETX, no change", "Loss 0.9", "Loss 0.9 + gate", "ML, no gate", "ML + gate (prop.)", "True labels"],
          "pdr", "Packet-delivery ratio (%)", "fig6_pdr_fc", 100, sty=STY)
    panel(["ETX+Loss-FC(0.7)", "ETX+Loss-FC(0.9)", "ETX+Consistency-FC", "ETX+ML-FC", "ETX+Loss-FC(0.9)+gate",
           "ETX+ML-FC+gate (proposed)"],
          ["Loss 0.7", "Loss 0.9", "Consistency check", "ML, no gate", "Loss 0.9 + gate", "ML + gate (prop.)"],
          "switches_per_s", "Changes per second", "fig7_switch_rate", logy=True, sty=STY)


def fig_follow():
    rows = json.load(open(os.path.join(RES, "follow.json")))
    fs = sorted({r["follow"] for r in rows})
    fig, ax = panel_fig()
    for i, (k, lab) in enumerate([("none", "No change"), ("ml", "ML, no gate"),
                                  ("mlgate", "Gate, true $F$"), ("mlgate50", "Gate, assumes 500 ms"),
                                  ("label", "True labels")]):
        v = np.array([[r[k] for r in rows if r["follow"] == f] for f in fs]) * 100
        ax.errorbar([f * 10 for f in fs], v.mean(1), yerr=1.96 * v.std(1, ddof=1) / np.sqrt(v.shape[1]),
                    color=C[[0, 3, 4, 1, 5][i]], marker=MK[i], ls=LS[i], label=lab, capsize=2, lw=1.4, ms=4)
    ax.set_xscale("log")
    ax.set_xticks([f * 10 for f in fs], [str(int(f * 10)) for f in fs])
    ax.minorticks_off()
    ax.set_xlabel("Jammer follow time $F$ (ms)"); ax.set_ylabel("Packet-delivery ratio (%)")
    save_panel(fig, ax, "fig8_follow")


if __name__ == "__main__":
    import sys
    what = sys.argv[1:] or ["arch", "detector", "routing", "follow"]
    for w in what:
        globals()["fig_" + w]()
