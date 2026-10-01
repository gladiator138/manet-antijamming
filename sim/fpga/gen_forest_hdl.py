#!/usr/bin/env python3
"""Random-Forest -> pipelined, synthesizable Verilog-2001 + bit-exact integer reference model.

    python3 gen_forest_hdl.py --model forest.pkl --quant quant.json --out DIR
                              [--bits 16] [--leaf-bits 12] [--thresh 0.60]
                              [--thr-mode floor|round] [--vectors test.npz|test.npy]
                              [--n-real 3000] [--n-stress 1000] [--seed 0]

quant.json: {"lo": [5 floats], "hi": [5 floats], "bits": 16, ("thresh": 0.6)}
(--bits on the command line overrides quant.json; --thresh overrides quant.json,
default 0.60 = run_experiments.THRESH).

Integer semantics (all exact Python ints)
-----------------------------------------
  scale_f = (hi_f - lo_f) / (2^B - 1)
  q_f     = clip(round_half_even((x_f - lo_f) / scale_f), 0, 2^B - 1)     (done upstream of the RTL)
  split   : sklearn  "x_f <= thr"   becomes   "q_f <= T",
            T = floor((thr - lo_f) / scale_f)          (--thr-mode floor, default)
            T = round((thr - lo_f) / scale_f)          (--thr-mode round; = run_experiments.fixedpoint())
            Nodes with T < 0 always go right, T >= 2^B-1 always go left; they are removed
            from the hardware tree (identical decisions), so every stored T fits in B bits.
  leaf    : v = round(p1 * (2^P - 1)), p1 = the leaf's class-1 fraction, P = --leaf-bits
  output  : score S = sum_t v_t ;  jammed = S >= K,  K = ceil(thresh * n_trees * (2^P - 1))
            (i.e. mean quantised probability >= thresh).

Hardware (rf_forest.v, top module "rf_forest")
------------------------------------------------
  * x_in = 5 x B-bit quantised features, feature f in bits [f*B +: B]; one new window per clock.
  * Stage L (L = 0..D-1) of every tree reads the node record of level L (registered ROM output),
    compares q[feat] <= T and forms the address of the child in level L+1; the ROM of level L+1
    is read synchronously (address combinational, data registered) -> one clock per tree level,
    maps onto block RAM (RAMB18/36) or LUT-ROM as the synthesiser chooses.
  * Record = {leaf(1), feat(3), thr_or_value(B), left_child_index(CW)}; right child = left + 1
    (children are laid out adjacently).  Leaves above the maximum depth set a "done" flag and
    their value is carried through the remaining stages.
  * All trees in parallel; registered pairwise adder tree; final compare register.
  * Latency (in_valid -> out_valid) = D + 3 + ceil(log2(n_trees)) clocks, D = forest depth.

Emitted files in --out:
  rf_forest.v, t<k>_l<L>.mem (ROM images), tb_rf_forest.v (self-checking-capable testbench),
  vectors.hex (quantised inputs), ref_int.txt (integer model: score jammed per vector),
  vectors.npz (X float, q, int score/decision, float prob/decision), meta.json.
"""
import argparse, json, math, os, pickle, sys
from fractions import Fraction
import numpy as np

NFEAT = 5
FEAT_BITS = 3


def clog2(n):
    return max(1, int(math.ceil(math.log2(max(n, 2)))))


# ---------------------------------------------------------------------------------------------
# quantisation
# ---------------------------------------------------------------------------------------------
class Quant:
    def __init__(self, lo, hi, bits):
        self.lo = np.asarray(lo, dtype=np.float64)
        self.hi = np.asarray(hi, dtype=np.float64)
        assert self.lo.shape == (NFEAT,) and self.hi.shape == (NFEAT,), "lo/hi must have 5 entries"
        self.bits = int(bits)
        self.levels = (1 << self.bits) - 1
        s = (self.hi - self.lo) / self.levels
        self.scale = np.where(s > 0, s, 1.0)  # degenerate (constant) feature: any positive scale

    def q(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.clip(np.round((X - self.lo) / self.scale), 0, self.levels).astype(np.int64)

    def T(self, f, thr, mode):
        u = (float(thr) - float(self.lo[f])) / float(self.scale[f])
        return int(math.floor(u)) if mode == "floor" else int(round(u))


# ---------------------------------------------------------------------------------------------
# integer trees
# ---------------------------------------------------------------------------------------------
class IntTree:
    """sklearn tree with integer thresholds (unclamped) and integer leaf values."""

    def __init__(self, est, qz, mode, pbits, cls1):
        t = est.tree_
        self.left = t.children_left.astype(np.int64)
        self.right = t.children_right.astype(np.int64)
        self.feat = np.where(t.feature >= 0, t.feature, 0).astype(np.int64)
        self.is_leaf = t.children_left < 0
        self.T = np.array([qz.T(int(t.feature[i]), t.threshold[i], mode) if not self.is_leaf[i] else 0
                           for i in range(t.node_count)], dtype=np.int64)
        val = t.value[:, 0, :].astype(np.float64)
        p1 = val[:, cls1] / np.maximum(val.sum(axis=1), 1e-300)
        self.p1 = p1
        self.v = np.round(p1 * ((1 << pbits) - 1)).astype(np.int64)
        self.levels = qz.levels

    def predict(self, Q):
        """Reference integer model: walk the ORIGINAL tree with q <= T (python-int semantics)."""
        n = np.zeros(len(Q), dtype=np.int64)
        rows = np.arange(len(Q))
        for _ in range(len(self.left) + 1):
            inner = ~self.is_leaf[n]
            if not inner.any():
                break
            go_left = Q[rows, self.feat[n]] <= self.T[n]
            nxt = np.where(go_left, self.left[n], self.right[n])
            n = np.where(inner, nxt, n)
        return self.v[n]

    def resolve(self, i):
        """Skip nodes whose integer decision is constant for every q in [0, 2^B-1]."""
        while not self.is_leaf[i]:
            if self.T[i] < 0:
                i = self.right[i]
            elif self.T[i] >= self.levels:
                i = self.left[i]
            else:
                break
        return i

    def layout(self):
        """Breadth-first level layout of the collapsed tree.
        Returns list of levels; each level is a list of records
        (leaf, feat, thr_or_value, left_child_index_in_next_level)."""
        levels = []
        cur = [self.resolve(0)]
        while cur:
            recs, nxt = [], []
            for i in cur:
                if self.is_leaf[i]:
                    recs.append((1, 0, int(self.v[i]), 0))
                else:
                    recs.append((0, int(self.feat[i]), int(self.T[i]), len(nxt)))
                    nxt.append(self.resolve(self.left[i]))
                    nxt.append(self.resolve(self.right[i]))
            levels.append(recs)
            cur = nxt
        return levels


def emulate_layout(layout, Q):
    """Cycle-free emulation of exactly what the RTL computes from the ROM tables."""
    n = len(Q)
    idx = np.zeros(n, dtype=np.int64)
    done = np.zeros(n, dtype=bool)
    val = np.zeros(n, dtype=np.int64)
    for L, recs in enumerate(layout):
        r = np.array(recs, dtype=np.int64)
        leaf, feat, thr, child = (r[idx, k] for k in range(4))
        newly = (~done) & (leaf == 1)
        val = np.where(newly, thr, val)
        done = done | (leaf == 1)
        go = Q[np.arange(n), feat] <= thr
        idx = np.where(done, 0, child + (~go).astype(np.int64))
    assert done.all()
    return val


# ---------------------------------------------------------------------------------------------
# Verilog emission
# ---------------------------------------------------------------------------------------------
def tree_verilog(k, layout, B, P, D, out, rom_style="auto"):
    """Emit module rf_tree_<k> and its .mem files. Returns the module text."""
    nlev = len(layout)            # levels 0..d_k
    dk = nlev - 1
    depth = [1 << clog2(len(l)) for l in layout]          # padded ROM depths (>= 2)
    aw = [clog2(len(l)) for l in layout]                  # address widths
    cw = [aw[L + 1] if L + 1 < nlev else 1 for L in range(nlev)]  # child-index width per level
    W = [1 + FEAT_BITS + B + cw[L] for L in range(nlev)]
    XW = NFEAT * B

    def pack(L, rec):
        leaf, f, t, c = rec
        assert 0 <= t < (1 << B) and 0 <= f < 8 and 0 <= c < (1 << cw[L])
        return (leaf << (FEAT_BITS + B + cw[L])) | (f << (B + cw[L])) | (t << cw[L]) | c

    v = []
    v.append(f"// tree {k}: {sum(len(l) for l in layout)} nodes, depth {dk}, levels {[len(l) for l in layout]}")
    v.append(f"module rf_tree_{k} (")
    v.append("  input  wire clk,")
    v.append(f"  input  wire [{max(D,1) * XW - 1}:0] xp,   // x pipeline: stage L at [L*{XW} +: {XW}]")
    v.append(f"  output reg  [{P - 1}:0] val_out")
    v.append(");")
    for L in range(nlev):
        v.append(f"  // ---- level {L}: {len(layout[L])} nodes, record width {W[L]}")
        if L == 0:
            v.append(f"  wire [{W[0] - 1}:0] rec0 = {W[0]}'h{pack(0, layout[0][0]):x};")
            v.append("  wire done0 = 1'b0;")
            v.append(f"  wire [{P - 1}:0] v0 = {P}'d0;")
        else:
            fn = f"t{k}_l{L}.mem"
            with open(os.path.join(out, fn), "w") as fh:
                hw = (W[L] + 3) // 4
                for i in range(depth[L]):
                    word = pack(L, layout[L][i]) if i < len(layout[L]) else 0
                    fh.write(f"{word:0{hw}x}\n")
            attr = "" if rom_style == "auto" else f"(* rom_style = \"{rom_style}\" *) "
            v.append(f"  {attr}reg [{W[L] - 1}:0] rom{L} [0:{depth[L] - 1}];")
            v.append(f"  initial $readmemh(\"{fn}\", rom{L});")
            v.append(f"  reg [{W[L] - 1}:0] rec{L};")
            v.append(f"  reg done{L};")
            v.append(f"  reg [{P - 1}:0] v{L};")
        v.append(f"  wire leaf{L} = rec{L}[{W[L] - 1}];")
        v.append(f"  wire [2:0] f{L} = rec{L}[{W[L] - 2}:{W[L] - 4}];")
        v.append(f"  wire [{B - 1}:0] t{L} = rec{L}[{B + cw[L] - 1}:{cw[L]}];")
        if L < dk:
            v.append(f"  wire [{cw[L] - 1}:0] c{L} = rec{L}[{cw[L] - 1}:0];")
            base = L * XW
            mux = " :\n                     ".join(
                f"(f{L} == 3'd{f}) ? xp[{base + f * B + B - 1}:{base + f * B}]" for f in range(NFEAT))
            v.append(f"  wire [{B - 1}:0] xs{L} = {mux} : {B}'d0;")
            v.append(f"  wire go{L} = (xs{L} <= t{L});")
            v.append(f"  wire [{cw[L] - 1}:0] a{L} = c{L} + {{{{{cw[L] - 1}{{1'b0}}}}, ~go{L}}};" if cw[L] > 1
                     else f"  wire a{L} = c{L} + ~go{L};")
    # sequential part
    v.append("  always @(posedge clk) begin")
    for L in range(dk):
        v.append(f"    rec{L + 1} <= rom{L + 1}[a{L}];")
        v.append(f"    done{L + 1} <= done{L} | leaf{L};")
        v.append(f"    v{L + 1} <= done{L} ? v{L} : t{L}[{P - 1}:0];")
    v.append("  end")
    # final value at stage dk, then pad to the forest depth D
    v.append(f"  wire [{P - 1}:0] vfin = done{dk} ? v{dk} : t{dk}[{P - 1}:0];")
    pad = D - dk
    if pad == 0:
        v.append("  always @(posedge clk) val_out <= vfin;")
    else:
        v.append(f"  // tree shallower than forest depth: {pad} extra delay stage(s)")
        v.append(f"  reg [{P - 1}:0] dly [0:{pad - 1}];")
        v.append("  integer i;")
        v.append("  always @(posedge clk) begin")
        v.append("    dly[0] <= vfin;")
        v.append(f"    for (i = 1; i < {pad}; i = i + 1) dly[i] <= dly[i-1];")
        v.append(f"    val_out <= dly[{pad - 1}];")
        v.append("  end")
    v.append("endmodule\n")
    return "\n".join(v), dict(nodes=sum(len(l) for l in layout), depth=dk,
                              rom_bits=sum(W[L] * len(layout[L]) for L in range(1, nlev)),
                              rom_bits_padded=sum(W[L] * depth[L] for L in range(1, nlev)))


def top_verilog(N, B, P, D, K, SW):
    XW = NFEAT * B
    A = clog2(N) if N > 1 else 0
    LAT = D + 3 + A
    v = [f"// rf_forest: {N} trees, depth {D}, {B}-bit features, {P}-bit leaves, latency {LAT}",
         "`timescale 1ns/1ps",
         "module rf_forest (",
         "  input  wire clk,",
         "  input  wire rst,",
         "  input  wire in_valid,",
         f"  input  wire [{XW - 1}:0] x_in,      // feature f (quantised, unsigned) at [f*{B} +: {B}]",
         "  output wire out_valid,",
         "  output reg  jammed,",
         f"  output reg  [{SW - 1}:0] score",
         ");",
         f"  localparam LATENCY = {LAT};",
         f"  localparam [{SW - 1}:0] K = {SW}'d{K};",
         f"  // feature pipeline x_0 .. x_{D - 1}",
         f"  reg [{XW - 1}:0] xr [0:{max(D, 1) - 1}];",
         "  integer i;",
         "  always @(posedge clk) begin",
         "    xr[0] <= x_in;",
         f"    for (i = 1; i < {max(D, 1)}; i = i + 1) xr[i] <= xr[i-1];",
         "  end",
         f"  wire [{max(D, 1) * XW - 1}:0] xp;",
         "  genvar g;",
         f"  generate for (g = 0; g < {max(D, 1)}; g = g + 1) begin : gx",
         f"    assign xp[g*{XW} +: {XW}] = xr[g];",
         "  end endgenerate",
         "  // valid pipeline",
         f"  reg [{LAT - 1}:0] vpipe;",
         "  always @(posedge clk)",
         f"    if (rst) vpipe <= {LAT}'d0; else vpipe <= {{vpipe[{LAT - 2}:0], in_valid}};" if LAT > 1 else
         "    if (rst) vpipe <= 1'b0; else vpipe <= in_valid;",
         f"  assign out_valid = vpipe[{LAT - 1}];",
         "  // trees"]
    for k in range(N):
        v.append(f"  wire [{P - 1}:0] tv{k};")
        v.append(f"  rf_tree_{k} u_t{k} (.clk(clk), .xp(xp), .val_out(tv{k}));")
    # adder tree
    v.append("  // registered pairwise adder tree")
    cur = [(f"tv{k}", P) for k in range(N)]
    lvl = 0
    while len(cur) > 1:
        nxt = []
        for j in range(0, len(cur), 2):
            if j + 1 < len(cur):
                (a, wa), (b, wb) = cur[j], cur[j + 1]
                w = max(wa, wb) + 1
                name = f"s{lvl}_{j // 2}"
                v.append(f"  reg [{w - 1}:0] {name};")
                v.append(f"  always @(posedge clk) {name} <= {{{w - wa}'d0, {a}}} + {{{w - wb}'d0, {b}}};")
            else:
                (a, wa) = cur[j]
                w = wa
                name = f"s{lvl}_{j // 2}"
                v.append(f"  reg [{w - 1}:0] {name};")
                v.append(f"  always @(posedge clk) {name} <= {a};")
            nxt.append((name, w))
        cur = nxt
        lvl += 1
    s, w = cur[0]
    v.append(f"  wire [{SW - 1}:0] ssum = {s};" if w == SW else f"  wire [{SW - 1}:0] ssum = {{{SW - w}'d0, {s}}};")
    v.append("  always @(posedge clk) begin")
    v.append("    score  <= ssum;")
    v.append("    jammed <= (ssum >= K);")
    v.append("  end")
    v.append("endmodule\n")
    return "\n".join(v), LAT


def tb_verilog(B, SW, n):
    XW = NFEAT * B
    return f"""`timescale 1ns/1ps
// Testbench: streams vectors.hex one per clock (with a few idle bubbles in the second half),
// writes "score jammed" per out_valid to rtl_out.txt.
module tb_rf_forest;
  localparam N = {n};
  reg clk = 0, rst = 1, in_valid = 0;
  reg [{XW - 1}:0] x_in = 0;
  wire out_valid, jammed;
  wire [{SW - 1}:0] score;
  reg [{XW - 1}:0] vec [0:N-1];
  integer fo, i, got, cyc, first_in, first_out;
  rf_forest dut(.clk(clk), .rst(rst), .in_valid(in_valid), .x_in(x_in),
                .out_valid(out_valid), .jammed(jammed), .score(score));
  always #5 clk = ~clk;
  always @(posedge clk) begin
    cyc = cyc + 1;
    if (out_valid) begin
      if (got == 0) first_out = cyc;
      $fwrite(fo, "%0d %0d\\n", score, jammed);
      got = got + 1;
    end
  end
  initial begin
    $readmemh("vectors.hex", vec);
    fo = $fopen("rtl_out.txt", "w");
    got = 0; cyc = 0; first_in = -1; first_out = -1;
    repeat (3) @(negedge clk);
    rst = 0;
    i = 0;
    while (i < N) begin
      @(negedge clk);
      if (i >= N/2 && ($random % 7 == 0)) begin in_valid = 0; x_in = {{{XW}{{1'bx}}}}; end
      else begin
        in_valid = 1; x_in = vec[i];
        if (i == 0) first_in = cyc + 1;
        i = i + 1;
      end
    end
    @(negedge clk); in_valid = 0; x_in = 0;
    repeat (200) @(negedge clk);
    $display("TB_DONE got=%0d expected=%0d latency_cycles=%0d", got, N, first_out - first_in);
    $fclose(fo);
    $finish;
  end
endmodule
"""


# ---------------------------------------------------------------------------------------------
def load_vectors(path):
    if path is None:
        return None
    if path.endswith(".npz"):
        d = np.load(path)
        return np.asarray(d["X"], dtype=np.float64)
    return np.asarray(np.load(path), dtype=np.float64)


def stress_vectors(forest_int, qz, n, rng):
    """Vectors whose features sit exactly at / next to integer split thresholds (q = T, T+1)
    plus the range extremes, to exercise every comparator boundary."""
    Ts = [[] for _ in range(NFEAT)]
    for t in forest_int:
        inner = ~t.is_leaf
        for f, T in zip(t.feat[inner], t.T[inner]):
            if 0 <= T < qz.levels:
                Ts[f].append(int(T))
    Q = rng.integers(0, qz.levels + 1, size=(n, NFEAT))
    for f in range(NFEAT):
        cand = np.array(Ts[f] + [0, qz.levels - 1]) if Ts[f] else np.array([0, qz.levels - 1])
        pick = cand[rng.integers(0, len(cand), n)] + rng.integers(0, 2, n)
        m = rng.random(n) < 0.8
        Q[m, f] = pick[m]
        Q[:, f] = np.clip(Q[:, f], 0, qz.levels)
    Q[:4] = [[0] * NFEAT, [qz.levels] * NFEAT, [0, qz.levels] * 2 + [0], [qz.levels, 0] * 2 + [qz.levels]]
    X = Q * qz.scale + qz.lo          # de-quantised float inputs (q(X) == Q exactly up to fp)
    return X


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--quant", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bits", type=int, default=None)
    ap.add_argument("--leaf-bits", type=int, default=12)
    ap.add_argument("--thresh", type=float, default=None)
    ap.add_argument("--thr-mode", choices=["floor", "round"], default="floor")
    ap.add_argument("--vectors", default=None, help=".npz with X (and optional y) or .npy of raw float features")
    ap.add_argument("--n-real", type=int, default=3000)
    ap.add_argument("--n-stress", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rom-style", choices=["auto", "block", "distributed"], default="auto",
                    help="auto: let the synthesiser pick BRAM/LUT-ROM per level (default)")
    a = ap.parse_args()

    rf = pickle.load(open(a.model, "rb"))
    if hasattr(rf, "steps"):          # sklearn Pipeline: take the last step
        rf = rf.steps[-1][1]
    assert hasattr(rf, "estimators_"), "model must be a fitted RandomForestClassifier"
    assert len(rf.classes_) == 2, "binary classifier required"
    assert rf.n_features_in_ == NFEAT, f"expected {NFEAT} features, got {rf.n_features_in_}"
    cls1 = 1
    qj = json.load(open(a.quant))
    B = int(a.bits if a.bits is not None else qj.get("bits", 16))
    P = int(a.leaf_bits)
    assert 1 <= P <= B, "leaf bits must be <= feature bits (leaf value is stored in the threshold field)"
    thresh = float(a.thresh if a.thresh is not None else qj.get("thresh", 0.60))
    qz = Quant(qj["lo"], qj["hi"], B)
    os.makedirs(a.out, exist_ok=True)
    for fn in os.listdir(a.out):       # remove stale ROM images from a previous (bigger) model
        if fn.endswith(".mem"):
            os.remove(os.path.join(a.out, fn))

    trees = [IntTree(e, qz, a.thr_mode, P, cls1) for e in rf.estimators_]
    layouts = [t.layout() for t in trees]
    N = len(trees)
    D = max(len(l) - 1 for l in layouts)
    PMAX = (1 << P) - 1
    K = math.ceil(Fraction(repr(thresh)) * N * PMAX)
    SW = (N * PMAX).bit_length()

    mods, tstats = [], []
    for k, lay in enumerate(layouts):
        txt, st = tree_verilog(k, lay, B, P, D, a.out, a.rom_style)
        mods.append(txt)
        tstats.append(st)
    top, LAT = top_verilog(N, B, P, D, K, SW)
    with open(os.path.join(a.out, "rf_forest.v"), "w") as fh:
        fh.write("// Generated by gen_forest_hdl.py -- do not edit\n")
        fh.write(top + "\n" + "\n".join(mods))

    # ----- vectors
    rng = np.random.default_rng(a.seed)
    Xr = load_vectors(a.vectors)
    parts, kinds = [], []
    if Xr is not None and a.n_real > 0:
        sel = rng.choice(len(Xr), min(a.n_real, len(Xr)), replace=False)
        parts.append(Xr[np.sort(sel)])
        kinds.append(np.zeros(len(sel), dtype=np.int8))
    if a.n_stress > 0 or not parts:
        ns = max(a.n_stress, 2000 if not parts else 0)
        parts.append(stress_vectors(trees, qz, ns, rng))
        kinds.append(np.ones(ns, dtype=np.int8))
    X = np.concatenate(parts)
    kind = np.concatenate(kinds)        # 0 = real data, 1 = synthetic boundary stress
    Q = qz.q(X)

    # ----- integer reference model (walks the original trees), and layout emulation cross-check
    V = np.stack([t.predict(Q) for t in trees], axis=1)
    S = V.sum(axis=1)
    dec_int = S >= K
    Ve = np.stack([emulate_layout(l, Q) for l in layouts], axis=1)
    assert np.array_equal(V, Ve), "internal error: ROM layout emulation != integer reference model"
    # ----- float model
    pf = rf.predict_proba(X)[:, cls1]
    dec_f = pf >= thresh

    XW = NFEAT * B
    with open(os.path.join(a.out, "vectors.hex"), "w") as fh:
        for row in Q:
            w = 0
            for f in range(NFEAT):
                w |= int(row[f]) << (f * B)
            fh.write(f"{w:0{(XW + 3) // 4}x}\n")
    with open(os.path.join(a.out, "ref_int.txt"), "w") as fh:
        for s_, d_ in zip(S, dec_int):
            fh.write(f"{int(s_)} {int(d_)}\n")
    np.savez(os.path.join(a.out, "vectors.npz"), X=X, Q=Q, kind=kind, score_int=S, dec_int=dec_int,
             prob_float=pf, dec_float=dec_f)
    with open(os.path.join(a.out, "tb_rf_forest.v"), "w") as fh:
        fh.write(tb_verilog(B, SW, len(Q)))

    def agree(m):
        return float((dec_int[m] == dec_f[m]).mean()) if m.any() else None
    meta = dict(trees=N, depth=D, bits=B, leaf_bits=P, thresh=thresh, K=K, score_width=SW,
                thr_mode=a.thr_mode, rom_style=a.rom_style, latency_cycles=LAT, n_vectors=int(len(Q)),
                n_real=int((kind == 0).sum()), n_stress=int((kind == 1).sum()),
                int_vs_float_agreement=agree(np.ones(len(Q), bool)),
                int_vs_float_agreement_real=agree(kind == 0),
                int_vs_float_agreement_stress=agree(kind == 1),
                prob_abs_err_max=float(np.abs(S / (N * PMAX) - pf).max()),
                prob_abs_err_mean=float(np.abs(S / (N * PMAX) - pf).mean()),
                nodes_sklearn=int(sum(e.tree_.node_count for e in rf.estimators_)),
                nodes_hw=int(sum(s["nodes"] for s in tstats)),
                rom_bits=int(sum(s["rom_bits"] for s in tstats)),
                rom_bits_padded=int(sum(s["rom_bits_padded"] for s in tstats)),
                per_tree=tstats, lo=qz.lo.tolist(), hi=qz.hi.tolist(),
                model=os.path.abspath(a.model), quant=os.path.abspath(a.quant))
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k not in ("per_tree", "lo", "hi")}, indent=1))


if __name__ == "__main__":
    main()
