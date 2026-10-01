# Random-forest FPGA flow (generate -> verify -> synthesize)

Turns a fitted binary scikit-learn `RandomForestClassifier` (5 features) into pipelined
Verilog-2001, proves the RTL bit-exact against a Python integer model, and synthesizes it with
Yosys `synth_xilinx -family xc7`. Results go to `../../results/fpga.json`.

Nothing in `../` (the simulator) is modified. The scripts only import `run_experiments.py` to make data.

## Quick start (production model)

```bash
cd sim/fpga
# 1. quant.json with the same lo/hi as run_experiments.fixedpoint() (0.01/99.99th pct of make_data(TRAIN_SEEDS)),
#    plus held-out test vectors (all 7 TEST_SEEDS, i.e. every jammer power); 2 worker processes.
python3 make_quant.py --out /path/quant.json --vectors /path/test_vectors.npz
# 2. full flow (pickle must be a fitted RandomForestClassifier, or a Pipeline ending in one)
bash run_fpga.sh /path/forest.pkl /path/quant.json /path/test_vectors.npz
```

Use `bash run_fpga.sh`, not `./run_fpga.sh`: the shared mount is `noexec`.
A 10-tree depth-12 forest with ~20k nodes takes about 4-6 min in total, using at most 2 cores:
two Yosys runs of about 1.5-2 min each, plus Icarus simulation of 4000 vectors (about 1.7 min),
which runs at the same time as synthesis.
The production 10-tree forest has to be pickled first. `run_experiments.fixedpoint()`/`compact()`
train it but do not save it. `results/rf.pkl` is the 120-tree model, which the flow also accepts,
but it needs about 12x the resources and synthesis time.

`quant.json` format: `{"lo": [5 floats], "hi": [5 floats], "bits": 16, "thresh": 0.6}`
(`thresh` is optional and defaults to 0.60 = `THRESH`).

Environment knobs for `run_fpga.sh`:

| var | default | meaning |
|---|---|---|
| `RESULT` | `../../results/fpga.json` | output json |
| `BUILD` | `build/<model name>` | generated HDL, logs, reports |
| `BITS` | quant.json `bits` | feature/threshold width B |
| `LEAF_BITS` | 12 | leaf probability width P |
| `THR_MODE` | `floor` | `floor`: T=floor((thr-lo)/s). `round`: T=round(...), the same as `fixedpoint()` |
| `ROM_STYLES` | `block auto` | synthesized variants. The first one is the primary result. `block` forces node ROMs into BRAM; `auto` lets Yosys choose (it mostly builds LUT-ROMs) |
| `SIM` | `auto` | `iverilog`, `cxxrtl` (Yosys CXXRTL + g++), `both`, or `auto` (iverilog if installed) |
| `N_REAL`/`N_STRESS` | 3000/1000 | real vectors sampled from the vectors file, plus synthetic boundary-stress vectors |

Tools: `python3 -m pip install yowasp-yosys` (the script installs it if it is missing), plus
`apt-get install iverilog` (optional; without it the CXXRTL fallback is used, which needs g++).

## Files

| file | purpose |
|---|---|
| `gen_forest_hdl.py` | model -> `rf_forest.v` + `t<k>_l<L>.mem` ROM images + testbench + vectors + integer reference outputs (`ref_int.txt`) + `meta.json` |
| `cxxrtl_tb.cc` | C++ testbench for the CXXRTL simulator fallback |
| `synth_report.py` | resource counts and a timing *estimate* from the mapped Yosys netlist |
| `collect_fpga.py` | compares RTL output with the integer model and writes `fpga.json` |
| `run_fpga.sh` | the driver |
| `make_quant.py` | quant.json and test vectors for the production model |
| `make_dev_model.py` | development forest (10 trees, depth 12, `make_data(TRAIN_SEEDS[:3], slots=600)`) -> `dev/` |

Generator CLI: `python3 gen_forest_hdl.py --model forest.pkl --quant quant.json --out DIR [--bits 16]
[--leaf-bits 12] [--thr-mode floor|round] [--vectors X.npz] [--rom-style auto|block|distributed]`.

## Integer semantics (bit-exact between Python and RTL)

* `scale_f = (hi_f - lo_f)/(2^B - 1)` and `q_f = clip(round((x_f - lo_f)/scale_f), 0, 2^B-1)`.
  Rounding is numpy's half-to-even. This conversion runs upstream of the core, in the
  feature extractor; the core takes `q`.
* A split `x_f <= thr` becomes `q_f <= T` with `T = floor((thr - lo_f)/scale_f)`, stored as a
  B-bit unsigned value. If `T < 0` the node always goes right, and if `T >= 2^B-1` it always goes
  left. Those nodes are removed at generation time, which gives identical decisions.
* Leaf value: `v = round(p1 * (2^P - 1))`, where `p1` is the leaf's class-1 fraction.
  The score is `S = sum_t v_t` and `jammed = S >= K`, where `K = ceil(0.60 * n_trees * (2^P-1))`.
  That is the same as mean quantized probability >= 0.60.
* The reference model (`IntTree.predict`) walks the **original** sklearn trees with Python ints.
  It is independent of the ROM layout. A second emulation of the ROM tables is asserted equal to it.

## Architecture

* One pipeline stage per tree level, and all trees run in parallel.
* Level L of each tree is a ROM of records `{leaf, feat[2:0], thr/value[B-1:0], left_child_idx}`.
  The right child is `left+1`.
* The ROM address is formed combinationally from the previous level's record and the feature
  compare. The ROM data is registered, so this is a synchronous read that can be inferred as
  BRAM, and each level costs one clock.
* A leaf reached above the maximum depth sets `done`, and its value is carried through the
  remaining stages.
* Trees shallower than the forest depth get delay registers.
* After the trees come a registered pairwise adder tree and a registered compare.
* Latency is `D + 3 + ceil(log2 n_trees)` clocks, which is 19 for 10 trees at depth 12.
  Throughput is 1 window per clock, and the testbench checks both.
* Ports: `clk, rst, in_valid, x_in[5B-1:0]` (feature f at `[f*B +: B]`) and `out_valid, jammed, score`.

## Verification

`run_fpga.sh` simulates `rf_forest.v` on all vectors: `N_REAL` real held-out windows plus
`N_STRESS` synthetic vectors placed exactly on and next to every integer split threshold and at
the range ends. The testbench streams one vector per clock and inserts idle bubbles in the
second half.

`fpga.json` records:

* `rtl_vs_int_match`: the RTL output must equal the integer model for every vector.
* `int_vs_float_agreement`: agreement of the jammed decision with the float sklearn forest
  (`predict_proba >= thresh`), reported for real and stress vectors separately.

## Synthesis and timing

Resources are counted from the flattened `synth_xilinx` netlist:

* LUTs include LUT-RAM and SRL.
* `bram36_equiv = RAMB36 + RAMB18/2`.
* IBUF/OBUF are excluded.

Fmax is **not measured**, because there is no vendor place-and-route. `synth_report.py` gives an
**estimate**: a longest-path search using the cell delays that ship with Yosys
(`share/xilinx/cells_sim.v`, which come from Project X-Ray Artix-7 -1 data). It reports two numbers:

* a zero-wire-delay bound, which is optimistic;
* a value with an assumed 450 ps per net hop.

Label these as estimates in the paper.

Yosys resource counts usually differ from Vivado's, often by 10-30% in LUTs. If numbers are
quoted as "Yosys synth_xilinx", say so.
