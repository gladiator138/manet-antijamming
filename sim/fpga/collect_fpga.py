#!/usr/bin/env python3
"""Compare RTL simulation output with the integer reference model and assemble fpga.json.

    python3 collect_fpga.py --build BUILD_DIR --primary block [--variants block auto] --result fpga.json

BUILD_DIR/<variant>/ holds meta.json, ref_int.txt, rtl_out_<sim>.txt, sim_<sim>.log,
synth_report.json produced by run_fpga.sh.
"""
import argparse, datetime, glob, hashlib, json, os, re, sys, time

# Artix-7 XC7A200T.  Logic cells (215,360), DSP slices (740) and block RAM (13,140 Kb) were
# confirmed on amd.com's Artix 7 product table (fetched 2026-09-30); LUT/FF counts follow from
# 33,650 slices x 4 LUT6 / x 8 FF (7-series DS180 product table; logic cells = 1.6 x LUTs).
DEVICE = dict(name="XC7A200T (Artix-7)", luts=134600, ffs=269200, slices=33650, bram36=365, bram18=730,
              bram_kb=13140, dsps=740, max_distributed_ram_kb=2888, logic_cells=215360,
              source="AMD Artix 7 product table (amd.com/en/products/adaptive-socs-and-fpgas/fpga/artix-7.html) "
                     "for logic cells/DSP/BRAM Kb (confirmed 2026-09-30); slices/LUT/FF/distributed-RAM from the DS180 7-Series "
                     "product table (not re-fetched: docs.amd.com unreachable from this sandbox; consistent with 215,360 LCs = 1.6 x 134,600 LUTs)")


def compare(ref, out):
    r = open(ref).read().split("\n")
    o = open(out).read().split("\n") if os.path.exists(out) else []
    r = [x for x in r if x.strip()]
    o = [x for x in o if x.strip()]
    same = sum(1 for a, b in zip(r, o) if a.split() == b.split())
    return dict(n_ref=len(r), n_rtl=len(o), n_equal=same, match=(same == len(r) == len(o)),
                match_fraction=same / len(r) if r else 0.0)


def sim_latency(log):
    if not os.path.exists(log):
        return None
    m = re.search(r"latency_cycles=(-?\d+)", open(log).read())
    return int(m.group(1)) if m else None


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True)
    ap.add_argument("--primary", default="block")
    ap.add_argument("--variants", nargs="+", default=["block", "auto"])
    ap.add_argument("--result", required=True)
    ap.add_argument("--yosys-version", default="")
    ap.add_argument("--command", default="")
    a = ap.parse_args()

    pdir = os.path.join(a.build, a.primary)
    meta = json.load(open(os.path.join(pdir, "meta.json")))
    sims = {}
    for out in sorted(glob.glob(os.path.join(pdir, "rtl_out_*.txt"))):
        sim = os.path.basename(out)[len("rtl_out_"):-4]
        c = compare(os.path.join(pdir, "ref_int.txt"), out)
        c["latency_measured"] = sim_latency(os.path.join(pdir, f"sim_{sim}.log"))
        sims[sim] = c
    rtl_match = bool(sims) and all(s["match"] for s in sims.values())
    lat_ok = all(s["latency_measured"] == meta["latency_cycles"] for s in sims.values())

    variants = {}
    for v in a.variants:
        f = os.path.join(a.build, v, "synth_report.json")
        for _ in range(60):            # the shared (rclone) mount can lag a few seconds behind the writer
            if os.path.exists(f):
                break
            time.sleep(2)
        if os.path.exists(f):
            variants[v] = json.load(open(f))
        else:
            print(f"WARNING: no synthesis report for variant {v}", file=sys.stderr)
    if a.primary not in variants:
        sys.exit("primary synthesis report missing")
    R = variants[a.primary]["resources"]
    T = variants[a.primary]["timing_estimate"]

    def util(r):
        return dict(luts_pct=round(100.0 * r["luts"] / DEVICE["luts"], 2),
                    ffs_pct=round(100.0 * r["ffs"] / DEVICE["ffs"], 2),
                    bram_pct=round(100.0 * r["bram36_equiv"] / DEVICE["bram36"], 2),
                    dsps_pct=round(100.0 * r["dsps"] / DEVICE["dsps"], 2))

    if T.get("available"):
        fmax_note = ("Fmax not measured (no vendor place-and-route/STA). ESTIMATE only, from a longest-path search "
                     "over the Yosys-mapped netlist with Yosys' xc7 cell delays (Project X-Ray, Artix-7 -1): "
                     f"logic-only (zero wire delay) period {T['logic_only']['period_ns']} ns "
                     f"(<= {T['logic_only']['fmax_mhz']} MHz, optimistic bound); with an assumed "
                     f"{T['route_ps_per_hop']} ps per net hop {T['with_routing']['period_ns']} ns "
                     f"(~{T['with_routing']['fmax_mhz']} MHz). Critical path: "
                     + " -> ".join(T["with_routing"]["path"]))
    else:
        fmax_note = "Fmax not measured"

    res = dict(
        generated=datetime.datetime.now().isoformat(timespec="seconds"),
        model=meta["model"], model_sha256_16=sha(meta["model"]), quant=meta["quant"],
        trees=meta["trees"], depth=meta["depth"], bits=meta["bits"], leaf_bits=meta["leaf_bits"],
        thresh=meta["thresh"], score_threshold_K=meta["K"], thr_mode=meta["thr_mode"],
        latency_cycles=meta["latency_cycles"], latency_measured_in_sim=lat_ok,
        throughput="1 window / clock",
        n_vectors=meta["n_vectors"], n_vectors_real=meta["n_real"], n_vectors_stress=meta["n_stress"],
        rtl_vs_int_match=rtl_match,
        rtl_vs_int=sims,
        int_vs_float_agreement=meta["int_vs_float_agreement"],
        int_vs_float_agreement_real=meta["int_vs_float_agreement_real"],
        int_vs_float_agreement_stress=meta["int_vs_float_agreement_stress"],
        nodes=meta["nodes_hw"], rom_bits=meta["rom_bits"],
        rom_style=a.primary,
        luts=R["luts"], lut_logic=R["lut_logic"], ffs=R["ffs"], bram36_equiv=R["bram36_equiv"],
        ramb36=R["ramb36"], ramb18=R["ramb18"], lutram=R["lutram"], lutram_as_srl=R["lutram_as_srl"],
        dsps=R["dsps"], carry4=R["carry4"], muxf7=R["muxf7"], muxf8=R["muxf8"],
        utilization=util(R),
        device=DEVICE,
        fmax_note=fmax_note,
        timing_estimate=T,
        variants={v: dict(resources={k: x for k, x in r["resources"].items() if k != "cell_types"},
                          utilization=util(r["resources"]),
                          fmax_est_logic_only_mhz=r["timing_estimate"].get("logic_only", {}).get("fmax_mhz"),
                          fmax_est_with_routing_mhz=r["timing_estimate"].get("with_routing", {}).get("fmax_mhz"))
                  for v, r in variants.items()},
        tools=dict(yosys=a.yosys_version, synth="synth_xilinx -family xc7 -top rf_forest -flatten",
                   simulators=sorted(sims)),
        command=a.command,
        caveats=["Yosys synth_xilinx resource counts, not Vivado; Vivado results typically differ (often fewer LUTs).",
                 "I/O buffers (IBUF/OBUF) excluded from LUT/FF counts; the core is meant to be embedded.",
                 "Inputs are already-quantised B-bit features; the float->fixed conversion is upstream.",
                 "Fmax is an estimate (see fmax_note), not a place-and-route result."],
    )
    os.makedirs(os.path.dirname(os.path.abspath(a.result)), exist_ok=True)
    json.dump(res, open(a.result, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("trees", "depth", "bits", "latency_cycles", "n_vectors", "rtl_vs_int_match",
                                          "int_vs_float_agreement", "luts", "ffs", "bram36_equiv", "lutram", "dsps",
                                          "carry4", "utilization", "fmax_note")}, indent=1))
    if not rtl_match:
        sys.exit(2)


if __name__ == "__main__":
    main()
