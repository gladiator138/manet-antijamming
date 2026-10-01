#!/usr/bin/env python3
"""Resource count + static-timing *estimate* for a Yosys synth_xilinx netlist.

    python3 synth_report.py --netlist net.json --top rf_forest [--cells-sim cells_sim.v]
                            [--route-ps 450] [--out synth_report.json]

Resources: counted from the (flattened) mapped netlist.
    LUTs   = LUT1..LUT6 + INV (Vivado implements INV as LUT1) + LUT-RAM/SRL LUTs
    lut_logic = LUT1..LUT6 + INV only
    LUTRAM = LUTs used as memory (RAM32M/RAM64M = 4, RAM128X1D = 4, RAM64X1D = 2, ... ; SRL16E/SRLC32E = 1)
    bram36_equiv = RAMB36E1 + RAMB18E1 / 2
Timing: a longest-path search over the mapped netlist using the cell delays that Yosys ships in
share/xilinx/cells_sim.v (specify blocks; Artix-7 -1 values derived from Project X-Ray SDF).
Two numbers are produced, both ESTIMATES, not vendor sign-off:
    logic-only   : zero interconnect delay -> an optimistic lower bound on the clock period
    with routing : + route_ps per net hop (an assumed average; real routed delay depends on
                   placement, fan-out and device utilisation)
I/O-pad paths (IBUF/OBUF) are ignored as the core would be embedded in a larger design.
"""
import argparse, json, os, re, sys
from collections import defaultdict

LUTRAM_LUTS = {"RAM32X1S": 1, "RAM64X1S": 1, "RAM128X1S": 2, "RAM256X1S": 4, "RAM32X1D": 2,
               "RAM64X1D": 2, "RAM128X1D": 4, "RAM256X1D": 8, "RAM32M": 4, "RAM64M": 4,
               "RAM32M16": 8, "RAM64M8": 8, "SRL16E": 1, "SRLC16E": 1, "SRLC32E": 1}
LUT_CELLS = {"LUT1", "LUT2", "LUT3", "LUT4", "LUT5", "LUT6", "INV", "LUT6_2"}
FF_CELLS = {"FDRE", "FDSE", "FDCE", "FDPE", "FDRE_1", "FDSE_1", "FDCE_1", "FDPE_1", "LDCE", "LDPE"}
IO_CELLS = {"IBUF", "OBUF", "IBUFG", "OBUFT", "IOBUF", "BUFG", "BUFGCTRL", "BUFH"}


def default_cells_sim():
    try:
        import yowasp_yosys
        p = os.path.join(os.path.dirname(yowasp_yosys.__file__), "share", "xilinx", "cells_sim.v")
        if os.path.exists(p):
            return p
    except ImportError:
        pass
    return None


def parse_pin(s):
    s = s.strip()
    m = re.match(r"^\\?([A-Za-z_][A-Za-z0-9_$]*)\s*(\[(\d+)\])?$", s)
    if not m:
        return None
    return m.group(1), (int(m.group(3)) if m.group(3) is not None else None)


def parse_specify(path):
    """Return {cell: dict(arcs=[(inpin,ibit,outpin,obit,ps)], clk2q={outpin:ps}, setup={inpin:ps})}"""
    lib = {}
    txt = open(path).read()
    for m in re.finditer(r"^\s*module\s+\\?(\$?[A-Za-z0-9_$]+)\s*[#(]?(.*?)^\s*endmodule", txt, re.S | re.M):
        name, body = m.group(1), m.group(2)
        arcs, clk2q, setup = [], defaultdict(int), defaultdict(int)
        for sp in re.findall(r"specify(.*?)endspecify", body, re.S):
            sp = re.sub(r"//[^\n]*", "", sp)
            for st in sp.split(";"):
                st = st.strip()
                if not st:
                    continue
                ms = re.match(r"^\$setup\s*\(\s*([^,]+),[^,]*,\s*(?:/\*.*?\*/)?\s*(-?\d+)\s*\)$", st, re.S)
                if ms:
                    p = parse_pin(ms.group(1))
                    if p:
                        setup[p[0]] = max(setup[p[0]], int(ms.group(2)))
                    continue
                me = re.search(r"\(\s*(?:posedge|negedge)\s+\w+\s*=>\s*\(\s*([^:\s]+)\s*:.*?\)\s*\)\s*=\s*([\d+ ]+)$", st, re.S)
                if me:
                    p = parse_pin(me.group(1))
                    if p:
                        clk2q[p[0]] = max(clk2q[p[0]], eval_sum(me.group(2)))
                    continue
                mc = re.search(r"\(\s*([^=()]+?)\s*=>\s*([^=()]+?)\s*\)\s*=\s*([^;]+)$", st, re.S)
                if mc:
                    a, b = parse_pin(mc.group(1)), parse_pin(mc.group(2))
                    d = eval_sum(mc.group(3))
                    if a and b and d is not None:
                        arcs.append((a[0], a[1], b[0], b[1], d))
        lib[name] = dict(arcs=arcs, clk2q=dict(clk2q), setup=dict(setup))
    return lib


def eval_sum(s):
    s = re.sub(r"/\*.*?\*/", "", s).strip()
    if not re.match(r"^[\d\s+]+$", s):
        return None
    return sum(int(x) for x in s.split("+") if x.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--netlist", required=True)
    ap.add_argument("--top", default="rf_forest")
    ap.add_argument("--cells-sim", default=default_cells_sim())
    ap.add_argument("--route-ps", type=int, default=450)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    nl = json.load(open(a.netlist))
    mod = nl["modules"][a.top]
    cells = mod["cells"]
    if any(c["type"] in nl["modules"] and not nl["modules"][c["type"]].get("attributes", {}).get("blackbox")
           for c in cells.values()):
        print("WARNING: netlist is hierarchical; counts cover the top module only -- synthesize with -flatten",
              file=sys.stderr)
    cnt = defaultdict(int)
    for c in cells.values():
        cnt[c["type"]] += 1
    lut_logic = sum(v for k, v in cnt.items() if k in LUT_CELLS)
    lutram = sum(v * LUTRAM_LUTS[k] for k, v in cnt.items() if k in LUTRAM_LUTS)
    lutram_srl = sum(v for k, v in cnt.items() if k.startswith("SRL"))
    ffs = sum(v for k, v in cnt.items() if k in FF_CELLS)
    res = dict(luts=lut_logic + lutram, lut_logic=lut_logic, lutram=lutram, lutram_as_srl=lutram_srl,
               lutram_as_rom_ram=lutram - lutram_srl, ffs=ffs,
               ramb36=cnt.get("RAMB36E1", 0), ramb18=cnt.get("RAMB18E1", 0),
               bram36_equiv=cnt.get("RAMB36E1", 0) + cnt.get("RAMB18E1", 0) / 2.0,
               dsps=cnt.get("DSP48E1", 0), carry4=cnt.get("CARRY4", 0),
               muxf7=cnt.get("MUXF7", 0), muxf8=cnt.get("MUXF8", 0),
               cell_types=dict(sorted(cnt.items())))

    # ------------------------------------------------------------------ timing estimate
    timing = dict(available=False)
    if a.cells_sim and os.path.exists(a.cells_sim):
        lib = parse_specify(a.cells_sim)
        arcs_out = defaultdict(list)      # net bit -> [(cell, out_bit, delay)]
        src = {}                          # net bit -> (clk2q, cell)
        sinks = []                        # (net bit, setup, cell)
        unknown = set()
        indeg = defaultdict(int)
        for cn, c in cells.items():
            t = c["type"]
            if t in IO_CELLS:
                continue
            L = lib.get(t)
            if L is None:
                unknown.add(t)
                continue
            conn, dirs = c["connections"], c["port_directions"]
            for p, bits in conn.items():
                if dirs.get(p) == "output" and p in L["clk2q"]:
                    for b in bits:
                        if isinstance(b, int):
                            src[b] = max(src.get(b, (0, cn))[0], L["clk2q"][p]), cn
                if dirs.get(p) == "input" and p in L["setup"]:
                    for b in bits:
                        if isinstance(b, int):
                            sinks.append((b, L["setup"][p], cn, p))
            for (ip, ib, op, ob, d) in L["arcs"]:
                if ip not in conn or op not in conn or dirs.get(op) != "output":
                    continue
                ibits = conn[ip] if ib is None else conn[ip][ib:ib + 1]
                obits = conn[op] if ob is None else conn[op][ob:ob + 1]
                for x in ibits:
                    if not isinstance(x, int):
                        continue
                    for y in obits:
                        if isinstance(y, int):
                            arcs_out[x].append((cn, y, d))
                            indeg[y] += 1
        # topological longest path, for both routing assumptions
        allnodes = set(arcs_out) | set(indeg) | set(src)
        results = {}
        for label, rp in (("logic_only", 0), ("with_routing", a.route_ps)):
            arr = {n: (src[n][0], None) for n in src}
            deg = dict(indeg)
            stack = [n for n in allnodes if deg.get(n, 0) == 0]
            order = []
            while stack:
                n = stack.pop()
                order.append(n)
                for (cn, y, d) in arcs_out.get(n, ()):
                    if n in arr:
                        t = arr[n][0] + rp + d
                        if y not in arr or t > arr[y][0]:
                            arr[y] = (t, (n, cn))
                    deg[y] -= 1
                    if deg[y] == 0:
                        stack.append(y)
            loop = len(order) < len(allnodes)
            best = (0, None, None, None)
            for (b, s, cn, p) in sinks:
                if b in arr:
                    t = arr[b][0] + rp + s
                    if t > best[0]:
                        best = (t, b, cn, p)
            # trace path
            path, n = [], best[1]
            while n is not None and arr.get(n, (0, None))[1] is not None:
                prev, cn = arr[n][1]
                path.append(cells[cn]["type"])
                n = prev
            if n is not None and n in src:
                path.append(cells[src[n][1]]["type"] + "(launch)")
            path.reverse()
            if best[2]:
                path.append(cells[best[2]]["type"] + "." + best[3] + "(capture)")
            results[label] = dict(period_ns=round(best[0] / 1000.0, 3),
                                  fmax_mhz=round(1e6 / best[0], 1) if best[0] > 0 else None,
                                  path=path, comb_cells_on_path=max(len(path) - 2, 0),
                                  lut_levels=sum(1 for x in path if x.startswith("LUT") or x == "INV"),
                                  graph_has_loop=loop)
        timing = dict(available=True, route_ps_per_hop=a.route_ps, cells_sim=a.cells_sim,
                      unknown_cell_types=sorted(unknown), **results)
    out = dict(resources=res, timing_estimate=timing)
    s = json.dumps(out, indent=1)
    if a.out:
        open(a.out, "w").write(s)
    print(s)


if __name__ == "__main__":
    main()
