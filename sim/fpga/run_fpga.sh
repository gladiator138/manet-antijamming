#!/usr/bin/env bash
# Random-Forest -> Verilog -> RTL simulation vs integer model -> Yosys synth_xilinx -> fpga.json
#
#   ./run_fpga.sh MODEL.pkl QUANT.json [VECTORS.npz|.npy]
#
# Environment knobs (all optional):
#   RESULT=...          output json   (default: <this dir>/../../results/fpga.json)
#   BUILD=...           build dir     (default: <this dir>/build/<model basename>)
#   BITS=16             feature bits  (default: quant.json "bits", else 16)
#   LEAF_BITS=12        leaf-probability bits
#   THR_MODE=floor      floor | round (round = what run_experiments.fixedpoint() emulates)
#   ROM_STYLES="block auto"   synthesised variants; the FIRST one is reported as primary
#   SIM=auto            auto | iverilog | cxxrtl | both   (auto: iverilog if installed, else cxxrtl)
#   N_REAL=3000 N_STRESS=1000   test vectors (real ones are sampled from VECTORS if given)
#   KEEP_NETLIST=0      1 keeps the (large) mapped netlists net.json
# Uses at most 2 CPU cores (Yosys/WASM and the simulators are single-threaded; synthesis and
# simulation run concurrently).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -L)"
[ $# -ge 2 ] || { sed -n '2,20p' "$0"; exit 1; }
MODEL="$(realpath -s "$1")"; QUANT="$(realpath -s "$2")"; VECTORS="${3:-}"
[ -n "$VECTORS" ] && VECTORS="$(realpath -s "$VECTORS")"
RESULT="${RESULT:-$HERE/../../results/fpga.json}"
NAME="$(basename "$MODEL" .pkl)"
BUILD="${BUILD:-$HERE/build/$NAME}"
LEAF_BITS="${LEAF_BITS:-12}"; THR_MODE="${THR_MODE:-floor}"
ROM_STYLES="${ROM_STYLES:-block auto}"; SIM="${SIM:-auto}"
N_REAL="${N_REAL:-3000}"; N_STRESS="${N_STRESS:-1000}"
PRIMARY="${ROM_STYLES%% *}"
log() { echo "[run_fpga $(date +%H:%M:%S)] $*"; }

# ---------------------------------------------------------------- tools
if ! command -v yowasp-yosys >/dev/null 2>&1; then
  log "installing yowasp-yosys (PyPI)"; python3 -m pip install -q yowasp-yosys
fi
YOSYS=yowasp-yosys
YVER="$($YOSYS -V 2>/dev/null | tail -1)"
if [ "$SIM" = auto ]; then
  if command -v iverilog >/dev/null 2>&1; then SIM=iverilog
  elif command -v g++ >/dev/null 2>&1; then SIM=cxxrtl
  else log "trying apt-get install iverilog"; (apt-get install -y iverilog >/dev/null 2>&1 || true)
       command -v iverilog >/dev/null 2>&1 && SIM=iverilog || { echo "no simulator (need iverilog or g++)"; exit 1; }
  fi
fi
case "$SIM" in both) SIMS="iverilog cxxrtl";; *) SIMS="$SIM";; esac
log "model=$MODEL quant=$QUANT vectors=${VECTORS:-<none: synthetic only>} sim=$SIMS yosys=$YVER"

# ---------------------------------------------------------------- generate (one dir per ROM style)
mkdir -p "$BUILD"
GEN_ARGS=(--model "$MODEL" --quant "$QUANT" --leaf-bits "$LEAF_BITS" --thr-mode "$THR_MODE"
          --n-real "$N_REAL" --n-stress "$N_STRESS")
[ -n "${BITS:-}" ] && GEN_ARGS+=(--bits "$BITS")
[ -n "$VECTORS" ] && GEN_ARGS+=(--vectors "$VECTORS")
for RS in $ROM_STYLES; do
  log "generate rom_style=$RS"
  mkdir -p "$BUILD/$RS"
  rm -f "$BUILD/$RS"/rtl_out_*.txt "$BUILD/$RS"/sim_*.log "$BUILD/$RS"/synth_report.json
  python3 "$HERE/gen_forest_hdl.py" "${GEN_ARGS[@]}" --rom-style "$RS" --out "$BUILD/$RS" > "$BUILD/$RS/gen.log" 2>&1 \
    || { cat "$BUILD/$RS/gen.log"; exit 1; }
done
XW=$(python3 -c "import json;print(5*json.load(open('$BUILD/$PRIMARY/meta.json'))['bits'])")

# ---------------------------------------------------------------- synthesis (background, 1 core)
synth() {
  local d="$1"
  ( cd "$d" && cat > synth.ys <<'EOF'
read_verilog rf_forest.v
synth_xilinx -family xc7 -top rf_forest -flatten
tee -q -o stat.txt stat
write_json net.json
EOF
    $YOSYS -q -l synth.log synth.ys > synth.stdout 2>&1
    python3 "$HERE/synth_report.py" --netlist net.json --top rf_forest --out synth_report.json > /dev/null
    gzip -f synth.log
    [ "${KEEP_NETLIST:-0}" = 1 ] || rm -f net.json )
}
(
  for RS in $ROM_STYLES; do log "synthesis rom_style=$RS (yosys synth_xilinx)"; synth "$BUILD/$RS"; log "synthesis $RS done"; done
) &
SYNTH_PID=$!
killtree() { local c; for c in $(pgrep -P "$1" 2>/dev/null); do killtree "$c"; done; kill "$1" 2>/dev/null || true; }
trap 'echo "run_fpga: failed, stopping synthesis" >&2; killtree $SYNTH_PID' ERR INT TERM

# ---------------------------------------------------------------- RTL simulation of the primary design
cd "$BUILD/$PRIMARY"
for S in $SIMS; do
  log "RTL simulation with $S"
  if [ "$S" = iverilog ]; then
    iverilog -g2001 -o sim.vvp rf_forest.v tb_rf_forest.v
    vvp -n sim.vvp > sim_iverilog.log
    mv rtl_out.txt rtl_out_iverilog.txt; rm -f sim.vvp
  else
    INC="$(python3 -c 'import yowasp_yosys,os;print(os.path.join(os.path.dirname(yowasp_yosys.__file__),"share","include","backends","cxxrtl","runtime"))')"
    CXTMP="$(mktemp -d /tmp/rf_cxxrtl.XXXXXX)"      # the shared mount is noexec: build/run binary in /tmp
    # (yowasp-yosys runs in a WASI sandbox that can only write below the current directory)
    $YOSYS -q -p "read_verilog rf_forest.v; hierarchy -top rf_forest; write_cxxrtl rf_forest_cxxrtl.cc" > /dev/null
    g++ -O1 -std=c++17 -DXW="$XW" -I"$INC" -I. -o "$CXTMP/tb_cxxrtl" "$HERE/cxxrtl_tb.cc"
    "$CXTMP/tb_cxxrtl" > sim_cxxrtl.log || true
    rm -rf "$CXTMP" rf_forest_cxxrtl.cc
    mv rtl_out.txt rtl_out_cxxrtl.txt
  fi
  tail -1 "sim_$S.log"
done
cd "$HERE"

wait $SYNTH_PID
trap - ERR INT TERM
log "collect -> $RESULT"
python3 "$HERE/collect_fpga.py" --build "$BUILD" --primary "$PRIMARY" --variants $ROM_STYLES --result "$RESULT" \
  --yosys-version "$YVER" --command "$0 $*  [ROM_STYLES='$ROM_STYLES' SIM=$SIM LEAF_BITS=$LEAF_BITS THR_MODE=$THR_MODE]"
log "done; artefacts in $BUILD"
