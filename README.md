# Cause- and severity-aware frequency change against jamming in tactical MANETs

Simulator, experiments and build scripts for the paper "Cause- and Severity-Aware Frequency Change against Jamming in
Tactical MANETs" (S. Singh, A. Madhukar, S. Sangwan). Every number, table and figure in the paper is generated from
`results/*.json` by the scripts below. `results/rf.pkl` (the 120-tree detector, 63 MB) is not included; regenerate it
with `python3 run_experiments.py detector`. Building the PDF also needs Springer's `llncs.cls` in
`springer_template/llncs/`.

Reproduce everything (Python 3.11, numpy, scipy, scikit-learn, matplotlib; pdflatex for the paper;
yowasp-yosys and iverilog for the FPGA flow):

    cd sim
    python3 run_experiments.py detector    # detector.json, rf.pkl (RF, logistic regression, consistency check, thresholds)
    python3 run_experiments.py detpoints   # detpoints.json (RF metrics at thresholds 0.5-0.8)
    python3 run_experiments.py routing     # routing.json (7 jammer powers x 10 seeds, all schemes)
    python3 run_experiments.py follow      # follow.json (follow time F, gate with true and with assumed F)
    python3 run_experiments.py tau         # tau.json (retune time tau = 5..40 slots)
    python3 run_experiments.py robust      # robust.json (7 unseen deployments x 5 seeds)
    python3 run_experiments.py cause       # cause.json, cause.pkl (jammer-type classifier; run before routing/jammers/robust)
    python3 run_experiments.py jammers     # jammers.json (spot, pulsed, sweep, two jammers)
    python3 run_experiments.py probe       # probe.json (probe every 1, 2, 5, 10 slots)
    python3 run_experiments.py compact     # compact.json (forest size vs quality)
    python3 run_experiments.py fixedpoint  # fixedpoint.json, rf_compact.pkl (16/12/8-bit quantisation)
    cd fpga && cp ../../results/quant.json ../../results/fpga_quant.json
    bash run_fpga.sh ../../results/rf_compact.pkl ../../results/fpga_quant.json ../../results/fpga_vectors.npy
    cd .. && python3 make_figures.py && python3 build_paper.py   # figures and the LaTeX paper; every number comes from results/

Model (v6): 40 nodes, RWP with 300 s warm-up, 10 ms slots, 10 flows x 1024 B every 10 slots (0.82 Mbit/s on an
11 Mbit/s radio), hidden-terminal co-channel interference (rho = 0.02, carrier-sense range 300 m), Rayleigh fading,
logistic PER; label E[PER] averaged over signal and jammer fading; BER feature from the measured (noisy) SINR; the
detector monitors every link with a full window. Trigger: alarm (RF at theta = 0.8, loss threshold or consistency
check) on >= 8 links for 3 consecutive slots; severity gate (Proposition 1 with jammer on/off times estimated online,
learned post-change PDR) applied as a sequential Bayesian test (P(p_J < p*) >= 0.95 over the current alarm episode,
at most 40 slots; episode prior 40 slots; chosen on design seeds 1-3). Control plane ('lossy'): alarms heard one slot
late per connected component, the order flooded hop by hop over the jammed links (3 relay rounds per slot, switch 3
slots ahead); nodes not reached are stranded and rejoin after 5 slots. The jammer follows F slots after the network
resumes on the new channel (v5 counted F from the change; fixed in v6). Optional cause classifier (sim/cause.py).
Seeds: detector train 1000-1013, test 2000-2006; trigger design 1-3; network results 101-110;
robustness and jammer types 301-305. Run with absolute paths if the shared mount loses the working directory.
