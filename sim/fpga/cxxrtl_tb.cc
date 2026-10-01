// CXXRTL testbench (fallback when Icarus Verilog is unavailable).
// Build: yowasp-yosys -p "read_verilog rf_forest.v; hierarchy -top rf_forest; write_cxxrtl rf_forest_cxxrtl.cc"
//        g++ -O1 -std=c++17 -DXW=<5*bits> -I<yowasp_yosys>/share/include/backends/cxxrtl/runtime cxxrtl_tb.cc
// Same stimulus/output format as tb_rf_forest.v: reads vectors.hex, writes rtl_out.txt ("score jammed").
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>
#include "rf_forest_cxxrtl.cc"

#ifndef XW
#error "define XW (= 5 * feature bits)"
#endif

int main() {
  cxxrtl_design::p_rf__forest top;
  std::ifstream in("vectors.hex");
  std::vector<std::string> vec;
  std::string s;
  while (in >> s) vec.push_back(s);
  const long N = (long)vec.size();
  FILE *fo = fopen("rtl_out.txt", "w");
  auto clock = [&]() {
    top.p_clk.set<bool>(false); top.step();
    top.p_clk.set<bool>(true);  top.step();
  };
  top.p_in__valid.set<bool>(false);
  top.p_rst.set<bool>(true);
  for (int k = 0; k < 3; k++) clock();
  top.p_rst.set<bool>(false);
  long i = 0, cyc = 0, got = 0, first_in = -1, first_out = -1;
  uint32_t lcg = 1;
  while (got < N && cyc < 2 * N + 1000) {
    bool bubble = false;
    if (i >= N / 2 && i < N) { lcg = lcg * 1103515245u + 12345u; bubble = ((lcg >> 16) % 7) == 0; }
    if (i < N && !bubble) {
      const std::string &h = vec[i];
      for (size_t c = 0; c < sizeof(top.p_x__in.data) / sizeof(top.p_x__in.data[0]); c++) {
        long end = (long)h.size() - 8 * (long)c;
        uint32_t w = 0;
        if (end > 0) {
          long beg = end - 8 < 0 ? 0 : end - 8;
          w = (uint32_t)std::stoul(h.substr(beg, end - beg), nullptr, 16);
        }
        top.p_x__in.data[c] = w;
      }
      top.p_in__valid.set<bool>(true);
      if (i == 0) first_in = cyc;
      i++;
    } else {
      top.p_in__valid.set<bool>(false);
    }
    // settle combinational logic with the new inputs, sample the (registered) outputs of the
    // current cycle, then apply the rising edge -- same sampling point as tb_rf_forest.v
    top.p_clk.set<bool>(false); top.step();
    if (top.p_out__valid.get<bool>()) {
      if (got == 0) first_out = cyc;
      fprintf(fo, "%u %u\n", (unsigned)top.p_score.get<uint32_t>(), (unsigned)top.p_jammed.get<bool>());
      got++;
    }
    top.p_clk.set<bool>(true); top.step();
    cyc++;
  }
  fclose(fo);
  printf("TB_DONE got=%ld expected=%ld latency_cycles=%ld\n", got, N, first_out - first_in);
  return got == N ? 0 : 1;
}
