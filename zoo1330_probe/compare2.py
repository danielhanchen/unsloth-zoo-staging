import json, sys, numpy as np
d = sys.argv[1]; V = ["fast", "pr_ternary", "clamp_arg", "precise"]
J = {v: json.load(open(f"{d}/p2_{v}.json")) for v in V}
Z = {v: np.load(f"{d}/p2_{v}.npz") for v in V}
n = len([k for k in Z["fast"].files if k.startswith("L")])
print("elementwise (2,000,001-pt grid on [-50,50] + edges):")
for v in V: print(" ", v, {k: J[v]["elementwise"][v][k] for k in ("nonfinite_outputs_on_finite_inputs", "max_abs_err_vs_f64", "bitdiff_vs_fast_on_|x|<=20")})
print("end-to-end,", n, "cells (4 scales x 3 dtypes x softcap{9,30} x dense/quant/quant-frozen):")
for v in V:
    same = lerr = gerr = ld = gd = 0; le = []; ge = []
    for i in range(n):
        L, G, Lb, Gb, RL, RG = (Z[v][f"L{i}"], Z[v][f"G{i}"], Z["fast"][f"L{i}"], Z["fast"][f"G{i}"], Z[v][f"RL{i}"], Z[v][f"RG{i}"])
        gs = max(1e-30, np.abs(RG).max())
        same += np.array_equal(L, Lb) and np.array_equal(G, Gb)
        ld = max(ld, np.abs(L - Lb).max()); gd = max(gd, np.abs(G - Gb).max() / gs)
        le.append(np.abs(L - RL).max()); ge.append(np.abs(G - RG).max() / gs)
    print(f"  {v:11s} bit-identical to fast: {same}/{n} | max|dLoss| vs fast {ld:.3g} | max rel dGrad vs fast {gd:.3g} |"
          f" err vs f64: loss median {np.median(le):.3g} max {max(le):.3g}, grad-rel median {np.median(ge):.3g} max {max(ge):.3g} |"
          f" cost {J[v]['cost_ms_softcap30_bf16'][v]:.1f} ms")
