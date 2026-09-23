import glob
import json
import os
import statistics
import sys
import xml.etree.ElementTree as ET

out = sys.argv[1]
arms = ["base", "head", "revert", "sentinel"]
P = {a: json.load(open(f"{out}/probe_{a}.json")) for a in arms if os.path.exists(f"{out}/probe_{a}.json")}
print("arms present:", list(P), "| mlx", next(iter(P.values()))["mlx"], "| device", next(iter(P.values()))["device"])

print("\n## pytest per arm (tests, failures, errors, skipped)")
for f in sorted(glob.glob(f"{out}/pytest_*.xml")):
    r = ET.parse(f).getroot()
    s = r if r.tag == "testsuite" else r.find("testsuite")
    failed = [c.get("name") for c in s.iter("testcase") if c.find("failure") is not None or c.find("error") is not None]
    print(os.path.basename(f), s.get("tests"), s.get("failures"), s.get("errors"), s.get("skipped"), failed[:12])

print("\n## tanh table (Metal, fp32)")
for row in P["head"]["tanh_table"]:
    print(row)

print("\n## saturation scenarios: arm -> loss_finite/grad_finite/loss_err/grad_rel_err")
keys = [(r["scenario"], r["dtype"], r["path"]) for r in P["head"]["scenarios"]]
for i, key in enumerate(keys):
    cells = []
    for a in arms:
        if a not in P:
            continue
        r = P[a]["scenarios"][i]
        if "error" in r:
            cells.append(f"{a}:ERR {r['error'][:60]}")
            continue
        c = f"{a}:L={'ok' if r['loss_finite'] else 'NaN%d' % r['loss_nan_rows']}"
        c += f",G={'ok' if r['grad_finite'] else 'NaN%d' % r['grad_nan']}"
        if "loss_max_abs_err" in r:
            c += f",lerr={r['loss_max_abs_err']:.3g}"
        if "grad_rel_err" in r:
            c += f",gerr={r['grad_rel_err']:.3g}"
        c += f",kern={r.get('metal_kernels_used')},ref_finite={r['ref_finite']}"
        cells.append(c)
    print(key, " | ".join(cells))

print("\n## bit parity base vs head on ordinary inputs")
same = diff = 0
for rb, rh in zip(P["base"]["parity"], P["head"]["parity"]):
    ident = rb.get("loss_hash") == rh.get("loss_hash") and rb.get("grad_hash") == rh.get("grad_hash") and "error" not in rb
    same += ident
    diff += not ident
    if not ident:
        print("DIFF", {k: rh.get(k) for k in ("seed", "dtype", "softcap", "path", "max_abs_ratio")}, rb.get("error"), rh.get("error"))
print(f"parity cells: {same + diff} total, {same} bit-identical, {diff} differ")
ratios = [r["max_abs_ratio"] for r in P["head"]["parity"] if r.get("max_abs_ratio")]
print("max |ratio| seen in parity cells:", max(ratios) if ratios else None, "; cells with |ratio|>20:",
      sum(1 for x in ratios if x > 20))

if "sentinel" in P:
    import math
    print("\n## sentinel arm (helper returns 0.5): forward/backward must move for softcap>0 only")
    moved = still = 0
    for rh, rs in zip(P["head"]["parity"], P["sentinel"]["parity"]):
        if rh["softcap"] == 0.0:
            assert rh.get("loss_hash") == rs.get("loss_hash"), ("softcap=0 moved under sentinel", rh)
            continue
        m = rh.get("loss_hash") != rs.get("loss_hash") and rh.get("grad_hash") != rs.get("grad_hash")
        moved += m
        still += not m
        if not m:
            print("SENTINEL DID NOT MOVE", rh)
    print(f"softcap>0 cells moved: {moved}, unmoved: {still}; softcap=0 cells unchanged (asserted)")
    ls = [r for r in P["sentinel"]["scenarios"] if r["path"] == "dense_label_smoothing"]
    lh = [r for r in P["head"]["scenarios"] if r["path"] == "dense_label_smoothing"]
    print("label-smoothing path unaffected by helper:",
          all(a.get("loss_hash") == b.get("loss_hash") for a, b in zip(ls, lh)))

print("\n## cost: fwd+bwd ms, median of per-process medians (3 processes x 15 reps)")
B = {}
for f in glob.glob(f"{out}/bench_*_r*.json"):
    d = json.load(open(f))
    for b in d.get("bench", []):
        B.setdefault((d["arm"], b["dtype"], b["softcap"]), []).append(b["median_ms"])
for key in sorted({(k[1], k[2]) for k in B}):
    bs, hs = B.get(("base",) + key, []), B.get(("head",) + key, [])
    if bs and hs:
        mb, mh = statistics.median(bs), statistics.median(hs)
        print(key, f"base {mb:.2f} ms {sorted(round(x, 2) for x in bs)} | head {mh:.2f} ms {sorted(round(x, 2) for x in hs)}"
              f" | delta {100 * (mh - mb) / mb:+.2f}%")
