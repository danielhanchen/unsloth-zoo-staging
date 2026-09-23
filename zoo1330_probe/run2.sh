set -u; mkdir -p probe_out2
for r in 1 2; do for v in fast pr_ternary clamp_arg precise; do python -u zoo1330_probe/probe2.py probe_out2/p2_${v}.json $v > probe_out2/log_${v}_r$r.txt 2>&1 || tail -20 probe_out2/log_${v}_r$r.txt; python -c "import json;d=json.load(open('probe_out2/p2_${v}.json'));print('$v r$r', d['cost_ms_softcap30_bf16'])"; done; done
python zoo1330_probe/compare2.py probe_out2 | tee probe_out2/COMPARE.txt
