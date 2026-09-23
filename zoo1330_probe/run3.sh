set -u; mkdir -p probe_out3; T=unsloth_zoo/mlx/cce/runtime_cce.py; cp $T probe_out3/orig.py
for arm in base head revert; do cp zoo1330_probe/arms/$arm.py $T; echo "=== ARM $arm"
python -m pytest tests/test_mlx_runtime_cce_compile.py -q -p no:cacheprovider -rA -k "finite_logits_past or saturated_softcap" 2>&1 | grep -E "PASSED|FAILED|assert|passed|failed" | cut -c1-220
python -m pytest tests/test_mlx_runtime_cce_compile.py -q -p no:cacheprovider 2>&1 | tail -1; done
cp probe_out3/orig.py $T
