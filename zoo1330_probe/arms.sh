#!/usr/bin/env bash
# Runs from the repo root of a checkout carrying the PR's tests. One fresh process per arm.
set -u
P=zoo1330_probe
OUT=probe_out
mkdir -p "$OUT"
TARGET=unsloth_zoo/mlx/cce/runtime_cce.py
cp "$TARGET" "$OUT/checkout_runtime_cce.py"
python -c "import mlx.core as mx; print('mlx', mx.__version__, 'metal', mx.metal.is_available(), mx.metal.device_info())"

for arm in base head revert sentinel; do
  cp "$P/arms/$arm.py" "$TARGET"
  echo "=== ARM $arm: $(shasum -a 256 $TARGET | cut -c1-16)"
  python -m pytest tests/test_mlx_runtime_cce_compile.py -q -p no:cacheprovider \
    --junitxml="$OUT/pytest_cce_compile_$arm.xml" 2>&1 | tail -25
  python -m pytest tests/test_mlx_cce_kernel.py -q -p no:cacheprovider \
    --junitxml="$OUT/pytest_cce_kernel_$arm.xml" 2>&1 | tail -8
  python -u "$P/probe.py" "$arm" "$OUT/probe_${arm}.json" --no-bench 2>&1 | tail -5
done

# Cost: base/head alternated, 3 rounds each, fresh process each time.
for round in 1 2 3; do
  for arm in base head; do
    cp "$P/arms/$arm.py" "$TARGET"
    python -u "$P/probe.py" "$arm" "$OUT/bench_${arm}_r${round}.json" 2>&1 | tail -2
  done
done
cp "$OUT/checkout_runtime_cce.py" "$TARGET"
python "$P/summarize.py" "$OUT" | tee "$OUT/SUMMARY.txt"
