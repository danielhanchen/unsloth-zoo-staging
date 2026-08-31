"""Disposable staging-CI probe for unsloth-zoo PR #1129 (sorted gather_qmm NAX guard).

Runs on GitHub's hosted Apple silicon, which is M1 class and has NO neural accelerator,
so `*_gather_qmm_rhs_nax` is never dispatched here and the corruption this guard exists
for cannot occur. Nothing below claims otherwise. What it does establish, on real Metal:

  A. `mx.gpu` is a DeviceType, so `_gather_qmm_target_device(...).type != mx.gpu` is a
     sound test rather than dead code (the open Codex P1 on utils.py:697).
  B. The quantization derivations match real mlx for all four modes -- in particular
     `k = w.shape[-1] * 32 // bits`, which the PR's own predicate test cannot check
     because it builds its weights from the same assumption.
  D. The probe does not arm on healthy silicon, and what it costs in time and memory --
     including the 32769-row branch, which the PR's tests compute but never execute.
  E. With the guard installed, a real differentiated sorted gather_qmm step produces
     bit-identical losses to the same step with the guard disabled, gated on evidence
     that the step actually reached the guard.

Exit 0 only if every check passes. Every failure names the measured value.
"""
import json
import math
import os
import platform
import subprocess
import sys
import time

import mlx.core as mx

from unsloth_zoo.mlx import utils as U

FAILURES = []
# Measured on mlx 0.32.1: affine defaults to a group size of 64, which leaves no K
# remainder for either the probe or any real call, so the k_remainder condition is
# unreachable there by construction. Probing affine at 32 is what exercises that path.
MODES = (("affine", 32), ("mxfp4", 32), ("mxfp8", 32), ("nvfp4", 16))
EXPECTED_DEFAULTS = {"affine": (64, 4), "mxfp4": (32, 4), "mxfp8": (32, 8),
                     "nvfp4": (16, 4)}


def check(name, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    if not ok:
        FAILURES.append(name)


def peak_memory():
    for holder in (mx, getattr(mx, "metal", None)):
        fn = getattr(holder, "get_peak_memory", None)
        if callable(fn):
            return fn()
    return -1


def reset_peak_memory():
    for holder in (mx, getattr(mx, "metal", None)):
        fn = getattr(holder, "reset_peak_memory", None)
        if callable(fn):
            fn()
            return


print(f"python {sys.version.split()[0]} | {platform.platform()} | {platform.processor()}")
print(f"mlx {getattr(mx, '__version__', '?')} | metal {mx.metal.is_available()} "
      f"| default device {mx.default_device()!r}", flush=True)

check("PRE this runner is Apple silicon with Metal",
      platform.machine() == "arm64" and mx.metal.is_available(),
      f"machine={platform.machine()} metal={mx.metal.is_available()}")
if FAILURES:
    print("Refusing to report on a runner without Metal.")
    sys.exit(1)

# --- A. the Codex P1 -------------------------------------------------------------------
device = mx.default_device()
print(f"A. type(mx.gpu)={type(mx.gpu).__name__} type(device.type)={type(device.type).__name__}")
check("A1 mx.gpu is a DeviceType, not a Device",
      not isinstance(mx.gpu, mx.Device) and type(device.type) is type(mx.gpu),
      f"{type(mx.gpu).__name__}; Device.type is {type(device.type).__name__}")
check("A2 the default device on a Metal runner reads as gpu",
      device.type == mx.gpu and device.type != mx.cpu, repr(device))
check("A3 the guard's own device resolution agrees",
      U._gather_qmm_target_device({}).type == mx.gpu,
      repr(U._gather_qmm_target_device({})))
check("A4 an explicit CPU stream disarms the predicate",
      U._gather_qmm_target_device({"stream": mx.default_stream(mx.cpu)}).type != mx.gpu,
      repr(U._gather_qmm_target_device({"stream": mx.default_stream(mx.cpu)})))

# --- B. quantization derivations against real mlx ---------------------------------------
for mode, expected in EXPECTED_DEFAULTS.items():
    try:
        probe = mx.zeros((1, 256), dtype=mx.bfloat16)
        packed, scales = mx.quantize(probe, bits=None, mode=mode)[:2]
        mx.eval(packed, scales)
        resolved = U._gather_qmm_resolved_quantization(None, mode)
        print(f"B {mode:7s} packed{tuple(packed.shape)} {packed.dtype} -> resolved {resolved}")
        check(f"B1 {mode} packs into uint32 (the *32 in the formula)",
              packed.dtype == mx.uint32, str(packed.dtype))
        check(f"B2 {mode} resolved (group_size, bits) matches mlx",
              resolved == expected, f"got {resolved}, want {expected}")
        width = 128   # divisible by every default group size above
        w = mx.random.normal((8, 192, width)).astype(mx.bfloat16)
        w_packed = mx.quantize(w, bits=None, mode=mode)[0]
        mx.eval(w_packed)
        derived = w_packed.shape[-1] * 32 // resolved[1]
        check(f"B3 {mode} the untransposed branch's K derivation recovers the true width",
              derived == width, f"packed{tuple(w_packed.shape)} -> {derived}, want {width}")
    except Exception as error:
        check(f"B0 {mode} mx.quantize(bits=None, mode={mode!r})", False, repr(error))

# --- D. probe cost and the healthy verdict ---------------------------------------------
U.apply_gather_qmm_nax_guard()          # populates _MLX_GATHER_QMM_ORIGINAL
costs = {}
for condition in (U._MLX_K_REMAINDER, U._MLX_ROW_OVERFLOW):
    for mode, group_size in MODES:
        k = U._gather_qmm_probe_k(condition, group_size)
        if k is None:
            print(f"D {condition}/{mode} gs={group_size}: unreachable by construction")
            continue
        rows = U._MLX_PROBE_ROWS[condition]
        reset_peak_memory()
        before = peak_memory()
        started = time.perf_counter()
        try:
            error = U._gather_qmm_canary_error(condition, group_size, None, mode, device)
        except Exception as failure:
            check(f"D0 {condition}/{mode} probe ran", False, repr(failure))
            continue
        elapsed = time.perf_counter() - started
        megabytes = (peak_memory() - before) / 2 ** 20
        costs[f"{condition}/{mode}"] = {"seconds": round(elapsed, 4),
                                        "peak_mib": round(megabytes, 1),
                                        "max_error": float(error)}
        print(f"D {condition:12s} {mode:7s} K={k:4d} rows={rows:6d} "
              f"{elapsed * 1e3:9.1f} ms  peak+{megabytes:8.1f} MiB  max_err={error:.4g}",
              flush=True)
        check(f"D1 {condition}/{mode} reads HEALTHY on non-NAX silicon",
              math.isfinite(error) and error < U._MLX_CANARY_ERROR_LIMIT,
              f"max_error {error:.4g} vs limit {U._MLX_CANARY_ERROR_LIMIT}")
        check(f"D2 {condition}/{mode} probe under 10 s", elapsed < 10.0, f"{elapsed:.2f} s")
        check(f"D3 {condition}/{mode} probe under 2048 MiB", megabytes < 2048,
              f"{megabytes:.0f} MiB")

# The verdict cache is what keeps this a one-off cost per distinct call shape.
first_started = time.perf_counter()
U._gather_qmm_canary_defective(U._MLX_ROW_OVERFLOW, 32, None, "affine", device)
first = time.perf_counter() - first_started
second_started = time.perf_counter()
U._gather_qmm_canary_defective(U._MLX_ROW_OVERFLOW, 32, None, "affine", device)
second = time.perf_counter() - second_started
check("D4 a cached verdict is far cheaper than the first probe",
      second < 1e-3 or second * 50 < first,
      f"first {first * 1e3:.1f} ms, second {second * 1e6:.1f} us")
check("D5 the guard did NOT arm anywhere on this machine",
      not any(U._MLX_GATHER_QMM_CANARIES.values()),
      json.dumps({str(key): value
                  for key, value in U._MLX_GATHER_QMM_CANARIES.items()}))

# --- E. loss identity, guard on versus off ----------------------------------------------
# Two subprocesses, because UNSLOTH_MLX_GATHER_QMM_GUARD is read at install time. The
# geometry is driven directly rather than through a model, so that "the step reached the
# guard" is a fact this script establishes rather than hopes for; E3/E4 make a vacuous
# pass impossible.
TRAIN = r'''
import json, os, mlx.core as mx
from unsloth_zoo.mlx import utils as U

seen = {"sorted_calls": 0, "armed_calls": 0, "conditions": []}
original_conditions = U._gather_qmm_conditions
def spy(x, w, args, kwargs):
    conditions = original_conditions(x, w, args, kwargs)
    if kwargs.get("sorted_indices"):
        seen["sorted_calls"] += 1
    if conditions:
        seen["armed_calls"] += 1
        if list(conditions) not in seen["conditions"]:
            seen["conditions"].append(list(conditions))
    return conditions
U._gather_qmm_conditions = spy
U.apply_gather_qmm_nax_guard()

mx.random.seed(0)
EXPERTS, WIDTH, K, ROWS, GROUP = 8, 64, 96, 64, 32     # K % 64 != 0 -> k_remainder
w = (mx.random.normal((EXPERTS, WIDTH, K)) * 0.1).astype(mx.bfloat16)
packed, scales, biases = mx.quantize(w, group_size=GROUP, bits=4, mode="affine")
indices = mx.sort(mx.random.randint(0, EXPERTS, (ROWS,)).astype(mx.uint32))
target = mx.random.normal((ROWS, 1, WIDTH)).astype(mx.bfloat16)
x = (mx.random.normal((ROWS, 1, K)) * 0.5).astype(mx.bfloat16)

def loss_fn(x):
    out = mx.gather_qmm(x, packed, scales, biases, rhs_indices=indices,
                        transpose=True, group_size=GROUP, bits=4,
                        sorted_indices=True)
    return mx.mean((out.astype(mx.float32) - target.astype(mx.float32)) ** 2)

losses = []
for _ in range(5):
    value, grad = mx.value_and_grad(loss_fn)(x)
    mx.eval(value, grad)
    losses.append(float(value))
    x = (x.astype(mx.float32) - 0.05 * grad.astype(mx.float32)).astype(mx.bfloat16)

print("RESULT " + json.dumps({
    "losses": losses, "seen": seen,
    "installed": U.is_gather_qmm_nax_guard_applied(),
    "env": os.environ.get("UNSLOTH_MLX_GATHER_QMM_GUARD", "<unset>")}))
'''


def run_training(guard_value):
    environment = dict(os.environ)
    if guard_value is None:
        environment.pop("UNSLOTH_MLX_GATHER_QMM_GUARD", None)
    else:
        environment["UNSLOTH_MLX_GATHER_QMM_GUARD"] = guard_value
    completed = subprocess.run([sys.executable, "-c", TRAIN], env=environment,
                               capture_output=True, text=True, timeout=900)
    for line in completed.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    raise RuntimeError(f"guard={guard_value} produced no RESULT\n"
                       f"stdout:\n{completed.stdout[-3000:]}\n"
                       f"stderr:\n{completed.stderr[-3000:]}")


try:
    guard_on, guard_off = run_training(None), run_training("0")
    print("E guard on :", json.dumps(guard_on))
    print("E guard off:", json.dumps(guard_off))
    check("E1 the guard installs when the kill switch is unset",
          guard_on["installed"] is True, str(guard_on["installed"]))
    check("E2 the kill switch prevents installation",
          guard_off["installed"] is False, str(guard_off["installed"]))
    check("E3 the step issued sorted gather_qmm calls",
          guard_on["seen"]["sorted_calls"] > 0, json.dumps(guard_on["seen"]))
    check("E4 those calls reached a non-empty condition set",
          guard_on["seen"]["armed_calls"] > 0, json.dumps(guard_on["seen"]["conditions"]))
    check("E5 losses are bit-identical with and without the guard",
          guard_on["losses"] == guard_off["losses"],
          f"{guard_on['losses']} vs {guard_off['losses']}")
except Exception as error:
    check("E0 the loss-identity harness ran", False, repr(error))

print("\n=== SUMMARY ===")
print(json.dumps({"failures": FAILURES, "probe_cost": costs}, indent=2, default=str))
print(f"=== {len(FAILURES)} failure(s) ===")
sys.exit(1 if FAILURES else 0)
