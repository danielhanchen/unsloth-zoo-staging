# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Staging-only probe for unslothai/unsloth-zoo#1128.

Underscore-prefixed so pytest never collects it; run directly. It answers the
questions the PR's monkeypatched unit tests structurally cannot, because they
replace streams with strings and mx.synchronize with a lambda:

  A. alias audit -- how many DISTINCT stream objects the candidate tuple yields,
     i.e. whether one VLM stream gets drained up to three times per phase.
  B. lazy-import audit -- is generation_stream a real module binding, or only
     reachable through a module __getattr__ (which would break the helper's
     "a cleanup path must not import mlx-vlm" comment).
  C. real-API audit -- does mx.synchronize accept these actual objects.
  D. thread affinity -- generation_stream is bound at module scope on whichever
     thread first imported it. mlx-lm#1181 / #1256 report wrong-thread use of
     such a stream raising "There is no Stream(gpu, N) in current thread". The
     entry drain sits OUTSIDE any try, so if that reaches synchronize, every
     generation burst dies where it previously worked. Only D can fail this job.
  E. cost -- the real per-burst synchronize count and wall time, against the
     PR's "~33 us, twice per generation burst" claim.
"""

import platform
import sys
import threading
import time
import traceback
import types

import mlx.core as mx

from unsloth_zoo.mlx.generate import (
    _VLM_STREAM_MODULES,
    _drain_generation_streams,
    _generation_cache_hygiene,
)

CANDIDATES = ("mlx_lm.generate",) + _VLM_STREAM_MODULES
failures = []


def section(title):
    print(f"\n===== {title} =====", flush=True)


def versions():
    section("environment")
    print(f"python   {sys.version.split()[0]}  machine={platform.machine()}")
    from importlib.metadata import PackageNotFoundError, version
    for name in ("mlx", "mlx-lm", "mlx-vlm"):
        try:
            print(f"{name:8} {version(name)}")
        except PackageNotFoundError:
            print(f"{name:8} ABSENT")
    print(f"default device {mx.default_device()}")


def import_backends():
    """Import on the MAIN thread, exactly as a Studio/trainer process would."""
    section("imports (main thread)")
    for name in CANDIDATES:
        try:
            __import__(name)
            print(f"imported {name}")
        except Exception as exc:
            print(f"not importable {name}: {type(exc).__name__}: {exc}")


def alias_and_lazy_audit():
    section("A/B: alias + lazy-import audit")
    seen = {}
    for name in CANDIDATES:
        module = sys.modules.get(name)
        if module is None:
            print(f"{name}: not in sys.modules")
            continue
        direct = "generation_stream" in vars(module)
        stream = getattr(module, "generation_stream", None)
        if stream is None:
            print(f"{name}: no generation_stream (direct_binding={direct})")
            continue
        seen.setdefault(id(stream), []).append(name)
        print(f"{name}: type={type(stream).__module__}.{type(stream).__name__} "
              f"id={id(stream)} repr={stream!r} direct_binding={direct} "
              f"truthy={bool(stream)}")
        if not direct:
            print(f"  NOTE {name}.generation_stream is NOT in vars(); it came through "
                  f"a module __getattr__, so the helper's no-import comment is too strong.")
        if not stream:
            print(f"  NOTE {name}.generation_stream is FALSEY; _wired_limit's "
                  f"`[stream] if stream else None` would drop it.")
    total = sum(len(v) for v in seen.values())
    print(f"\ncandidate hits={total}  distinct stream objects={len(seen)}")
    for sid, names in seen.items():
        if len(names) > 1:
            print(f"  ALIASED id={sid} <- {names}  (drained {len(names)}x per phase)")


def real_api_audit():
    section("C: mx.synchronize against the real objects")
    for name in CANDIDATES:
        stream = getattr(sys.modules.get(name), "generation_stream", None)
        if stream is None:
            continue
        try:
            mx.synchronize(stream)
            print(f"mx.synchronize({name}.generation_stream) OK")
        except Exception as exc:
            print(f"mx.synchronize({name}.generation_stream) RAISED "
                  f"{type(exc).__name__}: {exc}")
            failures.append(f"C: main-thread synchronize on {name} raised {type(exc).__name__}")


def _burst_then_hygiene():
    """Submit async work on the generation stream, drop the output, then clear."""
    stream = getattr(sys.modules.get("mlx_lm.generate"), "generation_stream", None)
    for _ in range(8):
        if stream is not None:
            with mx.stream(stream):
                out = mx.matmul(mx.random.normal((512, 512)), mx.random.normal((512, 512)))
                mx.async_eval(out)
                del out          # exactly the dropped-mid-flight output the PR is about
        with _generation_cache_hygiene():
            pass


def thread_affinity():
    section("D: worker-thread drain (the possible regression)")
    box = {}

    def run():
        try:
            _burst_then_hygiene()
            box["ok"] = True
        except BaseException as exc:               # noqa: BLE001 - reporting a probe
            box["exc"] = exc
            box["tb"] = traceback.format_exc()

    thread = threading.Thread(target=run, name="probe-worker")
    thread.start()
    thread.join(timeout=180)
    if thread.is_alive():
        print("HANG: worker thread did not finish in 180s")
        failures.append("D: worker-thread drain hung")
        return
    if "exc" in box:
        print(f"RAISED {type(box['exc']).__name__}: {box['exc']}\n{box['tb']}")
        failures.append(f"D: worker-thread drain raised {type(box['exc']).__name__}")
    else:
        print("worker-thread generation burst + cache hygiene completed cleanly")

    print("\nmain-thread control:")
    try:
        _burst_then_hygiene()
        print("main-thread generation burst + cache hygiene completed cleanly")
    except BaseException as exc:                   # noqa: BLE001 - reporting a probe
        print(f"RAISED {type(exc).__name__}: {exc}\n{traceback.format_exc()}")
        failures.append(f"D: main-thread drain raised {type(exc).__name__}")


def plain_stream_from_a_foreign_thread():
    """The case the fix exists for: a plain mx.new_stream drained off its own thread.

    mlx-vlm bound generation_stream this way until 0.5.0 and the pin floor 0.4.4 still
    does, so this is the supported range, not a hypothetical. mlx made command encoders
    thread local in 0.31.2, so synchronizing one of these from elsewhere raises.
    """
    section("F: plain (non thread-local) stream drained from a worker thread")
    stream = mx.new_stream(mx.default_device())
    with mx.stream(stream):
        mx.eval(mx.matmul(mx.random.normal((256, 256)), mx.random.normal((256, 256))))
    module = types.SimpleNamespace(generation_stream=stream)
    sys.modules["mlx_vlm.generate"] = module
    print(f"stream type={type(stream).__module__}.{type(stream).__name__} repr={stream!r}")

    direct = {}

    def raw():
        try:
            mx.synchronize(stream)
            direct["ok"] = True
        except BaseException as exc:                # noqa: BLE001 - reporting a probe
            direct["exc"] = exc

    thread = threading.Thread(target=raw, name="probe-raw")
    thread.start()
    thread.join(timeout=60)
    print("bare mx.synchronize(stream) from a worker thread: "
          + ("OK" if direct.get("ok") else f"{type(direct['exc']).__name__}: {direct['exc']}"))

    box = {}

    def guarded():
        try:
            with _generation_cache_hygiene():
                pass
            box["ok"] = True
        except BaseException as exc:                # noqa: BLE001 - reporting a probe
            box["exc"] = exc
            box["tb"] = traceback.format_exc()

    thread = threading.Thread(target=guarded, name="probe-guarded")
    thread.start()
    thread.join(timeout=120)
    del sys.modules["mlx_vlm.generate"]
    if box.get("ok"):
        print("_generation_cache_hygiene() survived it")
    else:
        print(f"RAISED {type(box['exc']).__name__}: {box['exc']}\n{box.get('tb')}")
        failures.append(f"F: hygiene raised {type(box['exc']).__name__} on a foreign plain stream")


def cost():
    section("E: real synchronize count + cost per drain")
    calls = []
    real = mx.synchronize
    mx.synchronize = lambda stream=None: (calls.append(stream), real(stream))[1]
    try:
        _drain_generation_streams(mx)                       # warm
        calls.clear()
        _drain_generation_streams(mx)
        per_drain = len(calls)
        samples = []
        for _ in range(200):
            t0 = time.perf_counter()
            _drain_generation_streams(mx)
            samples.append((time.perf_counter() - t0) * 1e6)
    finally:
        mx.synchronize = real
    samples.sort()
    print(f"mx.synchronize calls per drain: {per_drain}  "
          f"(PR describes the drain as running twice per burst, so "
          f"{per_drain * 2} calls per burst on this runtime)")
    print(f"idle drain us: p50={samples[100]:.1f} p95={samples[190]:.1f} "
          f"max={samples[-1]:.1f}")


def main():
    versions()
    import_backends()
    alias_and_lazy_audit()
    real_api_audit()
    cost()
    thread_affinity()
    plain_stream_from_a_foreign_thread()
    section("verdict")
    if failures:
        for line in failures:
            print(f"FAIL {line}")
        return 1
    print("no drain failure observed on this runtime")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
