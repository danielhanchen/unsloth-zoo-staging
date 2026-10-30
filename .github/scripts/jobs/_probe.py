"""Opt-in measurement probes for the jobs (`--probe`) and scripts/gc_bench.

Two halves:

1. `install()` patches, at class / module level and only while not compiling, the code paths whose use is
   otherwise invisible: which gradient checkpoint implementation ran (and how many bytes it sent to the
   host), whether the fused CE loss ran, how many dynamo recompiles happened. `snapshot()` reads them.
2. `analyze_trace(events, n_steps)` reads a torch profiler chrome trace (pure Python, CPU testable):
   D2H / H2D bytes per stream, whether copies left the compute stream, host syncs per step, copy and
   NCCL time not hidden behind compute kernels (exposed), attention and loss kernel families.

`ProfileWindow` profiles a step range with torch_debugging_utils.Profiler and returns analyze_trace's
dict. Nothing here changes what the job computes; the probe wrappers return the wrapped result as is.
"""

from __future__ import annotations

import gzip
import json
import os
import re
import sys
import weakref
from collections import defaultdict

STATE = {
    "installed": False,
    "gc": defaultdict(lambda: {"calls": 0, "offloaded": 0, "bytes": 0}),
    "loss": defaultdict(int),
    "recompiles": 0,
    "recompile_hook": False,
    "patched": [],
    "nonreentrant_instances": weakref.WeakSet() if hasattr(weakref, "WeakSet") else set(),
    "errors": [],
}


def _compiling():
    try:
        import torch
        return torch.compiler.is_compiling()
    except Exception:  # noqa: BLE001
        return False


def _tensor_bytes(x):
    try:
        return int(x.numel() * x.element_size())
    except Exception:  # noqa: BLE001
        return 0


def _wrap_function_forward(cls, key, offload_bytes_of):
    """Count cls.forward (an autograd.Function staticmethod); offload_bytes_of(ctx, args) -> bytes or 0."""
    orig = cls.__dict__.get("forward")
    if orig is None or getattr(orig, "_probe_wrapped", False):
        return
    fn = orig.__func__ if isinstance(orig, staticmethod) else orig

    def forward(ctx, *args, **kwargs):
        out = fn(ctx, *args, **kwargs)
        if not _compiling():
            rec = STATE["gc"][key]
            rec["calls"] += 1
            try:
                b = offload_bytes_of(ctx, args)
            except Exception:  # noqa: BLE001
                b = 0
            if b:
                rec["offloaded"] += 1
                rec["bytes"] += int(b)
        return out

    forward._probe_wrapped = True
    setattr(cls, "forward", staticmethod(forward))
    STATE["patched"].append(f"{cls.__module__}.{cls.__qualname__}.forward")


def _install_zoo_gc():
    try:
        import unsloth_zoo.gradient_checkpointing as ug
    except Exception as e:  # noqa: BLE001 - no zoo (plain HF job) or no accelerator
        STATE["errors"].append(f"zoo gc: {type(e).__name__}: {str(e)[:120]}")
        return None

    def ucf_bytes(ctx, args):
        meta = getattr(ctx, "_saved_metadata", None)
        return meta[0] if meta and meta[0] else 0

    if hasattr(ug, "UnslothCheckpointFunction"):
        _wrap_function_forward(ug.UnslothCheckpointFunction, "unsloth_reentrant", ucf_bytes)

    def first_tensor_bytes(ctx, args):
        import torch
        for a in args:
            if torch.is_tensor(a):
                return _tensor_bytes(a)
        return 0

    for name, key in (("Unsloth_Offloaded_Gradient_Checkpointer", "unsloth_offloaded_legacy"),
                      ("Unsloth_Gradient_Checkpointer", "unsloth_gc_legacy_no_offload")):
        cls = getattr(ug, name, None)
        if cls is not None:
            _wrap_function_forward(cls, key, first_tensor_bytes if "offloaded" in key else (lambda c, a: 0))

    cls = getattr(ug, "NonReentrantActivationOffloader", None)
    if cls is not None and not getattr(cls.pack, "_probe_wrapped", False):
        orig_pack, orig_init = cls.pack, cls.__init__

        def __init__(self, *a, **k):
            orig_init(self, *a, **k)
            try:
                STATE["nonreentrant_instances"].add(self)
            except TypeError:
                pass

        def pack(self, x):
            out = orig_pack(self, x)
            if not _compiling():
                key = "unsloth_nonreentrant_" + ("stream" if self.stream_mode else (
                    "side" if self.max_inflight > 0 else "compute"))
                rec = STATE["gc"][key]
                rec["calls"] += 1
                # a record = offloaded (or kept resident under stream mode); a tuple with a device =
                # the pageable fallback (non-CUDA or compiling); (None, x) = left alone
                if not isinstance(out, tuple) or out[0] is not None:
                    rec["offloaded"] += 1
                    rec["bytes"] += _tensor_bytes(x)
            return out

        pack._probe_wrapped = True
        cls.__init__, cls.pack = __init__, pack
        STATE["patched"].append("NonReentrantActivationOffloader.pack")
        live = getattr(ug, "NONREENTRANT_OFFLOADER", None)
        if live is not None:
            STATE["nonreentrant_instances"].add(live)
    return ug


def _install_torch_gc():
    import torch.utils.checkpoint as ck
    # torch >= 2.14: checkpoint() calls _checkpoint_without_reentrant_generator_impl and the old name is a
    # positional BC shim over it, so the _impl is the one every non-reentrant call reaches
    name = ("_checkpoint_without_reentrant_generator_impl"
            if hasattr(ck, "_checkpoint_without_reentrant_generator_impl")
            else "_checkpoint_without_reentrant_generator")
    gen = getattr(ck, name, None)
    if gen is not None and not getattr(gen, "_probe_wrapped", False):
        def counted_gen(*a, **k):
            if not _compiling():
                STATE["gc"]["torch_nonreentrant"]["calls"] += 1
            return gen(*a, **k)
        counted_gen._probe_wrapped = True
        setattr(ck, name, counted_gen)
        STATE["patched"].append(f"torch.utils.checkpoint.{name}")
    # torch's own reentrant Function (Unsloth may have swapped the module attribute already)
    orig_cls = getattr(ck, "_old_CheckpointFunction", None) or ck.CheckpointFunction
    if orig_cls.__name__ == "CheckpointFunction":
        _wrap_function_forward(orig_cls, "torch_reentrant", lambda c, a: 0)

    import torch.autograd.graph as ag
    soc = ag.save_on_cpu
    if not getattr(soc, "_probe_wrapped", False):
        class CountingSaveOnCpu(soc):
            _probe_wrapped = True

            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                pk = self.pack_hook

                def pack(t):
                    if not _compiling():
                        rec = STATE["gc"]["torch_save_on_cpu"]
                        rec["offloaded"] += 1
                        rec["bytes"] += _tensor_bytes(t)
                    return pk(t)
                self.pack_hook = pack
                if not _compiling():
                    STATE["gc"]["torch_save_on_cpu"]["calls"] += 1
        ag.save_on_cpu = CountingSaveOnCpu
        # transformers binds the name at import in modeling_utils when it offers offload
        try:
            import transformers.modeling_utils as mu
            if getattr(mu, "save_on_cpu", None) is soc:
                mu.save_on_cpu = CountingSaveOnCpu
        except Exception:  # noqa: BLE001
            pass
        STATE["patched"].append("torch.autograd.graph.save_on_cpu")


def _install_loss():
    try:
        import unsloth_zoo.fused_losses.cross_entropy_loss as ce
    except Exception as e:  # noqa: BLE001
        STATE["errors"].append(f"zoo fused ce: {type(e).__name__}: {str(e)[:120]}")
        return
    if hasattr(ce, "UnslothFusedLoss"):
        orig = ce.UnslothFusedLoss.__dict__.get("forward")
        if orig is not None and not getattr(orig, "_probe_wrapped", False):
            fn = orig.__func__ if isinstance(orig, staticmethod) else orig

            def forward(ctx, *a, **k):
                if not _compiling():
                    STATE["loss"]["UnslothFusedLoss.forward"] += 1
                return fn(ctx, *a, **k)
            forward._probe_wrapped = True
            ce.UnslothFusedLoss.forward = staticmethod(forward)
            STATE["patched"].append("UnslothFusedLoss.forward")
    targets = [("unsloth_zoo.fused_losses.cross_entropy_loss", "unsloth_fused_ce_loss"),
               ("unsloth_zoo.loss_utils", "unsloth_fused_ce_loss"),
               ("unsloth_zoo.fused_losses.forward_adapter", "unsloth_fused_lm_head_loss"),
               ("unsloth.models.llama", "unsloth_fused_ce_loss"),
               ("unsloth.models.llama", "fast_cross_entropy_loss"),
               ("unsloth.kernels.cross_entropy_loss", "fast_cross_entropy_loss")]
    # Unsloth's generated modules (FastModel path) bind the loss names at their import; their functions
    # read module globals at call time, so rebinding the module attribute reaches them
    for mod_name in list(sys.modules):
        if "unsloth_compiled_module" in mod_name or mod_name.startswith("unsloth_compiled_cache"):
            for attr in ("unsloth_fused_ce_loss", "unsloth_fused_lm_head_loss", "fused_linear_cross_entropy",
                         "unsloth_count_aware_cross_entropy"):
                targets.append((mod_name, attr))
    for mod_name, attr in targets:
        mod = sys.modules.get(mod_name)
        fn = getattr(mod, attr, None) if mod is not None else None
        if fn is None or getattr(fn, "_probe_wrapped", False):
            continue

        short = "compiled_module" if "unsloth_compiled" in mod_name else mod_name.rsplit(".", 1)[-1]

        def make(fn=fn, tag=f"{short}.{attr}"):
            def wrapper(*a, **k):
                if not _compiling():
                    STATE["loss"][tag] += 1
                return fn(*a, **k)
            wrapper._probe_wrapped = True
            wrapper.__wrapped__ = fn
            return wrapper
        setattr(mod, attr, make())
        STATE["patched"].append(f"{mod_name}.{attr}")


def _install_recompile_hook():
    try:
        from torch._dynamo.utils import register_hook_for_recompile_user_context
    except Exception:  # noqa: BLE001 - torch < 2.8: fall back to guard_failures at snapshot
        return

    def hook():
        STATE["recompiles"] += 1
        return ""
    register_hook_for_recompile_user_context(hook)
    STATE["recompile_hook"] = True


def install():
    """Idempotent; call after `import unsloth` (and the model load) so the zoo modules exist."""
    if STATE["installed"]:
        _install_loss()  # modules imported since the first call
        return STATE
    STATE["installed"] = True
    for f in (_install_torch_gc, _install_zoo_gc, _install_loss, _install_recompile_hook):
        try:
            f()
        except Exception as e:  # noqa: BLE001 - a probe must never break the job
            STATE["errors"].append(f"{f.__name__}: {type(e).__name__}: {str(e)[:160]}")
    return STATE


def dynamo_live():
    """Positive control: compile one throwaway function (eager backend) and check the frame counter
    moves, so a 0 graph-break / 0 frame report is read as "nothing compiled", not a dead counter. The
    control's own frame is not in the dynamo_snapshot taken before it."""
    try:
        import torch
        from torch._dynamo.utils import counters
        # stats.unique_graphs, not frames.total: the frames counter is not bumped on every path
        # (eager backend on CPU leaves it at 0 on torch 2.13 - 2.15)
        n0 = int(counters.get("stats", {}).get("unique_graphs", 0))
        # a fresh code object per call: a second call on the same one is a dynamo cache hit
        STATE["live_checks"] = STATE.get("live_checks", 0) + 1
        ns = {}
        exec(f"def _probe_live_{STATE['live_checks']}(x):\n    return x + {STATE['live_checks']}", ns)
        fn = torch.compile(next(v for k, v in ns.items() if k.startswith("_probe_live")), backend="eager",
                           fullgraph=True)
        fn(torch.zeros(3))
        return int(counters.get("stats", {}).get("unique_graphs", 0)) > n0
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {str(e)[:120]}"


def dynamo_snapshot():
    try:
        from torch._dynamo.utils import counters
    except Exception:  # noqa: BLE001
        return None
    gb = counters.get("graph_break", {})
    out = {"graph_breaks": int(sum(gb.values())),
           "unique_graphs": int(counters.get("stats", {}).get("unique_graphs", 0)),
           "frames_total": int(counters.get("frames", {}).get("total", 0)),
           "calls_captured": int(counters.get("stats", {}).get("calls_captured", 0)),
           "recompiles": STATE["recompiles"] if STATE["recompile_hook"] else None}
    try:
        from torch._dynamo import convert_frame
        gf = getattr(convert_frame, "guard_failures", None)
        if gf is not None:
            out["guard_failures"] = int(sum(len(v) for v in gf.values()))
    except Exception:  # noqa: BLE001
        pass
    return out


def snapshot():
    gc = {k: dict(v) for k, v in STATE["gc"].items() if v["calls"] or v["offloaded"]}
    nr = []
    for o in list(STATE["nonreentrant_instances"]):
        nr.append({"stream_mode": bool(getattr(o, "stream_mode", False)),
                   "max_inflight": getattr(o, "max_inflight", None),
                   "keep_last": getattr(o, "keep_last", None),
                   "side_stream_d2h": bool(getattr(o, "d2h_streams", None)),
                   "side_stream_h2d": bool(getattr(o, "h2d_streams", None)),
                   "stats": dict(getattr(o, "stats", {}) or {})})
    zoo = {}
    try:
        import torch.utils.checkpoint as ck
        g = getattr(ck, "_checkpoint_without_reentrant_generator_impl", None) or getattr(
            ck, "_checkpoint_without_reentrant_generator", None)
        zoo["torch_generator_probe_in_place"] = bool(getattr(g, "_probe_wrapped", False))
    except Exception:  # noqa: BLE001
        pass
    ug = sys.modules.get("unsloth_zoo.gradient_checkpointing")
    if ug is not None:
        for k in ("USE_DOUBLE_BUFFER", "GC_PREFETCH", "MINIMUM_SIZE", "CPU_INDEX", "LAST_GC_INDEX",
                  "GC_PINNED_BYTES"):
            v = getattr(ug, k, None)
            if isinstance(v, (bool, int, float, str)) or v is None:
                zoo[k] = v
        zoo["checkpoint_function"] = __import__("torch").utils.checkpoint.CheckpointFunction.__name__
        zoo["checkpoint"] = getattr(__import__("torch").utils.checkpoint.checkpoint, "__name__", None)
    ran = sorted(k for k, v in gc.items() if v["calls"])
    offloading = sorted(k for k, v in gc.items() if v["offloaded"])
    return {"gc": gc, "gc_impl_ran": ran, "gc_impl_offloaded": offloading, "nonreentrant": nr,
            "zoo_gc_state": zoo, "loss": dict(STATE["loss"]), "dynamo": dynamo_snapshot(),
            "patched": list(STATE["patched"]), "errors": list(STATE["errors"])}


# ---------------------------------------------------------------- trace analysis (pure Python)

SYNC_RUNTIME = ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize", "cudaMemcpy",
                "cudaMemcpy2D", "hipStreamSynchronize", "hipDeviceSynchronize", "hipEventSynchronize",
                "hipMemcpy")

ATTN_OPS = (  # cpu_op name -> family (op level is unambiguous; kernel names are a cross-check)
    ("aten::_scaled_dot_product_flash_attention", "sdpa_flash"),
    ("aten::_scaled_dot_product_efficient_attention", "sdpa_mem_efficient"),
    ("aten::_scaled_dot_product_cudnn_attention", "sdpa_cudnn"),
    ("aten::_scaled_dot_product_attention_math", "sdpa_math"),
    ("aten::_scaled_dot_product_fused_attention_overrideable", "sdpa_overrideable"),
    ("aten::_flash_attention_forward", "sdpa_flash"),
    ("aten::_efficient_attention_forward", "sdpa_mem_efficient"),
    ("flash_attn::", "flash_attn_pkg"),
    ("_flash_attn_varlen_forward", "flash_attn_pkg_varlen"),
    ("_flash_attn_forward", "flash_attn_pkg"),
    ("xformers", "xformers"),
    ("flex_attention", "flex"),
)


def _kernel_family(name):
    n = name.lower()
    if "nccl" in n:
        return "nccl"
    if "cudnn" in n and any(t in n for t in ("sdpa", "fmha", "flash", "attn", "mha", "fprop", "dgrad")):
        return "attn_cudnn"
    if "flash" in n and ("fwd" in n or "bwd" in n or "attn" in n or "kernel" in n):
        return "attn_flash"
    if "fmha" in n or "efficient_attention" in n or "mem_efficient" in n:
        return "attn_mem_efficient"
    if "flex_attention" in n or ("triton_tem" in n and "flex" in n):
        return "attn_flex"
    if "cross_entropy" in n or "crossentropy" in n:
        return "cross_entropy"
    return None


def _union(iv):
    out = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            if e > out[-1][1]:
                out[-1][1] = e
        else:
            out.append([s, e])
    return out


def _length(iv):
    return sum(e - s for s, e in iv)


def _intersect(a, b):
    """Both unions (sorted, disjoint)."""
    i = j = 0
    out = []
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if s < e:
            out.append([s, e])
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def exposed_time(target, cover):
    """|union(target)| - |union(target) & union(cover)|, in the intervals' unit."""
    t, c = _union(target), _union(cover)
    return _length(t) - _length(_intersect(t, c))


def _memcpy_kind(name):
    m = re.search(r"Memcpy (\w+)", name)
    kind = m.group(1) if m else "other"
    dst = re.search(r"-> (\w+)\)", name)
    return kind, (dst.group(1) if dst else None)


COMPILE_SITE_MARKERS = ("dynamo_timed", "CachingAutotuner", "CUDAGraphNode.record", "cudagraphify",
                        "## Call CompiledFxGraph", "compile_fx", "autotune")


def _sync_sites(events, syncs, top=8, all_sites=False):
    """Innermost cpu_op (same thread) enclosing each sync runtime call: where the host waits."""
    import bisect
    ops = defaultdict(list)
    for ev in events:
        if ev.get("ph") == "X" and ev.get("cat") == "cpu_op":
            ops[ev.get("tid")].append((float(ev.get("ts", 0.0)), float(ev.get("ts", 0.0)) + float(ev.get("dur", 0.0)),
                                       ev.get("name", "")))
    for v in ops.values():
        v.sort()
    starts = {t: [o[0] for o in v] for t, v in ops.items()}
    sites = defaultdict(int)
    for ev in syncs:
        t, ts = ev.get("tid"), float(ev.get("ts", 0.0))
        v = ops.get(t, [])
        i = bisect.bisect_right(starts.get(t, []), ts) - 1
        best = None
        for j in range(i, max(-1, i - 4000), -1):
            s0, e0, nm = v[j]
            if s0 <= ts <= e0 and (best is None or s0 >= best[0]):
                best = (s0, nm)
                break
        sites[f"{ev.get('name')} <- {best[1] if best else '(no op: harness / python)'}"] += 1
    ranked = sorted(sites.items(), key=lambda kv: -kv[1])
    return dict(ranked if all_sites else ranked[:top])


def analyze_trace(events, n_steps=1, harness_device_syncs=0):
    """Metrics from chrome-trace events (list of dicts as torch's export_chrome_trace writes).

    Per device: the compute stream is the stream with the most kernel time. Copies (Memcpy DtoH / HtoD)
    are split by stream; `exposed_*_ms` is time a copy (or NCCL kernel) ran with no non-NCCL kernel of the
    same device running on any stream. Host syncs: synchronising runtime calls plus DtoH copies into
    pageable memory (always synchronous). Times in ms, per step where named so.
    """
    n_steps = max(1, int(n_steps or 1))
    sync_events = []
    kernels = defaultdict(list)          # device -> [(s, e, stream, name)]
    copies = defaultdict(list)           # device -> [(s, e, stream, kind, dst, bytes)]
    runtime = defaultdict(int)
    attn_ops = defaultdict(int)
    for ev in events:
        if ev.get("ph") != "X":
            continue
        cat, name = ev.get("cat", ""), ev.get("name", "")
        args = ev.get("args") or {}
        ts, dur = float(ev.get("ts", 0.0)), float(ev.get("dur", 0.0))
        if cat == "kernel":
            kernels[args.get("device", 0)].append((ts, ts + dur, args.get("stream"), name))
        elif cat == "gpu_memcpy":
            kind, dst = _memcpy_kind(name)
            copies[args.get("device", 0)].append((ts, ts + dur, args.get("stream"), kind, dst,
                                                  int(args.get("bytes", 0) or 0)))
        elif cat in ("cuda_runtime", "cuda_driver"):
            if name in SYNC_RUNTIME:
                runtime[name] += 1
                sync_events.append(ev)
        elif cat == "cpu_op":
            for pat, fam in ATTN_OPS:
                if pat in name:
                    attn_ops[fam] += 1
                    break
    # the profiler's own stop (and a harness's) synchronises the device inside the window: not the job's
    excl = 0
    for nm in ("cudaDeviceSynchronize", "hipDeviceSynchronize"):
        take = min(runtime.get(nm, 0), int(harness_device_syncs) - excl)
        if take > 0:
            runtime[nm] -= take
            excl += take
    out = {"n_steps": n_steps, "devices": {}, "runtime_syncs": {k: v for k, v in runtime.items() if v},
           "harness_device_syncs_excluded": excl}
    total_syncs = sum(runtime.values())
    pageable_d2h = 0
    all_exposed_comm = all_exposed_copy = 0.0
    kernel_fams = defaultdict(lambda: {"count": 0, "ms": 0.0})
    fam_names = defaultdict(set)
    for dev in sorted(set(kernels) | set(copies), key=str):
        ks, cs = kernels.get(dev, []), copies.get(dev, [])
        by_stream = defaultdict(float)
        for s, e, st, nm in ks:
            if "nccl" not in nm.lower():
                by_stream[st] += e - s
            fam = _kernel_family(nm)
            if fam:
                fam_names[fam].add(nm[:120])
                kernel_fams[fam]["count"] += 1
                kernel_fams[fam]["ms"] += (e - s) / 1e3
        compute_stream = max(by_stream, key=by_stream.get) if by_stream else None
        compute_iv = [(s, e) for s, e, st, nm in ks if "nccl" not in nm.lower()]
        nccl_iv = [(s, e) for s, e, st, nm in ks if "nccl" in nm.lower()]
        copy_stats = defaultdict(lambda: {"count": 0, "bytes": 0, "ms": 0.0, "on_compute_stream_bytes": 0,
                                          "side_stream_bytes": 0, "streams": set()})
        copy_iv = []
        for s, e, st, kind, dst, nb in cs:
            c = copy_stats[kind]
            c["count"] += 1
            c["bytes"] += nb
            c["ms"] += (e - s) / 1e3
            c["streams"].add(st)
            if st == compute_stream:
                c["on_compute_stream_bytes"] += nb
            else:
                c["side_stream_bytes"] += nb
            if kind == "DtoH" and dst == "Pageable":
                pageable_d2h += 1
            if kind in ("DtoH", "HtoD"):
                copy_iv.append((s, e))
        ex_comm = exposed_time(nccl_iv, compute_iv) / 1e3
        ex_copy = exposed_time(copy_iv, compute_iv) / 1e3
        all_exposed_comm += ex_comm
        all_exposed_copy += ex_copy
        busy = _length(_union(compute_iv)) / 1e3
        out["devices"][str(dev)] = {
            "compute_stream": compute_stream,
            "kernel_streams": {str(k): round(v / 1e3, 3) for k, v in by_stream.items()},
            "compute_busy_ms": round(busy, 3),
            "nccl_ms": round(_length(_union(nccl_iv)) / 1e3, 3),
            "exposed_comm_ms": round(ex_comm, 3),
            "exposed_copy_ms": round(ex_copy, 3),
            "copies": {k: {**{kk: vv for kk, vv in v.items() if kk != "streams"},
                           "ms": round(v["ms"], 3), "streams": sorted(map(str, v["streams"]))}
                       for k, v in copy_stats.items()},
        }
    # syncs issued by compilation / autotuning / CUDA-graph recording inside the window: the window was
    # not steady state (profile later, or after more warmup); reported apart so they are not read as
    # per-step model syncs
    compile_syncs = 0
    if sync_events:
        for site, n in _sync_sites(events, sync_events, all_sites=True).items():
            if any(m in site for m in COMPILE_SITE_MARKERS):
                compile_syncs += n
    out["compile_syncs_in_window"] = compile_syncs
    out["steady_state_window"] = compile_syncs == 0
    d2h = sum(d["copies"].get("DtoH", {}).get("bytes", 0) for d in out["devices"].values())
    h2d = sum(d["copies"].get("HtoD", {}).get("bytes", 0) for d in out["devices"].values())
    side = sum(d["copies"].get(k, {}).get("side_stream_bytes", 0) for d in out["devices"].values()
               for k in ("DtoH", "HtoD"))
    out.update({
        "d2h_bytes_per_step": d2h // n_steps, "h2d_bytes_per_step": h2d // n_steps,
        "side_stream_copy_bytes_per_step": side // n_steps,
        "host_syncs_per_step": round((total_syncs + pageable_d2h) / n_steps, 2),
        "host_syncs_per_step_excl_compile": round((total_syncs + pageable_d2h - compile_syncs) / n_steps, 2),
        "pageable_d2h_per_step": round(pageable_d2h / n_steps, 2),
        "exposed_comm_ms_per_step": round(all_exposed_comm / n_steps, 3),
        "exposed_copy_ms_per_step": round(all_exposed_copy / n_steps, 3),
        "sync_sites": _sync_sites(events, sync_events) if sync_events else {},
        "kernel_family_examples": {k: sorted(v)[:3] for k, v in fam_names.items()},
        "attention_ops": dict(attn_ops),
        "kernel_families": {k: {"count": v["count"], "ms": round(v["ms"], 3)} for k, v in kernel_fams.items()},
    })
    return out


def load_trace(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as fh:
        tr = json.load(fh)
    return tr["traceEvents"] if isinstance(tr, dict) else tr


class ProfileWindow:
    """Profile steps [start, start + n) of a loop: call begin(step) before and end(step) after each step.
    On the last step it stops, exports the trace (gzipped when keep_trace) and stores analyze_trace's dict
    in .result."""

    # torch.profiler's stop runs one device synchronize inside the window
    PROFILER_DEVICE_SYNCS = 1

    def __init__(self, start, n, trace_path=None, keep_trace=False, harness_device_syncs=0):
        self.start, self.n = int(start), int(n)
        self.harness_device_syncs = int(harness_device_syncs)
        self.trace_path, self.keep_trace = trace_path, keep_trace
        self.prof, self.result, self.error = None, None, None

    def begin(self, step):
        if step != self.start or self.prof is not None or self.n <= 0:
            return
        try:
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if root not in sys.path:
                sys.path.insert(0, root)
            from torch_debugging_utils import Profiler
            self.prof = Profiler()
            self.prof.__enter__()
        except Exception as e:  # noqa: BLE001
            self.error, self.prof = f"{type(e).__name__}: {e}", None

    def active(self, step):
        return self.prof is not None and self.start <= step < self.start + self.n

    def end(self, step):
        if self.prof is None or step != self.start + self.n - 1:
            return
        try:
            self.prof.__exit__(None, None, None)
            path = self.trace_path or os.path.join(os.environ.get("TMPDIR", "/tmp"), f"probe_{os.getpid()}.json")
            self.prof.export_chrome_trace(path)
            self.result = analyze_trace(load_trace(path), self.n,
                                        harness_device_syncs=self.PROFILER_DEVICE_SYNCS + self.harness_device_syncs)
            self.result["steps_profiled"] = [self.start, self.start + self.n - 1]
            if self.keep_trace:
                with open(path, "rb") as src, gzip.open(path + ".gz", "wb") as dst:
                    dst.write(src.read())
                self.result["trace"] = path + ".gz"
            os.remove(path)
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"
        finally:
            self.prof = None
