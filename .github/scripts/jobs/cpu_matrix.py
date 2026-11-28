"""CPU training matrix: SFT LoRA, full fine-tuning, DPO, GRPO (HF generate) and compiled SFT on tiny models.

Each (model, mode) cell runs in its own subprocess so a crash or a hang stays in its cell and peak RSS is per cell.
Records pass / fail, s/step (wall + process CPU), peak RSS, warning lines, and whether the CPU fast paths engaged
(chunked CE / chunked log-probs, fused AdamW, torch.compile).

    python jobs/cpu_matrix.py --out cpu_matrix.json                 # all cells
    python jobs/cpu_matrix.py --modes sft_lora,dpo --models qwen3   # subset
    python jobs/cpu_matrix.py --threads 2                           # equal thread count on every OS

macOS Apple Silicon: Unsloth picks MLX when `mlx` is installed; this forces the torch CPU path
(UNSLOTH_FORCE_GPU_PATH=1) so the CPU trainers are what runs. `--tiny` is accepted and ignored (CI passes it).
"""

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time

MODELS = {
    "qwen3": "trl-internal-testing/tiny-Qwen3ForCausalLM",
    "llama": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    "qwen3_06b": "unsloth/Qwen3-0.6B",  # real size: is the speed / memory reasonable, not only does it run
    "gemma3": "hf-internal-testing/tiny-random-Gemma3ForCausalLM",
    "gemma3_270m": "unsloth/gemma-3-270m",  # 262k vocab: the LM head / CE dominates a CPU SFT step
}
# sft_masked: prompt / completion SFT (completion-only loss), about 75% of the label tokens masked.
MODES = ("sft_lora", "full_ft", "dpo", "grpo", "sft_compile", "sft_masked")
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
RESULT_TAG = "CPU_MATRIX_RESULT "
WARN_RE = re.compile(r"warn|deprecat|error|exception|traceback|slow|fallback|not supported|staying eager", re.I)


def peak_rss_mib():
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        k32 = ctypes.WinDLL("kernel32")
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi")
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
        return pmc.PeakWorkingSetSize / 2**20
    import resource
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 1024  # bytes on macOS, KiB on Linux


def cfg(cls, **kw):
    """Build a TRL config with only the fields this TRL version has (dropped names are recorded)."""
    import dataclasses
    names = {f.name for f in dataclasses.fields(cls)}
    dropped = sorted(k for k in kw if k not in names)
    return cls(**{k: v for k, v in kw.items() if k in names}), dropped


def tok_kwarg(trainer_cls, tok):
    import inspect
    params = inspect.signature(trainer_cls.__init__).parameters
    return {"processing_class": tok} if "processing_class" in params else {"tokenizer": tok}


class Counters:
    def __init__(self):
        self.n = {"chunked_ce": 0, "chunked_logps": 0, "chunked_logps_ckpt": 0}

    def wrap_forward(self, cls, key):
        orig = cls.forward

        def forward(*args, **kwargs):
            self.n[key] += 1
            return orig(*args, **kwargs)
        cls.forward = staticmethod(forward)

    def wrap_func(self, mod, name, key):
        orig = getattr(mod, name)

        def f(*args, **kwargs):
            self.n[key] += 1
            return orig(*args, **kwargs)
        setattr(mod, name, f)


def instrument(counters):
    try:
        from unsloth_zoo.cpu import ops
    except Exception as e:  # noqa: BLE001
        return f"no unsloth_zoo.cpu.ops: {e!r}"
    for cls_name, key in (("_ChunkedLinearCrossEntropy", "chunked_ce"), ("_ChunkedLogProbs", "chunked_logps")):
        if hasattr(ops, cls_name):
            counters.wrap_forward(getattr(ops, cls_name), key)
    if hasattr(ops, "_logprob_piece"):
        counters.wrap_func(ops, "_logprob_piece", "chunked_logps_ckpt")
    try:  # round 11: frozen projections moved to fp32 for a train-mode rollout (sum of modules moved)
        from unsloth_zoo.cpu import loader as _ld
        orig = getattr(_ld, "set_bf16_stored_dtype", None)
        if orig is not None:
            import torch

            def moved(model, dtype, _orig=orig):
                n = _orig(model, dtype)
                if dtype == torch.float32 and n:
                    counters.n["rollout_fp32_moves"] = counters.n.get("rollout_fp32_moves", 0) + n
                return n
            _ld.set_bf16_stored_dtype = moved
    except Exception:  # noqa: BLE001
        pass
    return None


def shift_upcast(w):
    """bf16 -> fp32 by a 16-bit shift (exact, NaN / inf preserved): integer ops are vectorised where bf16 is not."""
    import torch
    return (w.view(torch.int16).to(torch.int32) << 16).view(torch.float32)


def patch_fast_upcast():
    """Experiment arm: the frozen bf16 projections upcast with shift_upcast instead of .to(float32)."""
    import torch
    import torch.nn.functional as F
    from unsloth_zoo.cpu import ops
    fn = getattr(ops, "_Bf16StoredLinearFunction", None)
    if fn is None:
        return "no _Bf16StoredLinearFunction"

    def up(w, dtype):
        return shift_upcast(w) if w.dtype == torch.bfloat16 and dtype == torch.float32 else w.to(dtype)

    def forward(ctx, x, weight, bias):
        ctx.save_for_backward(weight)
        return F.linear(x, up(weight, x.dtype), bias)

    def backward(ctx, grad):
        (weight,) = ctx.saved_tensors
        return grad @ up(weight, grad.dtype), None, None
    fn.forward = staticmethod(forward)
    fn.backward = staticmethod(backward)
    return "patched"


def step_callback(rec, profile_last=False):
    from transformers import TrainerCallback

    class CB(TrainerCallback):
        prof = None

        def on_step_begin(self, args, state, control, **kw):
            if profile_last and state.global_step == (state.max_steps or 0) - 1:
                import torch
                deep = os.environ.get("CPU_MATRIX_PROFILE_DEEP", "0") == "1"
                self.prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                                   record_shapes=deep, with_stack=deep)
                self.deep = deep
                self.prof.start()
            self.t, self.c, self.g = time.perf_counter(), time.process_time(), _GEN["s"]

        def on_step_end(self, args, state, control, **kw):
            rec["step_wall_s"].append(round(time.perf_counter() - self.t, 4))
            rec["step_cpu_s"].append(round(time.process_time() - self.c, 4))
            rec.setdefault("step_gen_s", []).append(round(_GEN["s"] - self.g, 4))  # rollout seconds in this step
            if self.prof is not None:
                self.prof.stop()
                ev = sorted(self.prof.key_averages(), key=lambda e: e.self_cpu_time_total, reverse=True)
                total = sum(e.self_cpu_time_total for e in ev) or 1
                rec["profile_top"] = [[e.key, round(e.self_cpu_time_total / 1e3, 1), round(100 * e.self_cpu_time_total / total, 1),
                                       e.count] for e in ev[:20]]
                rec["profile_note"] = "profiled step (excluded from s/step median): [op, self CPU ms, %, calls]"
                if getattr(self, "deep", False):  # CPU_MATRIX_PROFILE_DEEP=1: per-shape and per-stack attribution
                    sh = sorted(self.prof.key_averages(group_by_input_shape=True), key=lambda e: e.self_cpu_time_total, reverse=True)
                    rec["profile_shapes"] = [[e.key, str(e.input_shapes)[:200], round(e.self_cpu_time_total / 1e3, 1), e.count] for e in sh[:40]]
                    st = sorted(self.prof.key_averages(group_by_stack_n=12), key=lambda e: e.self_cpu_time_total, reverse=True)
                    rec["profile_stacks"] = [[e.key, round(e.self_cpu_time_total / 1e3, 1), e.count, list(e.stack or [])[:12]] for e in st[:40]]
                rec["step_wall_s"][-1] = None
                rec["step_cpu_s"][-1] = None
                self.prof = None

        def on_log(self, args, state, control, logs=None, **kw):
            if logs and "loss" in logs:
                rec["losses"].append(round(float(logs["loss"]), 5))
    return CB()


def optimizer_info(trainer):
    opt = getattr(trainer, "optimizer", None)
    inner = getattr(opt, "optimizer", opt)
    if inner is None:
        return {"optim_arg": str(getattr(trainer.args, "optim", None))}
    g = inner.param_groups[0] if inner.param_groups else {}
    return {"optim_arg": str(getattr(trainer.args, "optim", None)), "optimizer": type(inner).__name__,
            "fused": bool(g.get("fused") or inner.defaults.get("fused")),
            "foreach": g.get("foreach", inner.defaults.get("foreach"))}


def compile_info(model):
    import torch
    compiled = sum(1 for m in model.modules() if getattr(m, "_compiled_call_impl", None) is not None
                   or type(m).__name__ == "OptimizedModule")
    info = {"compiled_modules": compiled, "UNSLOTH_CPU_COMPILE": os.environ.get("UNSLOTH_CPU_COMPILE")}
    try:
        from torch._dynamo.utils import counters
        info["dynamo_unique_graphs"] = int(counters["stats"]["unique_graphs"])
        info["dynamo_graph_breaks"] = int(sum(counters["graph_break"].values()))
        info["dynamo_frames_ok"] = int(counters["frames"]["ok"])
    except Exception:  # noqa: BLE001
        pass
    info["torch_compiler_available"] = hasattr(torch, "compile")
    return info


_GEN = {"n": 0, "s": 0.0}  # generate() calls / seconds, read per step by the step callback


def instrument_generate():
    """Seconds spent in HF generate() (GRPO rollouts): tells the decode share of a step from the train share."""
    acc = _GEN
    try:
        from transformers.generation.utils import GenerationMixin
        orig = GenerationMixin.generate

        def generate(self, *a, **k):
            import torch
            gen_threads = int(os.environ.get("CPU_MATRIX_GEN_THREADS", "0") or 0)  # torch threads for rollouts only
            old = torch.get_num_threads()
            if gen_threads:
                torch.set_num_threads(gen_threads)
            undo = lmhead_chunk_install(self)
            undo16 = fp16w_install(self)
            t0 = time.perf_counter()
            try:
                return orig(self, *a, **k)
            finally:
                if undo:
                    undo()
                if undo16:
                    undo16()
                acc["n"] += 1
                acc["s"] += time.perf_counter() - t0
                if gen_threads:
                    torch.set_num_threads(old)
        GenerationMixin.generate = generate
    except Exception:  # noqa: BLE001
        pass
    return acc


LMHEAD_CALLS = [0, 0]  # calls, most rows in one call
FP16W = {"calls": 0, "head_calls": 0, "max_rows": 0, "fallback_rows": 0, "packed": 0, "skipped": 0, "pack_s": 0.0,
         "error": None}
_FP16W_PROJ = frozenset(("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"))


def fp16w_install(model):
    """Round 16 arm CPU_MATRIX_FP16W=1 (projections) / 2 (+ LM head): during generate, frozen bf16-stored or fp32
    projections whose values convert to fp16 without overflow are run through FBGEMM fp16-packed weights
    (quantized.linear_dynamic_fp16: fp16 weight bytes, fp32 activations and accumulation). Packed once, cached on
    the module (keyed on weight version); calls with more than CPU_MATRIX_FP16W_ROWS rows keep the normal path."""
    level = int(os.environ.get("CPU_MATRIX_FP16W", "0") or 0)
    if not level:
        return None
    import torch
    max_rows = int(os.environ.get("CPU_MATRIX_FP16W_ROWS", "64") or 64)
    try:
        prepack, run = torch.ops.quantized.linear_prepack_fp16, torch.ops.quantized.linear_dynamic_fp16
    except Exception as e:  # noqa: BLE001
        FP16W["error"] = repr(e)[:200]
        return None
    head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    targets = []
    for name, m in model.named_modules():
        w = getattr(m, "weight", None)
        if not isinstance(m, torch.nn.Linear) or w is None or w.requires_grad or w.dtype not in (torch.bfloat16, torch.float32):
            continue
        is_head = m is head
        if not is_head and name.replace(".base_layer", "").rsplit(".", 1)[-1] not in _FP16W_PROJ:
            continue
        if is_head and level < 2:
            continue
        b = getattr(m, "bias", None)
        if b is not None and b.requires_grad:
            continue
        key = (w._version, w.data_ptr(), w.dtype)
        cache = getattr(m, "_r16_fp16w", None)
        if cache is None or cache[0] != key:
            t0 = time.perf_counter()
            wf = w.detach().float()
            if not bool(wf.abs().max() <= 65504):
                FP16W["skipped"] += 1
                continue
            try:
                packed = prepack(wf, None if b is None else b.detach().float())
            except Exception as e:  # noqa: BLE001
                FP16W["error"] = repr(e)[:200]
                return None
            FP16W["pack_s"] += time.perf_counter() - t0
            FP16W["packed"] += 1
            cache = (key, packed)
            m._r16_fp16w = cache
        targets.append((m, cache[1], is_head))

    def make(m, packed, is_head, orig_forward):
        def forward(x, *a, **k):
            if x.dtype != torch.float32 or torch.is_grad_enabled() or a or k:
                return orig_forward(x, *a, **k)
            x2 = x.reshape(-1, x.shape[-1])
            if x2.shape[0] > max_rows:
                FP16W["fallback_rows"] += 1
                return orig_forward(x)
            FP16W["head_calls" if is_head else "calls"] += 1
            FP16W["max_rows"] = max(FP16W["max_rows"], x2.shape[0])
            y = run(x2.contiguous(), packed)
            return y.view(*x.shape[:-1], y.shape[-1])
        return forward
    for m, packed, is_head in targets:
        m.forward = make(m, packed, is_head, m.forward)

    def undo():
        for m, _, _ in targets:
            m.__dict__.pop("forward", None)
    return undo




def lmhead_chunk_install(model):
    """Experiment arm CPU_MATRIX_LMHEAD=<KiB>: during generate, an fp32 LM head is computed from a bf16 copy of its
    weight (exact: bf16 checkpoint values), upcast <KiB> at a time into a reused buffer, x @ W.T per vocab chunk."""
    kib = int(os.environ.get("CPU_MATRIX_LMHEAD", "0") or 0)
    if not kib:
        return None
    import torch
    import torch.nn.functional as F
    head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    if head is None or getattr(head, "weight", None) is None or head.weight.dtype != torch.float32 \
            or getattr(head, "bias", None) is not None:
        return None
    w = head.weight
    cache = getattr(head, "_r15_bf16", None)
    if cache is None or cache[0] != w._version:
        wb = w.detach().bfloat16()
        if not torch.equal(wb.float(), w.detach()):
            return None  # not bf16-exact values: skip
        cache = (w._version, wb, torch.empty(kib * 256))
        head._r15_bf16 = cache
    _, wb, buf = cache
    n, kk = wb.shape
    rows = max(1, buf.numel() // kk)

    def forward(x):
        x2 = x.reshape(-1, kk)
        out = torch.empty(x2.shape[0], n, dtype = x2.dtype)
        for r in range(0, n, rows):
            rr = min(rows, n - r)
            out[:, r:r + rr] = F.linear(x2, buf[:rr * kk].view(rr, kk).copy_(wb[r:r + rr]))
        LMHEAD_CALLS[0] += 1
        LMHEAD_CALLS[1] = max(LMHEAD_CALLS[1], x2.shape[0])
        return out.view(*x.shape[:-1], n)
    head.forward = forward

    def undo():
        del head.forward
    return undo


def instrument_op_dtypes(trainer):
    """GEMM-family ops by input dtypes and FLOPs over every training step (forward, recompute, backward), plus GRPO
    rollouts: tells a bf16 (oneDNN / AMX) GEMM from an fp32 one (MKL sgemm) that a dtype cast forced."""
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    gemm = {"mm", "addmm", "bmm", "baddbmm", "matmul", "linear", "_scaled_dot_product_flash_attention_for_cpu",
            "_scaled_dot_product_flash_attention_for_cpu_backward", "mkldnn_linear", "_addmm_activation", "addbmm"}
    acc = {}

    def flops(name, args):
        try:
            if name in ("mm", "addmm", "_addmm_activation"):
                a, b = (args[0], args[1]) if name == "mm" else (args[1], args[2])
                return 2 * a.shape[0] * a.shape[1] * b.shape[1]
            if name in ("bmm", "baddbmm"):
                a, b = (args[0], args[1]) if name == "bmm" else (args[1], args[2])
                return 2 * a.shape[0] * a.shape[1] * a.shape[2] * b.shape[2]
        except Exception:  # noqa: BLE001
            pass
        return 0

    class M(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            name = func._schema.name.split("::")[-1]
            if name in gemm:
                dts = ",".join(str(x.dtype).replace("torch.", "") for x in args if isinstance(x, torch.Tensor))
                k = f"{name}[{dts}]"
                e = acc.setdefault(k, [0, 0])
                e[0] += 1
                e[1] += flops(name, args)
            return func(*args, **(kwargs or {}))
    orig = trainer.training_step

    def training_step(*a, **k):
        with M():
            return orig(*a, **k)
    trainer.training_step = training_step
    return acc  # filled in place: {op[dtypes]: [calls, flops]}


def threads_info(trainer):
    """What the defaults picked: torch intra / inter-op pools, OMP / MKL env, Unsloth's fit, dataloader, Inductor."""
    import torch
    info = {"torch_intra": torch.get_num_threads(), "torch_interop": torch.get_num_interop_threads(),
            "cpu_count": os.cpu_count(), "dataloader_num_workers": getattr(trainer.args, "dataloader_num_workers", None)}
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "UNSLOTH_CPU_THREADS", "TORCHINDUCTOR_COMPILE_THREADS"):
        info[k] = os.environ.get(k)
    try:
        info["affinity"] = len(os.sched_getaffinity(0))
    except Exception:  # noqa: BLE001
        info["affinity"] = None
    try:
        import psutil
        info["physical_cores"] = psutil.cpu_count(logical=False)
    except Exception:  # noqa: BLE001
        info["physical_cores"] = None
    try:
        from unsloth_zoo.cpu import runtime as rt
        info["unsloth_fit"] = list(rt._THREADS_FITTED) if rt._THREADS_FITTED else None
        info["unsloth_compute_cores"] = rt.cpu_compute_cores()
    except Exception as e:  # noqa: BLE001
        info["unsloth_fit"] = f"? {e!r}"
    try:
        import torch._inductor.config as ic
        info["inductor_compile_threads"] = ic.compile_threads
        info["inductor_worker_start"] = getattr(ic, "worker_start_method", None)
    except Exception as e:  # noqa: BLE001
        info["inductor_compile_threads"] = f"? {e!r}"
    try:
        pi = torch.__config__.parallel_info()
        info["parallel_info"] = [ln.strip() for ln in pi.splitlines() if "threads" in ln.lower() or "backend" in ln.lower()][:8]
    except Exception:  # noqa: BLE001
        pass
    return info


def instrument_compile():
    """Where a cold CPU compile goes: C++ builds (count + seconds, in this process), the MSVC env load, the probe,
    the layer compile call. Dynamo's own phase timers are read after training."""
    prof = {"cpp_build_n": 0, "cpp_build_s": 0.0, "cpp_build_max_s": 0.0}

    def timed(obj, name, key):
        orig = getattr(obj, name, None)
        if orig is None:
            return

        def f(*a, **k):
            t0 = time.perf_counter()
            try:
                return orig(*a, **k)
            finally:
                dt = time.perf_counter() - t0
                prof[key + "_n"] = prof.get(key + "_n", 0) + 1
                prof[key + "_s"] = round(prof.get(key + "_s", 0.0) + dt, 3)
                if key == "cpp_build":
                    prof["cpp_build_max_s"] = round(max(prof["cpp_build_max_s"], dt), 3)
        f.__wrapped__ = orig
        setattr(obj, name, f)
    try:
        from torch._inductor import cpp_builder
        timed(cpp_builder.CppBuilder, "build", "cpp_build")
    except Exception as e:  # noqa: BLE001
        prof["cpp_builder_error"] = repr(e)[:120]
    try:
        from unsloth_zoo.cpu import loader
        timed(loader, "_load_msvc_environment", "msvc_env")
        timed(loader, "_compile_layers", "layer_compile_call")
        if hasattr(loader._cpu_compile_unavailable, "__wrapped__"):
            timed(loader, "_cpu_compile_unavailable", "compile_probe")
    except Exception as e:  # noqa: BLE001
        prof["loader_error"] = repr(e)[:120]
    try:
        import torch._inductor.config as ic
        prof["compile_threads"] = ic.compile_threads
        prof["cache_dir"] = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
        prof["precompile_headers"] = getattr(ic, "cpp_cache_precompile_headers", None)
    except Exception:  # noqa: BLE001
        pass
    return prof


def compile_prof_summary(prof):
    try:
        from torch._dynamo.utils import compilation_time_metrics
        tot = {k: round(sum(v), 2) for k, v in compilation_time_metrics.items() if v}
        prof["dynamo_phases_s"] = dict(sorted(tot.items(), key=lambda kv: -kv[1])[:12])
    except Exception as e:  # noqa: BLE001
        prof["dynamo_phases_error"] = repr(e)[:120]
    try:
        from torch._dynamo.utils import counters
        prof["inductor_counters"] = {k: int(v) for k, v in counters["inductor"].items()
                                     if any(x in k for x in ("fxgraph", "cache", "kernel"))}
        prof["aot_autograd_counters"] = {k: int(v) for k, v in counters["aot_autograd"].items() if "cache" in k}
        import torch._inductor.config as ic
        from torch._inductor.runtime.cache_dir_utils import cache_dir
        prof["fx_graph_cache"] = getattr(ic, "fx_graph_cache", None)
        prof["autograd_cache"] = getattr(getattr(__import__("torch")._functorch, "config", None), "enable_autograd_cache", None)
        prof["resolved_cache_dir"] = cache_dir()
    except Exception:  # noqa: BLE001
        pass
    return prof


def sdpa_probe(ctx):
    """Which CPU SDPA kernels a Qwen3-0.6B-shaped GQA causal call (fwd + bwd) dispatches to under ctx."""
    try:
        import torch
        import torch.nn.functional as F
        q = torch.randn(1, 16, 32, 128, requires_grad = True)
        k = torch.randn(1, 8, 32, 128, requires_grad = True)
        v = torch.randn(1, 8, 32, 128, requires_grad = True)
        with ctx, torch.profiler.profile() as p:
            F.scaled_dot_product_attention(q, k, v, is_causal = True, enable_gqa = True).sum().backward()
        return sorted({e.name for e in p.events() if "scaled_dot_product" in e.name})
    except Exception as e:  # noqa: BLE001
        return repr(e)[:200]


def run_child(mode, model_key, steps, threads, seq):
    rec = {"mode": mode, "model": model_key, "step_wall_s": [], "step_cpu_s": [], "losses": [], "dropped_cfg": []}
    # Test-harness switches a user never sets (CI test env): they bypass the CPU backend selection.
    rec["cleared_env"] = [k for k in ("UNSLOTH_ZOO_DISABLE_GPU_INIT", "UNSLOTH_ALLOW_CPU", "UNSLOTH_IS_PRESENT",
                                      "UNSLOTH_COMPILE_DISABLE") if os.environ.pop(k, None) is not None]
    if sys.platform == "darwin":
        rec["mlx_installed"] = __import__("importlib.util").util.find_spec("mlx") is not None
        os.environ.setdefault("UNSLOTH_FORCE_GPU_PATH", "1")  # torch CPU path, not MLX
    if mode == "sft_compile":
        os.environ["UNSLOTH_CPU_COMPILE"] = "1"
    t0 = time.perf_counter()
    import unsloth  # noqa: F401  (before transformers / trl)
    from unsloth import FastLanguageModel
    import torch
    from datasets import Dataset
    rec["import_s"] = round(time.perf_counter() - t0, 2)
    import unsloth_zoo
    rec["zoo_file"] = unsloth_zoo.__file__  # which tree this arm imported
    try:
        import unsloth as _u
        rec["unsloth_file"] = _u.__file__
    except Exception:  # noqa: BLE001
        pass
    try:
        from unsloth_zoo import device_type as dt
        rec["device_type"] = dt.DEVICE_TYPE
    except Exception as e:  # noqa: BLE001
        rec["device_type"] = f"? {e!r}"
    counters = Counters()
    rec["instrument_error"] = instrument(counters)
    gen_t = instrument_generate()
    if os.environ.get("CPU_MATRIX_FAST_UPCAST", "0") == "1":
        rec["fast_upcast"] = patch_fast_upcast()
    if os.environ.get("CPU_MATRIX_FP32_ROLLOUT", "0") == "1":  # round 15: fp32 frozen copy for rollouts on any CPU
        try:
            from unsloth_zoo.cpu import loader as _ld
            _ld._slow_bf16_upcast = lambda: True
            rec["fp32_rollout_patch"] = True
        except Exception as e:  # noqa: BLE001
            rec["fp32_rollout_patch"] = repr(e)[:200]
    if os.environ.get("CPU_MATRIX_FP16W", "0") not in ("", "0"):  # round 16: fp16 packs supersede #2935's fp32 copy
        try:
            from unsloth_zoo.cpu import loader as _ld
            if hasattr(_ld, "_avx2_fp32_rollout"):
                _ld._avx2_fp32_rollout = lambda *a, **k: False
        except Exception:  # noqa: BLE001
            pass
    arm_keys = ("UNSLOTH_CPU_FROZEN_BF16", "CPU_MATRIX_FAST_UPCAST", *os.environ.get("CPU_MATRIX_ARM_KEYS", "").split(","))
    rec["arm_env"] = {k: os.environ.get(k) for k in arm_keys if k and k in os.environ}
    rec["compile_prof"] = instrument_compile() if mode == "sft_compile" else None
    if threads:
        torch.set_num_threads(threads)
    rec["torch_threads"] = torch.get_num_threads()

    t0 = time.perf_counter()
    full = mode == "full_ft"
    model, tok = FastLanguageModel.from_pretrained(model_name=MODELS[model_key], max_seq_length=seq,
                                                   load_in_4bit=False, dtype=None, full_finetuning=full)
    if not full:
        model = FastLanguageModel.get_peft_model(model, r=8, lora_alpha=8, lora_dropout=0,
                                                 target_modules=LORA_TARGETS,
                                                 use_gradient_checkpointing="unsloth", random_state=0)
    rec["load_s"] = round(time.perf_counter() - t0, 2)
    t = getattr(tok, "tokenizer", tok)
    if t.pad_token is None:
        t.pad_token = t.eos_token
    rec["param_dtype"] = str(next(model.parameters()).dtype)
    rec["chunked_ce_patched"] = any(vars(m).get("_unsloth_cpu_chunked_ce", False) for m in model.modules())

    out_dir = tempfile.mkdtemp(prefix=f"cpu_matrix_{mode}_")
    common = dict(output_dir=out_dir, max_steps=steps, per_device_train_batch_size=int(os.environ.get("CPU_MATRIX_BS", "2")),
                  gradient_accumulation_steps=1,
                  learning_rate=1e-4, logging_steps=1, save_strategy="no", report_to="none", seed=0,
                  )  # dataloader_num_workers left at the library default (recorded in threads_info)
    words = "the quick brown fox jumps over the lazy dog while a small cat sleeps in the warm sun".split()

    def sentence(i, n=24):
        return " ".join(words[(i + j) % len(words)] for j in range(n))

    if mode in ("sft_lora", "full_ft", "sft_compile"):
        from trl import SFTConfig, SFTTrainer
        ds = Dataset.from_list([{"text": sentence(i)} for i in range(32)])
        c, rec["dropped_cfg"] = cfg(SFTConfig, dataset_text_field="text", max_length=seq, max_seq_length=seq,
                                    packing=False, **common)
        trainer = SFTTrainer(model=model, train_dataset=ds, args=c, **tok_kwarg(SFTTrainer, tok))
    elif mode == "sft_masked":
        from trl import SFTConfig, SFTTrainer
        n = max(8, int(seq * 0.85))
        ds = Dataset.from_list([{"prompt": sentence(i, n * 3 // 4), "completion": " " + sentence(i + 5, n // 4)}
                                for i in range(32)])
        c, rec["dropped_cfg"] = cfg(SFTConfig, max_length=seq, max_seq_length=seq, packing=False,
                                    completion_only_loss=True, **common)
        trainer = SFTTrainer(model=model, train_dataset=ds, args=c, **tok_kwarg(SFTTrainer, tok))
        try:
            batch = next(iter(trainer.get_train_dataloader()))
            lab = batch["labels"]
            rec["masked_label_frac"] = round(float((lab == -100).float().mean()), 3)
        except Exception as e:  # noqa: BLE001
            rec["masked_label_frac"] = f"? {e!r}"[:120]
    elif mode == "dpo":
        from trl import DPOConfig, DPOTrainer
        ds = Dataset.from_list([{"prompt": sentence(i, 8), "chosen": " " + sentence(i + 3, 12),
                                 "rejected": " " + sentence(i + 7, 10)} for i in range(32)])
        c, rec["dropped_cfg"] = cfg(DPOConfig, max_length=seq, max_prompt_length=seq // 2, beta=0.1, **common)
        trainer = DPOTrainer(model=model, ref_model=None, train_dataset=ds, args=c, **tok_kwarg(DPOTrainer, tok))
    elif mode == "grpo":
        from trl import GRPOConfig, GRPOTrainer
        ds = Dataset.from_list([{"prompt": sentence(i, 8)} for i in range(32)])

        import hashlib
        rec["completion_sha"] = []

        def length_reward(completions, **kw):  # sampled-token fingerprint per step (equal tokens across arms)
            rec["completion_sha"].append(hashlib.sha1(repr(completions).encode()).hexdigest()[:12])
            return [float(len(str(x))) / 100.0 for x in completions]
        c, rec["dropped_cfg"] = cfg(GRPOConfig, num_generations=int(os.environ.get("CPU_MATRIX_GRPO_GENS", "2")),
                                    max_completion_length=int(os.environ.get("CPU_MATRIX_GRPO_LEN", "16")),
                                    max_prompt_length=seq // 2, use_vllm=False, temperature=1.0, beta=0.0, **common)
        trainer = GRPOTrainer(model=model, reward_funcs=[length_reward], train_dataset=ds, args=c,
                              **tok_kwarg(GRPOTrainer, tok))
    else:
        raise SystemExit(f"unknown mode {mode}")
    profile = os.environ.get("CPU_MATRIX_PROFILE", "0") == "1"
    if os.environ.get("CPU_MATRIX_OPDTYPES", "0") == "1":
        rec["op_dtypes"] = instrument_op_dtypes(trainer)
    rec["threads_info"] = threads_info(trainer)
    trainer.add_callback(step_callback(rec, profile_last=profile))
    import contextlib
    sdpa = os.environ.get("CPU_MATRIX_SDPA", "").strip().lower()  # round 15: force a CPU SDPA backend for training
    ctx = contextlib.nullcontext  # a factory: a generator context manager cannot be entered twice
    if sdpa:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        ctx = lambda: sdpa_kernel([{"math": SDPBackend.MATH, "flash": SDPBackend.FLASH_ATTENTION}[sdpa]])  # noqa: E731
    rec["sdpa_ops"] = sdpa_probe(ctx())
    t0 = time.perf_counter()
    with ctx():
        trainer.train()
    rec["train_s"] = round(time.perf_counter() - t0, 2)
    rec["optimizer"] = optimizer_info(trainer)
    try:  # engagement of round-10 levers: CPU grad-norm patch installed
        from torch.nn.utils import clip_grad as _cg
        rec["grad_norm_patched"] = bool(getattr(getattr(_cg, "_get_total_norm", None), "_unsloth_cpu", False))
    except Exception:  # noqa: BLE001
        pass
    try:  # engagement of round-11 lever: frozen projections still bf16-stored (upcast per call) vs float32
        rec["frozen_bf16_stored"] = sum(1 for m in trainer.model.modules() if type(m).__name__ == "Bf16StoredLinear"
                                        and getattr(m, "weight", None) is not None and m.weight.dtype == torch.bfloat16)
    except Exception:  # noqa: BLE001
        pass
    rec["compile"] = compile_info(trainer.model)
    rec["fast_paths"] = dict(counters.n)
    rec["generate"] = {"calls": gen_t["n"], "s": round(gen_t["s"], 3), "lmhead_chunk_calls": LMHEAD_CALLS[0], "lmhead_max_rows": LMHEAD_CALLS[1],
                       "fp16w": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in FP16W.items()}}
    for _name, _mod in (("growable_kv", "kv_cache"), ("fast_multinomial", "sampling")):  # CPU decode levers engaged
        try:
            rec["generate"][_name] = __import__(f"unsloth_zoo.cpu.{_mod}", fromlist=["ENGAGED"]).ENGAGED[0]
        except Exception:  # noqa: BLE001
            rec["generate"][_name] = None
    rec["peak_rss_mib"] = round(peak_rss_mib(), 1)
    rec["threads_after_train"] = torch.get_num_threads()
    if rec["compile_prof"] is not None:
        rec["compile_prof"] = compile_prof_summary(rec["compile_prof"])
    acc = getattr(trainer, "accelerator", None)
    rec["precision"] = {"bf16": bool(getattr(trainer.args, "bf16", False)), "fp16": bool(getattr(trainer.args, "fp16", False)),
                        "accelerate_mixed_precision": str(getattr(acc, "mixed_precision", None))}
    try:
        from unsloth_zoo.cpu import runtime as rt
        rec["precision"]["cpu_autocast_dtype"] = str(rt.cpu_autocast_dtype())
    except Exception as e:  # noqa: BLE001
        rec["precision"]["cpu_autocast_dtype"] = f"? {e!r}"
    wall = [x for x in rec["step_wall_s"] if x is not None]
    cpu = [x for x in rec["step_cpu_s"] if x is not None]
    steady_w, steady_c = wall[1:] or wall, cpu[1:] or cpu
    med = lambda v: sorted(v)[len(v) // 2] if v else None  # noqa: E731
    rec["s_per_step"] = med(steady_w)
    rec["cpu_s_per_step"] = med(steady_c)
    rec["first_step_s"] = wall[0] if wall else None
    finite = all(x == x and abs(x) != float("inf") for x in rec["losses"])
    rec["losses_finite"] = finite
    print(RESULT_TAG + json.dumps(rec), flush=True)


def env_info():
    info = {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(),
            "cpu_count": os.cpu_count()}
    for pkg in ("torch", "transformers", "trl", "peft", "accelerate", "datasets", "unsloth", "unsloth_zoo", "mlx"):
        try:
            from importlib.metadata import version
            info[pkg] = version(pkg.replace("_", "-") if pkg == "unsloth_zoo" else pkg)
        except Exception:  # noqa: BLE001
            info[pkg] = None
    return info


def diag(threads):
    """CPU identity + GEMM throughput per dtype at a Qwen3-0.6B MLP shape: tells hardware from software."""
    import torch
    out = {"processor": platform.processor()}
    try:
        if sys.platform == "darwin":
            out["brand"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        elif sys.platform == "win32":
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            out["brand"] = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        else:
            with open("/proc/cpuinfo") as f:
                info = f.read()
            out["brand"] = next((ln.split(":", 1)[1].strip() for ln in info.splitlines() if ln.startswith("model name")), "")
            flags = next((ln for ln in info.splitlines() if ln.startswith("flags")), "")
            out["flags"] = [x for x in ("avx2", "avx512f", "avx512_bf16", "amx_bf16", "avx512_fp16", "amx_fp16") if x in flags.split()]
    except Exception as e:  # noqa: BLE001
        out["brand_error"] = repr(e)
    if threads:
        torch.set_num_threads(threads)
    out["threads"] = torch.get_num_threads()
    out["mkldnn"] = bool(torch.backends.mkldnn.is_available())
    out["mkl"] = bool(torch.backends.mkl.is_available())
    try:
        out["blas"] = [ln.strip()[:240] for ln in torch.__config__.show().splitlines() if "BLAS" in ln or "LAPACK" in ln][:4]
    except Exception:  # noqa: BLE001
        pass
    gemm = {}
    for name, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16), ("fp16", torch.float16)):
        try:
            a, b = torch.randn(256, 1024, dtype=dt), torch.randn(1024, 3072, dtype=dt)
            a @ b
            reps, t0 = 0, time.perf_counter()
            while time.perf_counter() - t0 < 1.5:
                a @ b
                reps += 1
            dt_s = (time.perf_counter() - t0) / reps
            gemm[name] = round(2 * 256 * 1024 * 3072 / dt_s / 1e9, 1)
        except Exception as e:  # noqa: BLE001
            gemm[name] = repr(e)[:100]
    out["gemm_gflops_256x1024x3072"] = gemm
    try:
        w = torch.randn(3072, 1024).to(torch.bfloat16)
        assert torch.equal(shift_upcast(w), w.float())

        def best(f):
            f()
            ts = []
            for _ in range(7):
                t0 = time.perf_counter()
                f()
                ts.append(time.perf_counter() - t0)
            return round(min(ts) * 1e3, 3)
        w32 = w.float()
        out["upcast_ms_3072x1024"] = {"to_float": best(lambda: w.to(torch.float32)), "shift": best(lambda: shift_upcast(w)),
                                      "fp32_clone": best(lambda: w32.clone())}
    except Exception as e:  # noqa: BLE001
        out["upcast_error"] = repr(e)[:200]
    try:
        q = torch.randn(2, 16, 128, 128)
        k = torch.randn(2, 8, 128, 128)
        import torch.nn.functional as F
        qq = q.requires_grad_()
        t0 = time.perf_counter()
        for _ in range(5):
            F.scaled_dot_product_attention(qq, k.repeat_interleave(2, 1), k.repeat_interleave(2, 1), is_causal=True).sum().backward()
        out["sdpa_fwd_bwd_ms_fp32"] = round((time.perf_counter() - t0) / 5 * 1e3, 2)
    except Exception as e:  # noqa: BLE001
        out["sdpa_error"] = repr(e)[:100]
    return out


def zoo_checkout(ref, cache={}, repo="unsloth-zoo"):
    """A git checkout of unsloth-zoo (or `repo`) at `ref` for a base arm in the same job (same runner, interleaved)."""
    if (repo, ref) not in cache:
        d = tempfile.mkdtemp(prefix=f"cpu_matrix_{repo.replace('-', '_')}_")
        url = f"https://github.com/unslothai/{repo}"
        subprocess.run(["git", "init", "-q", d], check=True)
        for attempt in range(4):  # a transient fetch failure (exit 128) must not fail the whole job
            if subprocess.run(["git", "-C", d, "fetch", "-q", "--depth", "1", url, ref]).returncode == 0:
                break
            time.sleep(15 * (attempt + 1))
        else:
            subprocess.run(["git", "-C", d, "fetch", "-q", "--depth", "1", url, ref], check=True)
        subprocess.run(["git", "-C", d, "checkout", "-q", "FETCH_HEAD"], check=True)
        sha = subprocess.run(["git", "-C", d, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        cache[(repo, ref)] = (d, sha)
    return cache[(repo, ref)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="cpu_matrix.json")
    p.add_argument("--modes", default=",".join(MODES))
    p.add_argument("--models", default=",".join(MODELS))
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--seq", type=int, default=64)
    p.add_argument("--threads", type=int, default=2, help="torch threads per cell (0 = leave the default)")
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--tiny", action="store_true", help="ignored (always tiny)")
    p.add_argument("--no-fail", action="store_true", help="exit 0 even when a cell fails")
    p.add_argument("--diag", action="store_true", help="add CPU identity + per-dtype GEMM throughput to the report")
    p.add_argument("--profile", action="store_true", help="torch.profiler op table of each cell's last step")
    p.add_argument("--arms", default="", help="A/B arms 'name=K=V+K=V;name2=...' run interleaved per cell")
    p.add_argument("--repeats", type=int, default=1, help="interleaved rounds per arm")
    p.add_argument("--child", nargs=2, metavar=("MODE", "MODEL"), help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.child:
        return run_child(a.child[0], a.child[1], a.steps, a.threads, a.seq)

    report = {"env": env_info(), "cells": []}
    print(json.dumps(report["env"]), flush=True)
    if a.diag:
        report["diag"] = diag(a.threads)
        print("diag: " + json.dumps(report["diag"]), flush=True)
    if a.profile:
        os.environ["CPU_MATRIX_PROFILE"] = "1"
    arms = [("default", {})]
    if a.arms:
        arms = []
        for spec in a.arms.split(";"):
            name, _, kv = spec.partition("=")
            arms.append((name, dict(x.split("=", 1) for x in kv.split("+") if x)))
    plan = [(m, mode, arm, rep) for m in a.models.split(",") for mode in a.modes.split(",")
            for rep in range(a.repeats) for arm in arms]
    for model_key, mode, (arm_name, arm_env), rep in plan:
        if True:
            cmd = [sys.executable, "-u", os.path.abspath(__file__), "--child", mode, model_key,
                   "--steps", str(a.steps), "--threads", str(a.threads), "--seq", str(a.seq)]
            env = dict(os.environ, PYTHONIOENCODING="utf-8", **arm_env)
            env["CPU_MATRIX_ARM_KEYS"] = ",".join(arm_env)
            if env.get("TORCHINDUCTOR_CACHE_DIR") == "fresh":  # a cold Inductor cache for this cell only
                env["TORCHINDUCTOR_CACHE_DIR"] = tempfile.mkdtemp(prefix="cpu_matrix_ind_")
            elif env.get("TORCHINDUCTOR_CACHE_DIR") == "shared":  # one dir per job: cold on its first use, then warm
                env["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(tempfile.gettempdir(), "cpu_matrix_ind_shared")
            zoo_ref = env.pop("CPU_MATRIX_ZOO_REF", None)
            if zoo_ref:
                zdir, zsha = zoo_checkout(zoo_ref)
                env["PYTHONPATH"] = zdir + os.pathsep + env.get("PYTHONPATH", "")
                report.setdefault("zoo_refs", {})[zoo_ref] = zsha
            u_ref = env.pop("CPU_MATRIX_UNSLOTH_REF", None)
            if u_ref:  # base unsloth tree for this arm: PYTHONPATH precedes site-packages
                udir, usha = zoo_checkout(u_ref, repo="unsloth")
                env["PYTHONPATH"] = udir + os.pathsep + env.get("PYTHONPATH", "")
                report.setdefault("unsloth_refs", {})[u_ref] = usha
            t0 = time.perf_counter()
            try:
                pr = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                                    timeout=a.timeout, text=True, encoding="utf-8", errors="replace")
                out, rc = pr.stdout, pr.returncode
            except subprocess.TimeoutExpired as e:
                out, rc = (e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or ""), "timeout"
            cell = {"mode": mode, "model": model_key, "arm": arm_name, "rep": rep, "rc": rc, "wall_s": round(time.perf_counter() - t0, 1)}
            res = [ln[len(RESULT_TAG):] for ln in out.splitlines() if ln.startswith(RESULT_TAG)]
            if res:
                cell.update(json.loads(res[-1]))
            cell["ok"] = rc == 0 and bool(res) and cell.get("losses_finite", False) and bool(cell.get("step_wall_s"))
            seen, warns = set(), []
            for ln in out.splitlines():
                s = ln.strip()
                if s and not s.startswith(RESULT_TAG) and WARN_RE.search(s) and s[:160] not in seen:
                    seen.add(s[:160])
                    warns.append(s[:400])
            cell["warnings"] = warns[:40]
            if not cell["ok"]:
                cell["tail"] = out.splitlines()[-60:]
            report["cells"].append(cell)
            fp = cell.get("fast_paths", {})
            print(f"[{'PASS' if cell['ok'] else 'FAIL'}] {model_key:6s} {mode:12s} {arm_name}#{rep} rc={rc} "
                  f"s/step={cell.get('s_per_step')} cpu/step={cell.get('cpu_s_per_step')} "
                  f"first={cell.get('first_step_s')} rss={cell.get('peak_rss_mib')}MiB dev={cell.get('device_type')} "
                  f"ce={fp.get('chunked_ce')} logps={fp.get('chunked_logps')}/{fp.get('chunked_logps_ckpt')} "
                  f"fused={cell.get('optimizer', {}).get('fused')} compiled={cell.get('compile', {}).get('compiled_modules')} "
                  f"thr={cell.get('threads_info', {}).get('torch_intra')} warns={len(warns)}", flush=True)
            if not cell["ok"]:
                print("\n".join("    | " + x for x in cell["tail"][-25:]), flush=True)
            with open(a.out, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=1)
    bad = [c for c in report["cells"] if not c["ok"]]
    print(f"cpu_matrix: {len(report['cells']) - len(bad)} / {len(report['cells'])} cells passed -> {a.out}")
    return 1 if bad and not a.no_fail else 0


if __name__ == "__main__":
    sys.exit(main())
