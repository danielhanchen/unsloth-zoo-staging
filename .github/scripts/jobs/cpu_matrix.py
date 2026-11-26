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
}
MODES = ("sft_lora", "full_ft", "dpo", "grpo", "sft_compile")
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
                self.prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU])
                self.prof.start()
            self.t, self.c = time.perf_counter(), time.process_time()

        def on_step_end(self, args, state, control, **kw):
            rec["step_wall_s"].append(round(time.perf_counter() - self.t, 4))
            rec["step_cpu_s"].append(round(time.process_time() - self.c, 4))
            if self.prof is not None:
                self.prof.stop()
                ev = sorted(self.prof.key_averages(), key=lambda e: e.self_cpu_time_total, reverse=True)
                total = sum(e.self_cpu_time_total for e in ev) or 1
                rec["profile_top"] = [[e.key, round(e.self_cpu_time_total / 1e3, 1), round(100 * e.self_cpu_time_total / total, 1),
                                       e.count] for e in ev[:20]]
                rec["profile_note"] = "profiled step (excluded from s/step median): [op, self CPU ms, %, calls]"
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


def run_child(mode, model_key, steps, threads, seq):
    rec = {"mode": mode, "model": model_key, "step_wall_s": [], "step_cpu_s": [], "losses": [], "dropped_cfg": []}
    # Test-harness switches a user never sets (CI test env): they bypass the CPU backend selection.
    rec["cleared_env"] = [k for k in ("UNSLOTH_ZOO_DISABLE_GPU_INIT", "UNSLOTH_ALLOW_CPU", "UNSLOTH_IS_PRESENT",
                                      "UNSLOTH_COMPILE_DISABLE") if os.environ.pop(k, None) is not None]
    if sys.platform == "darwin":
        rec["mlx_installed"] = __import__("importlib").util.find_spec("mlx") is not None
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
        from unsloth_zoo import device_type as dt
        rec["device_type"] = dt.DEVICE_TYPE
    except Exception as e:  # noqa: BLE001
        rec["device_type"] = f"? {e!r}"
    counters = Counters()
    rec["instrument_error"] = instrument(counters)
    if os.environ.get("CPU_MATRIX_FAST_UPCAST", "0") == "1":
        rec["fast_upcast"] = patch_fast_upcast()
    rec["arm_env"] = {k: os.environ.get(k) for k in ("UNSLOTH_CPU_FROZEN_BF16", "CPU_MATRIX_FAST_UPCAST") if k in os.environ}
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
    common = dict(output_dir=out_dir, max_steps=steps, per_device_train_batch_size=2, gradient_accumulation_steps=1,
                  learning_rate=1e-4, logging_steps=1, save_strategy="no", report_to="none", seed=0,
                  dataloader_num_workers=0)
    words = "the quick brown fox jumps over the lazy dog while a small cat sleeps in the warm sun".split()

    def sentence(i, n=24):
        return " ".join(words[(i + j) % len(words)] for j in range(n))

    if mode in ("sft_lora", "full_ft", "sft_compile"):
        from trl import SFTConfig, SFTTrainer
        ds = Dataset.from_list([{"text": sentence(i)} for i in range(32)])
        c, rec["dropped_cfg"] = cfg(SFTConfig, dataset_text_field="text", max_length=seq, max_seq_length=seq,
                                    packing=False, **common)
        trainer = SFTTrainer(model=model, train_dataset=ds, args=c, **tok_kwarg(SFTTrainer, tok))
    elif mode == "dpo":
        from trl import DPOConfig, DPOTrainer
        ds = Dataset.from_list([{"prompt": sentence(i, 8), "chosen": " " + sentence(i + 3, 12),
                                 "rejected": " " + sentence(i + 7, 10)} for i in range(32)])
        c, rec["dropped_cfg"] = cfg(DPOConfig, max_length=seq, max_prompt_length=seq // 2, beta=0.1, **common)
        trainer = DPOTrainer(model=model, ref_model=None, train_dataset=ds, args=c, **tok_kwarg(DPOTrainer, tok))
    elif mode == "grpo":
        from trl import GRPOConfig, GRPOTrainer
        ds = Dataset.from_list([{"prompt": sentence(i, 8)} for i in range(32)])

        def length_reward(completions, **kw):
            return [float(len(str(x))) / 100.0 for x in completions]
        c, rec["dropped_cfg"] = cfg(GRPOConfig, num_generations=2, max_completion_length=16,
                                    max_prompt_length=seq // 2, use_vllm=False, temperature=1.0, beta=0.0, **common)
        trainer = GRPOTrainer(model=model, reward_funcs=[length_reward], train_dataset=ds, args=c,
                              **tok_kwarg(GRPOTrainer, tok))
    else:
        raise SystemExit(f"unknown mode {mode}")
    profile = os.environ.get("CPU_MATRIX_PROFILE", "0") == "1"
    trainer.add_callback(step_callback(rec, profile_last=profile))
    t0 = time.perf_counter()
    trainer.train()
    rec["train_s"] = round(time.perf_counter() - t0, 2)
    rec["optimizer"] = optimizer_info(trainer)
    rec["compile"] = compile_info(trainer.model)
    rec["fast_paths"] = dict(counters.n)
    rec["peak_rss_mib"] = round(peak_rss_mib(), 1)
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


def zoo_checkout(ref, cache={}):
    """A git checkout of unsloth-zoo at `ref` for a base arm in the same job (same runner, interleaved)."""
    if ref not in cache:
        d = tempfile.mkdtemp(prefix="cpu_matrix_zoo_")
        url = "https://github.com/unslothai/unsloth-zoo"
        subprocess.run(["git", "init", "-q", d], check=True)
        subprocess.run(["git", "-C", d, "fetch", "-q", "--depth", "1", url, ref], check=True)
        subprocess.run(["git", "-C", d, "checkout", "-q", "FETCH_HEAD"], check=True)
        sha = subprocess.run(["git", "-C", d, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        cache[ref] = (d, sha)
    return cache[ref]


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
            zoo_ref = env.pop("CPU_MATRIX_ZOO_REF", None)
            if zoo_ref:
                zdir, zsha = zoo_checkout(zoo_ref)
                env["PYTHONPATH"] = zdir + os.pathsep + env.get("PYTHONPATH", "")
                report.setdefault("zoo_refs", {})[zoo_ref] = zsha
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
                  f"warns={len(warns)}", flush=True)
            if not cell["ok"]:
                print("\n".join("    | " + x for x in cell["tail"][-25:]), flush=True)
            with open(a.out, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=1)
    bad = [c for c in report["cells"] if not c["ok"]]
    print(f"cpu_matrix: {len(report['cells']) - len(bad)} / {len(report['cells'])} cells passed -> {a.out}")
    return 1 if bad and not a.no_fail else 0


if __name__ == "__main__":
    sys.exit(main())
