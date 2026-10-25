"""Real Unsloth LoRA SFT on the CPU backend: train, eval probe, save, reload in a fresh process, generate.

    python jobs/cpu_sft.py                                  # unsloth/Qwen3-0.6B, 10 steps
    python jobs/cpu_sft.py --model unsloth/Qwen3.5-0.8B
    python jobs/cpu_sft.py --tiny                           # tiny Qwen3, quick local check
    python jobs/cpu_sft.py --env UNSLOTH_CPU_BF16=0         # force fp32 (non-AMX picture)

Product defaults are kept (optimizer, bf16 resolution, chunked CE); only LoRA shape, data and steps
are fixed. On Apple Silicon the MLX backend is bypassed (UNSLOTH_FORCE_CPU + UNSLOTH_FORCE_GPU_PATH)
so the torch CPU path is what runs. Records step times, peak RSS per phase, dtypes and the loss route.
"""

import os
import platform
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as c  # noqa: E402

p = c.base_parser("cpu_sft", "unsloth/Qwen3-0.6B", "trl-internal-testing/tiny-Qwen3ForCausalLM", default_steps=10)
p.set_defaults(max_seq_length=256, backend="unsloth")
p.add_argument("--env", action="append", default=[], metavar="KEY=VALUE", help="set before importing unsloth")
p.add_argument("--load-in-4bit", type=int, default=0, choices=(0, 1))
p.add_argument("--grad-ckpt", default="unsloth", choices=("unsloth", "true", "false"))
p.add_argument("--lora-r", type=int, default=8)
p.add_argument("--batch", type=int, default=1)
p.add_argument("--grad-accum", type=int, default=2)
p.add_argument("--rows", type=int, default=32)
p.add_argument("--skip-reload", action="store_true")
p.add_argument("--profile", action="store_true", help="torch.profiler top ops for one optimizer step")
p.add_argument("--attn", default=None, help="attn_implementation passed to from_pretrained (eager / sdpa)")
p.add_argument("--frozen-bf16", action="store_true",
               help="experiment: frozen fp32 Linear weights stored in bf16, upcast per call for an fp32 matmul")
p.add_argument("--reload", default=None, help="internal: child mode, reload this adapter dir")
a = p.parse_args()

for item in a.env:
    key, sep, value = item.partition("=")
    if not sep:
        raise SystemExit(f"--env {item!r}: expected KEY=VALUE")
    os.environ[key] = value
if platform.system() == "Darwin" and platform.machine() == "arm64":
    os.environ.setdefault("UNSLOTH_FORCE_CPU", "1")
    os.environ.setdefault("UNSLOTH_FORCE_GPU_PATH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

if os.environ.get("UNSLOTH_ZOO_DISABLE_GPU_INIT") == "1":
    # The zoo's test opt-out reports "cpu" without device detection and leaves core on its GPU loaders.
    raise SystemExit("cpu_sft: UNSLOTH_ZOO_DISABLE_GPU_INIT=1 is set; unset it to test the real CPU path")
a.model = a.model or (a._tiny_model if a.tiny else a._default_model)
a.out = a.out or f"outputs/jobs/{a._job}.json"
PROMPT = "What is the capital of France?"


def peak_rss_mb():
    """Process high-water RSS in MiB (Windows peak working set)."""
    try:
        import psutil
        info = psutil.Process().memory_info()
        if hasattr(info, "peak_wset"):
            return round(info.peak_wset / 2**20, 1)
    except Exception:
        pass
    try:
        import resource
        v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(v / 2**20 if platform.system() == "Darwin" else v / 1024, 1)
    except Exception:
        return None


def host_info():
    info = {"system": platform.system(), "machine": platform.machine(), "python": platform.python_version(),
            "cpu_count": os.cpu_count()}
    try:
        import psutil
        vm = psutil.virtual_memory()
        info.update(ram_total_gb=round(vm.total / 2**30, 2), ram_available_gb=round(vm.available / 2**30, 2),
                    physical_cores=psutil.cpu_count(logical=False))
    except Exception:
        pass
    try:
        from unsloth_zoo.cpu.runtime import cpu_capabilities
        info["capabilities"] = cpu_capabilities()
    except Exception as e:  # noqa: BLE001
        info["capabilities_error"] = repr(e)
    if platform.system() == "Linux":
        try:
            text = open("/proc/cpuinfo").read()
            for key in ("model name", "CPU part"):
                for line in text.splitlines():
                    if line.startswith(key):
                        info["cpu_model"] = line.split(":", 1)[1].strip()
                        break
                if "cpu_model" in info:
                    break
        except OSError:
            pass
    elif platform.system() == "Windows":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            info["cpu_model"] = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            info["cpu_model"] = platform.processor()
    else:
        info["cpu_model"] = platform.processor()
    return info


def build_info(torch):
    """Which BLAS / oneDNN / ACL this torch wheel carries, its ATen CPU dispatch level, and the C++ compiler
    Inductor would find (torch.compile on CPU needs one)."""
    import re
    import shutil
    cfg = torch.__config__.show()
    keys = ("BLAS_INFO", "LAPACK_INFO", "USE_MKL", "USE_MKLDNN", "USE_MKLDNN_ACL", "USE_OPENMP", "TORCH_VERSION")
    out = {k: (re.search(rf"{k}=([^,\n]+)", cfg) or [None, None])[1] for k in keys}
    out["libraries"] = [ln.strip("- ").strip() for ln in cfg.splitlines()
                        if any(t in ln for t in ("Math Kernel", "MKL-DNN", "OpenMP", "Compute Library", "KleidiAI", "OpenBLAS", "Accelerate"))]
    for name, fn in (("cpu_capability", lambda: torch.backends.cpu.get_cpu_capability()),
                     ("mkl", lambda: torch.backends.mkl.is_available()),
                     ("mkldnn", lambda: torch.backends.mkldnn.is_available()),
                     ("mkldnn_acl", lambda: torch.backends.mkldnn.is_acl_available())):
        try:
            out[name] = fn()
        except Exception as e:  # noqa: BLE001
            out[name] = f"n/a ({type(e).__name__})"
    out["compilers"] = {cc: shutil.which(cc) for cc in ("cl", "clang-cl", "clang++", "g++", "c++") if shutil.which(cc)}
    return out


def profile_callback(rec, step=3):
    from transformers import TrainerCallback
    from torch.profiler import ProfilerActivity, profile

    class _P(TrainerCallback):
        prof = None

        def on_step_begin(self, args, state, control, **kw):
            if state.global_step == step:
                self.prof = profile(activities=[ProfilerActivity.CPU])
                self.prof.__enter__()

        def on_step_end(self, args, state, control, **kw):
            if self.prof is not None and state.global_step == step + 1:
                self.prof.__exit__(None, None, None)
                rows = sorted(self.prof.key_averages(), key=lambda e: -e.self_cpu_time_total)[:25]
                total = sum(e.self_cpu_time_total for e in self.prof.key_averages()) or 1
                rec.summary(profile=[{"op": e.key, "self_ms": round(e.self_cpu_time_total / 1e3, 2),
                                      "pct": round(100 * e.self_cpu_time_total / total, 1), "calls": e.count} for e in rows])
                print(self.prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=25), flush=True)
                self.prof = None
    return _P()


def store_frozen_linears_in_bf16(model):
    """Experiment for fp32-only CPUs: half the frozen weight memory, fp32 math. Backward only needs
    dX = dY @ W, so the custom function saves the bf16 weight and upcasts it again there."""
    import torch
    import torch.nn.functional as F

    class _Bf16Stored(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, bias):
            ctx.save_for_backward(weight)
            return F.linear(x, weight.to(x.dtype), bias)

        @staticmethod
        def backward(ctx, grad):
            (weight,) = ctx.saved_tensors
            return grad @ weight.to(grad.dtype), None, None

    # Loaded in bf16 (the checkpoint's own dtype, so storage is lossless); everything except the frozen
    # decoder Linear weights goes to fp32 so activations, norms, embeddings and LoRA stay fp32.
    count = 0
    stored = set()
    for name, module in model.named_modules():
        if type(module) is torch.nn.Linear and not module.weight.requires_grad and "lm_head" not in name:
            module.forward = (lambda m: (lambda x: _Bf16Stored.apply(x, m.weight, m.bias)))(module)
            stored.add(id(module.weight))
            count += 1
    for p in model.parameters():
        if id(p) not in stored and p.dtype == torch.bfloat16:
            p.data = p.data.float()
    for b_name, b in model.named_buffers():
        if b.dtype == torch.bfloat16:
            b.data = b.data.float()
    return count


def probe_batch(tok, n=4):
    import torch
    t = getattr(tok, "tokenizer", tok)
    rows = [r["text"] for r in k.render_chat(tok, k.chat_fixture(n))]
    enc = t(rows, return_tensors="pt", padding=True, truncation=True, max_length=a.max_seq_length)
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels}


def probe_loss(model, batch):
    import torch
    with torch.no_grad():
        return float(model(**batch).loss)


def generate(model, tok):
    import torch
    t = getattr(tok, "tokenizer", tok)
    msgs = [{"role": "user", "content": PROMPT}]
    if getattr(t, "chat_template", None):
        text = t.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    else:
        text = f"user: {PROMPT}\nassistant:"
    enc = t(text, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=16, do_sample=False, pad_token_id=t.pad_token_id)
    return t.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def load(name):
    import torch
    from unsloth import FastLanguageModel
    model, tok = FastLanguageModel.from_pretrained(
        model_name=name, max_seq_length=a.max_seq_length, load_in_4bit=bool(a.load_in_4bit),
        dtype=torch.bfloat16 if getattr(a, "frozen_bf16", False) else None,
        **({"attn_implementation": a.attn} if a.attn else {}),
        use_gradient_checkpointing={"unsloth": "unsloth", "true": True, "false": False}[a.grad_ckpt])
    return FastLanguageModel, model, tok


if a.reload:
    # Child: fresh process, reload the saved adapter, same probe, greedy generate.
    import json
    import time
    import unsloth  # noqa: F401
    import _compat as k
    t0 = time.perf_counter()
    FLM, model, tok = load(a.reload)
    k.ensure_pad(tok)
    loss = probe_loss(model, probe_batch(tok))
    FLM.for_inference(model)
    text = generate(model, tok)
    print("RELOAD_RESULT " + json.dumps({"probe_loss": loss, "generation": text, "load_s": round(time.perf_counter() - t0, 2),
                                          "peak_rss_mb": peak_rss_mb(),
                                          "lora_modules": sum(1 for n, _ in model.named_modules() if n.endswith("lora_A"))}))
    sys.exit(0)

c.seed_everything(a.seed)
with c.JobRecorder(a) as rec:
    import json
    import subprocess
    import time
    rec.summary(host=host_info(), env={kk: os.environ.get(kk) for kk in (
        "UNSLOTH_FORCE_CPU", "UNSLOTH_FORCE_GPU_PATH", "UNSLOTH_CPU_BF16", "UNSLOTH_CPU_FP16", "UNSLOTH_CPU_COMPILE",
        "UNSLOTH_CPU_CHUNKED_CE", "UNSLOTH_ZOO_DISABLE_GPU_INIT", "OMP_NUM_THREADS", "UNSLOTH_CPU_COMPILE_DYNAMIC",
        "UNSLOTH_CPU_FROZEN_DTYPE", "LD_PRELOAD")}, rss_start_mb=peak_rss_mb())
    t0 = time.perf_counter()
    import unsloth
    import torch
    import _compat as k
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    import unsloth_zoo.cpu.ops as cpu_ops

    rec.summary(build=build_info(torch))
    device_type = getattr(unsloth, "DEVICE_TYPE", None)
    rec.set_backend(f"unsloth-{device_type}")
    import unsloth_zoo
    rec.summary(import_s=round(time.perf_counter() - t0, 2), device_type=device_type, torch_threads=torch.get_num_threads(),
                zoo_file=os.path.abspath(unsloth_zoo.__file__))
    if not rec.check("device_is_cpu", device_type == "cpu", f"unsloth.DEVICE_TYPE = {device_type!r}"):
        raise SystemExit("not on the CPU backend")

    # Count which loss route actually runs (module globals are looked up at call time).
    route = {"chunked_ce": 0, "trl_fused_head_cpu": 0}
    for fn_name, key in (("chunked_linear_cross_entropy", "chunked_ce"), ("cpu_chunked_log_probs", "trl_fused_head_cpu")):
        orig = getattr(cpu_ops, fn_name)

        def counted(*args, _orig=orig, _key=key, **kw):
            route[_key] += 1
            return _orig(*args, **kw)
        setattr(cpu_ops, fn_name, counted)

    t1 = time.perf_counter()
    FLM, model, tok = load(a.model)
    model = FLM.get_peft_model(model, r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0, target_modules=k.LORA_TARGETS,
                               use_gradient_checkpointing={"unsloth": "unsloth", "true": True, "false": False}[a.grad_ckpt],
                               random_state=a.seed)
    k.ensure_pad(tok)
    if a.frozen_bf16:
        import gc
        rec.summary(frozen_bf16_linears=store_frozen_linears_in_bf16(model))
        gc.collect()
    load_s = time.perf_counter() - t1
    params = list(model.named_parameters())
    off_cpu = [n for n, p_ in params if p_.device.type != "cpu"]
    rec.check("params_on_cpu", not off_cpu, off_cpu[:5] or f"{len(params)} params on cpu")
    frozen = [p_ for _, p_ in params if not p_.requires_grad and p_.ndim == 2]
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    rec.summary(load_s=round(load_s, 2), rss_after_load_mb=peak_rss_mb(), model_class=type(base).__name__,
                frozen_weight_dtype=str(frozen[0].dtype) if frozen else None,
                chunked_ce_installed=bool(getattr(base, "_unsloth_cpu_chunked_ce", False)),
                lora_modules=sum(1 for n, _ in model.named_modules() if n.endswith("lora_A")),
                trainable_params=sum(p_.numel() for _, p_ in params if p_.requires_grad),
                gradient_checkpointing=bool(getattr(base, "is_gradient_checkpointing", False)),
                linear4bit_modules=sum(1 for m in model.modules() if type(m).__name__ == "Linear4bit"))
    if a.load_in_4bit:
        rec.summary(load_in_4bit_engaged=rec.data["summary"]["linear4bit_modules"] > 0)

    batch = probe_batch(tok)
    probe_before = probe_loss(model, batch)
    ds = Dataset.from_list(k.render_chat(tok, k.chat_fixture(a.rows)))
    dropped = []
    cfg = k.make(SFTConfig, dropped, output_dir=os.path.join(os.path.dirname(a.out) or ".", "cpu_sft_run"),
                 max_steps=a.max_steps, per_device_train_batch_size=a.batch, gradient_accumulation_steps=a.grad_accum,
                 learning_rate=a.lr, warmup_steps=0, lr_scheduler_type="constant", logging_steps=1, save_strategy="no",
                 seed=a.seed, report_to="none", max_grad_norm=1.0, include_num_input_tokens_seen=True,
                 dataloader_num_workers=0, dataset_text_field="text", max_length=a.max_seq_length,
                 max_seq_length=a.max_seq_length, packing=False, **c.cfg_overrides(a))
    callbacks = [c.metrics_callback(rec)] + ([profile_callback(rec)] if a.profile else [])
    trainer = SFTTrainer(model=model, train_dataset=ds, args=cfg, callbacks=callbacks,
                         **k.tokenizer_kwarg(SFTTrainer, tok))
    rec.summary(optim=str(getattr(trainer.args.optim, "value", trainer.args.optim)), bf16=bool(trainer.args.bf16),
                fp16=bool(trainer.args.fp16), dropped_config_args=dropped)
    before = c.adapter_fingerprint(model)
    frozen_before = frozen[0].detach().clone() if frozen else None
    t2 = time.perf_counter()
    out = trainer.train()
    train_s = time.perf_counter() - t2
    after = c.adapter_fingerprint(model)
    probe_after = probe_loss(model, batch)
    steps_ms = sorted(s["time_ms"] for s in rec.data["steps"] if s.get("time_ms"))
    rec.summary(train_s=round(train_s, 2), final_loss=out.training_loss, rss_after_train_mb=peak_rss_mb(),
                probe_loss_before=probe_before, probe_loss_after=probe_after, loss_route=dict(route),
                p95_step_ms=steps_ms[min(len(steps_ms) - 1, int(0.95 * len(steps_ms)))] if steps_ms else None)
    if os.environ.get("UNSLOTH_CPU_COMPILE") == "1":
        try:
            from torch._inductor.cpp_builder import get_cpp_compiler
            rec.summary(inductor_cxx=get_cpp_compiler())
        except Exception as e:  # noqa: BLE001
            rec.summary(inductor_cxx=repr(e))
    rec.standard_training_checks()
    rec.check("adapter_changed", *c.adapter_changed(before, after))
    if frozen_before is not None:
        rec.check("frozen_base_unchanged", torch.equal(frozen_before, frozen[0].detach()), "first frozen 2-D weight")
    rec.check("probe_loss_decreased", probe_after < probe_before, f"{probe_before:.4f} -> {probe_after:.4f}")
    rec.check("loss_route_bounded", route["chunked_ce"] + route["trl_fused_head_cpu"] > 0 or a.tiny,
              f"{route} (0 = full logits materialised)")

    FLM.for_inference(model)
    gen = generate(model, tok)
    rec.summary(generation=gen, rss_after_generate_mb=peak_rss_mb())
    rec.check("generate_nonempty", bool(gen.strip()), repr(gen))

    if not a.skip_reload:
        save_dir = os.path.join(os.path.dirname(os.path.abspath(a.out)), f"cpu_sft_adapter_{os.getpid()}")
        model.save_pretrained(save_dir)
        tok.save_pretrained(save_dir)
        cmd = [sys.executable, os.path.abspath(__file__), "--reload", save_dir, "--model", a.model,
               "--max-seq-length", str(a.max_seq_length), "--load-in-4bit", str(a.load_in_4bit),
               "--grad-ckpt", a.grad_ckpt] + [x for e in a.env for x in ("--env", e)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RELOAD_RESULT ")), None)
        if line is None:
            rec.check("reload_ran", False, (proc.stdout[-1500:] + proc.stderr[-2500:]))
        else:
            child = json.loads(line.split(" ", 1)[1])
            rec.summary(reload=child)
            tol = 1e-3 if not (trainer.args.bf16 or trainer.args.fp16 or a.load_in_4bit) else 3e-2
            rel = abs(child["probe_loss"] - probe_after) / max(abs(probe_after), 1e-8)
            rec.check("reload_probe_matches", rel <= tol, f"trained {probe_after:.5f} vs reloaded {child['probe_loss']:.5f} (rel {rel:.2e}, tol {tol})")
            rec.check("reload_lora_attached", child["lora_modules"] == rec.data["summary"]["lora_modules"],
                      f"{child['lora_modules']} vs {rec.data['summary']['lora_modules']}")
            rec.check("reload_generate_nonempty", bool(child["generation"].strip()), repr(child["generation"]))
        import shutil
        shutil.rmtree(save_dir, ignore_errors=True)
    rec.summary(peak_rss_mb=peak_rss_mb())
