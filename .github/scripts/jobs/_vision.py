# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Shared pieces of the vision jobs (vision_sft, vision_grpo, vision_dpo): --mode lora|qlora|full, image
fixtures with a second image size, and the engagement probes behind the M / C / K cell properties.

Probes (all counts, recorded into the job JSON summary under "probes"):
- gc: Unsloth checkpoint forwards / recomputes per component (language / vision / other) and how many
  forwards offloaded their saved activation to the host (UnslothCheckpointFunction ctx._saved_metadata).
- fused_ce: top-level forwards that returned empty logits (Unsloth fused CE engaged) vs full logits.
- grpo: fused_linear_grpo_loss forward-gradient calls (fused_linear.CALLS["grpo_fwd_grad"]) and route stats.
- dpo: chunked DPO head calls / fallback reasons, and TRL >= 1.15 fused_lm_head forwards.
- block: block-compile layer census (full / split / off per layer) and its install flag.
- compile: graph breaks, unique graphs and recompile reasons (torch._dynamo.convert_frame hook).
"""

MODES = ("lora", "qlora", "full")

# For processors shipping no chat template (PaliGemma): image placeholder then text, one turn per line.
MINIMAL_CHAT_TEMPLATE = (
    "{% for m in messages %}{% for c in m['content'] %}{% if c['type'] == 'image' %}<image>{% else %}"
    "{{ c['text'] }}{% endif %}{% endfor %}\n{% endfor %}"
)


def add_mode_arg(p, default="lora"):
    p.add_argument("--mode", choices=MODES, default=default,
                   help="lora: 16-bit base + LoRA; qlora: 4-bit base + LoRA; full: full finetuning (no adapter)")


def load_kwargs(mode):
    return dict(load_in_4bit=mode == "qlora", full_finetuning=mode == "full")


def trainable_fingerprint(model):
    """Like _common.adapter_fingerprint, over every trainable parameter (full finetuning has no adapter)."""
    import _common as c
    return c.adapter_fingerprint(model)


def drawn_images(n, sizes=(224,), seed=0):
    """n coloured-square PIL images; sizes cycles per row so a second size exercises shape changes
    (pattern A, A, B, A, B for two sizes)."""
    from PIL import Image, ImageDraw
    colors = [(220, 40, 40), (40, 160, 60), (40, 80, 220), (230, 200, 40)]
    names = ["red", "green", "blue", "yellow"]
    pattern = [0, 0, 1, 0, 1] if len(sizes) == 2 else list(range(len(sizes)))
    out = []
    for i in range(n):
        size = sizes[pattern[i % len(pattern)] % len(sizes)]
        img = Image.new("RGB", (size, size), (255, 255, 255))
        d = ImageDraw.Draw(img)
        d.rectangle([size // 4, size // 4, 3 * size // 4, 3 * size // 4], fill=colors[(i + seed) % 4])
        d.text((4, 4), str(i % 10), fill=(0, 0, 0))
        out.append((img, names[(i + seed) % 4]))
    return out


def _component(module):
    name = type(module).__name__.lower()
    if any(s in name for s in ("vision", "visual", "siglip", "vit", "clip", "pixtral", "image", "mllamavision")):
        return "vision"
    if any(s in name for s in ("decoderlayer", "textlayer", "languagemodel")):
        return "language"
    return "other:" + type(module).__name__


class Probes:
    def __init__(self, model):
        self.model = model
        self.gc = {}
        self.fused_ce = {"empty_logits": 0, "full_logits": 0, "with_labels": 0}
        self.recompiles = []
        self._undo = []
        self._install_gc()
        self._install_logits_hook()
        self._install_recompile_hook()
        self._install_layer_hooks()
        self.grpo0 = self._grpo_calls()

    # -- per-layer call counts: with gradient checkpointing every layer runs twice per step (forward, recompute) --
    def _install_layer_hooks(self):
        import torch
        self.layer_calls = {}
        self.gc_funcs = {}
        calls = self.layer_calls
        for name, mod in self.model.named_modules():
            cls = type(mod).__name__
            if not cls.endswith(("DecoderLayer", "EncoderLayer", "VisionBlock", "VisionLayer", "Block", "TextLayer",
                                 "SelfAttentionDecoderLayer", "CrossAttentionDecoderLayer")):
                continue
            low = name.lower()
            if any(t in low for t in ("vision", "visual", "tower", "image", "encoder", "projector")):
                comp = "vision"
            elif cls.endswith(("DecoderLayer", "TextLayer")) or "language_model" in low:
                comp = "language"
            else:
                comp = "other"
            key = f"{comp}:{cls}"
            calls.setdefault(key, {"modules": 0, "grad_calls": 0, "nograd_calls": 0})
            calls[key]["modules"] += 1
            gf = getattr(mod, "_gradient_checkpointing_func", None)
            if gf is not None:
                self.gc_funcs[key] = getattr(gf, "__qualname__", None) or repr(gf)[:80]

            def pre(module, args, _key=key):
                calls[_key]["grad_calls" if torch.is_grad_enabled() else "nograd_calls"] += 1
            self._undo.append(mod.register_forward_pre_hook(pre).remove)

    # -- gradient checkpointing --
    def _install_gc(self):
        try:
            import unsloth_zoo.gradient_checkpointing as g
        except Exception:
            return
        fn = getattr(g, "UnslothCheckpointFunction", None)
        if fn is None:
            return
        orig_f, orig_b = fn.forward, fn.backward
        gc = self.gc

        import torch

        def comp(run_function):
            seen, todo = set(), [run_function]
            while todo:
                f = todo.pop()
                if f is None or id(f) in seen:
                    continue
                seen.add(id(f))
                if isinstance(f, torch.nn.Module):
                    return _component(f)
                todo += [getattr(f, "__self__", None), getattr(f, "func", None), getattr(f, "__wrapped__", None)]
                for cell in getattr(f, "__closure__", None) or ():
                    try:
                        todo.append(cell.cell_contents)
                    except ValueError:
                        pass
            return "fn:" + getattr(run_function, "__qualname__", repr(run_function))[:80]

        def forward(ctx, run_function, preserve_rng_state, *args):
            out = orig_f(ctx, run_function, preserve_rng_state, *args)
            d = gc.setdefault(comp(run_function), {"forward": 0, "recompute": 0, "offloaded": 0})
            d["forward"] += 1
            meta = getattr(ctx, "_saved_metadata", None)
            if meta is not None and any(x is not None for x in meta):
                d["offloaded"] += 1
            ctx._w12_comp = comp(run_function)
            return out

        def backward(ctx, *grads):
            d = gc.setdefault(getattr(ctx, "_w12_comp", "?"), {"forward": 0, "recompute": 0, "offloaded": 0})
            d["recompute"] += 1
            return orig_b(ctx, *grads)

        fn.forward, fn.backward = staticmethod(forward), staticmethod(backward)
        self._undo.append(lambda: (setattr(fn, "forward", staticmethod(orig_f)),
                                   setattr(fn, "backward", staticmethod(orig_b))))

    # -- fused CE witness: Unsloth's fused loss returns empty logits --
    def _install_logits_hook(self):
        top = self.model
        stats = self.fused_ce

        def hook(module, args, kwargs, output):
            if kwargs.get("labels") is None:
                return
            stats["with_labels"] += 1
            try:
                logits = output.logits
                empty = logits is None or logits.numel() == 0
            except Exception:  # Unsloth's EMPTY_LOGITS raises on any use
                empty = True
            if empty:
                stats["empty_logits"] += 1
            else:
                stats["full_logits"] += 1
        h = top.register_forward_hook(hook, with_kwargs=True)
        self._undo.append(h.remove)

    def _install_recompile_hook(self):
        try:
            import torch._dynamo.convert_frame as cfm
        except Exception:
            return
        orig = getattr(cfm, "get_and_maybe_log_recompilation_reasons", None)
        if orig is None:
            return
        rec = self.recompiles

        def wrapped(*a, **k):
            r = orig(*a, **k)
            try:
                code = a[1].f_code if len(a) > 1 and hasattr(a[1], "f_code") else None
                where = f"{code.co_name}:{code.co_firstlineno}" if code else "?"
            except Exception:
                where = "?"
            rec.append(f"{where}: {(r[0] if r else 'unknown')[:200]}")
            return r
        cfm.get_and_maybe_log_recompilation_reasons = wrapped
        self._undo.append(lambda: setattr(cfm, "get_and_maybe_log_recompilation_reasons", orig))

    @staticmethod
    def _grpo_calls():
        try:
            import unsloth_zoo.fused_losses.fused_linear as fl
            return dict(fl.CALLS)
        except Exception:
            return {}

    def block_census(self):
        try:
            import unsloth.models._block_compile as M
        except Exception as e:
            return {"error": repr(e)}
        names = {M._OFF: "off", M._FULL: "full", M._SPLIT: "split"}
        out = {"modes": {"full": 0, "split": 0, "off": 0, "unset": 0}, "classes": {}, "forced_float32": None}
        try:
            layers = M._layers(self.model)
        except Exception as e:
            return {"error": repr(e)}
        for layer, _ in layers:
            out["classes"][type(layer).__name__] = bool(getattr(type(layer).forward, M._MARK, False))
            out["modes"][names.get(layer.__dict__.get("_unsloth_block_mode"), "unset")] += 1
            if out["forced_float32"] is None:
                out["forced_float32"] = layer.__dict__.get("_unsloth_forced_float32")
        out["n_layers"] = len(layers)
        out["compiled_classes"] = sum(out["classes"].values())
        # Towers are never in the decoder lists: name every *Layer / *Block class outside them.
        found = {id(layer) for layer, _ in layers}
        tower = {}
        for name, mod in self.model.named_modules():
            cls = type(mod).__name__
            if id(mod) not in found and cls.endswith(("EncoderLayer", "VisionBlock", "VisionLayer", "Block")) \
                    and M._in_tower(name):
                tower[cls] = tower.get(cls, 0) + 1
        out["tower_layers_uncompiled"] = tower
        return out

    def callback(self, parity_batch=None):
        """TrainerCallback recording per-step recompiles / unique graphs / graph breaks / GC flags; after
        step 1 it runs gc_parity on parity_batch() (inside training, where the GC flags are live)."""
        from transformers import TrainerCallback
        probes = self
        probes.per_step = []
        probes.gc_parity = None

        class _CB(TrainerCallback):
            def on_step_end(self, args, state, control, model=None, **kw):
                import _common as c
                cs = c.compile_stats() or {}
                flags = [m.gradient_checkpointing for m in probes.model.modules()
                         if isinstance(getattr(m, "gradient_checkpointing", None), bool)]
                probes.per_step.append({"step": state.global_step, "recompiles": len(probes.recompiles),
                                        "unique_graphs": cs.get("unique_graphs"),
                                        "graph_breaks": cs.get("graph_breaks"),
                                        "gc_flags_on": sum(flags), "gc_flags": len(flags)})
                if state.global_step == 1 and parity_batch is not None:
                    try:
                        probes.gc_parity = gc_parity(probes.model, parity_batch())
                    except Exception as e:  # noqa: BLE001
                        probes.gc_parity = {"error": repr(e)[:400]}
        return _CB()

    def summary(self, trainer=None):
        import _common as c
        calls = self._grpo_calls()
        delta = {k: calls.get(k, 0) - self.grpo0.get(k, 0) for k in calls if calls.get(k, 0) != self.grpo0.get(k, 0)}
        out = {"gc": self.gc, "layer_calls": self.layer_calls, "gc_funcs": self.gc_funcs, "fused_ce": self.fused_ce, "fused_linear_calls": delta,
               "block": self.block_census(), "compile": c.compile_stats(),
               "recompiles": {"count": len(self.recompiles), "reasons": self.recompiles[:16]},
               "per_step": getattr(self, "per_step", []), "gc_parity": getattr(self, "gc_parity", None)}
        try:
            from unsloth_zoo.rl_replacements import GRPO_FUSED_ROUTE_STATS
            out["grpo_fused_route_stats"] = dict(GRPO_FUSED_ROUTE_STATS)
        except Exception:
            pass
        if trainer is not None:
            for k in ("_unsloth_dpo_chunked_calls", "_unsloth_dpo_chunked_fallbacks"):
                v = trainer.__dict__.get(k)
                if v is not None:
                    out[k.lstrip("_")] = v
        return out

    def close(self):
        for f in reversed(self._undo):
            try:
                f()
            except Exception:
                pass


def gc_parity(model, batch):
    """One forward + backward with gradient checkpointing as configured, then with every module's
    gradient_checkpointing flag off: tensors / bytes saved for backward (a structural count, not a
    memory benchmark), loss, and the max relative gradient difference between the two."""
    import torch
    flags = [(m, m.gradient_checkpointing) for m in model.modules()
             if isinstance(getattr(m, "gradient_checkpointing", None), bool)]

    def run():
        model.zero_grad(set_to_none=True)
        n = {"tensors": 0, "bytes": 0}

        def pack(t):
            n["tensors"] += 1
            n["bytes"] += t.numel() * t.element_size()
            return t
        torch.manual_seed(0)
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            out = model(**batch)
            loss = out.loss
        loss.backward()
        g = {k: p.grad.detach().float().clone() for k, p in model.named_parameters() if p.grad is not None}
        return n, float(loss), g

    was_training = model.training
    model.train()
    try:
        on = run()
        for m, _ in flags:
            m.gradient_checkpointing = False
        off = run()
    finally:
        for m, f in flags:
            m.gradient_checkpointing = f
        model.zero_grad(set_to_none=True)
        model.train(was_training)
    rel = 0.0
    worst = None
    for k, a in on[2].items():
        b = off[2].get(k)
        if b is None:
            continue
        d = float((a - b).abs().max() / (b.abs().max() + 1e-12))
        if d > rel:
            rel, worst = d, k
    return {"flags": len(flags), "flags_on": sum(f for _, f in flags), "saved_gc_on": on[0], "saved_gc_off": off[0],
            "loss_on": on[1], "loss_off": off[1], "grads": len(on[2]), "max_rel_grad_diff": rel, "worst": worst}
