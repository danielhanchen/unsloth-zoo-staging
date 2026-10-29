"""unsloth#15889 UI evidence on Apple Silicon: one Studio arm (before / after), tiny LFM3-Audio.

Installs Studio from unslothai/unsloth at --unsloth (install.sh --local), puts unsloth-zoo at --zoo
and mlx-vlm 0.7.11 in its venv, builds a tiny random LFM3-Audio from the real config, tokenizer and
speech codec, loads it, asks a spoken question in the chat UI with a spoken reply, and records the
status fields, the API reply shape and a screenshot (base64 PNG) into --out.
"""
import argparse, asyncio, base64, io, json, os, platform, shutil, subprocess, sys, time, traceback, urllib.request, wave
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True)
ap.add_argument("--unsloth", required=True)
ap.add_argument("--zoo", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--port", type=int, default=8931)
a = ap.parse_args()

ROOT = Path.home() / f"pr15889_{a.arm}"
SRC, HOME, MODEL = ROOT / "src", ROOT / "home", Path.home() / "lfm3_audio_tiny"
res = {"job": "mlx_pr15889_ui", "arm": a.arm, "unsloth": a.unsloth, "zoo": a.zoo,
       "machine": platform.machine(), "system": platform.system(), "steps": [], "checks": {}}


def save():
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))


def sh(cmd, **kw):
    t = time.time()
    p = subprocess.run(cmd, shell=True, text=True, capture_output=True, **kw)
    res["steps"].append({"cmd": cmd[:200], "rc": p.returncode, "s": round(time.time() - t, 1),
                         "tail": (p.stdout + p.stderr)[-1500:]})
    save()
    if p.returncode:
        raise RuntimeError(f"{cmd[:120]} -> rc {p.returncode}: {(p.stdout + p.stderr)[-800:]}")
    return p.stdout


def http(method, path, body=None, tok=None, timeout=600):
    req = urllib.request.Request(f"http://127.0.0.1:{a.port}{path}", None if body is None else json.dumps(body).encode(),
                                 {"Content-Type": "application/json"}, method=method)
    if tok:
        req.add_header("Authorization", "Bearer " + tok)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode(errors="replace")[:2000]}


def wav_b64(seconds=1.5, rate=16000, hz=440):
    import math
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(rate)
        w.writeframes(b"".join(int(8000 * math.sin(2 * math.pi * hz * i / rate)).to_bytes(2, "little", signed=True)
                               for i in range(int(seconds * rate))))
    return base64.b64encode(buf.getvalue()).decode()


BUILD = r'''
import json, sys
from pathlib import Path
import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_vlm.models.lfm3_audio import Model, ModelConfig
d = Path(sys.argv[1]); cfg = json.loads((d / "config.json").read_text())
def shrink(c, layers):
    c.update(hidden_size=64, intermediate_size=128, num_attention_heads=2, num_key_value_heads=1, head_dim=32, block_multiple_of=32, num_hidden_layers=layers)
    if isinstance(c.get("layer_types"), list): c["layer_types"] = c["layer_types"][:layers]
shrink(cfg["lfm"], 2); shrink(cfg["encoder"]["lfm"], 1); shrink(cfg["depthformer"], 1)
cfg["encoder"].update(dim_conv=32, dim_out=64)
mx.random.seed(0)
model = Model(ModelConfig.from_dict(cfg))
w = {k: (v * 0.02 if v.ndim > 1 else v) for k, v in tree_flatten(model.parameters())}
mx.save_safetensors(str(d / "model.safetensors"), w)
(d / "config.json").write_text(json.dumps(cfg))
print("BUILT", sum(v.size for v in w.values()))
'''


async def drive_ui(tok, shot):
    from playwright.async_api import async_playwright
    clip = Path.home() / "question.wav"
    clip.write_bytes(base64.b64decode(wav_b64()))
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        ctx = await browser.new_context(viewport={"width": 1280, "height": 860})
        await ctx.add_init_script(f"localStorage.setItem('unsloth_auth_token', {json.dumps(tok)});")
        page = await ctx.new_page()
        await page.goto(f"http://127.0.0.1:{a.port}/chat", wait_until="networkidle")
        await page.wait_for_timeout(4000)
        await page.locator('form:has(textarea) input[type="file"]').first.set_input_files(str(clip))
        await page.wait_for_timeout(1500)
        box = page.locator("form:has(textarea) textarea").first
        await box.click()
        await box.fill("What is the capital of France?")
        await box.press("Enter")
        found = False
        for _ in range(90):
            await page.wait_for_timeout(2000)
            if await page.locator("main audio, [data-message-role='assistant'] audio, audio").count():
                found = True
                break
        await page.wait_for_timeout(1500)
        text = await page.locator("main").inner_text()
        await page.screenshot(path=str(shot), full_page=False)
        await browser.close()
        return found, text[-1500:]


studio = None
try:
    os.environ.update(UNSLOTH_STUDIO_HOME=str(HOME), UNSLOTH_SKIP_AUTOSTART="1", UNSLOTH_DISABLE_AUTO_UPDATES="1")
    shutil.rmtree(ROOT, ignore_errors=True)
    ROOT.mkdir(parents=True)
    sh(f"git init -q {SRC} && git -C {SRC} fetch -q --depth 1 https://github.com/unslothai/unsloth {a.unsloth} && git -C {SRC} checkout -q FETCH_HEAD")
    sh("./install.sh --local < /dev/null", cwd=SRC, timeout=2400)
    py = next(p for p in (HOME / "unsloth_studio/bin/python", HOME / ".venv/bin/python") if p.exists())
    sh(f"uv pip install --python {py} --no-deps 'unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo@{a.zoo}'")
    sh(f"uv pip install --python {py} 'mlx-vlm==0.7.11'")
    res["versions"] = sh(f"{py} -m pip list 2>/dev/null | grep -iE '^(mlx|mlx-lm|mlx-vlm|transformers|unsloth|unsloth.zoo) '")

    if not (MODEL / "model.safetensors").exists():
        from huggingface_hub import snapshot_download
        snapshot_download("LiquidAI/LFM3-Audio-2B", local_dir=str(MODEL), allow_patterns=[
            "*.json", "*.jinja", "tokenizer*", "audio_tokenizer/*"])
        (Path.home() / "build_tiny.py").write_text(BUILD)
        sh(f"{py} {Path.home() / 'build_tiny.py'} {MODEL}")

    bin_ = next(p for p in (HOME / "bin/unsloth", HOME / "unsloth_studio/bin/unsloth") if p.exists())
    log = open(ROOT / "studio.log", "w")
    studio = subprocess.Popen([str(bin_), "studio", "-H", "127.0.0.1", "-p", str(a.port)], stdout=log,
                              stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(120):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{a.port}/healthz", timeout=5)
            break
        except Exception:
            time.sleep(5)
    boot = (HOME / "auth/.bootstrap_password").read_text().strip()
    _, b = http("POST", "/api/auth/login", {"username": "unsloth", "password": boot})
    new = "Pr15889-Evidence-1!"
    http("POST", "/api/auth/change-password", {"current_password": boot, "new_password": new}, b["access_token"])
    _, b = http("POST", "/api/auth/login", {"username": "unsloth", "password": new})
    tok = b["access_token"]

    code, load = http("POST", "/api/inference/load", {"model_path": str(MODEL), "max_seq_length": 2048}, tok, timeout=900)
    res["load"] = {"code": code, **{k: load.get(k) for k in ("status", "is_vision", "has_audio_input", "audio_type", "speaks", "error", "detail")}}
    _, st = http("GET", "/api/inference/status", None, tok)
    res["status"] = {k: st.get(k) for k in ("active_model", "loaded", "is_vision", "has_audio_input", "audio_type", "speaks", "is_mlx")}

    code, chat = http("POST", "/v1/chat/completions", {
        "model": str(MODEL), "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "audio_base64": wav_b64(), "modalities": ["text", "audio"], "max_tokens": 120, "stream": False}, tok, timeout=900)
    msg = ((chat.get("choices") or [{}])[0].get("message") or {}) if code == 200 else {}
    audio = msg.get("audio") or {}
    res["api_spoken_reply"] = {"code": code, "text": (msg.get("content") or "")[:200], "has_audio": bool(audio.get("data")),
                               "sample_rate": audio.get("sample_rate"), "wav_bytes": len(base64.b64decode(audio["data"])) if audio.get("data") else 0,
                               "error": chat.get("error") or chat.get("detail")}

    sh(f"{sys.executable} -m pip install -q 'playwright<2' && {sys.executable} -m playwright install chromium")
    shot = ROOT / "chat.png"
    found, text = asyncio.run(drive_ui(tok, shot))
    res["ui"] = {"audio_player": found, "main_text_tail": text}
    res["screenshot_png_b64"] = base64.b64encode(shot.read_bytes()).decode()
    res["checks"] = {"ui_ran": {"ok": True}, "audio_player": {"ok": found}}
    res["passed"] = True
except Exception as e:
    res["error"] = f"{type(e).__name__}: {e}"[:3000]
    res["tb"] = traceback.format_exc()[-3000:]
    res["passed"] = False
finally:
    if studio is not None:
        os.killpg(os.getpgid(studio.pid), 15)
    log_path = ROOT / "studio.log"
    if log_path.exists():
        res["studio_log_tail"] = log_path.read_text(errors="replace")[-4000:]
    save()
print("JOB_RESULT " + json.dumps({k: res.get(k) for k in ("arm", "passed", "load", "status", "api_spoken_reply", "ui", "error")}, default=str))
sys.exit(0)
