"""cpu_sft in the environment a platform without a pip-installable stack needs, built from scratch.

    python jobs/cpu_altenv.py --model unsloth/Qwen3-0.6B

- Intel macOS: PyTorch stopped x86_64 macOS wheels at 2.2 (Unsloth needs 2.4+); conda-forge still
  builds current PyTorch for osx-64, so a micromamba env with conda-forge pytorch + pip for the rest.
- Windows on Arm: native win_arm64 has torch wheels but no pyarrow (so no `datasets`, which TRL and
  unsloth_zoo require); an x86_64 Python under the OS emulator installs every wheel.
- Anything else: nothing to do (the regular cpu_sft job covers it); writes a skipped JSON, exits 0.

The staged checkout (cwd) is installed editable: unsloth_zoo when cwd is the zoo repo, else unsloth.
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser()
p.add_argument("--model", default="unsloth/Qwen3-0.6B")
p.add_argument("--max-steps", type=int, default=6)
p.add_argument("--tiny", action="store_true", help="accepted for the staging harness; ignored")
p.add_argument("--out", default="outputs/jobs/cpu_altenv.json")
a, rest = p.parse_known_args()

system, machine = platform.system(), platform.machine().lower()
mode = ("conda-forge" if system == "Darwin" and machine == "x86_64"
        else "x64-emulation" if system == "Windows" and machine in ("arm64", "aarch64") else None)
out = os.path.abspath(a.out)
os.makedirs(os.path.dirname(out), exist_ok=True)
if mode is None:
    json.dump({"job": "cpu_altenv", "skipped": f"{system} {machine}: covered by cpu_sft", "passed": True}, open(out, "w"))
    print(f"cpu_altenv: nothing to do on {system} {machine}")
    sys.exit(0)

work = tempfile.mkdtemp(prefix="cpu_altenv_")
is_zoo = os.path.isdir("unsloth_zoo")
zoo_git = "unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo"
core_git = "unsloth @ git+https://github.com/unslothai/unsloth"
log = []


def run(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, text=True, capture_output=True, **kw)
    log.append({"cmd": " ".join(cmd), "rc": proc.returncode, "tail": (proc.stdout[-1200:] + proc.stderr[-1800:])})
    print(proc.stdout[-3000:], proc.stderr[-3000:], flush=True)
    if proc.returncode != 0:
        json.dump({"job": "cpu_altenv", "mode": mode, "passed": False, "error": f"step failed: {' '.join(cmd)}",
                   "log": log}, open(out, "w"), indent=2)
        sys.exit(1)
    return proc


if mode == "conda-forge":
    url = "https://micro.mamba.pm/api/micromamba/osx-64/latest"
    tarball = os.path.join(work, "mm.tar.bz2")
    urllib.request.urlretrieve(url, tarball)
    run(["tar", "-xjf", tarball, "-C", work, "bin/micromamba"])
    mm = os.path.join(work, "bin", "micromamba")
    env_dir = os.path.join(work, "env")
    run([mm, "create", "-y", "-q", "-p", env_dir, "-c", "conda-forge", "python=3.12", "pytorch", "pip"],
        env=dict(os.environ, MAMBA_ROOT_PREFIX=os.path.join(work, "root")))
    py = os.path.join(env_dir, "bin", "python")
    pip = [py, "-m", "pip", "install", "-q"]
else:
    uv = shutil.which("uv")
    if uv is None:
        run([sys.executable, "-m", "pip", "install", "-q", "uv"])
        uv = shutil.which("uv") or os.path.join(os.path.dirname(sys.executable), "Scripts", "uv.exe")
    env_dir = os.path.join(work, "venv")
    run([uv, "venv", "-q", "--python", "cpython-3.12-windows-x86_64-none", env_dir])
    py = os.path.join(env_dir, "Scripts", "python.exe")
    run([uv, "pip", "install", "-q", "--python", py, "--index-url", "https://download.pytorch.org/whl/cpu", "torch"])
    pip = [uv, "pip", "install", "-q", "--python", py]

run([py, "-c", "import platform, sys, torch; print(platform.machine(), sys.version, torch.__version__)"])
# The rest from PyPI; torch is already satisfied by the conda / emulated install above.
run(pip + ["transformers", "trl", "peft", "accelerate", "datasets", "psutil", "sentencepiece", "protobuf"])
if is_zoo:
    run(pip + ["-e", "."])
    run(pip + ["--no-deps", core_git])
else:
    run(pip + [zoo_git])
    run(pip + ["--no-deps", "-e", "."])
child_out = os.path.join(work, "cpu_sft.json")
proc = subprocess.run([py, os.path.join(HERE, "cpu_sft.py"), "--model", a.model, "--max-steps", str(a.max_steps),
                       "--out", child_out, *rest], text=True)
try:
    d = json.load(open(child_out))
except Exception:  # noqa: BLE001
    d = {"passed": False, "error": "cpu_sft wrote no JSON"}
d.update(job="cpu_altenv", mode=mode, setup_log=log)
json.dump(d, open(out, "w"), indent=2, default=str)
print("JOB_RESULT " + json.dumps({"job": "cpu_altenv", "mode": mode, "passed": d.get("passed")}))
sys.exit(0 if d.get("passed") and proc.returncode == 0 else 1)
