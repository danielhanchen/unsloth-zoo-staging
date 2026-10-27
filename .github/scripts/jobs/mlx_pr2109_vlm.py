"""TEMPORARY (unsloth-zoo#2109 review): MLX VLM GRPO / KTO / CPO / DPO-reread / generation on Apple Silicon.

Runs each scenario of mlx_pr2109_e2e.py in its own process on a tiny random Gemma 4; exit 0 when every
expected-OK scenario passed. Delete after the review.
"""
import argparse, json, os, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECT = {  # scenario -> should succeed
    "gen_stop_ids": True, "grpo_vlm": True, "grpo_vlm_text": True, "kto_vlm": True,
    "cpo_vlm_text": True, "cpo_vlm_image": False, "dpo_vlm_hf": True,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="trl-internal-testing/tiny-Gemma4ForConditionalGeneration")
    p.add_argument("--out", default="mlx_pr2109_vlm.json")
    a, _ = p.parse_known_args()
    results, ok = {}, True
    for s, want in EXPECT.items():
        o = Path(f"_pr2109_{s}.json")
        proc = subprocess.run([sys.executable, "-u", str(HERE / "mlx_pr2109_e2e.py"), s, a.model, str(o)],
                              capture_output=True, text=True, timeout=1800)
        r = json.loads(o.read_text()) if o.exists() else {"ok": False, "error": proc.stderr[-2000:]}
        r["expected_ok"] = want
        r["pass"] = bool(r.get("ok")) == want
        if s == "gen_stop_ids" and r.get("ok"):
            r["pass"] = r["pass"] and r["stopped"]["finish"] == "stop" and r["shared_eos_unchanged"] and r["prepared_equals_rendered"]
        if s.startswith("grpo") or s in ("kto_vlm", "dpo_vlm_hf"):
            r["pass"] = r["pass"] and r.get("train_steps") == 2 and (r.get("lora_moved") or [0])[0] > 0
        if s in ("grpo_vlm", "kto_vlm", "dpo_vlm_hf"):
            r["pass"] = r["pass"] and r.get("vision_forwards", 0) > 0
        print(s, "PASS" if r["pass"] else "FAIL", {k: v for k, v in r.items() if k != "tb"}, flush=True)
        if not r["pass"] and r.get("tb"):
            print(r["tb"], flush=True)
        results[s] = r
        ok = ok and r["pass"]
    summary = {"job": "mlx_pr2109_vlm", "model": a.model, "passed": ok, "results": results}
    Path(a.out).write_text(json.dumps(summary, indent=1, default=str))
    print("JOB_RESULT " + json.dumps({"passed": ok, "scenarios": {k: v["pass"] for k, v in results.items()}}))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
