# SPDX-License-Identifier: AGPL-3.0-or-later
import argparse
import ast
import json
import os
import statistics
from pathlib import Path
import subprocess
import sys
import tempfile
import time

BASE = 'f53aa65cb57a5ae6222c42334387c51504723d54'


def arm(args):
    os.environ['UNSLOTH_ZOO_DISABLE_GPU_INIT'] = '1'
    sys.path.insert(0, str(Path(args.tree).resolve()))
    import mlx.core as mx
    import unsloth_zoo
    assert Path(unsloth_zoo.__file__).resolve().is_relative_to(Path(args.tree).resolve())
    if not args.cpu:
        assert mx.metal.is_available(), 'Metal required for this evidence'
    from mlx_vlm.models.qwen3_5 import LanguageModel, TextConfig, ModelConfig, VisionConfig
    from unsloth_zoo.mlx.generate import SamplingParams
    from types import SimpleNamespace
    mx.random.seed(3407)
    cfg = TextConfig(model_type='qwen3_5_text', hidden_size=128, intermediate_size=256,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1, head_dim=64,
        linear_num_value_heads=2, linear_num_key_heads=1, linear_key_head_dim=64,
        linear_value_head_dim=64, linear_conv_kernel_dim=4, rms_norm_eps=1e-6,
        vocab_size=128, max_position_embeddings=2048, full_attention_interval=2,
        rope_parameters={'type':'default', 'mrope_section':[2,3,3], 'rope_theta':10000,
                         'partial_rotary_factor':0.25})
    lm = LanguageModel(cfg, ModelConfig(text_config=cfg, vision_config=VisionConfig(depth=1, hidden_size=64, num_heads=2, intermediate_size=128, out_hidden_size=128), model_type="qwen3_5"))
    lm.eval()
    mx.eval(lm.parameters())
    model = SimpleNamespace(language_model=lm)
    source = Path(args.helpers).read_text(encoding='utf-8')
    names = {'_prefill', '_solo', '_run', '_script', '_head_in_step', '_parity'}
    tree = ast.parse(source)
    selected = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
    ns = {'mx':mx, 'SimpleNamespace':SimpleNamespace}
    exec(compile(selected, args.helpers, 'exec'), ns)
    prompt = list(range(8, 32))*2
    sampling = SamplingParams()
    from unsloth_zoo.mlx.speculative import _language_call
    _language_call('qwen3_5')
    plain = ns['_solo'](model, prompt, 256, sampling)
    mx.synchronize()
    prefill_times=[]
    for _ in range(20):
        started=time.perf_counter()
        cache,first,hidden=ns['_prefill'](lm,prompt,sampling)
        mx.synchronize()
        prefill_times.append(time.perf_counter()-started)
    prefill_s=statistics.median(prefill_times)
    from unsloth_zoo.mlx.speculative import EngineRow, SpeculativeEngine
    def decode():
        engine = SpeculativeEngine(model, ns['_script'](('plain',4)))
        cache, first, hidden = ns['_prefill'](lm,prompt,sampling)
        engine.add(EngineRow(cache=cache,pending=first,prompt=prompt,max_tokens=256,sampling=sampling,hidden=hidden,uid=0))
        mx.synchronize()
        out = [first]
        started = time.perf_counter()
        while engine.rows:
            for step in engine.step():
                out.extend(step.tokens)
        mx.synchronize()
        return out,time.perf_counter()-started
    cold,cold_s=decode()
    assert cold==plain
    timings=[]
    for _ in range(2):
        assert decode()[0]==plain
    if mx.metal.is_available():
        mx.reset_peak_memory()
    for _ in range(5):
        seq,elapsed=decode()
        assert seq==plain
        timings.append(elapsed)
    elapsed=statistics.median(timings)
    inference_peak=mx.get_peak_memory() if mx.metal.is_available() else None
    checks={'plain_repeatable':{'ok':True}}
    result = {'tree':args.tree,'import':unsloth_zoo.__file__,'mlx':mx.__version__,
        'metal':mx.metal.is_available(),'plain':plain,'checks':checks,
        'prefill_s':prefill_s,'prefill_tokens_per_s':len(prompt)/prefill_s,
        'cold_decode_s':cold_s,'decode_s':elapsed,'decode_tokens_per_s':255/elapsed,'decode_samples_s':timings,'prefill_samples_s':prefill_times,
        'peak_bytes':inference_peak}
    Path(args.out).write_text(json.dumps(result,indent=2),encoding='utf-8')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',required=True)
    p.add_argument('--tree')
    p.add_argument('--helpers')
    p.add_argument('--cpu',action='store_true')
    a=p.parse_args()
    if a.tree:
        arm(a)
        return
    head=Path.cwd()
    helpers=head/'tests/test_mlx_speculative_metal.py'
    subprocess.run(['git','fetch','--no-tags','--depth','1','https://github.com/unslothai/unsloth-zoo.git',BASE],check=True)
    base=Path(tempfile.mkdtemp(prefix='pr2359-base-'))
    subprocess.run(['git','worktree','add','--detach',str(base),BASE],check=True)
    results=[]
    try:
        for i,side in enumerate((['base']*3 + ['base','head','head','base']*3)):
            print(f'ARM {i} {side}',flush=True)
            tree=base if side=='base' else head
            target=Path(a.out).resolve().with_name(f'job_perf2359_{i}_{side}.json')
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--tree',str(tree),
                '--helpers',str(helpers),'--out',str(target)],check=True,cwd=tree)
            result=json.loads(target.read_text(encoding='utf-8'))
            result['side']=side
            results.append(result)
        assert all(r['plain']==results[0]['plain'] for r in results)
        assert all(all(c['ok'] for c in r['checks'].values()) for r in results if r['side']=='head')
        assert all(r['checks']['plain_repeatable']['ok'] for r in results)
        final={'passed':True,'results':results}
    finally:
        subprocess.run(['git','worktree','remove',str(base)],check=True)
    Path(a.out).write_text(json.dumps(final,indent=2),encoding='utf-8')
    print(json.dumps(final),flush=True)


if __name__=='__main__':
    main()
