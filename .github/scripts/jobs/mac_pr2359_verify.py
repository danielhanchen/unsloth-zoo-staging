# SPDX-License-Identifier: AGPL-3.0-or-later
import argparse
import ast
import json
import os
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
        vocab_size=128, max_position_embeddings=512, full_attention_interval=2,
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
    plain = ns['_solo'](model, prompt, 32, sampling)
    mx.synchronize()
    started = time.perf_counter()
    cache, first, hidden = ns['_prefill'](lm, prompt, sampling)
    mx.synchronize()
    prefill_s = time.perf_counter()-started
    from unsloth_zoo.mlx.speculative import EngineRow, SpeculativeEngine
    def decode():
        engine = SpeculativeEngine(model, ns['_script'](('plain',4)))
        cache, first, hidden = ns['_prefill'](lm,prompt,sampling)
        engine.add(EngineRow(cache=cache,pending=first,prompt=prompt,max_tokens=32,sampling=sampling,hidden=hidden,uid=0))
        mx.synchronize()
        out = [first]
        started = time.perf_counter()
        while engine.rows:
            for step in engine.step():
                out.extend(step.tokens)
        mx.synchronize()
        return out,time.perf_counter()-started
    cold, cold_s = decode()
    if mx.metal.is_available():
        mx.reset_peak_memory()
    seq, elapsed = decode()
    inference_peak = mx.get_peak_memory() if mx.metal.is_available() else None
    assert seq == cold == plain
    checks = {}
    try:
        favour = ns['_parity']()
        expected = ns['_solo'](model, prompt, 32, sampling, [ns['_parity']()])
        out, _, engine = ns['_run'](model,[prompt],32,ns['_script'](('copy',4),('plain',2)),sampling,processors={0:[favour]})
        assert out == [expected]
        assert engine.controller is not None and favour.committed[-1] == out[0]
        class Cycle:
            def __init__(self):
                self.committed = []
                self.start = None
            def __call__(self,tokens,logits):
                if self.start is None:
                    self.start = len(tokens)
                want = int(tokens[-1].item()) % 4 + 1
                return mx.where(mx.arange(logits.shape[-1]) == want, logits, -1e9)
            def settle(self,tokens):
                self.committed.append(tokens[self.start:].tolist())
        grammar = Cycle()
        cycle_prompt = [1,2,3,4]*8
        actual,drafted,_ = ns['_run'](model,[cycle_prompt],32,ns['_script'](('copy',4),('plain',2)),sampling,processors={0:[grammar]})
        assert actual == [[1,2,3,4]*8]
        assert drafted[0][1] > 0
        assert grammar.committed[-1] == actual[0]
        assert all(b[:len(a)] == a for a,b in zip(grammar.committed,grammar.committed[1:]))
        checks['settling'] = {'ok':True,'drafted':drafted[0][0],'accepted':drafted[0][1]}

    except Exception as e:
        checks['settling'] = {'ok':False,'error':repr(e)}
    try:
        from unsloth_zoo.mlx.speculative import _join
        from unsloth_zoo.mlx.generate import _row_cache
        from mlx_vlm.models.cache import QuantizedKVCache, BatchQuantizedKVCache
        entries = []
        for length in [5,7,3]:
            c = QuantizedKVCache()
            x = mx.random.normal((1,1,length,64))
            c.update_and_fetch(x,x)
            entries.append([c])
        batch = None
        for size,row in enumerate(entries):
            batch = _join(batch,row,size)
        assert isinstance(batch[0],BatchQuantizedKVCache)
        assert [_row_cache(batch,i)[0].offset for i in range(3)] == [5,7,3]
        prefill = ns['_prefill']
        def packed(lm,*args,**kwargs):
            cache,first,hidden = prefill(lm,*args,**kwargs)
            return [c.to_quantized(group_size=64,bits=8) if hasattr(c,'to_quantized') else c for c in cache],first,hidden
        ns['_prefill'] = packed
        try:
            prompts = [prompt,prompt[:-3]]
            expected = [ns['_solo'](model,ids,24,sampling) for ids in prompts]
            actual,_,_ = ns['_run'](model,prompts,24,ns['_script'](('copy',4),('plain',2)),sampling)
            assert actual == expected
        finally:
            ns['_prefill'] = prefill
        checks['affine'] = {'ok':True}
    except Exception as e:
        checks['affine'] = {'ok':False,'error':repr(e)}
    result = {'tree':args.tree,'import':unsloth_zoo.__file__,'mlx':mx.__version__,
        'metal':mx.metal.is_available(),'plain':plain,'checks':checks,
        'prefill_s':prefill_s,'prefill_tokens_per_s':len(prompt)/prefill_s,
        'cold_decode_s':cold_s,'decode_s':elapsed,'decode_tokens_per_s':31/elapsed,
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
            target=Path(a.out).resolve().with_name(f'job_pr2359_{i}_{side}.json')
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--tree',str(tree),
                '--helpers',str(helpers),'--out',str(target)],check=True,cwd=tree)
            result=json.loads(target.read_text(encoding='utf-8'))
            result['side']=side
            results.append(result)
        assert all(r['plain']==results[0]['plain'] for r in results)
        assert all(all(c['ok'] for c in r['checks'].values()) for r in results if r['side']=='head')
        assert all(not r['checks']['settling']['ok'] and not r['checks']['affine']['ok'] for r in results if r['side']=='base')
        final={'passed':True,'results':results}
    finally:
        subprocess.run(['git','worktree','remove',str(base)],check=True)
    Path(a.out).write_text(json.dumps(final,indent=2),encoding='utf-8')
    print(json.dumps(final),flush=True)


if __name__=='__main__':
    main()
