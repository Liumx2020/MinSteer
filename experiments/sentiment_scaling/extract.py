"""Use the original extraction function; only extend its fixed scenario prompts."""
from pathlib import Path
import importlib.util, json, hashlib, collections, os, platform
from datetime import datetime,timezone
import importlib.metadata
import torch

E=Path(__file__).resolve().parent
ROOT=E.parent.parent
SRC=ROOT/'sentiment/extract_cached.py'
POOL=ROOT/'prompts/sentiment_pool50.json'
contract=json.loads((ROOT/'configs/sentiment_scaling.json').read_text())
assert hashlib.sha256(SRC.read_bytes()).hexdigest()==contract['extraction_source_sha256']
assert hashlib.sha256(POOL.read_bytes()).hexdigest()==contract['prompt_pool_sha256']
spec=importlib.util.spec_from_file_location('original_minsteer_extraction',SRC)
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
rows=json.loads(POOL.read_text())
lookup={(r['scenario'],r['sample_idx']):r for r in rows}
counters=collections.Counter()
OUT=ROOT/'runs/sentiment_scaling/raw'
OUT.mkdir(parents=True,exist_ok=False)
old.DETAILED_RESIDUALS_OUTPUT_DIR=str(OUT)
base_prompt=old.create_emotion_prompt
def extended_prompt(emotion_type='positive',scenario='office_worker'):
    assert emotion_type=='negative'
    idx=counters[scenario]; counters[scenario]+=1
    r=lookup[(scenario,idx)]
    original_system,original_user=base_prompt(emotion_type,scenario)
    assert r['system']==original_system
    assert r['user']==original_user or r['user'].startswith(original_user+'\nContext: ')
    return r['system'],r['user']
old.create_emotion_prompt=extended_prompt
base_generation=old.token_wise_generation_with_residuals
trace=(OUT/'generation_trace.jsonl').open('x')
call=0
def record_generation(*args,**kwargs):
    global call
    model,tokenizer=args[:2]
    ids=kwargs['input_ids'][0].tolist()
    inherited=kwargs.get('past_key_values') is not None
    result=base_generation(*args,**kwargs)
    trace.write(json.dumps(dict(call=call,inherited_cache=inherited,input_ids=ids,
                               generated_ids=result['generated_token_ids'],
                               generated_text=result['generated_text'],
                               residual_shapes={k:list(v.shape) for k,v in result['residuals'].items()}),ensure_ascii=False)+'\n')
    trace.flush();call+=1
    if call==1:
        runtime=dict(started_at=datetime.now(timezone.utc).isoformat(),
                     packages={p:importlib.metadata.version(p) for p in ['torch','transformers','tokenizers','numpy']},
                     python=platform.python_version(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                     model_config=model.config.to_dict(),generation_config=model.generation_config.to_dict(),
                     tokenizer_init=tokenizer.init_kwargs,attention_implementation=model.config._attn_implementation,
                     extraction_source_sha256=contract['extraction_source_sha256'],prompt_pool_sha256=contract['prompt_pool_sha256'])
        (OUT/'runtime.json').write_text(json.dumps(runtime,ensure_ascii=False,indent=2,default=str)+'\n')
    return result
old.token_wise_generation_with_residuals=record_generation
old.run_emotion_style_analysis(emotion_type='negative',sampling_method='argmax',top_p=0.9,temperature=0.7,
                              max_samples=10,scenarios=sorted(counters.keys() or {r['scenario'] for r in rows}),no_save=False,model='llama3')
trace.close()
assert counters=={s:10 for s in sorted({r['scenario'] for r in rows})}
assert call==100
files=sorted(OUT.glob('*.pt'));assert len(files)==50
data_hashes={}
for p in files:
    data_hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
(OUT/'completion.json').write_text(json.dumps(dict(completed_at=datetime.now(timezone.utc).isoformat(),count=50,generation_calls=call,files=data_hashes),indent=2)+'\n')
print('COMPLETE: 50 prompts, 100 generation stages, 50 raw tensor files.',flush=True)
