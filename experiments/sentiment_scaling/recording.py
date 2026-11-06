from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, platform, time
import importlib.metadata

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):
            h.update(chunk)
    return h.hexdigest()

class Recorder:
    def __init__(self,args):
        self.path=Path(args.output_file)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.raw_path=self.path.with_suffix('.raw.jsonl')
        self.raw=self.raw_path.open('x')
        self.meta={'started_at':datetime.now(timezone.utc).isoformat(),'args':vars(args),
                   'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
                   'python':platform.python_version(),
                   'packages':{name:importlib.metadata.version(name) for name in ['torch','transformers','tokenizers','datasets','numpy','scikit-learn']},
                   'code_sha256':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}}
        self.t0=time.monotonic(); self.count=0
        self.save_meta()

    def save_meta(self):
        self.path.with_suffix('.receipt.json').write_text(json.dumps(self.meta,indent=2,ensure_ascii=False,default=str)+'\n')

    def runtime(self,model,tokenizer):
        import torch
        self.meta.update(model_config=model.config.to_dict(),generation_config=model.generation_config.to_dict(),
                         tokenizer_class=type(tokenizer).__name__,tokenizer_init=tokenizer.init_kwargs,
                         gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,
                         attention_implementation=model.config._attn_implementation,
                         model_training=model.training)
        self.save_meta()

    def dataset(self,ds,eval_set):
        items=list(eval_set)
        root=Path(__file__).resolve().parents[2]
        expected=json.loads((root/'configs/data_contract.json').read_text())['cached_eval500_row_sha256']
        observed=[hashlib.sha256(json.dumps({'text':r['text'],'label':r['label']},ensure_ascii=False,sort_keys=True).encode()).hexdigest() for r in items]
        assert 1 <= len(items) <= 500 and observed==expected[:len(items)], 'Dataset differs from the original cached HF test strings'
        self.meta['dataset']={'fingerprint':ds._fingerprint,'cache_files':ds.cache_files,
                              'rows_sha256':hashlib.sha256(json.dumps(items,ensure_ascii=False,sort_keys=True).encode()).hexdigest(),
                              'count':len(items),'label_counts':{str(i):sum(x['label']==i for x in items) for i in range(3)}}
        self.save_meta()

    def vector(self,path,vector):
        self.meta.setdefault('vectors', []).append({'path':str(path),'sha256':sha(path),'shape':list(vector.shape),'dtype':str(vector.dtype),'norm_fp32':vector.float().norm().item()})
        self.save_meta()

    def row(self,n,alpha,i,item,prompt,inputs,output_ids,response,pred):
        row={'n':n,'alpha':alpha,'index':i,'text':item['text'],'original_label':item['label'],
             'prompt':prompt,'input_ids':inputs.input_ids[0].tolist(),
             'generated_ids':output_ids[0][inputs.input_ids.shape[1]:].tolist(),
             'response':response,'prediction':pred}
        self.raw.write(json.dumps(row,ensure_ascii=False)+'\n'); self.raw.flush(); self.count+=1
        if self.count%50==0:
            print(f'AUDIT rows={self.count} elapsed_s={time.monotonic()-self.t0:.1f}',flush=True)

    def finish(self,results):
        self.raw.close()
        assert self.count==self.meta['args']['num_eval']*len(self.meta['args']['n_values'])*len(self.meta['args']['alpha_values'])
        self.meta.update(finished_at=datetime.now(timezone.utc).isoformat(),row_count=self.count,
                         raw_sha256=sha(self.raw_path),csv_sha256=sha(self.path),results=results,complete=True)
        self.save_meta()
