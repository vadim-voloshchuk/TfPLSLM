"""Rebuild synthetic data with delayed overwrite/forget after the corpus job."""
import json,random,time
from pathlib import Path
import sentencepiece as spm
from tfplslm.data import Writer
from tfplslm.synthetic import CATEGORIES,token_task
root=Path('data');sp=spm.SentencePieceProcessor(model_file=str(root/'tokenizer.model'))
rng=random.Random(20261007);w=Writer(root,'synthetic');start=time.time()
while w.offsets[-1]<50_000_000:
    category=CATEGORIES[len(w.hashes)%7]
    distance=rng.choice([64,128,256,512,512,512,512,1024,2048])
    p,a,_=token_task(sp,rng,category,distance,'train');w.add(p+a)
result=w.close();manifest=json.loads((root/'manifest.json').read_text())
manifest['synthetic']=result;manifest['synthetic_generator']='v2: obsolete/replacement facts separated by 512 tokens'
(root/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(result|{'seconds':time.time()-start}),flush=True)
