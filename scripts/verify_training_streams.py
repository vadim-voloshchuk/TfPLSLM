"""Audit completed controls without allocating GPU memory."""
import gc
import hashlib
import json
from pathlib import Path
import torch

rows=[]
for name in ['poc','ssm','transformer']:
    root=Path('artifacts')/name
    if not (root/'final_training.json').exists():continue
    final=json.loads((root/'final_training.json').read_text())
    if final['stop_reason']!='token_budget':continue
    ckpt=torch.load(root/'last.pt',map_location='cpu',weights_only=False)
    sampler=json.dumps(ckpt['batcher'],sort_keys=True,separators=(',',':')).encode()
    rows.append({'model':name,'step':ckpt['step'],'tokens_seen':ckpt['tokens_seen'],
        'source_tokens':final['source_tokens'],'sampler_sha256':hashlib.sha256(sampler).hexdigest(),
        'seed':ckpt['config']['seed'],'data':ckpt['config']['data'],
        'checkpoint_source_commit':ckpt['git_commit']})
    del ckpt
    gc.collect()
keys=['step','tokens_seen','source_tokens','sampler_sha256','seed','data']
match=bool(rows) and all(all(r[k]==rows[0][k] for k in keys) for r in rows)
result={'matched':match,'runs':rows,
    'scope':'Same deterministic sampler, seed and prepared data; final RNG/cursors and token counts match'}
Path('artifacts/training_stream_comparison.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
if not match:raise RuntimeError('Completed controls did not consume the same training stream')
