"""Generate the same seeded prompts from saved training checkpoints."""
import argparse
import gc
from pathlib import Path
import sentencepiece as spm
import torch
from tfplslm.model import make_model
from tfplslm.train import seed_all,write_json
from tfplslm.evaluate import samples,transformer_samples

p=argparse.ArgumentParser()
p.add_argument('--checkpoint',action='append',required=True)
p.add_argument('--out',default='artifacts/checkpoint_samples')
args=p.parse_args()
for path in args.checkpoint:
    seed_all(9072026)
    ckpt=torch.load(path,map_location='cpu',weights_only=False)
    model=make_model(ckpt['config']['model']).cuda()
    model.load_state_dict(ckpt['model']);model.eval()
    sp=spm.SentencePieceProcessor(model_file=ckpt['config']['data']+'/tokenizer.model')
    out=Path(args.out)/f"step_{ckpt['step']}_tokens_{ckpt['tokens_seen']}"
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/'checkpoint_info.json',{'step':ckpt['step'],'tokens_seen':ckpt['tokens_seen'],
        'source_commit':ckpt['git_commit'],'sampling_seed':9072026,'temperature':.8,'top_k':40})
    (transformer_samples if model.cfg.architecture=='transformer' else samples)(model,sp,out)
    print('SAMPLES_WRITTEN',out,flush=True)
    del model,ckpt
    gc.collect();torch.cuda.empty_cache()
