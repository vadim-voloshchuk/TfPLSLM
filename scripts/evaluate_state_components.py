"""Post-hoc content intervention; execute after training jobs release the GPU."""
import json
from pathlib import Path
import sentencepiece as spm
import torch
from tfplslm.model import make_model
from tfplslm.train import seed_all,write_json
from tfplslm.evaluate import memory_eval

seed_all(9072026)
ckpt=torch.load('artifacts/poc/last.pt',map_location='cpu',weights_only=False)
model=make_model(ckpt['config']['model']).cuda()
model.load_state_dict(ckpt['model']);model.eval()
sp=spm.SentencePieceProcessor(model_file=ckpt['config']['data']+'/tokenizer.model')
config=json.loads(Path('configs/eval_state_components.json').read_text())
out=Path('artifacts/poc/state_components');out.mkdir(parents=True,exist_ok=True)
write_json(out/'config.json',config)
memory_eval(model,sp,config,out)
