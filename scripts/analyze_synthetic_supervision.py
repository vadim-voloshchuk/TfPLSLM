import json
from pathlib import Path
import numpy as np
import sentencepiece as spm
sp=spm.SentencePieceProcessor(model_file='data/tokenizer.model')
t=np.memmap('data/synthetic.bin',mode='r',dtype=np.uint16)
idx=np.load('data/synthetic.idx.npy')
colon=sp.encode('Ответ:')[-1]
assert sp.decode([colon]).strip()==':'
answer=0
for a,b in zip(idx[:-1],idx[1:]):
    doc=t[a:b];positions=np.flatnonzero(doc[-64:]==colon)
    assert len(positions)
    answer+=min(64,len(doc))-int(positions[-1])-1
total=int(len(t)-(len(idx)-1))
result={'documents':len(idx)-1,'answer_tokens_including_eos':answer,
        'total_supervised_synthetic_tokens':total,'synthetic_answer_fraction':answer/total,
        'expected_mixed_training_answer_fraction':answer/total/7,
        'method':'Tokens after the final question colon, including EOS; prepared synthetic corpus'}
Path('artifacts/synthetic_answer_weight.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
