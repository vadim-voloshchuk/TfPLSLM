"""Pinned streaming corpus, SentencePiece, uint16 documents, sequential batches."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import sentencepiece as spm
from .synthetic import CATEGORIES, token_task

DATASET = 'deepvk/cultura_ru_edu'
REVISION = '55788b7e4a9b1f64c95e4bdd040b6ea93aa4e403'


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1<<20), b''): h.update(chunk)
    return h.hexdigest()


def rows(split):
    from datasets import load_dataset
    # Explicit files avoid enumerating or prefetching the entire massive corpus.
    files = [f'https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/data/' +
             (f'train-{i:03d}.parquet' if split=='train' else 'validation.parquet')
             for i in (range(8) if split=='train' else [0])]
    return load_dataset('parquet', data_files={split:files}, split=split, streaming=True)


class Writer:
    def __init__(self, root, name):
        self.root=Path(root); self.name=name; self.f=open(self.root/f'{name}.bin','wb')
        self.offsets=[0]; self.hashes=[]
    def add(self, ids, text_hash=''):
        np.asarray(ids,dtype=np.uint16).tofile(self.f)
        self.offsets.append(self.offsets[-1]+len(ids)); self.hashes.append(text_hash)
    def close(self):
        self.f.close()
        np.save(self.root/f'{self.name}.idx.npy',np.asarray(self.offsets,dtype=np.int64))
        (self.root/f'{self.name}.hashes.json').write_text(json.dumps(self.hashes))
        return {'documents':len(self.hashes),'tokens':self.offsets[-1],
                'sha256':sha256(self.root/f'{self.name}.bin'),
                'index_sha256':sha256(self.root/f'{self.name}.idx.npy')}


def prepare(args):
    root=Path(args.out); root.mkdir(parents=True,exist_ok=True)
    started=time.time(); manifest={'dataset':DATASET,'revision':REVISION,'seed':args.seed}
    sample_path=root/'tokenizer_sample.txt'
    train_iter=iter(rows('train'))
    sample=[]; seen=set(); chars=0
    while len(sample)<args.tokenizer_docs and chars<30_000_000:
        text=next(train_iter)['text'].strip()
        h=hashlib.sha256(text.encode()).hexdigest()
        if len(text)<100 or h in seen: continue
        seen.add(h); sample.append(text); chars+=len(text)
    sample_path.write_text('\n'.join(t.replace('\n',' ')[:16000] for t in sample),encoding='utf-8')
    manifest['tokenizer_sample']={'documents':len(sample),'characters':chars,'sha256':sha256(sample_path)}
    spm.SentencePieceTrainer.train(input=str(sample_path),model_prefix=str(root/'tokenizer'),
        vocab_size=args.vocab_size,model_type='bpe',character_coverage=.9995,
        byte_fallback=True,normalization_rule_name='identity',split_digits=True,
        pad_id=0,bos_id=1,eos_id=2,unk_id=3,num_threads=8,
        shuffle_input_sentence=False,max_sentence_length=200000)
    sp=spm.SentencePieceProcessor(model_file=str(root/'tokenizer.model'))
    manifest['tokenizer_sha256']=sha256(root/'tokenizer.model')
    print('TOKENIZER_READY',json.dumps(manifest),flush=True)
    val=Writer(root,'validation'); val_hashes=set()
    for row in rows('validation'):
        text=row['text'].strip(); h=hashlib.sha256(text.encode()).hexdigest()
        if len(text)<100 or h in seen or h in val_hashes: continue
        val_hashes.add(h); val.add([1]+sp.encode(text)+[2],h)
        if val.offsets[-1]>=args.validation_tokens: break
    manifest['validation']=val.close()
    train=Writer(root,'train'); train_seen=set()
    def add_text(text):
        h=hashlib.sha256(text.encode()).hexdigest()
        if len(text)<100 or h in val_hashes or h in train_seen: return
        train_seen.add(h); train.add([1]+sp.encode(text)+[2],h)
    for text in sample: add_text(text)
    for row in train_iter:
        add_text(row['text'].strip())
        if len(train.hashes)%10000==0:
            print('DATA_PROGRESS',train.offsets[-1],len(train.hashes),flush=True)
        if train.offsets[-1]>=args.train_tokens: break
    manifest['train']=train.close()
    synth=Writer(root,'synthetic'); rng=random.Random(args.seed)
    while synth.offsets[-1]<args.synthetic_tokens:
        category=CATEGORIES[len(synth.hashes)%len(CATEGORIES)]
        # Fact/question adjacent across the 512-token boundary for the majority
        # of examples; longer examples explicitly exercise truncated BPTT limits.
        distance=rng.choice([64,128,256,512,512,512,512,1024,2048])
        prompt,answer,_=token_task(sp,rng,category,distance,'train')
        synth.add(prompt+answer)
    manifest['synthetic']=synth.close()
    manifest['seconds']=time.time()-started
    manifest['exact_train_validation_overlap']=len(train_seen & val_hashes)
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('DATA_READY',json.dumps(manifest),flush=True)


class Documents:
    def __init__(self, root, name):
        root=Path(root)
        self.tokens=np.memmap(root/f'{name}.bin',dtype=np.uint16,mode='r')
        self.offsets=np.load(root/f'{name}.idx.npy')
    def __len__(self): return len(self.offsets)-1
    def doc(self, i): return self.tokens[self.offsets[i]:self.offsets[i+1]]


class SequentialBatcher:
    """Each row keeps its document across updates; new documents reset state.

Padding is masked. A document never shares one recurrent trajectory with the
next document. No JSON parsing or tokenization occurs in the training hot path.
"""
    def __init__(self, corpora, batch_size, length, seed=1, synthetic_fraction=1/7):
        self.corpora=corpora; self.batch=batch_size; self.length=length
        self.rng=np.random.default_rng(seed); self.synthetic_fraction=synthetic_fraction
        self.rows=[None]*batch_size

    def next(self):
        x=np.zeros((self.batch,self.length),dtype=np.int64)
        y=np.full((self.batch,self.length),-100,dtype=np.int64)
        reset=np.zeros(self.batch,dtype=bool)
        sources=np.zeros(self.batch,dtype=np.int64)
        for i, cursor in enumerate(self.rows):
            if cursor is None:
                source=int(len(self.corpora)>1 and self.rng.random()<self.synthetic_fraction)
                docid=int(self.rng.integers(len(self.corpora[source])))
                cursor=[source,docid,0]; reset[i]=True
            source,docid,pos=cursor; doc=self.corpora[source].doc(docid)
            count=min(self.length,len(doc)-1-pos)
            x[i,:count]=doc[pos:pos+count]; y[i,:count]=doc[pos+1:pos+count+1]
            sources[i]=source
            self.rows[i]=None if pos+count>=len(doc)-1 else [source,docid,pos+count]
        return x,y,reset,sources

    def state_dict(self): return {'rows':self.rows,'rng':self.rng.bit_generator.state}
    def load_state_dict(self,d): self.rows=d['rows']; self.rng.bit_generator.state=d['rng']


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',default='data')
    p.add_argument('--train-tokens',type=int,default=300_000_000)
    p.add_argument('--synthetic-tokens',type=int,default=50_000_000)
    p.add_argument('--validation-tokens',type=int,default=1_000_000)
    p.add_argument('--tokenizer-docs',type=int,default=12000)
    p.add_argument('--vocab-size',type=int,default=16384); p.add_argument('--seed',type=int,default=20261007)
    prepare(p.parse_args())
