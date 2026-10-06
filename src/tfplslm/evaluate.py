"""Streaming long-memory evaluation and paired causal state interventions."""
import argparse
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import sentencepiece as spm
import torch
from .model import make_model, tree_map
from .train import seed_all, validate, state_stats, write_json
from .synthetic import token_task, FILLERS


def state_bytes(state):
    values=[]
    tree_map(lambda x:values.append(x.numel()*x.element_size()) or x,state)
    return sum(values)


@torch.no_grad()
def consume(model,ids,state=None,disable_memory=False):
    last=None
    start=0
    while start<ids.shape[1]:
        remaining=model.cfg.memory_chunk-(state['position'] if state else 0)
        part=ids[:,start:start+remaining]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            hidden,state,_=model(part,state,disable_memory=disable_memory,return_hidden=True)
            last=model.lm_head(hidden[:,-1:]).float()
        start+=part.shape[1]
    return last,state


@torch.no_grad()
def score_answers(model,last,state,answers,disable_memory=False):
    length=max(map(len,answers)); batch=len(answers)
    target=torch.full((batch,length),-100,device='cuda',dtype=torch.long)
    for i,answer in enumerate(answers): target[i,:len(answer)]=torch.tensor(answer,device='cuda')
    inputs=target[:,:-1].clamp_min(0)
    predictions=[last]
    start=0
    while start<inputs.shape[1]:
        remaining=model.cfg.memory_chunk-state['position']
        part=inputs[:,start:start+remaining]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            hidden,state,_=model(part,state,disable_memory=disable_memory,return_hidden=True)
            predictions.append(model.lm_head(hidden).float())
        start+=part.shape[1]
    logits=torch.cat(predictions,1)
    loss=torch.nn.functional.cross_entropy(logits.flatten(0,1),target.flatten(),ignore_index=-100,reduction='none').reshape(batch,length)
    return (loss.sum(1)/(target!=-100).sum(1)).cpu().tolist()


@torch.no_grad()
def generate_from_state(model,last,state,max_tokens,eos,disable_memory=False,temperature=0.):
    generated=[]; done=torch.zeros(last.shape[0],dtype=torch.bool,device=last.device)
    for _ in range(max_tokens):
        if temperature>0:
            logits=last[:,-1]/temperature
            top=torch.topk(logits,40,dim=-1).values[:,-1:]
            ids=torch.multinomial(logits.masked_fill(logits<top,-float('inf')).softmax(-1),1)
        else: ids=last[:,-1].argmax(-1,keepdim=True)
        ids=torch.where(done[:,None],torch.full_like(ids,eos),ids)
        generated.append(ids); done|=ids[:,0]==eos
        if done.all(): break
        with torch.autocast('cuda',dtype=torch.bfloat16):
            h,state,_=model(ids,state,disable_memory=disable_memory,return_hidden=True)
            last=model.lm_head(h).float()
    return torch.cat(generated,1),state


def memory_eval(model,sp,config,out):
    rng=random.Random(config['seed']); started=time.time(); results=[]; details=[]
    filler=sp.encode(FILLERS[0])*100
    for distance in config['distances']:
        for category in config['categories']:
            cases=[token_task(sp,rng,category,distance,'test') for _ in range(config['examples_per_category'])]
            length=max(len(c[0]) for c in cases)
            prompts=[[sp.bos_id()]+filler[:length-len(p)]+p[1:] for p,a,m in cases]
            ids=torch.tensor(prompts,device='cuda')
            # Split before the final chunk; all facts are earlier than this split
            # for tested distances >=512. The same continuation is used for all
            # state interventions, which makes the comparison causal and paired.
            cut=((length-1)//model.cfg.memory_chunk)*model.cfg.memory_chunk
            for mode in config['ablations']:
                disabled=mode=='no_memory'
                torch.cuda.reset_peak_memory_stats(); begin=time.time()
                if mode=='zero':
                    state=model.initial_state(len(cases))
                elif mode=='reset':
                    # Premature reset: retain just the last complete history chunk.
                    _,state=consume(model,ids[:,max(0,cut-model.cfg.memory_chunk):cut],disable_memory=disabled)
                else:
                    _,state=consume(model,ids[:,:cut],disable_memory=disabled)
                    if mode=='shuffle':
                        # Rotation is a derangement; metadata records the donor answer.
                        state=tree_map(lambda x:x.roll(1,0),state)
                before=state_bytes(state)
                last,state=consume(model,ids[:,cut:],state,disable_memory=disabled)
                answer_nll=score_answers(model,last,state,[a for p,a,m in cases],disable_memory=disabled)
                predicted,_=generate_from_state(model,last,state,32,sp.eos_id(),disable_memory=disabled)
                decoded=[sp.decode(row.tolist()).strip() for row in predicted.cpu()]
                correct=[]; donor_match=[]
                for i,(text,(_,_,meta)) in enumerate(zip(decoded,cases)):
                    # Exact whole answer, with a terminal full stop allowed equally
                    # for every category. No substring or numeric extraction scoring.
                    cleaned=text.strip().rstrip('.').strip()
                    hit=cleaned==meta['answer']; correct.append(hit)
                    donor=cases[(i-1)%len(cases)][2]['answer']
                    donor_match.append(cleaned==donor)
                    details.append({**meta,'mode':mode,'prediction':text,'correct':hit,'donor_answer':donor,
                                    'prompt_tokens':length,'history_state_bytes':before,'index':i,'answer_nll':answer_nll[i]})
                n=len(correct); hits=sum(correct); acc=hits/n
                # Wilson interval prevents presenting tiny samples as precise.
                z=1.96; center=(acc+z*z/(2*n))/(1+z*z/n)
                half=z*math.sqrt(acc*(1-acc)/n+z*z/(4*n*n))/(1+z*z/n)
                metric={'distance':distance,'category':category,'mode':mode,'correct':hits,'n':n,
                        'accuracy':acc,'ci95':[max(0,center-half),min(1,center+half)],
                        'mean_answer_nll':float(np.mean(answer_nll)),
                        'donor_answer_accuracy':sum(donor_match)/n,'state_bytes':before,
                        'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'seconds':time.time()-begin}
                results.append(metric); print(json.dumps(metric),flush=True)
                write_json(out/'memory_results.json',results)
            write_json(out/'memory_details.json',details)
    write_json(out/'memory_timing.json',{'seconds':time.time()-started})
    return results


@torch.no_grad()
def memory_scaling(model,out):
    state=None; processed=0; rows=[]
    chunk=torch.randint(4,model.cfg.vocab_size,(1,model.cfg.memory_chunk),device='cuda')
    for distance in [512,2048,8192,32768,65536,131072]:
        torch.cuda.reset_peak_memory_stats()
        while processed<distance:
            _,state=consume(model,chunk,state); processed+=chunk.shape[1]
        torch.cuda.synchronize()
        rows.append({'history_tokens':processed,'state_bytes':state_bytes(state),
                     'allocated_bytes':torch.cuda.memory_allocated(),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                     'reserved_bytes':torch.cuda.memory_reserved()})
    write_json(out/'memory_scaling.json',rows)


def samples(model,sp,out):
    prompts=['Однажды вечером в маленьком городе','Почему небо кажется синим?',
             'Научный эксперимент показал, что','История России тесно связана с',
             'Анна открыла дверь и увидела']
    rows=[]
    for prompt in prompts:
        ids=torch.tensor([[sp.bos_id()]+sp.encode(prompt)],device='cuda')
        last,state=consume(model,ids)
        generated,_=generate_from_state(model,last,state,160,sp.eos_id(),temperature=.8)
        rows.append({'prompt':prompt,'continuation':sp.decode(generated[0].tolist())})
    write_json(out/'generation_samples.json',rows)


def counterfactual(model,sp,out):
    # Identical continuation, two different factual prefixes, interventions on
    # SSM-only vs explicit-memory-only as well as the complete recurrent state.
    texts=['Машина была красной.','Машина была синей.']
    fillers=sp.encode(' '.join(FILLERS))*100
    prefixes=[]
    for text in texts:
        fact=[sp.bos_id()]+sp.encode(text)
        prefixes.append(fact+fillers[:512-len(fact)])
    ids=torch.tensor(prefixes,device='cuda'); _,states=consume(model,ids)
    q=torch.tensor([sp.encode('Какого цвета была машина? Ответ:')]*2,device='cuda')
    rows=[]
    for mode in ['normal','swap_all','swap_memory','swap_ssm','zero']:
        s=tree_map(lambda x:x.clone(),states)
        if mode=='swap_all': s=tree_map(lambda x:x.flip(0),s)
        elif mode=='swap_memory': s['memory']=s['memory'].flip(0)
        elif mode=='swap_ssm': s['layers']=tree_map(lambda x:x.flip(0),s['layers'])
        elif mode=='zero': s=model.initial_state(2)
        logits,s=consume(model,q,s); generation,_=generate_from_state(model,logits,s,16,sp.eos_id())
        rows.append({'mode':mode,'outputs':[sp.decode(t.tolist()) for t in generation.cpu()],
                     'next_token_logits':logits[:,-1].cpu().tolist()})
    normal=torch.tensor(rows[0].pop('next_token_logits'))
    for row in rows[1:]:
        v=torch.tensor(row.pop('next_token_logits'))
        row['mean_absolute_logit_change']=(v-normal).abs().mean().item()
        row['kl_from_normal']=torch.nn.functional.kl_div(v.log_softmax(-1),normal.softmax(-1),reduction='batchmean').item()
    write_json(out/'counterfactual.json',rows)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True)
    p.add_argument('--config',default='configs/eval_memory.json'); p.add_argument('--out',required=True)
    p.add_argument('--skip-memory',action='store_true')
    args=p.parse_args(); seed_all(9072026)
    ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model=make_model(ckpt['config']['model']).cuda(); model.load_state_dict(ckpt['model']); model.eval()
    sp=spm.SentencePieceProcessor(model_file=ckpt['config']['data']+'/tokenizer.model')
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    write_json(out/'validation.json',validate(model,ckpt['config']['data'],batches=64)); model.eval()
    samples(model,sp,out); counterfactual(model,sp,out); memory_scaling(model,out)
    if not args.skip_memory: memory_eval(model,sp,json.loads(Path(args.config).read_text()),out)
