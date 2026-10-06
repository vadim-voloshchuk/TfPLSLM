"""Bounded training, performance gate, tiny overfit and complete resume state."""
import argparse
import json
import math
import os
import random
import subprocess
import signal
import time
from pathlib import Path
import numpy as np
import psutil
import torch
from .model import make_model, detach_state, tree_map
from .data import Documents, SequentialBatcher


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True


def optimizer_for(model, lr, wd):
    decay=[]; no_decay=[]
    for name,p in model.named_parameters():
        (no_decay if p.ndim<2 or any(k in name for k in ['A_log','dt_bias','.D']) else decay).append(p)
    return torch.optim.AdamW([{'params':decay,'weight_decay':wd},
                              {'params':no_decay,'weight_decay':0.}],lr=lr,betas=(.9,.95),eps=1e-8,fused=True)


def reset_rows(state, reset):
    if state is None: return state
    return tree_map(lambda x:x * (~reset).reshape(-1,*([1]*(x.ndim-1))),state)


def loss_unroll(model,x,y,state,unroll):
    total=(y!=-100).sum().clamp_min(1)
    loss=0.; diagnostics={}
    for xx,yy in zip(x.chunk(unroll,1),y.chunk(unroll,1)):
        hidden,state,diagnostics=model(xx,state,return_hidden=True)
        # Reduction=sum avoids NaN for an entirely padded second chunk.
        logits=model.lm_head(hidden)
        loss=loss+torch.nn.functional.cross_entropy(logits.float().flatten(0,1),yy.flatten(),
                                                    ignore_index=-100,reduction='sum')/total
    return loss,state,diagnostics


def cpu_tree(x): return tree_map(lambda t:t.detach().cpu(),x)


def save_checkpoint(path,model,opt,state,batcher,step,tokens,config,extra=None):
    payload={'model':model.state_dict(),'optimizer':opt.state_dict(),'state':cpu_tree(state),
             'batcher':batcher.state_dict() if batcher else None,'step':step,'tokens_seen':tokens,
             'config':config,'scheduler':{'step':step,'tokens_seen':tokens},'scaler':None,
             'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
             'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
             'git_commit':git_commit(),'extra':extra or {}}
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp'); torch.save(payload,tmp); os.replace(tmp,path)


def git_commit():
    try: return subprocess.check_output(['git','rev-parse','HEAD'],stderr=subprocess.DEVNULL,text=True).strip()
    except Exception:
        p=Path('SOURCE_COMMIT'); return p.read_text().strip() if p.exists() else 'uncommitted'


def load_checkpoint(path,model,opt=None,batcher=None):
    ckpt=torch.load(path,map_location='cpu',weights_only=False)
    model.load_state_dict(ckpt['model'])
    if opt is not None: opt.load_state_dict(ckpt['optimizer'])
    if batcher is not None and ckpt['batcher']: batcher.load_state_dict(ckpt['batcher'])
    random.setstate(ckpt['python_rng']); np.random.set_state(ckpt['numpy_rng'])
    torch.set_rng_state(ckpt['torch_rng']); torch.cuda.set_rng_state_all(ckpt['cuda_rng'])
    state=tree_map(lambda x:x.cuda(),ckpt['state'])
    return ckpt,state


def state_stats(state,diag):
    if state is None: return {}
    mem=state['memory'].detach().float()
    flat=torch.cat([s[0].detach().flatten() for s in state['layers']])
    result={'ssm_rms':flat.square().mean().sqrt().item(),'ssm_max_abs':flat.abs().max().item(),
            'ssm_mean':flat.mean().item(),'ssm_std':flat.std().item(),
            'memory_rms':mem.square().mean().sqrt().item() if mem.numel() else 0.,
            'memory_max_abs':mem.abs().max().item() if mem.numel() else 0.}
    if mem.numel():
        norms=mem.square().mean(-1).sqrt()
        result['slot_rms']=norms.mean(0).tolist()
        result['dead_slot_fraction']=(norms<1e-5).float().mean().item()
    for k,v in diag.items():
        result[k+'_mean']=v.detach().float().mean().item()
        result[k+'_saturation']=((v<.01)|(v>.99)).float().mean().item()
    return result


def gpu_stats():
    try:
        s=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True)
        util,mem=map(float,s.strip().split(',')); return {'gpu_utilization':util,'nvidia_used_mib':mem}
    except Exception: return {}


def write_json(path,value): Path(path).write_text(json.dumps(value,indent=2),encoding='utf-8')


@torch.no_grad()
def validate(model, data_root, batches=8):
    model.eval(); batcher=SequentialBatcher([Documents(data_root,'validation')],8,model.cfg.memory_chunk*2,seed=989)
    total_loss=0.; total_tokens=0; state=None
    for _ in range(batches):
        xx,yy,reset,_=batcher.next(); x=torch.from_numpy(xx).cuda(); y=torch.from_numpy(yy).cuda()
        state=reset_rows(state,torch.from_numpy(reset).cuda())
        with torch.autocast('cuda',dtype=torch.bfloat16): loss,state,_=loss_unroll(model,x,y,state,2)
        n=int((yy!=-100).sum()); total_loss+=loss.item()*n; total_tokens+=n
    model.train(); loss=total_loss/total_tokens
    return {'validation_loss':loss,'perplexity':math.exp(min(loss,30)),'validation_tokens':total_tokens}


def smoke(config,out):
    model=make_model(config['model']).cuda(); opt=optimizer_for(model,config['lr'],0.)
    chunk=model.cfg.memory_chunk; unroll=config['unroll_chunks']
    tokens=torch.randint(4,model.cfg.vocab_size,(config['batch_size'],chunk*unroll+1),device='cuda')
    x,y=tokens[:,:-1],tokens[:,1:]
    losses=[]; started=time.time()
    for step in range(config['steps']):
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16): loss,state,diag=loss_unroll(model,x,y,None,unroll)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True); opt.step()
        losses.append(loss.item())
        if step%20==0: print(json.dumps({'step':step,'loss':losses[-1]}),flush=True)
    save_checkpoint(out/'smoke.pt',model,opt,None,None,config['steps'],x.numel()*config['steps'],config)
    # Exact optimizer/RNG resume test: one update then reload and repeat it.
    def step_once():
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16): loss,_,_=loss_unroll(model,x,y,None,unroll)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
        return loss.item(),{k:v.detach().clone() for k,v in model.state_dict().items()}
    first,weights=step_once(); load_checkpoint(out/'smoke.pt',model,opt); second,_=step_once()
    maxdiff=max((weights[k]-v).abs().max().item() for k,v in model.state_dict().items())
    result={'first_loss':losses[0],'final_loss':losses[-1],'losses':losses,
            'resume_loss_difference':abs(first-second),'resume_max_parameter_difference':maxdiff,
            'seconds':time.time()-started,'passed':losses[-1]<min(.5,losses[0]*.15) and maxdiff<1e-5}
    write_json(out/'smoke.json',result); print(json.dumps(result | {'losses':[]}),flush=True)
    if not result['passed']: raise RuntimeError('Tiny overfit/resume gate failed')


def benchmark(config,out,steps=110):
    model=make_model(config['model']).cuda(); opt=optimizer_for(model,config['lr'],config['weight_decay'])
    B,L=config['batch_size'],model.cfg.memory_chunk*config['unroll_chunks']
    tokens=torch.randint(4,model.cfg.vocab_size,(B,L+1),device='cuda'); x,y=tokens[:,:-1],tokens[:,1:]
    times=[]; losses=[]; state=None; torch.cuda.reset_peak_memory_stats()
    for step in range(steps):
        torch.cuda.synchronize(); start=time.perf_counter(); opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16): loss,state,diag=loss_unroll(model,x,y,state,config['unroll_chunks'])
        loss.backward(); grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True); opt.step()
        state=detach_state(state); torch.cuda.synchronize(); elapsed=time.perf_counter()-start
        losses.append(loss.item())
        if step>=10: times.append(elapsed)
        if step%10==0: print(json.dumps({'step':step,'seconds':elapsed,'tokens_per_sec':B*L/elapsed,'loss':losses[-1]}),flush=True)
    speed=float(B*L/np.mean(times))
    result={'batch_size':B,'unroll_chunks':config['unroll_chunks'],'sequence_chunk':model.cfg.memory_chunk,
            'parameters':sum(p.numel() for p in model.parameters()),'stable_steps':len(times),
            'tokens_per_sec':speed,'step_seconds_mean':float(np.mean(times)),
            'step_seconds_p50':float(np.median(times)),'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
            'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'loss':losses[-1],
            'gradient_norm':grad.item(),'cpu_percent':psutil.cpu_percent(),
            'dataloader_wait_seconds':0.,'input':'preallocated random tokens; real loader measured during train',
            'estimated_350m_hours':350e6/speed/3600,'passed':speed>=25000,**gpu_stats(),**state_stats(state,diag)}
    write_json(out/'benchmark.json',result); print(json.dumps(result),flush=True)


def train(config,out,resume=None):
    started=time.time(); model=make_model(config['model']).cuda(); opt=optimizer_for(model,config['lr'],config['weight_decay'])
    batcher=SequentialBatcher([Documents(config['data'],'train'),Documents(config['data'],'synthetic')],
        config['batch_size'],model.cfg.memory_chunk*config['unroll_chunks'],config['seed'])
    step=tokens=0; state=None; source_tokens=[0,0]
    if resume:
        ckpt,state=load_checkpoint(resume,model,opt,batcher); step=ckpt['step']; tokens=ckpt['tokens_seen']
        source_tokens=ckpt.get('extra',{}).get('source_tokens',[0,0])
    last_save=last_log=time.time(); last_tokens=tokens; loader_wait=0.; loss_sum=0.; interval=0
    initial=validate(model,config['data']); write_json(out/'initial_validation.json',initial)
    print(json.dumps(initial),flush=True); torch.cuda.reset_peak_memory_stats()
    stop_reason='token_budget'
    slow_intervals=0
    stop_requested=[False]
    signal.signal(signal.SIGTERM,lambda *_:stop_requested.__setitem__(0,True))
    signal.signal(signal.SIGINT,lambda *_:stop_requested.__setitem__(0,True))
    with open(out/'metrics.jsonl','a',buffering=1) as log:
        while tokens<config['tokens']:
            if stop_requested[0] or time.time()-started>config['max_seconds'] or (out/'STOP').exists():
                stop_reason='wall_clock_or_stop_file'; break
            tick=time.perf_counter(); xx,yy,rr,src=batcher.next()
            x=torch.from_numpy(xx).cuda(); y=torch.from_numpy(yy).cuda()
            state=reset_rows(state,torch.from_numpy(rr).cuda()); loader_wait+=time.perf_counter()-tick
            progress=min(1.,tokens/config['tokens']); warm=min(1.,(step+1)/config['warmup_steps'])
            lr=config['lr']*warm*(config['min_lr_ratio']+(1-config['min_lr_ratio'])*.5*(1+math.cos(math.pi*progress)))
            for group in opt.param_groups: group['lr']=lr
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16): loss,state,diag=loss_unroll(model,x,y,state,config['unroll_chunks'])
            loss.backward(); grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True); opt.step()
            state=detach_state(state); step+=1
            count=int((yy!=-100).sum()); tokens+=count
            for source in [0,1]: source_tokens[source]+=int(((yy!=-100)*(src==source)[:,None]).sum())
            loss_sum+=loss.detach(); interval+=1
            if step%config['log_every']==0:
                torch.cuda.synchronize(); now=time.time()
                metric={'step':step,'tokens_seen':tokens,'source_tokens':source_tokens.copy(),
                        'loss':(loss_sum/interval).item(),'lr':lr,'gradient_norm':grad.item(),
                        'tokens_per_sec':(tokens-last_tokens)/(now-last_log),'elapsed_seconds':now-started,
                        'dataloader_wait_seconds':loader_wait,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                        'cpu_percent':psutil.cpu_percent(),**gpu_stats(),**state_stats(state,diag)}
                log.write(json.dumps(metric)+'\n'); print(json.dumps(metric),flush=True)
                if not math.isfinite(metric['loss']) or metric.get('ssm_max_abs',0)>1e6:
                    raise RuntimeError('Nonfinite loss or exploding state')
                last_log=now; last_tokens=tokens; loader_wait=0.; loss_sum=0.; interval=0
                slow_intervals=slow_intervals+1 if step>=60 and metric['tokens_per_sec']<15000 else 0
                if slow_intervals>=3:
                    stop_reason='sustained_throughput_below_15k'; break
            if time.time()-last_save>=config['checkpoint_seconds']:
                save_checkpoint(out/'last.pt',model,opt,state,batcher,step,tokens,config,{'source_tokens':source_tokens})
                val=validate(model,config['data']); val.update(step=step,tokens_seen=tokens)
                log.write(json.dumps(val)+'\n'); print(json.dumps(val),flush=True); last_save=time.time()
        save_checkpoint(out/'last.pt',model,opt,state,batcher,step,tokens,config,{'source_tokens':source_tokens})
        final=validate(model,config['data'],batches=32)
        final.update(step=step,tokens_seen=tokens,source_tokens=source_tokens,seconds=time.time()-started,stop_reason=stop_reason)
        write_json(out/'final_training.json',final); print(json.dumps(final),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True)
    p.add_argument('--mode',choices=['smoke','benchmark','train'],required=True)
    p.add_argument('--out',required=True); p.add_argument('--resume'); p.add_argument('--steps',type=int,default=110)
    args=p.parse_args(); config=json.loads(Path(args.config).read_text()); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    write_json(out/'config.json',config); seed_all(config['seed'])
    {'smoke':lambda:smoke(config,out),'benchmark':lambda:benchmark(config,out,args.steps),
     'train':lambda:train(config,out,args.resume)}[args.mode]()
