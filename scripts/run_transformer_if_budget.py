"""Optional third baseline, admitted only after smoke and measured budget gates."""
import json,os,sys,time,subprocess
from pathlib import Path
os.chdir(Path(__file__).resolve().parents[1]);os.environ['PYTHONPATH']='src';os.environ['OMP_NUM_THREADS']='8'
def read(p): return json.loads(Path(p).read_text())
def save(p,d): Path(p).write_text(json.dumps(d,indent=2))
def run(args,path):
    print('RUN',' '.join(args),flush=True)
    with open(path,'w') as f: subprocess.run([sys.executable,'-u',*args],stdout=f,stderr=subprocess.STDOUT,check=True)
budget=read('artifacts/environment/budget.json');deadline=budget['hard_stop_epoch']
if not Path('artifacts/experiment_complete.json').exists(): raise RuntimeError('Primary and SSM experiment must finish first')
if deadline-time.time()<3600:
    save('artifacts/transformer_skipped.json',{'reason':'less than one hour remains'});sys.exit(0)
small=read('configs/smoke.json');small['model']['architecture']='transformer';small['model']['memory_slots']=0
save('configs/smoke_transformer.json',small)
run(['-m','tfplslm.train','--mode','smoke','--config','configs/smoke_transformer.json','--out','artifacts/transformer_smoke'],'artifacts/transformer_smoke.log')
if not read('artifacts/transformer_smoke/smoke.json')['passed']: raise RuntimeError('Transformer tiny-overfit gate failed')
run(['-m','tfplslm.train','--mode','benchmark','--config','configs/baseline_transformer.json','--out','artifacts/transformer_benchmark'],'artifacts/transformer_benchmark.log')
bench=read('artifacts/transformer_benchmark/benchmark.json')
previous=read('artifacts/ssm/final_training.json')
fraction=previous['tokens_seen']/(previous['step']*64*1024)
effective=bench['tokens_per_sec']*fraction*.9
expected=350_000_000/effective;reserve=1800
decision={'benchmark_tokens_per_sec':bench['tokens_per_sec'],'estimated_useful_tokens_per_sec':effective,
          'estimated_seconds':expected,'remaining_seconds':deadline-time.time(),'reserve_seconds':reserve}
decision['admitted']=bench['passed'] and deadline-time.time()>expected+reserve
save('artifacts/transformer_budget_decision.json',decision);print(json.dumps(decision),flush=True)
if not decision['admitted']:sys.exit(0)
c=read('configs/baseline_transformer.json');c['max_seconds']=int(deadline-time.time()-reserve)
save('configs/transformer_run.json',c)
run(['-m','tfplslm.train','--mode','train','--config','configs/transformer_run.json','--out','artifacts/transformer'],'artifacts/transformer.log')
run(['-m','tfplslm.evaluate','--checkpoint','artifacts/transformer/last.pt','--out','artifacts/transformer/eval'],'artifacts/transformer_eval.log')
save('artifacts/transformer_complete.json',{'completed_at':time.time()})
