"""Sequential experiment controller with saved gates and rental deadline checks."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ['PYTHONPATH']=str(ROOT/'src')
os.environ['OMP_NUM_THREADS']='8'


def read(path): return json.loads(Path(path).read_text())


def run(args,log):
    print('RUN',' '.join(args),flush=True)
    with open(log,'w') as f:
        result=subprocess.run([sys.executable,'-u',*args],stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'{args[0]} failed; inspect {log}')


budget=read('artifacts/environment/budget.json')
deadline=budget['hard_stop_epoch']
while not Path('data/manifest.json').exists():
    if time.time()>deadline-3*3600: raise RuntimeError('Data preparation consumed training reserve')
    time.sleep(20)
if 'v2:' not in read('data/manifest.json').get('synthetic_generator',''):
    run(['scripts/prepare_synthetic_v2.py'],'artifacts/synthetic_v2.log')
manifest=read('data/manifest.json')
assert manifest['exact_train_validation_overlap']==0
assert read('artifacts/smoke/smoke.json')['passed']
bench=read('artifacts/benchmark64/benchmark.json')
assert bench['passed'] and bench['stable_steps']>=100
short=read('artifacts/short_real/final_training.json')
assert short['validation_loss']<read('artifacts/short_real/initial_validation.json')['validation_loss']
if not Path('artifacts/short_real/eval/memory_scaling.json').exists():
    raise RuntimeError('End-to-end generation and evaluation smoke not complete')

for name,base_config in [('poc','configs/poc_70m.json'),('ssm','configs/baseline_ssm.json')]:
    remaining=deadline-time.time()
    # Padding and evaluation overhead are budgeted using the real-data pilot.
    pilot_metrics=[json.loads(line) for line in Path('artifacts/short_real/metrics.jsonl').read_text().splitlines()]
    rates=[m['tokens_per_sec'] for m in pilot_metrics if m.get('step',0)>=20 and 'tokens_per_sec' in m]
    actual_rate=sum(rates)/len(rates) if rates else short['tokens_seen']/short['seconds']
    expected=350e6/max(actual_rate,25000)
    reserve=45*60
    if remaining<expected+reserve:
        Path(f'artifacts/{name}_skipped.json').write_text(json.dumps({'reason':'insufficient budget',
            'remaining_seconds':remaining,'estimated_seconds':expected,'reserve_seconds':reserve},indent=2))
        break
    cfg=read(base_config);cfg['max_seconds']=int(min(remaining-reserve,4*3600))
    cfg_path=f'configs/{name}_run.json';Path(cfg_path).write_text(json.dumps(cfg,indent=2))
    run(['-m','tfplslm.train','--mode','train','--config',cfg_path,'--out',f'artifacts/{name}'],f'artifacts/{name}.log')
    run(['-m','tfplslm.evaluate','--checkpoint',f'artifacts/{name}/last.pt',
         '--config','configs/eval_memory.json' if name=='poc' else 'configs/eval_ssm.json',
         '--out',f'artifacts/{name}/eval'],f'artifacts/{name}_eval.log')
Path('artifacts/experiment_complete.json').write_text(json.dumps({'completed_at':time.time()}))
print('EXPERIMENT_COMPLETE',flush=True)
