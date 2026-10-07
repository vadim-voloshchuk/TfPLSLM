"""Run bounded final diagnostics after both training controllers finish."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.chdir(Path(__file__).resolve().parents[1])
os.environ['PYTHONPATH']='src';os.environ['OMP_NUM_THREADS']='8'
deadline=json.loads(Path('artifacts/environment/budget.json').read_text())['hard_stop_epoch']
while True:
    status=subprocess.run(['supervisorctl','status','tfplslm-experiment','tfplslm-transformer'],capture_output=True,text=True).stdout
    if 'RUNNING' not in status:break
    if time.time()>deadline-180:raise RuntimeError('Insufficient time for final GPU diagnostics')
    time.sleep(15)
if not Path('artifacts/experiment_complete.json').exists():raise RuntimeError('Main controller did not finish')
if not Path('artifacts/transformer_complete.json').exists():
    decision=Path('artifacts/transformer_budget_decision.json')
    skipped=Path('artifacts/transformer_skipped.json').exists() or decision.exists() and not json.loads(decision.read_text())['admitted']
    if not skipped:raise RuntimeError('Optional controller failed; review it before final diagnostics')

history={'poc-early.pt':'3ca7e117a6be464a06d7e6c866673a1eb5777460888e5cb54978aafa586189ce',
         'poc-76m.pt':'d0fa7de0d40d7ccb5581a0b58efa7e2566f678977b77ab84e643df34724f45d3'}
paths=[]
for name,expected in history.items():
    path=Path('/workspace/TfPLSLM-history')/name
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    if h.hexdigest()!=expected:raise RuntimeError('Historical checkpoint hash mismatch: '+name)
    paths.extend(['--checkpoint',str(path)])

def run(args,log):
    print('RUN',' '.join(args),flush=True)
    with open(log,'w') as f:subprocess.run([sys.executable,'-u',*args],stdout=f,stderr=subprocess.STDOUT,check=True)
run(['scripts/sample_checkpoints.py',*paths],'artifacts/history_samples.log')
run(['scripts/evaluate_state_components.py'],'artifacts/state_components.log')
run(['scripts/verify_training_streams.py'],'artifacts/training_stream_comparison.log')
run(['scripts/build_report.py'],'artifacts/build_report.log')
Path('artifacts/final_diagnostics_complete.json').write_text(json.dumps({'completed_at':time.time()}))
print('FINAL_DIAGNOSTICS_COMPLETE',flush=True)
