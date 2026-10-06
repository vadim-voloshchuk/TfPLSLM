"""Rental deadline safety net. Run only on the explicitly authorized instance.

Secrets are read from the injected environment and never printed or persisted.
Stop preserves the container disk; it does not destroy the instance. Storage
charges can continue. Normal completion should back up artifacts and stop sooner.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

p=argparse.ArgumentParser(); p.add_argument('--deadline',type=float,required=True)
args=p.parse_args()
while time.time()<args.deadline: time.sleep(min(30,max(0,args.deadline-time.time())))
root=Path('/workspace/TfPLSLM/artifacts')
for path in root.iterdir():
    if path.is_dir(): (path/'STOP').touch()
print('Rental deadline reached; allowing checkpoints to finish.',flush=True)
time.sleep(120)
subprocess.run(['vastai','stop','instance',os.environ['CONTAINER_ID'],
                '--api-key',os.environ['CONTAINER_API_KEY']],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
