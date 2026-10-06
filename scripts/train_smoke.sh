#!/usr/bin/env bash
set -euo pipefail
python -m pytest -q
python -m tfplslm.train --mode smoke --config configs/smoke.json --out artifacts/smoke
