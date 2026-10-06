#!/usr/bin/env bash
set -euo pipefail
python -m tfplslm.train --mode train --config configs/poc_70m.json --out artifacts/poc "$@"
