#!/usr/bin/env bash
set -euo pipefail
python -m tfplslm.data --out data "$@"
