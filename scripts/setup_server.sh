#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pip install -e . pytest==8.3.5 matplotlib==3.10.1
if [ ! -d vendor/mamba/.git ]; then
  mkdir -p vendor
  git clone https://github.com/state-spaces/mamba.git vendor/mamba
fi
git -C vendor/mamba checkout 95d8aba8a8c75aedcaa6143713b11e745e7cd0d9
