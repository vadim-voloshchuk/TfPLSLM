#!/usr/bin/env bash
set -euo pipefail
destination="${1:-results.tar.gz}"
tar -czf "$destination" --exclude='*.pt' --exclude='short_data' --exclude='*.bin' \
    artifacts reports configs data/manifest.json data/tokenizer.model data/tokenizer.vocab
printf 'Metadata archive: %s\nCheckpoints remain under artifacts/<run>/last.pt\n' "$destination"
