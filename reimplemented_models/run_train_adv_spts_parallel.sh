#!/usr/bin/env bash
# Adv train + SPTS, nhiều GPU. Ví dụ: ./reimplemented_models/run_train_adv_spts_parallel.sh 0 1 2 3
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec "$ROOT/reimplemented_models/parallel_train_on_gpus.sh" train_all_adv_with_spts.py "$@"
