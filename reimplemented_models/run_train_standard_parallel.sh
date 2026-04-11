#!/usr/bin/env bash
# Train chuẩn (CE) trên nhiều GPU. Ví dụ 4 GPU:
#   ./reimplemented_models/run_train_standard_parallel.sh 0 1 2 3
# Tham số thêm cho Python sau --:
#   ./reimplemented_models/run_train_standard_parallel.sh 0 1 -- --epochs 5 --datasets ton
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec "$ROOT/reimplemented_models/parallel_train_on_gpus.sh" train_all_standard_models.py "$@"
