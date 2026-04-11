#!/usr/bin/env bash
# Chạy huấn luyện 20 model chuẩn (2 dataset × 5 kiến trúc × t/s) vào thư mục reimplemented_models/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
exec python -u reimplemented_models/train_all_standard_models.py "$@"
