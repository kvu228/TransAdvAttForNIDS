#!/usr/bin/env bash
# Huấn luyện hàng loạt vào reimplemented_models/: chuẩn → adv SPTS → adv normal.
# Có thể truyền thêm tham số cho từng bước, ví dụ: ./run_train_all.sh --epochs 5
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# -u: in log ngay (unbuffered), tránh treo im lặng khi đang đọc CSV / train
python -u reimplemented_models/train_all_standard_models.py "$@"
python -u reimplemented_models/train_all_adv_with_spts.py "$@"
python -u reimplemented_models/train_all_adv_normal.py "$@"
