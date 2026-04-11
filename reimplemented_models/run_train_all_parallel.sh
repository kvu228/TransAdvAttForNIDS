#!/usr/bin/env bash
# Chuẩn -> adv SPTS -> adv normal, mỗi giai đoạn song song trên các GPU được liệt kê.
# Ví dụ: ./reimplemented_models/run_train_all_parallel.sh 0 1 2 3
#         ./reimplemented_models/run_train_all_parallel.sh 0 1 -- --epochs 5
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LAUNCH="$ROOT/reimplemented_models/parallel_train_on_gpus.sh"

GPUS=()
EXTRA=()
while [ "$#" -gt 0 ]; do
  if [ "$1" = "--" ]; then
    shift
    EXTRA=("$@")
    break
  fi
  GPUS+=("$1")
  shift
done

if [ "${#GPUS[@]}" -lt 1 ]; then
  echo "Usage: $0 GPU_ID [GPU_ID ...] [-- extra python args]" >&2
  exit 1
fi

echo "=== Phase 1/3: standard CE ==="
"$LAUNCH" train_all_standard_models.py "${GPUS[@]}" -- "${EXTRA[@]}"
echo "=== Phase 2/3: adv SPTS ==="
"$LAUNCH" train_all_adv_with_spts.py "${GPUS[@]}" -- "${EXTRA[@]}"
echo "=== Phase 3/3: adv normal ==="
"$LAUNCH" train_all_adv_normal.py "${GPUS[@]}" -- "${EXTRA[@]}"
echo "Hoàn tất cả 3 phase."
