#!/usr/bin/env bash
# Chạy một script train Python trên nhiều GPU song song (mỗi GPU một shard).
#
# Usage (từ root repo):
#   ./reimplemented_models/parallel_train_on_gpus.sh train_all_standard_models.py 0 1 2 3
#   ./reimplemented_models/parallel_train_on_gpus.sh train_all_standard_models.py 0 1 -- --epochs 3 --datasets ton
#
# Mọi tham số sau -- được truyền nguyên cho từng process Python.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SCRIPT_NAME="${1:?Tên script trong reimplemented_models/, vd train_all_standard_models.py}"
shift

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
  echo "Cần ít nhất một GPU id. Ví dụ: $0 train_all_standard_models.py 0 1 2 3" >&2
  exit 1
fi

N=${#GPUS[@]}
SCRIPT_PATH="$ROOT/reimplemented_models/$SCRIPT_NAME"
if [ ! -f "$SCRIPT_PATH" ]; then
  echo "Không tìm thấy: $SCRIPT_PATH" >&2
  exit 1
fi

echo "parallel_train_on_gpus: $SCRIPT_NAME | $N GPU(s): ${GPUS[*]} | extra: ${EXTRA[*]:-}"
LOGDIR="$ROOT/reimplemented_models/parallel_logs"
mkdir -p "$LOGDIR"
TS="$(date +%Y%m%d_%H%M%S)"

for i in "${!GPUS[@]}"; do
  gid="${GPUS[$i]}"
  log="$LOGDIR/${SCRIPT_NAME%.py}_shard${i}_gpu${gid}_${TS}.log"
  echo "  shard $i/$N -> CUDA_VISIBLE_DEVICES=$gid  (log: $log)"
  CUDA_VISIBLE_DEVICES="$gid" python -u "$SCRIPT_PATH" \
    --shard-index "$i" --shard-count "$N" "${EXTRA[@]}" >>"$log" 2>&1 &
done

wait
echo "Xong $N tiến trình ($SCRIPT_NAME). Xem log trong $LOGDIR"
