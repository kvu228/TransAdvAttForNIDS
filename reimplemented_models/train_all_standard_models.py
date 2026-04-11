#!/usr/bin/env python3
"""
Huấn luyện lại toàn bộ NIDS chuẩn (CE, không adversarial training) và lưu .pth vào thư mục này.

Cấu hình mặc định (khớp ``train_NIDS/training.py``):
  - Dataset: ton, ids18 (cần ``{ds}_sam_train_{t|s}.csv`` trong ``STORAGE_DIR/dataset``)
  - Kiến trúc: mlp, cnn, rescnn, lstm, Selfattention × target (t) × surrogate (s)
  - Tên file: ``{dataset}_{arch}_{t|s}.pth`` (vd. ``ton_mlp_t.pth``) để tương thích ``load_net`` / web app.

Chạy:
  python reimplemented_models/train_all_standard_models.py
  python reimplemented_models/train_all_standard_models.py --datasets ton --epochs 15

Song song nhiều GPU (mỗi process một GPU, cùng thư mục output — an toàn vì mỗi job ghi file khác tên):
  ./reimplemented_models/run_train_standard_parallel.sh 0 1 2 3
  # hoặc tay: CUDA_VISIBLE_DEVICES=2 python -u ... --shard-index 2 --shard-count 4
"""

from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import torch
from torch.utils.data import DataLoader

from utils.utils import CustomDataset, init_net, STORAGE_DIR

import train_jobs

ARCHITECTURES = ("mlp", "cnn", "rescnn", "lstm", "Selfattention")
DATASETS = ("ton", "ids18")
MODEL_TYPES = ("t", "s")


def train_one(
    dataset_name: str,
    arch: str,
    model_type: str,
    fp_output: str,
    dev: torch.device,
    lr: float,
    epochs: int,
    batch_size: int,
) -> str:
    fp_fea = os.path.join(STORAGE_DIR, "dataset", f"fea_{model_type}.csv")
    fp_minmax = os.path.join(STORAGE_DIR, "dataset", f"{dataset_name}_minmax_{model_type}.csv")
    fp_dataset = os.path.join(STORAGE_DIR, "dataset", f"{dataset_name}_sam_train_{model_type}.csv")

    for p in (fp_fea, fp_minmax, fp_dataset):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Thiếu file dữ liệu: {p}")

    os.makedirs(fp_output, exist_ok=True)
    model_key = f"{arch}_{model_type}"
    tag = f"{dataset_name}_{arch}_{model_type}"
    print(f">>> [{tag}] Khởi tạo mạng + đọc CSV (bước này có thể vài phút) ...", flush=True)
    net = init_net(model_type, model_key)
    if net is None:
        raise RuntimeError(f"init_net không khởi tạo được: type={model_type}, name={model_key}")

    dataset = CustomDataset(fp_dataset, fp_minmax, fp_fea)
    print(f">>> [{tag}] Đã nạp {len(dataset)} mẫu — train {epochs} epoch.", flush=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    net.to(dev)
    criterion = CrossEntropyLoss()
    optimizer = Adam(net.parameters(), lr=lr, betas=(0.99, 0.99))

    for ep in range(epochs):
        net.train()
        seen = 0
        last_loss = 0.0
        for data, labels in dataloader:
            seen += len(data)
            data, labels = data.to(dev), labels.to(dev)
            optimizer.zero_grad()
            loss = criterion(net(data), labels)
            loss.backward()
            optimizer.step()
            last_loss = float(loss)
            print(
                f"\r[{tag}] Epoch {ep + 1}/{epochs}  {seen}/{len(dataset)}  loss={last_loss:.4f}",
                end="   ",
                flush=True,
            )
        print(f"\n[{tag}] Xong epoch {ep + 1}/{epochs}, loss_cuối_batch={last_loss:.4f}", flush=True)

    out_path = os.path.join(fp_output, f"{dataset_name}_{arch}_{model_type}.pth")
    torch.save(net.state_dict(), out_path)
    return out_path


def main() -> None:
    default_out = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Train all standard target/surrogate NIDS into reimplemented_models/")
    parser.add_argument(
        "--output-dir",
        default=default_out,
        help="Thư mục lưu .pth (mặc định: cùng thư mục với script)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS),
        choices=DATASETS,
        help="Tập dữ liệu (mặc định: ton ids18)",
    )
    parser.add_argument(
        "--architectures",
        nargs="+",
        default=list(ARCHITECTURES),
        choices=ARCHITECTURES,
        help="Kiến trúc (mặc định: cả 5)",
    )
    parser.add_argument(
        "--model-types",
        nargs="+",
        default=list(MODEL_TYPES),
        choices=MODEL_TYPES,
        help="t=target, s=surrogate",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cuda", "cpu"),
        help="auto: CUDA nếu có, ngược lại CPU",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Chỉ số shard 0..shard_count-1 (dùng khi chia job lên nhiều GPU)",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Số process song song; 1 = chạy hết job trên một process (mặc định)",
    )
    args = parser.parse_args()

    if args.device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(args.device)

    print(f"STORAGE_DIR={STORAGE_DIR}", flush=True)
    print(f"Output: {os.path.abspath(args.output_dir)}  device={dev}", flush=True)

    jobs = train_jobs.sharded_jobs(
        list(args.datasets),
        list(args.architectures),
        list(args.model_types),
        args.shard_index,
        args.shard_count,
    )
    print(
        f"Shard {args.shard_index}/{args.shard_count}: {len(jobs)} job(s) trong lô này.",
        flush=True,
    )

    done = []
    for ds, arch, mt in jobs:
        path = train_one(
            ds,
            arch,
            mt,
            args.output_dir,
            dev,
            args.lr,
            args.epochs,
            args.batch_size,
        )
        done.append(path)
        print(f"Saved: {path}", flush=True)

    print(f"\nHoàn tất {len(done)} checkpoint (shard này).", flush=True)


if __name__ == "__main__":
    main()
