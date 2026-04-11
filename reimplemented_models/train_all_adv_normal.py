#!/usr/bin/env python3
"""
Lặp toàn bộ cấu hình giống ``train_NIDS/normal_adv_training.py`` (MI-FGSM không SPTS).

Tên file: ``normal_advtrain_{dataset}_{arch}_{t|s}.pth``

Chạy: python reimplemented_models/train_all_adv_normal.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.MIFGSM_noSPTS import MIFGSM_noSPTS
from utils.utils import CustomDataset, init_net, STORAGE_DIR

import train_jobs

ARCHITECTURES = ("mlp", "cnn", "rescnn", "lstm", "Selfattention")
DATASETS = ("ton", "ids18")
MODEL_TYPES = ("t", "s")


def get_step_len(fp_minmax: str) -> torch.Tensor:
    df_minmax = pd.read_csv(fp_minmax, header=0, index_col=None)
    step_len = 140 / (df_minmax.loc[1] - df_minmax.loc[0])
    step_len = step_len.replace([np.inf, -np.inf], 0)
    step_len.loc[step_len >= 1.0] = 0.0
    return torch.from_numpy(step_len.values).float()


def train_one_normal(
    dataset_name: str,
    arch: str,
    model_type: str,
    fp_output: str,
    dev: torch.device,
    k: float,
    lr: float,
    epochs: int,
    batch_size: int,
) -> str:
    fp_fea = os.path.join(STORAGE_DIR, "dataset", f"fea_{model_type}.csv")
    fp_minmax = os.path.join(STORAGE_DIR, "dataset", f"{dataset_name}_minmax_{model_type}.csv")
    fp_dataset = os.path.join(STORAGE_DIR, "dataset", f"{dataset_name}_sam_train_{model_type}.csv")
    for p in (fp_fea, fp_minmax, fp_dataset):
        if not os.path.isfile(p):
            raise FileNotFoundError(p)

    tag = f"{dataset_name}_{arch}_{model_type}_normalAdv"
    print(f">>> [{tag}] Đọc CSV + CustomDataset (có thể vài phút) ...", flush=True)
    dataset = CustomDataset(fp_dataset, fp_minmax, fp_fea)
    print(f">>> [{tag}] {len(dataset)} mẫu — normal adv train {epochs} epoch.", flush=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    step_len = get_step_len(fp_minmax).to(dev)

    model_key = f"{arch}_{model_type}"
    net = init_net(model_type, model_key)
    if net is None:
        raise RuntimeError(f"init_net failed: {model_key}")
    net.to(dev)
    criterion = CrossEntropyLoss()
    optimizer = Adam(net.parameters(), lr=lr, betas=(0.99, 0.99))

    os.makedirs(fp_output, exist_ok=True)

    net.train()
    last_loss = 0.0
    for ep in range(epochs):
        seen = 0
        for data, labels in dataloader:
            seen += len(data)
            data, labels = data.to(dev), labels.to(dev)
            adv_data = MIFGSM_noSPTS(net, data.detach().clone(), labels, criterion, 7, step_len, dev)
            optimizer.zero_grad()
            loss = k * criterion(net(data), labels) + (1 - k) * criterion(net(adv_data), labels)
            loss.backward()
            optimizer.step()
            last_loss = float(loss)
            print(
                f"\r[{tag}] Epoch {ep + 1}/{epochs} {seen}/{len(dataset)} loss={last_loss:.4f}",
                end="   ",
                flush=True,
            )
        print(f"\n[{tag}] Xong epoch {ep + 1}/{epochs}, loss_cuối_batch={last_loss:.4f}", flush=True)

    out = os.path.join(fp_output, f"normal_advtrain_{dataset_name}_{arch}_{model_type}.pth")
    torch.save(net.state_dict(), out)
    return out


def main() -> None:
    default_out = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=default_out)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    ap.add_argument("--architectures", nargs="+", default=list(ARCHITECTURES), choices=ARCHITECTURES)
    ap.add_argument("--model-types", nargs="+", default=list(MODEL_TYPES), choices=MODEL_TYPES)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--k", type=float, default=0.9, help="Trọng số loss trên clean (như script gốc)")
    ap.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    args = ap.parse_args()

    if args.device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(args.device)

    jobs = train_jobs.sharded_jobs(
        list(args.datasets),
        list(args.architectures),
        list(args.model_types),
        args.shard_index,
        args.shard_count,
    )
    print(
        f"Shard {args.shard_index}/{args.shard_count}: {len(jobs)} job(s) (normal adv).",
        flush=True,
    )

    saved = []
    for ds, arch, mt in jobs:
        p = train_one_normal(ds, arch, mt, args.output_dir, dev, args.k, args.lr, args.epochs, args.batch_size)
        saved.append(p)
        print(f"Saved: {p}", flush=True)
    print(f"Done {len(saved)} checkpoints (shard này).", flush=True)


if __name__ == "__main__":
    main()
