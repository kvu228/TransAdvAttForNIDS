#!/usr/bin/env python3
"""
Lặp toàn bộ cấu hình giống ``train_NIDS/adv_training_with_SPTS.py`` (MI-FGSM + mask SPTS).

Tên file lưu: ``advtrain_withSPTS_{dataset}_{arch}_{t|s}.pth`` (thêm hậu tố _t/_s so với script gốc).

Chạy: python reimplemented_models/train_all_adv_with_spts.py
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.MIFGSM_forAdvTrain import MIFGSM_forAdvTrain
from utils.utils import init_net, STORAGE_DIR

import train_jobs

ARCHITECTURES = ("mlp", "cnn", "rescnn", "lstm", "Selfattention")
DATASETS = ("ton", "ids18")
MODEL_TYPES = ("t", "s")


def get_mask(list_col: list, batch_size: int) -> torch.Tensor:
    df_temp = pd.DataFrame([[0.0] * len(list_col)], columns=list_col)
    df_temp["Fwd Pkt Len Max"] = 1.0
    df_temp["Fwd Pkt Len Min"] = 1.0
    df_temp["Fwd IAT Max"] = 1.0
    df_temp["Fwd IAT Min"] = 1.0
    return torch.from_numpy(df_temp.loc[0].values).repeat(batch_size, 1)


def train_one_spts(
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

    tag = f"{dataset_name}_{arch}_{model_type}_SPTS"
    print(f">>> [{tag}] Đọc CSV {fp_dataset} (có thể lâu) ...", flush=True)
    df_raw_data = pd.read_csv(fp_dataset, header=0, index_col=None)
    df_raw_data["label"] = 1
    df_raw_data.loc[df_raw_data["Label"] == "Benign", "label"] = 0
    list_fea = pd.read_csv(fp_fea, header=0, index_col=None).columns.tolist()
    df_minmax = pd.read_csv(fp_minmax, header=0, index_col=None)

    model_key = f"{arch}_{model_type}"
    net = init_net(model_type, model_key)
    if net is None:
        raise RuntimeError(f"init_net failed: {model_key}")
    net.to(dev)
    criterion = CrossEntropyLoss()
    optimizer = Adam(net.parameters(), lr=lr, betas=(0.99, 0.99))

    os.makedirs(fp_output, exist_ok=True)
    print(f">>> [{tag}] {len(df_raw_data)} dòng — adv train {epochs} epoch (mỗi epoch rất chậm do MI-FGSM).", flush=True)

    net.train()
    last_loss = 0.0
    for ep in range(epochs):
        df_raw_data = df_raw_data.sample(frac=1.0, replace=False).reset_index(drop=True)
        pos_row = 0
        while pos_row < len(df_raw_data):
            df_part_flow = df_raw_data.iloc[pos_row : pos_row + batch_size]
            pos_row += len(df_part_flow)
            mask = get_mask(list_fea, len(df_part_flow)).to(dev)

            labels = torch.from_numpy(df_part_flow["label"].values).to(dev)
            df_part_flow_66fea = df_part_flow[list_fea]
            df_part_flow_66fea = (
                (df_part_flow_66fea - df_minmax.loc[0]) / (df_minmax.loc[1] - df_minmax.loc[0])
            ).fillna(0)
            tensor1 = torch.from_numpy(df_part_flow_66fea.values).float().to(dev)

            df_adv_flow = MIFGSM_forAdvTrain(
                net,
                criterion,
                df_part_flow[list_fea],
                labels,
                mask,
                7,
                140,
                dev,
                df_minmax.loc[0],
                df_minmax.loc[1],
            )
            df_adv_flow = (
                (df_adv_flow - df_minmax.loc[0]) / (df_minmax.loc[1] - df_minmax.loc[0])
            ).fillna(0).replace([np.inf, -np.inf], [1.0, -1.0])
            pos_benign = df_part_flow["label"] == 0
            df_adv_flow.loc[pos_benign] = df_part_flow_66fea.loc[pos_benign]
            tensor2 = torch.clamp(torch.from_numpy(df_adv_flow.values).float().to(dev), 0.0, 1.0)

            optimizer.zero_grad()
            loss = k * criterion(net(tensor1), labels) + k * criterion(net(tensor2), labels)
            loss.backward()
            optimizer.step()
            last_loss = float(loss)
            print(
                f"\r[{tag}] Epoch {ep + 1}/{epochs} row {pos_row}/{len(df_raw_data)} loss={last_loss:.4f}",
                end="   ",
                flush=True,
            )
        print(f"\n[{tag}] Xong epoch {ep + 1}/{epochs}, loss_cuối_batch={last_loss:.4f}", flush=True)

    out = os.path.join(fp_output, f"advtrain_withSPTS_{dataset_name}_{arch}_{model_type}.pth")
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
    ap.add_argument("--k", type=float, default=0.8, help="Hệ số loss clean/adv (như script gốc)")
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
        f"Shard {args.shard_index}/{args.shard_count}: {len(jobs)} job(s) (SPTS).",
        flush=True,
    )

    saved = []
    for ds, arch, mt in jobs:
        p = train_one_spts(ds, arch, mt, args.output_dir, dev, args.k, args.lr, args.epochs, args.batch_size)
        saved.append(p)
        print(f"Saved: {p}", flush=True)
    print(f"Done {len(saved)} checkpoints (shard này).", flush=True)


if __name__ == "__main__":
    main()
