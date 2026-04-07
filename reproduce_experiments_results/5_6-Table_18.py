"""Mục 5.6 — Table 18 (hardening: adv_train_with_SPTS).

Giống Table 17 nhưng checkpoint lấy từ ``pre-trained_models/adv_train_with_SPTS`` —
huấn luyện adversarial kèm **SPTS** (theo paper). Cùng dữ liệu đánh giá AAT và ``load_net(66)``.
"""
import os, sys
project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root_dir)
import pandas as pd
from utils.utils import CustomDataset, load_net, STORAGE_DIR
import torch
from torch.utils.data import DataLoader
import math

def main(adv_train_type, dsn, an, mnt, fp_dataset, fp_minmax, fp_fea, fp_model):
    """Recall trên adv; xem Table_17."""
    dev = torch.device('cuda')
    batch_size = 128
    dataset = CustomDataset(fp_dataset, fp_minmax, fp_fea)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    net = load_net(66, f'{dsn}_{mnt}', fp_model)
    net.to(dev)

    net.eval()
    TP, FP, TN, FN, curr_iter = 0, 0, 0, 0, 0
    for flows, labels in dataloader:
        flows, labels = flows.to(dev), labels.to(dev)
        curr_iter += len(labels)

        with torch.no_grad():
            pred = net(flows).argmax(1)

        TP += ((pred == 1) & (labels == 1)).sum().item()
        FN += ((pred == 0) & (labels == 1)).sum().item()
        acc = TP / (TP + FN)
        print(f"\r {adv_train_type}, Sur_model:mlp_s, Tar_model:{mnt}, attack:{an}, Progress:{curr_iter}/{len(dataset)}, Acc: {acc:.3f}", end="")
    print()
    return acc

if __name__ == '__main__':
    adv_train_type = 'adv_train_with_SPTS'
    dataset_names = ['ids18', 'ton']
    model_names_t = ['mlp_t', 'cnn_t', 'rescnn_t', 'lstm_t', 'Selfattention_t']
    attack_names = ['MIFGSM', 'SIM', 'VMIFGSM', 'DGM']

    lst_col = []
    for mnt in model_names_t:
        lst_col.append(f'{mnt}(IDS18/TON)')

    res = []
    for dsn in dataset_names:
        df = pd.DataFrame([[0.] * len(model_names_t)] * len(attack_names), index=attack_names, columns=lst_col)
        print(df)

        for an in attack_names:
            for mnt in model_names_t:
                fp_dataset = os.path.join(STORAGE_DIR, 'AAT', f'{dsn}_mlp_s', an, '7_140.csv')
                fp_minmax = os.path.join(STORAGE_DIR, 'dataset', f'{dsn}_minmax_t.csv')
                fp_fea = os.path.join(STORAGE_DIR, 'dataset', 'fea_t.csv')
                fp_model = os.path.join(STORAGE_DIR, 'pre-trained_models', adv_train_type, f'{dsn}_{mnt}.pth')
                acc = main(adv_train_type, dsn, an, mnt, fp_dataset, fp_minmax, fp_fea, fp_model)
                df.loc[an, f'{mnt}(IDS18/TON)'] = round(acc * 100, 1) if round(acc * 100, 1) != 100 else math.floor(acc * 1000) / 10
            print(df)
        df['Average(IDS18/TON)'] = df.mean(axis=1).round(1)
        print(df)
        res.append(df)

    val = res[0].values.astype(str) + '/' + res[1].values.astype(str)
    df_res = pd.DataFrame(val, columns=df.columns, index=df.index)
    print(df_res)
