"""Mục 5.4.4 — Figure 9 (TON_IoT).

Cùng thiết kế ``5_4_4-Figure_8.py`` với ``dsn='ton'``: trung bình 3 experiment_no,
đường cong Recall theo số đặc trưng surrogate cho từng attack × target.

Đầu ra: ``output/figures/fig9.png``.
"""
import os, sys
project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root_dir)
import pandas as pd
from utils.utils import CustomDataset, load_net, STORAGE_DIR
from utils.plot_line2 import plot_line2
import torch
from torch.utils.data import DataLoader

def main(dsn, mnt, an, fea_num, fp_dataset, fp_minmax, fp_fea, fp_model):
    """Recall trên adv theo ``fea_num``; xem module docstring Figure_8."""

    dev = torch.device('cuda')
    batch_size = 128
    dataset = CustomDataset(fp_dataset, fp_minmax, fp_fea)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    model_name = f'{dsn}_{mnt}'
    net = load_net(60, model_name, fp_model)
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
        print(f"\r Dataset:{dsn}, Tar_model:{mnt}, Att:{an}, Num of input fea of sur model:{fea_num}, Progress:{curr_iter}/{len(dataset)} Acc: {acc:.3f}", end="")
    print()
    return acc

if __name__ == '__main__':
    dsn = 'ton'
    experiment_no = [0, 1, 2]
    model_names_t = ['mlp_t', 'cnn_t', 'rescnn_t', 'lstm_t', 'Selfattention_t']
    attack_names = ['MIFGSM', 'SIM', 'VMIFGSM', 'DGM']
    fea_nums = [i for i in range(27, 67, 3)]

    lst_idx = []
    for mnt in model_names_t:
        for an in attack_names:
            lst_idx.append((mnt, an))
    midx = pd.MultiIndex.from_tuples(lst_idx, names=['Target Model', 'Attack'])

    res = []
    for exno in experiment_no:
        df = pd.DataFrame([[0.] * len(fea_nums)] * len(lst_idx), index=midx, columns=fea_nums)
        print(df)

        for mnt, an in lst_idx:
            for fea_num in fea_nums:

                fp_dataset = os.path.join(STORAGE_DIR, '5_4_4', f'{exno}_{dsn}_adv_flow', f'{an}_{fea_num}.csv')
                fp_minmax = os.path.join(STORAGE_DIR, 'dataset', f'{dsn}_minmax_t.csv')
                fp_fea = os.path.join(STORAGE_DIR, 'dataset', 'fea_t.csv')
                fp_model = os.path.join(STORAGE_DIR, 'pre-trained_models', 'normal_train', f'{dsn}_{mnt}.pth')
                acc = main(dsn, mnt, an, fea_num, fp_dataset, fp_minmax, fp_fea, fp_model)
                df.loc[(mnt, an), fea_num] = round(acc * 100, 1)
            print(df)
        res.append(df)

    val = (res[0].values + res[1].values + res[2].values) / 3
    df_res = pd.DataFrame(val, index=df.index, columns=df.columns).round(1)
    print(df_res)
    fp_fig = os.path.join(project_root_dir, 'output', 'figures', 'fig9.png')
    plot_line2(df_res, fp_fig)
