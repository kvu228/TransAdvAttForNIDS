"""Mục 5.3.1 — Table 8.

Ma trận Recall attack của **target** (huấn luyện sạch, 60 fea) trên:
    - Adversarial sinh từ từng surrogate ``*_s`` + từng attack (MIFGSM, SIM, VMIFGSM, DGM),
      file cố định ``7_140.csv`` (iteration=7, step=140 — cấu hình chuẩn thí nghiệm transfer).
    - Baseline ``raw_att_traffic`` (không qua surrogate): ``{dsn}_raw_att.csv``.

Hyperparameter: ``batch_size=128``; ``load_net(60, ...)`` khớp target 60 đặc trưng.
In ra bảng ghép IDS18/TON (chuỗi ``x/y``) và cột trung bình theo target.
"""
import os, sys
project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root_dir)
import pandas as pd
from utils.utils import CustomDataset, load_net, STORAGE_DIR
import torch
from torch.utils.data import DataLoader

def main(dsn, mn_s, an, mn_t, fp_dataset, fp_minmax, fp_fea, fp_model):
    """Recall TP/(TP+FN) của ``mn_t`` trên ``fp_dataset`` (adv hoặc raw)."""
    # hyper
    dev = torch.device('cuda')
    batch_size = 128
    dataset = CustomDataset(fp_dataset, fp_minmax, fp_fea)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    model_name = f'{dsn}_{mn_t}'
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
        print(f"\r Dataset:{dsn}, Sur_model:{mn_s}, Att:{an}, Tar:{mn_t}, Progress:{curr_iter}/{len(dataset)} Acc: {acc:.3f}", end="")
    print()
    return acc

if __name__ == '__main__':
    dataset_names = ['ids18', 'ton']
    model_names_t = ['mlp_t', 'cnn_t', 'rescnn_t', 'lstm_t', 'Selfattention_t']
    model_names_s = ['mlp_s', 'cnn_s', 'rescnn_s', 'lstm_s', 'Selfattention_s']
    attack_names = ['MIFGSM', 'SIM', 'VMIFGSM', 'DGM']

    lst_idx = []
    for mn in model_names_s:
        for an in attack_names:
            lst_idx.append((mn, an))
    lst_idx.append(('None', 'raw_att_traffic'))
    midx = pd.MultiIndex.from_tuples(lst_idx, names=['Model', 'Attack'])

    lst_col = []
    for mn in model_names_t:
        lst_col.append(f'{mn}(IDS18/TON)')

    res = []
    for dsn in dataset_names:
        df = pd.DataFrame([[0.] * len(model_names_t)] * len(lst_idx), index=midx, columns=lst_col)
        print(df)

        for mn_s, an in lst_idx:
            if mn_s != 'None':
                fp_dataset = os.path.join(STORAGE_DIR, 'AAT', f'{dsn}_{mn_s}', an, '7_140.csv')
            else:
                fp_dataset = os.path.join(STORAGE_DIR, 'dataset', f'{dsn}_raw_att.csv')
            fp_minmax = os.path.join(STORAGE_DIR, 'dataset', f'{dsn}_minmax_t.csv')
            fp_fea = os.path.join(STORAGE_DIR, 'dataset', 'fea_t.csv')

            for mn_t in model_names_t:
                fp_model = os.path.join(STORAGE_DIR, 'pre-trained_models', 'normal_train', f'{dsn}_{mn_t}.pth')
                acc = main(dsn, mn_s, an, mn_t, fp_dataset, fp_minmax, fp_fea, fp_model)
                df.loc[(mn_s, an), f'{mn_t}(IDS18/TON)'] = round(acc * 100, 1)
            print(df)
        df['Average(IDS18/TON)'] = df.mean(axis=1).round(1)
        print(df)
        res.append(df)
    
    val = res[0].values.astype(str) + '/' + res[1].values.astype(str)

    df_res = pd.DataFrame(val, columns=df.columns, index=df.index)
    print(df_res)
