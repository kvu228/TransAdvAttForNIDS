"""Mục 5.4.3 — Table 10.

Huấn luyện / đánh giá mô hình trên **tập test có trộn tỷ lệ** adversarial (``ratios`` 0.1–0.4):
mỗi cặp (model, ratio) dùng CSV ``5_4_3/dataset/{dsn}_raw_test_dataset_{t|s}_{ra}.csv``,
minmax ``{dsn}_minmax_{t|s}_{ra}``, checkpoint ``5_4_3/pre-trained_models/{dsn}_{mn}_{ra}.pth``.

Metrics: Acc, Pre, Rec, F1 (đầy đủ ma trận nhầm lẫn). ``mlp_s`` nằm cùng lưới với ``*_t``.

Hyperparameter: ``batch_size=128``; ``load_net(66, ...)`` cho chiều vào 66 trong setup này.
"""
import os, sys
project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root_dir)
import pandas as pd
from utils.utils import CustomDataset, load_net, STORAGE_DIR
import torch
from torch.utils.data import DataLoader
import math

def main(dsn, mn, ra, fp_dataset, fp_fea, fp_minmax, fp_model):
    """Đánh giá một checkpoint tại tỷ lệ trộn ``ra``; trả về (acc, pre, rec, f1)."""
    dev = torch.device('cuda')
    batch_size = 128

    dataset = CustomDataset(fp_dataset, fp_minmax, fp_fea)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    model_name = f'{dsn}_{mn}'
    net = load_net(66, model_name, fp_model)
    net.to(dev)

    net.eval()
    TP, FP, TN, FN, curr_iter = 0, 0, 0, 0, 0
    acc, pre, rec, f1 = None, None, None, None
    for flows, labels in dataloader:
        flows, labels = flows.to(dev), labels.to(dev)
        curr_iter += len(labels)

        with torch.no_grad():
            pred = net(flows).argmax(1)
        TP += ((pred == 1) & (labels == 1)).sum().item()
        FP += ((pred == 1) & (labels == 0)).sum().item()
        TN += ((pred == 0) & (labels == 0)).sum().item()
        FN += ((pred == 0) & (labels == 1)).sum().item()

        acc = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) !=0 else 0
        pre = TP / (TP + FP) if (TP + FP) != 0 else 0
        rec = TP / (TP + FN) if (TP + FN) != 0 else 0
        f1 = 2 * (pre * rec) / (pre + rec) if (pre + rec) != 0 else 0

        print(f"\r{model_name}, ratio:{ra}, Progress:{curr_iter}/{len(dataset)}, TP|FP|TN|FN:{TP}|{FP}|{TN}|{FN}, Acc: {acc:.3f}, Pre: {pre:.3f}, Rec: {rec:.3f}, F1: {f1:.3f}", end="")
    print()
    return acc, pre, rec, f1

if __name__ == '__main__':
    dataset_names = ['ids18', 'ton']
    lst_performance = ['Acc(IDS18/TON)', 'Pre(IDS18/TON)','Rec(IDS18/TON)','F1(IDS18/TON)',]
    ratios = [0.1, 0.2, 0.3, 0.4]

    model_names = ['mlp_t', 'cnn_t', 'rescnn_t', 'lstm_t', 'Selfattention_t', 'mlp_s']

    lst_idx = []
    for mn in model_names:
        for ra in ratios:
            lst_idx.append((mn, ra))

    midx = pd.MultiIndex.from_tuples(lst_idx, names=['Model', 'Ratio'])
    

    res = []
    for dsn in dataset_names:
        df = pd.DataFrame([[0.] * len(lst_performance)] * len(lst_idx), columns=lst_performance, index=midx)
        print(df)
        

        for mn, ra in lst_idx:
            ts = mn[-1]
            fp_dataset = os.path.join(STORAGE_DIR, '5_4_3', 'dataset', f'{dsn}_raw_test_dataset_{ts}_{ra}.csv')

            fp_fea = os.path.join(STORAGE_DIR, 'dataset', f'fea_{ts}.csv')
            fp_minmax = os.path.join(STORAGE_DIR, '5_4_3', 'dataset', f'{dsn}_minmax_{ts}_{ra}.csv')

            fp_model = os.path.join(STORAGE_DIR, '5_4_3', 'pre-trained_models', f'{dsn}_{mn}_{ra}.pth')

            acc, pre, rec, f1 = main(dsn, mn, ra, fp_dataset, fp_fea, fp_minmax, fp_model)
            df.loc[(mn, ra), 'Acc(IDS18/TON)'] = round(acc * 100, 1)
            df.loc[(mn, ra), 'Pre(IDS18/TON)'] = round(pre * 100, 1) if round(pre * 100, 1) != 100 else math.floor(pre * 1000) / 10
            df.loc[(mn, ra), 'Rec(IDS18/TON)'] = round(rec * 100, 1)
            df.loc[(mn, ra), 'F1(IDS18/TON)'] = round(f1 * 100, 1)
            print(df)
    
        res.append(df)
    
    val = res[0].values.astype(str) + '/' + res[1].values.astype(str)

    df_res = pd.DataFrame(val, columns=df.columns, index=df.index)
    print(df_res)


