"""Mục 5.2 — Table 5.

Đánh giá **tập test sạch** (không adversarial) cho IDS2018 và TON_IoT: Acc, Precision,
Recall, F1 theo số đặc trưng và loại mô hình (target 78/66 fea, surrogate 60 fea).

Hyperparameter / thiết kế:
    - ``fea_nums = [78, 66, 60]``: ba cấu hình đặc trưng trong paper; 78 dùng thư mục
      ``normal_train_with_78_fea`` và file ``fea_78`` / minmax riêng (``5_2``).
    - ``batch_size=128``: đồng nhất các script.
    - Kết quả in ra: cột dạng ``Acc(IDS18/TON)`` ghép hai dataset sau khi chạy xong cả hai.

``main`` dùng đầy đủ TP/FP/TN/FN (khác các script chỉ báo Recall trên attack).
"""
import os, sys
project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root_dir)
import pandas as pd
from utils.utils import CustomDataset, load_net, STORAGE_DIR
import torch
from torch.utils.data import DataLoader
import math

def main(dsn, fea_num, mn, fp_dataset, dir_model, fp_fea, fp_minmax):
    """Đánh giá một checkpoint trên ``fp_dataset``.

    Args:
        dsn: Tên dataset (tiền tố model).
        fea_num: Chiều vào của mạng (78, 66 hoặc 60).
        mn: Tên model (``*_t`` hoặc ``*_s``).
        fp_dataset: CSV test (``{dsn}_test_{t|s}.csv``).
        dir_model: Thư mục chứa ``{dsn}_{mn}.pth``.
        fp_fea, fp_minmax: Danh sách cột và min-max tương ứng ``fea_num``.

    Returns:
        tuple: (acc, pre, rec, f1) dạng float [0,1].
    """
    dev = torch.device('cuda')
    batch_size = 128

    dataset = CustomDataset(fp_dataset, fp_minmax, fp_fea)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    model_name = f'{dsn}_{mn}'
    fp_model = os.path.join(dir_model, f'{model_name}.pth')
    net = load_net(fea_num, model_name, fp_model)
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

        print(f"\r {fea_num}, {model_name}, Progress:{curr_iter}/{len(dataset)}, TP|FP|TN|FN:{TP}|{FP}|{TN}|{FN}, Acc: {acc:.3f}, Pre: {pre:.3f}, Rec: {rec:.3f}, F1: {f1:.3f}", end="")
    print()
    return acc, pre, rec, f1

if __name__ == '__main__':
    dataset_names = ['ids18', 'ton']
    fea_nums = [78, 66, 60]
    lst_performance = ['Acc(IDS18/TON)', 'Pre(IDS18/TON)','Rec(IDS18/TON)','F1(IDS18/TON)',]

    model_names_t = ['mlp_t', 'cnn_t', 'rescnn_t', 'lstm_t', 'Selfattention_t']
    model_names_s = ['mlp_s', 'cnn_s', 'rescnn_s', 'lstm_s', 'Selfattention_s']
    model_types = ['t', 's']

    lst_idx = []
    for fea_num in fea_nums:
        if fea_num !=60 :
            for mn in model_names_t:
                lst_idx.append((fea_num, mn))
        else:
            for mn in model_names_s:
                lst_idx.append((fea_num, mn))

    midx = pd.MultiIndex.from_tuples(lst_idx, names=['Features number', 'Model'])

    res = []
    for dsn in dataset_names:

        df = pd.DataFrame([[0.] * len(lst_performance)] * len(lst_idx), columns=lst_performance, index=midx)
        print(df)
        
        for fea_num, mn in lst_idx:
            ts = mn[-1]
            fp_dataset = os.path.join(STORAGE_DIR, 'dataset', f'{dsn}_test_{ts}.csv')

            if fea_num == 78:
                dir_model = os.path.join(STORAGE_DIR, 'pre-trained_models', 'normal_train_with_78_fea')
                fp_fea = os.path.join(STORAGE_DIR, '5_2', 'fea_78.csv')
                fp_minmax = os.path.join(STORAGE_DIR, '5_2', f'{dsn}_minmax_{ts}_78.csv')
            else:
                dir_model = os.path.join(STORAGE_DIR, 'pre-trained_models', 'normal_train')
                fp_fea = os.path.join(STORAGE_DIR, 'dataset', f'fea_{ts}.csv')
                fp_minmax = os.path.join(STORAGE_DIR, 'dataset', f'{dsn}_minmax_{ts}.csv')

            acc, pre, rec, f1 = main(dsn, fea_num, mn, fp_dataset, dir_model, fp_fea, fp_minmax)
            df.loc[(fea_num, mn), 'Acc(IDS18/TON)'] = round(acc * 100, 1)
            df.loc[(fea_num, mn), 'Pre(IDS18/TON)'] = round(pre * 100, 1) if round(pre * 100, 1) != 100 else math.floor(pre * 1000) / 10
            df.loc[(fea_num, mn), 'Rec(IDS18/TON)'] = round(rec * 100, 1)
            df.loc[(fea_num, mn), 'F1(IDS18/TON)'] = round(f1 * 100, 1)
            print(df)
    
        res.append(df)
    
    val = res[0].values.astype(str) + '/' + res[1].values.astype(str)

    df_res = pd.DataFrame(val, columns=df.columns, index=df.index)
    print(df_res)


