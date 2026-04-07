"""Mục 5.3.2 — Table 9 (TON_IoT, baseline TANTRA).

So sánh transfer attack lên target TON: surrogate × attack, thêm ``TANTRA`` và ``raw_att_traffic``
(không surrogate). Chạy ``repeat_nums = [0,1,2]`` rồi **trung bình** để ổn định kết quả.

Hyperparameter:
    - ``load_net(66, 'ton_{mn_t}', ...)``: target TON dùng 66 đặc trưng trong thí nghiệm này.
    - ``batch_size=128``.

Đường dẫn adv: ``5_3_2/adv_flow/`` theo quy ước trong vòng lặp.
"""
import os, sys
project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root_dir)
import pandas as pd
from utils.utils import CustomDataset, load_net, STORAGE_DIR
import torch
from torch.utils.data import DataLoader

def main(mn_s, an, mn_t, fp_dataset, fp_minmax, fp_fea, fp_model):
    """Recall trên một file adv/raw; model luôn checkpoint ``ton_{mn_t}``."""
    dev = torch.device('cuda')
    batch_size = 128
    dataset = CustomDataset(fp_dataset, fp_minmax, fp_fea)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    model_name = f'ton_{mn_t}'
    net = load_net(66, model_name, fp_model)
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
        print(f"\rSur_model:{mn_s}, Att:{an}, Tar:{mn_t}, Progress:{curr_iter}/{len(dataset)} Acc: {acc:.3f}", end="")
    print()
    return acc

if __name__ == '__main__':
    model_names_t = ['mlp_t', 'cnn_t', 'rescnn_t', 'lstm_t', 'Selfattention_t']
    model_names_s = ['mlp_s', 'cnn_s', 'rescnn_s', 'lstm_s', 'Selfattention_s']
    attack_names = ['MIFGSM', 'SIM', 'VMIFGSM', 'DGM']
    repeat_nums = [0, 1, 2]

    lst_idx = []
    for mn in model_names_s:
        for an in attack_names:
            lst_idx.append((mn, an))
    lst_idx.append(('None', 'TANTRA'))
    lst_idx.append(('None', 'raw_att_traffic'))

    midx = pd.MultiIndex.from_tuples(lst_idx, names=['Model', 'Attack'])

    res = []
    for rn in repeat_nums:
        df = pd.DataFrame([[0.] * len(model_names_t)] * len(lst_idx), index=midx, columns=model_names_t)
        print(df)

        for mn_s, an in lst_idx:
            if mn_s == 'None':
                fp_dataset = os.path.join(STORAGE_DIR, '5_3_2', 'adv_flow', f'{rn}_{mn_s}_{an}.csv')
            else:
                fp_dataset = os.path.join(STORAGE_DIR, '5_3_2', 'adv_flow', f'{rn}_{mn_s[:-2]}_{an}.csv')
            fp_minmax = os.path.join(STORAGE_DIR, 'dataset', f'ton_minmax_t.csv')
            fp_fea = os.path.join(STORAGE_DIR, 'dataset', 'fea_t.csv')

            for mn_t in model_names_t:
                fp_model = os.path.join(STORAGE_DIR, 'pre-trained_models', 'normal_train', f'ton_{mn_t}.pth')
                acc = main(mn_s, an, mn_t, fp_dataset, fp_minmax, fp_fea, fp_model)
                df.loc[(mn_s, an), mn_t] = round(acc * 100, 1)
                print(df)
        
        df['Average'] = df.mean(axis=1).round(1)
        print(df)
        res.append(df)
    
    val = (res[0].values + res[1].values + res[2].values) / 3
    df_res = pd.DataFrame(val, columns=df.columns, index=df.index).round(1)
    print(df_res)

