"""Mục 5.4.3 — Table 14.

Cùng thiết kế Table 11 với **``ratio_t=0.4``**."""
import os, sys
project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root_dir)
import pandas as pd
from utils.utils import CustomDataset, load_net, STORAGE_DIR
import torch
from torch.utils.data import DataLoader

def main(ratio_t, dsn, ras, an, mn_t, fp_dataset, fp_minmax, fp_fea, fp_model):
    """Recall target; xem Table_11."""
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
        print(f"\r Dataset:{dsn}, ratio_t:{ratio_t}, Sur_model:MLP, ratio_s:{ras}, Att:{an}, Tar:{mn_t}, Progress:{curr_iter}/{len(dataset)} Acc: {acc:.3f}", end="")
    print()
    return acc

if __name__ == '__main__':
    ratio_t = 0.4
    ratios_s = [0.1, 0.2, 0.3, 0.4]
    dataset_names = ['ids18', 'ton']
    model_names_t = ['mlp_t', 'cnn_t', 'rescnn_t', 'lstm_t', 'Selfattention_t']
    attack_names = ['MIFGSM', 'SIM', 'VMIFGSM', 'DGM', 'raw_att_traffic']

    lst_idx = []
    for ras in ratios_s:
        for an in attack_names:
            lst_idx.append((ras, an))
    midx = pd.MultiIndex.from_tuples(lst_idx, names=['Ratio', 'Attack'])

    lst_col = []
    for mn in model_names_t:
        lst_col.append(f'{mn}(IDS18/TON)')

    res = []
    for dsn in dataset_names:
        df = pd.DataFrame([[0.] * len(lst_col)] * len(lst_idx), index=midx, columns=lst_col)
        print(df)

        for ras, an in lst_idx:
            
            fp_dataset = os.path.join(STORAGE_DIR, '5_4_3', 'adv_flow', dsn, f'{an}_{ras}.csv')
            fp_minmax = os.path.join(STORAGE_DIR, '5_4_3', 'dataset', f'{dsn}_minmax_t_{ratio_t}.csv')
            fp_fea = os.path.join(STORAGE_DIR, 'dataset', 'fea_t.csv')

            for mn_t in model_names_t:
                fp_model = os.path.join(STORAGE_DIR, '5_4_3', 'pre-trained_models', f'{dsn}_{mn_t}_{ratio_t}.pth')
                acc = main(ratio_t, dsn, ras, an, mn_t, fp_dataset, fp_minmax, fp_fea, fp_model)
                df.loc[(ras, an), f'{mn_t}(IDS18/TON)'] = round(acc * 100, 1)
            print(df)
        df['Average(IDS18/TON)'] = df.mean(axis=1).round(1)
        print(df)
        res.append(df)
    
    val = res[0].values.astype(str) + '/' + res[1].values.astype(str)

    df_res = pd.DataFrame(val, columns=df.columns, index=df.index)
    print(df_res)
