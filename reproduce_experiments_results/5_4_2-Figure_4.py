"""Mục 5.4.2 — Figure 4 (IDS2018).

Đánh giá ảnh hưởng của số bản sao (copies) trong DGM lên khả năng làm giảm Recall target.
Mỗi surrogate ``*_s`` sinh adversarial với ``dropout_rate`` cố định; thay đổi ``cpn`` ∈ {1,3,5,7,9}.

Hyperparameter:
    - ``dropout_rate=0.2``: một điểm cố định trên trục dropout (Fig.6/7 khảo sát thêm các mức khác).
    - ``batch_size=128``: nhất quán pipeline.
    - File AAT: ``{cpn}_{dropout_rate}.csv`` trong thư mục ``DGM`` của từng surrogate.

Đầu ra: ``output/figures/fig4.png`` (line plot, thêm cột Average).
"""
import os, sys
project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root_dir)
import pandas as pd
from utils.utils import CustomDataset, load_net, STORAGE_DIR
from utils.plot_line import plot_line
import torch
from torch.utils.data import DataLoader

def main(mn_s, cpn, dropout_rate, mn_t, fp_dataset, fp_minmax, fp_fea, fp_model):
    """Đo Recall attack của target trên adversarial theo (surrogate, copies, dropout).

    Args:
        mn_s: Loại surrogate (vd. ``mlp_s``); chỉ để log, đường dẫn đã nằm trong ``fp_dataset``.
        cpn: Số copies trong cấu hình DGM (tên file ``{cpn}_{dropout_rate}.csv``).
        dropout_rate: Tỷ lệ dropout trong sinh DGM.
        mn_t: Target; ``load_net(66, f'{dsn}_{mn_t}', ...)``.
        fp_dataset, fp_minmax, fp_fea, fp_model: CSV AAT và checkpoint target.

    Returns:
        float: TP/(TP+FN) trên tập adversarial.
    """
    dev = torch.device('cuda')
    batch_size = 128

    dataset = CustomDataset(fp_dataset, fp_minmax, fp_fea)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    model_name = f'{dsn}_{mn_t}'
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
        print(f"\rSurr_model:{mn_s}, attack:DGM, Tar_model:{mn_t}, copies:{cpn}, dropout_rate:{dropout_rate}, Progress:{curr_iter}/{len(dataset)}, Acc: {acc:.3f}", end="")
    print()
    return acc

if __name__ == '__main__':
    dsn = 'ids18'
    model_names_t = ['mlp_t', 'cnn_t', 'rescnn_t', 'lstm_t', 'Selfattention_t']
    model_names_s = ['mlp_s', 'cnn_s', 'rescnn_s', 'lstm_s', 'Selfattention_s']
    dropout_rate = 0.2
    copies = [1, 3, 5, 7, 9]

    lst_idx = []
    for mn_s in model_names_s:
        for cpn in copies:
            lst_idx.append((mn_s, cpn))
    midx = pd.MultiIndex.from_tuples(lst_idx, names=['Model', 'Copies'])

    df = pd.DataFrame([[0.] * len(model_names_t)] * len(lst_idx), index=midx, columns=model_names_t)
    print(df)

    for mn_s, cpn in lst_idx:
        for mn_t in model_names_t:
            fp_dataset = os.path.join(STORAGE_DIR, 'AAT', f'{dsn}_{mn_s}', 'DGM', f'{cpn}_{dropout_rate}.csv')
            fp_minmax = os.path.join(STORAGE_DIR, 'dataset', f'{dsn}_minmax_t.csv')
            fp_fea = os.path.join(STORAGE_DIR, 'dataset', 'fea_t.csv')
            fp_model = os.path.join(STORAGE_DIR, 'pre-trained_models', 'normal_train', f'{dsn}_{mn_t}.pth')
            acc = main(mn_s, cpn, dropout_rate, mn_t, fp_dataset, fp_minmax, fp_fea, fp_model)
            df.loc[(mn_s, cpn), mn_t] = round(acc * 100, 1)
            print(df)

    df['Average'] = df.mean(axis=1).round(1)
    print(df)
    fp_fig = os.path.join(project_root_dir, 'output', 'figures', 'fig4.png')
    plot_line(df, fp_fig)