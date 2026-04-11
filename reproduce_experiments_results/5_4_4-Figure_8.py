"""Mục 5.4.4 — Figure 8 (IDS2018).

Khảo sát **số đặc trưng đầu vào của surrogate** (27, 30, …, 66) khi sinh adversarial
(MIFGSM, SIM, VMIFGSM, DGM) và đo Recall của target (66 fea) trên các file trong
``STORAGE_DIR/5_4_4/{exno}_{dsn}_adv_flow/{Attack}_{fea_num}.csv``.

Hyperparameter:
    - ``experiment_no = [0,1,2]``: ba lần lặp độc lập; kết quả figure là **trung bình** 3 run
      để giảm nhiễu ngẫu nhiên của tấn công/huấn luyện.
    - ``fea_nums = range(27, 67, 3)``: lưới thưa, đủ phủ từ thiếu đặc trưng tới đủ 66.
    - ``load_net(60, ...)``: target vẫn 60-dim theo checkpoint thí nghiệm mục này (khác Fig.2).

Đầu ra: ``output/figures/fig8.png``.
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
    """Recall attack của target trên adversarial sinh với ``fea_num`` đặc trưng surrogate.

    Args:
        dsn: ``ids18`` hoặc ``ton``.
        mnt, an: Target và tên tấn công (khớp thư mục/file CSV).
        fea_num: Số chiều đặc trưng surrogate khi tạo adv (cột trong ma trận kết quả).
        fp_dataset, fp_minmax, fp_fea, fp_model: Đường dẫn dữ liệu và model target.

    Returns:
        float: TP/(TP+FN).
    """
    # hyper
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
    dsn = 'ids18'
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
    fp_fig = os.path.join(project_root_dir, 'output', 'figures', 'fig8.png')
    plot_line2(df_res, fp_fig)
