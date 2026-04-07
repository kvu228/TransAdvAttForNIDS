"""Mục 5.4.1 — Figure 2 (IDS2018).

Đánh giá độ bền của các mục tiêu (target) khi traffic đối kháng do tấn công DGM tạo ra,
với surrogate cố định là ``mlp_s``. Lưới thử: ``step_size`` × ``iteration`` (siêu tham số
của DGM trong pipeline sinh dữ liệu AAT), mỗi ô là tỷ lệ phát hiện attack (Recall trên lớp
attack: TP/(TP+FN)) của một target đã huấn luyện trên traffic sạch.

Hyperparameter trong script:
    - ``batch_size=128``: cân bằng throughput GPU và ổn định bộ nhớ; đồng bộ các script reproduce.
    - ``step_sizes`` [80..180] bước 20, ``iterations`` [3,5,...,13]: lưới theo paper để xem
      độ nhạy của hiệu quả tấn công theo bước và số vòng lặp DGM.
    - Target dùng 66 đặc trưng (``load_net(66, ...)``) và minmax/fea của tập ``_t``.

Đầu ra: ``output/figures/fig2.png`` (heatmap).
"""
import os, sys
project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root_dir)
import pandas as pd
from utils.utils import CustomDataset, load_net, STORAGE_DIR
from utils.plot_heatmap import plot_hm
import torch
from torch.utils.data import DataLoader

def main(mn_t, ss, ite, fp_dataset, fp_minmax, fp_fea, fp_model):
    """Chạy inference một lần trên CSV adversarial và trả về recall phát hiện attack.

    Args:
        mn_t: Tên target (ví dụ ``mlp_t``); dùng cùng biến global ``dsn`` để tạo ``model_name``.
        ss: Step size DGM (khớp tên file ``{ite}_{ss}.csv``).
        ite: Số iteration DGM (khớp tên file).
        fp_dataset: CSV luồng adversarial (AAT).
        fp_minmax, fp_fea: Chuẩn hóa theo tập target.
        fp_model: Checkpoint target (normal_train).

    Returns:
        float: ``TP / (TP + FN)`` trên toàn tập (chỉ quan tâm phát hiện đúng attack).
    """
    # hyper
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
        print(f"\rSur_model:mlp-s, Att:DGM, Tar:{mn_t}, setp_size:{ss}, iteration:{ite}, Progress:{curr_iter}/{len(dataset)} Acc: {acc:.3f}", end="")
    print()
    return acc

if __name__ == '__main__':
    dsn = 'ids18'
    model_names_t = ['mlp_t', 'cnn_t', 'rescnn_t', 'lstm_t', 'Selfattention_t']
    step_sizes = [80, 100, 120, 140, 160, 180]
    iterations = [3, 5, 7, 9, 11, 13]

    lst_idx = []
    for mn_t in model_names_t:
        for ss in step_sizes:
            lst_idx.append((mn_t, ss))
    midx = pd.MultiIndex.from_tuples(lst_idx, names=['Model', 'step_size'])
    
    df = pd.DataFrame([[0.] * len(iterations)] * len(lst_idx), index=midx, columns=iterations)
    print(df)

    for mn_t, ss in lst_idx:
        
        for ite in iterations:
            fp_dataset = os.path.join(STORAGE_DIR, 'AAT', f'{dsn}_mlp_s', 'DGM', f'{ite}_{ss}.csv')
            fp_minmax = os.path.join(STORAGE_DIR, 'dataset', f'{dsn}_minmax_t.csv')
            fp_fea = os.path.join(STORAGE_DIR, 'dataset', 'fea_t.csv')
            fp_model = os.path.join(STORAGE_DIR, 'pre-trained_models', 'normal_train', f'{dsn}_{mn_t}.pth')
            acc = main(mn_t, ss, ite, fp_dataset, fp_minmax, fp_fea, fp_model)
            df.loc[(mn_t, ss), ite] = round(acc * 100, 1)
            print(df)
    fp_fig = os.path.join(project_root_dir, 'output', 'figures', 'fig2.png')
    plot_hm(df, fp_fig)

    
