"""Mục 5.5 — Table 15 (chi phí tấn công: MIFGSM, SIM).

Đo **chi phí thực thi** khi sinh adversarial từ traffic attack thô: thời gian (µs/mẫu),
độ dài payload trung bình, độ trễ IAT — lũy kế trên toàn ``{dsn}_raw_att.csv`` theo chunk.

Hyperparameter:
    - ``batch_size=128``: đồng bộ với huấn luyện/đánh giá; chunk đọc CSV.
    - ``7, 140`` truyền vào hàm attack: **số iteration và step size** (cùng cấu hình chuẩn
      như file ``7_140.csv`` trong AAT) để số liệu overhead so sánh được với thí nghiệm transfer.
    - ``get_mask``: cố định một số trường độ dài/IAT (Fwd ...) = 1 — **ràng buộc chỉnh sửa**
      theo thiết kế tấn công trong paper (chỉ perturb các đặc trưng được phép).

Surrogate LSTM chạy ``train()`` (dropout hoạt động) theo convention dự án; các kiến trúc khác ``eval()``.
"""
import os, sys
project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root_dir)
import pandas as pd
from torch.nn import CrossEntropyLoss
from utils.surrogate_models import cnn_s, mlp_s, ResCNN_s, lstm_s, SelfAttention_s
import torch
from utils.DGM import DGM
from utils.MIFGSM import MIFGSM
from utils.SIM import SIM
from utils.VMIFGSM import VMIFGSM
from utils.utils import load_net, STORAGE_DIR
import os, glob
import subprocess


def get_mask(list_col:list, batch_size):
    """Tạo tensor mask [B, F]: 0 = có thể perturb; 1 = khóa (Fwd Len/IAT min-max).

    Args:
        list_col: Tên cột trùng ``fea_s.csv``.
        batch_size: Số hàng lặp mask.

    Returns:
        torch.Tensor: mask CPU, caller đưa lên GPU.
    """
    df_temp = pd.DataFrame([[0.] * len(list_col)], columns=list_col)
    df_temp["Fwd Pkt Len Max"] = 1.
    df_temp["Fwd Pkt Len Min"] = 1.
    df_temp["Fwd IAT Max"] = 1.
    df_temp["Fwd IAT Min"] = 1.
    tensor_temp = torch.from_numpy(df_temp.loc[0].values).repeat(batch_size, 1)
    return tensor_temp


def main(dsn, mns, an):
    """Chạy attack ``an`` trên toàn raw attack flows; trả về (time_ave µs, payload_ave, iat_ave).

    ``time_ave`` = (tổng thời gian / số mẫu) * 1e6; các giá trị payload/IAT do hàm attack trả về.
    """
    batch_size = 128
    dev = torch.device("cuda")
    att = None
    if an == 'MIFGSM':
        att = MIFGSM
    if an == 'SIM':
        att = SIM
    if an == 'VMIFGSM':
        att = VMIFGSM
    if an == 'DGM':
        att = DGM

    fp_minmax = os.path.join(STORAGE_DIR, 'dataset', f'{dsn}_minmax_s.csv')
    df_minmax = pd.read_csv(fp_minmax)

    lossfn = CrossEntropyLoss()

    fp_raw_att_flows = os.path.join(STORAGE_DIR, 'dataset', f'{dsn}_raw_att.csv')
    list_sm_col = pd.read_csv(os.path.join(STORAGE_DIR, 'dataset', 'fea_s.csv'), header=0, index_col=None).columns.tolist()
    mask = get_mask(list_sm_col, batch_size)
    mask = mask.to(dev)

    fp_model = os.path.join(STORAGE_DIR, 'pre-trained_models', 'normal_train', f'{dsn}_{mns}.pth')
    surrogate_model = load_net(60, f'{dsn}_{mns}', fp_model)
    surrogate_model.to(dev)
    if 'lstm' in mns:
        surrogate_model.train()
    else:
        surrogate_model.eval()

    cunt, cunt2 = 0, 0
    time_sum, payload_sum, iat_sum = 0, 0, 0
    for raw_flow in pd.read_csv(fp_raw_att_flows, header=0, index_col=None, chunksize=batch_size):
        cunt += len(raw_flow)
        
        if len(raw_flow) < 128:
            mask = mask[0: len(raw_flow)]
        
        df_flow = raw_flow[list_sm_col]
        df_adv_flow, (time_temp, payload_temp, iat_temp) = att(surrogate_model, lossfn, df_flow, None, mask, 7, 140, dev, df_minmax.loc[0], df_minmax.loc[1])
        
        time_sum += time_temp
        payload_sum += payload_temp
        iat_sum += iat_temp

        time_ave = (time_sum / cunt) * 1000 * 1000
        payload_ave = payload_sum / cunt
        iat_ave = iat_sum / cunt

        print(f'\r {dsn}, Sur_model:{mns}, attack:{an}, Progress:{cunt}, Time cost_ave:{time_ave:.1f} us, Length_ave:{payload_ave:.1f}, Delay:{iat_ave:.1f} us', end='')
        
    print()
    return time_ave, payload_ave, iat_ave

if __name__ == "__main__":

    dataset_names = ['ids18', 'ton']
    model_names_s = ['mlp_s', 'cnn_s', 'rescnn_s', 'lstm_s', 'Selfattention_s']

    attack_names = ['MIFGSM', 'SIM']

    list_col = []
    for an in attack_names:
        list_col.append(an + '_Time cost(IDS18/TON)')
        list_col.append(an + '_Length(IDS18/TON)')
        list_col.append(an + '_Delay(IDS18/TON)')

    res = []
    for dsn in dataset_names:
        df = pd.DataFrame([[0.] * len(attack_names) * 3] * len(model_names_s), index=model_names_s,  columns=list_col)
        print(df)

        for mns in model_names_s:
            for an in attack_names:
                time_ave, payload_ave, iat_ave = main(dsn, mns, an)
                df.loc[mns, an+'_Time cost(IDS18/TON)'] = round(time_ave, 1)
                df.loc[mns, an+'_Length(IDS18/TON)'] = round(payload_ave, 1)
                df.loc[mns, an+'_Delay(IDS18/TON)'] = round(iat_ave, 1)
                print(df)
        res.append(df)
    
    val = res[0].values.astype(str) + '/' + res[1].values.astype(str)
    df_res = pd.DataFrame(val, index=df.index, columns=df.columns).round(1)
    print(df_res)
