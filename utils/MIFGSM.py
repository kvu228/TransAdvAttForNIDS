"""MI-FGSM trên bảng đặc trưng flow (momentum, dấu gradient, bước ``step_length``).

Chuẩn hóa min-max chỉ để tính gradient trên surrogate; **perturbation cộng trực tiếp** lên
giá trị gốc (không inverse min-max) tránh mất số — ví dụ thời lượng rất nhỏ có thể về 0 sau
de-normalize. Sau mỗi bước gọi ``rectify_adv_flows`` để cập nhật đặc trưng phụ thuộc.

Trả về ``(adv_flows, (thời_gian, tổng_delta_payload, tổng_delta_iat))`` phục vụ đo overhead.
"""
import os, sys
project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root_dir not in sys.path: sys.path.append(project_root_dir)
import torch
import pandas as pd
from utils.utils import rectify_adv_flows
import time


def MIFGSM(model, lossfn, flows, labels, mask1, step, step_length, device, min_val, max_val):
    """Chạy ``step`` vòng MI-FGSM; ``mask1`` khóa các chiều không được perturb.

    Args:
        model: Surrogate trên GPU.
        lossfn: Thường là CrossEntropyLoss.
        flows: DataFrame đặc trưng (đơn vị thực, chưa chuẩn hóa).
        labels: Không dùng trong vòng lặp (API thống nhất với attack khác).
        mask1: Tensor [B, F] (0 = cho phép sửa, 1 = khóa) nhân với sign(momentum).
        step: Số iteration (vd. 7).
        step_length: Độ lớn bước (vd. 140).
        min_val, max_val: Hai hàng min/max (Series/DataFrame) cho chuẩn hóa gradient.
    """
    t1 = time.perf_counter()
    momentum = 0.
    mu = 1.5
    adv_flows = flows.copy(deep=True)
    
    for _ in range(step):
        adv_df = adv_flows.copy(deep=True)
        
        # Chuẩn hóa [0,1] chỉ để forward/backward trên surrogate
        adv_df = (adv_df - min_val) / (max_val - min_val)
        adv_df = adv_df.fillna(0)
        adv_tensor = torch.from_numpy(adv_df.values).float().to(device)

        labels_tensor = torch.ones(adv_tensor.size(0), dtype=torch.long).to(device)

        adv_tensor.requires_grad_(True)
        # mask_loc1 = torch.from_numpy(np.random.choice([0, 1], size=adv_tensor.shape, p=[0.5, 0.5])).to(device)
        # # loss = lossfn(model(adv_tensor * mask_loc1), labels_tensor)

        loss = lossfn(model(adv_tensor), labels_tensor)
        loss.backward()

        momentum = mu * momentum + adv_tensor.grad / torch.norm(adv_tensor.grad, p=1)
        perturbation_direction = torch.sign(momentum) * mask1

        # Cộng nhiễu trên **thang gốc** (không inverse min-max) để tránh làm tròn/mất giá trị nhỏ
        pert = step_length * perturbation_direction
        pert = pd.DataFrame(pert.to("cpu").numpy(), columns=adv_flows.columns)

        # Chỉ cho phép hướng: IAT Max/Len Max không giảm; IAT Min/Len Min không tăng
        pert.loc[pert['Fwd IAT Max'] < 0, ['Fwd IAT Max']] = 0.
        pert.loc[pert['Fwd IAT Min'] > 0, ['Fwd IAT Min']] = 0.
        pert.loc[pert['Fwd Pkt Len Max'] < 0, ['Fwd Pkt Len Max']] = 0.
        pert.loc[pert['Fwd Pkt Len Min'] > 0, ['Fwd Pkt Len Min']] = 0.

        adv_flows += pert.values
        rectify_adv_flows(adv_flows, flows, pert)

    t2 = time.perf_counter()
    res_time = t2 - t1

    res_payload = (adv_flows['Fwd Pkt Len Max'] - flows['Fwd Pkt Len Max']).sum()
    res_iat = (adv_flows['Fwd IAT Max'] - flows['Fwd IAT Max']).sum()

    # are_equal = (flows.reset_index(drop=True) == adv_flows.reset_index(drop=True)).all(axis=1)
    # count_equal_rows = are_equal.sum()
    # print(count_equal_rows)
        
    return adv_flows, (res_time, res_payload, res_iat)
