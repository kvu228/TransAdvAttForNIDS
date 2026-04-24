"""NI-FGSM (Nesterov Iterative FGSM) trên bảng đặc trưng flow.

Biến thể của MI-FGSM sử dụng Nesterov accelerated gradient: tính gradient tại vị trí
"nhìn trước" (look-ahead) thay vì vị trí hiện tại. Điều này giúp hội tụ nhanh hơn và
tìm được hướng tấn công tổng quát hơn, cải thiện transferability.

Tham khảo: Lin et al., "Nesterov Accelerated Gradient and Scale Invariance for
Adversarial Attacks", ICLR 2020.

Trả về ``(adv_flows, (thời_gian, tổng_delta_payload, tổng_delta_iat))`` phục vụ đo overhead.
"""
import os, sys
project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root_dir not in sys.path: sys.path.append(project_root_dir)
import torch
import pandas as pd
from utils.utils import rectify_adv_flows
import time


def NIFGSM(model, lossfn, flows, labels, mask1, step, step_length, device, min_val, max_val):
    """Chạy ``step`` vòng NI-FGSM; ``mask1`` khóa các chiều không được perturb.

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

        # --- Nesterov look-ahead ---
        # Tính gradient tại vị trí "tương lai": x + mu * momentum
        # thay vì tại vị trí hiện tại x như MI-FGSM
        nes_tensor = adv_tensor + mu * momentum if not isinstance(momentum, float) else adv_tensor.clone()
        nes_tensor = nes_tensor.detach().requires_grad_(True)

        loss = lossfn(model(nes_tensor), labels_tensor)
        loss.backward()

        # Cập nhật momentum bằng gradient tại vị trí look-ahead
        momentum = mu * momentum + nes_tensor.grad / torch.norm(nes_tensor.grad, p=1)
        perturbation_direction = torch.sign(momentum) * mask1

        # Cộng nhiễu trên **thang gốc** (không inverse min-max)
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

    return adv_flows, (res_time, res_payload, res_iat)
