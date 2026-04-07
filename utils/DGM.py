"""DGM (Dense surrogate Gradient Mix): tấn công đối kháng trên đặc trưng flow.

Kết hợp momentum, trung bình gradient trên nhiều biến thể forward (nhân gradient, cộng nhiễu,
dropout ngẫu nhiên trên tensor, scale) và mask dropout theo ``dropoutp``. Mỗi bước: chuẩn hóa
min-max → backward → cập nhật theo sign(momentum) * ``mask``, ràng buộc 4 đặc trưng Level-1
(IAT/Len fwd max-min) rồi ``rectify_adv_flows`` để đồng bộ đặc trưng phụ thuộc.

Tham số ``nums_of_noise`` (mặc định 5) và ``dropoutp`` (0.2) khớp thí nghiệm paper; ``step`` /
``step_length`` giống các attack khác (vd. 7, 140).
"""
import random
import time

import numpy as np
import pandas as pd
import torch

from utils.utils import rectify_adv_flows


def DGM(model, lossfn, flows, labels, mask, step, step_length, device, min_val, max_val, nums_of_noise=5, dropoutp=0.2):
    """Sinh adversarial flow; trả về ``(adv_df, (elapsed_s, sum_delta_len, sum_delta_iat))`` như MIFGSM."""
    t1 = time.perf_counter()
    momentum = 0.
    mu = 1.
    adv_flows = flows.copy(deep=True)

    for _ in range(step):
        adv_df = adv_flows.copy(deep=True)
        adv_df = (adv_df - min_val) / (max_val - min_val)
        adv_df = adv_df.fillna(0)
        adv_tensor = torch.from_numpy(adv_df.values).float().to(device)

        # Nhãn cho loss (toàn 1); cùng convention với MIFGSM/SIM/VMIFGSM trong repo.
        labels_tensor = torch.ones(adv_tensor.size(0), dtype=torch.long).to(device)

        adv_tensor.requires_grad_(True)
        loss = lossfn(model(adv_tensor), labels_tensor)
        loss.backward()

        agg_grad = 0.
        for j in range(nums_of_noise):
            noise1 = torch.rand(adv_tensor.shape).to(device)

            mask_loc1 = torch.from_numpy(
                np.random.choice([0, 1], size=adv_tensor.shape, p=[dropoutp, 1 - dropoutp])
            ).to(device)

            adv_temp = adv_tensor.data.clone().requires_grad_(True)

            r = random.random()
            if r < (1/4):
                loss = lossfn(model(adv_temp * adv_tensor.grad), labels_tensor)
            elif r < (2/4):
                loss = lossfn(model(adv_temp + noise1), labels_tensor)
            elif r < (3/4):
                loss = lossfn(model(adv_temp * mask_loc1), labels_tensor)
            else:
                loss = lossfn(model(adv_temp / (j+1)), labels_tensor)

            loss.backward()
            agg_grad += adv_temp.grad

        g = agg_grad / nums_of_noise

        momentum = mu * momentum + g / torch.norm(g, p=1)

        perturbation_direction = torch.sign(momentum) * mask

        pert = step_length * perturbation_direction
        pert = pd.DataFrame(pert.to("cpu").numpy(), columns=adv_flows.columns)

        # Ràng buộc hướng perturb trên 4 đặc trưng Level-1 (đồng bộ MIFGSM/SIM/VMIFGSM)
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