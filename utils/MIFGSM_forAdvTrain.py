"""MI-FGSM cho **huấn luyện adversarial**: dùng nhãn thật ``labels`` (không cố định toàn 1).

Tensor đầu vào được clamp [0,1] sau chuẩn hóa; thay ``inf`` để tránh NaN. Phù hợp stage
``adv_train`` / ``normal_adv_train`` trong repo.
"""
import os, sys
project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root_dir not in sys.path: sys.path.append(project_root_dir)
import torch
import pandas as pd
from utils.utils import rectify_adv_flows
import numpy as np


def MIFGSM_forAdvTrain(model, lossfn, flows:pd.DataFrame, labels:torch.Tensor, mask1, step, step_length, device, min_val, max_val):
    """Trả về chỉ ``adv_flows`` (không đo thời gian/payload như ``MIFGSM``)."""

    momentum = 0.
    mu = 1.5
    adv_flows = flows.copy(deep=True)

    for _ in range(step):
        adv_df = adv_flows.copy(deep=True)

        adv_df = (adv_df - min_val) / (max_val - min_val)
        adv_df = adv_df.fillna(0).replace([np.inf, -np.inf], [1.0, -1.])
        adv_tensor = torch.from_numpy(adv_df.values).float().to(device)
        adv_tensor = torch.clamp(adv_tensor, 0., 1.)

        adv_tensor.requires_grad_(True)
        loss = lossfn(model(adv_tensor), labels)
        loss.backward()

        momentum = mu * momentum + adv_tensor.grad / torch.norm(adv_tensor.grad, p=1)
        perturbation_direction = torch.sign(momentum) * mask1

        pert = step_length * perturbation_direction
        pert = pd.DataFrame(pert.to("cpu").numpy(), columns=adv_flows.columns)

        pert.loc[pert['Fwd IAT Max'] < 0, ['Fwd IAT Max']] = 0.
        pert.loc[pert['Fwd IAT Min'] > 0, ['Fwd IAT Min']] = 0.
        pert.loc[pert['Fwd Pkt Len Max'] < 0, ['Fwd Pkt Len Max']] = 0.
        pert.loc[pert['Fwd Pkt Len Min'] > 0, ['Fwd Pkt Len Min']] = 0.

        adv_flows += pert.values
        rectify_adv_flows(adv_flows, flows, pert)

    return adv_flows
