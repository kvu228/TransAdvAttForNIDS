"""SIM (Scale-Invariant attack Method): trung bình gradient trên nhiều bản scale của input.

``nums_of_copies=5``: loss trên ``adv_tensor_temp / (2**i)`` rồi gộp gradient; momentum + sign,
mask giống MI-FGSM. Perturb trên thang đặc trưng gốc + ``rectify_adv_flows``.
"""
import os, sys
project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root_dir not in sys.path: sys.path.append(project_root_dir)
import torch
import pandas as pd
from utils.utils import rectify_adv_flows
import time


def SIM(model, lossfn, flows, labels, mask, step, step_length, device, min_val, max_val):
    """Tham số tương tự ``MIFGSM``; ``mask`` khóa chiều perturb."""
    t1 = time.perf_counter()
    momentum = 0.
    mu = 1.
    adv_flows = flows.copy(deep=True)
    nums_of_copies = 5

    for _ in range(step):
        adv_df = adv_flows.copy(deep=True)
        adv_df = (adv_df - min_val) / (max_val - min_val)
        adv_df = adv_df.fillna(0)
        adv_tensor = torch.from_numpy(adv_df.values).float().to(device)
        labels_tensor = torch.ones(adv_tensor.size(0), dtype=torch.long).to(device)

        g_agg = torch.zeros(adv_tensor.data.shape).to(device)
        for i in range(nums_of_copies):
            adv_tensor_temp = adv_tensor.detach().clone().to(device).requires_grad_(True)
            loss = lossfn(model(adv_tensor_temp / (2**i)), labels_tensor)
            loss.backward()
            g_agg += adv_tensor_temp.grad
        g = g_agg / nums_of_copies
        momentum = mu * momentum + g / torch.norm(g, p=1)

        perturbation_direction = torch.sign(momentum) * mask

        pert = step_length * perturbation_direction
        pert = pd.DataFrame(pert.to("cpu").numpy(), columns=adv_flows.columns)

        pert.loc[pert['Fwd IAT Max'] < 0, ['Fwd IAT Max']] = 0.
        pert.loc[pert['Fwd IAT Min'] > 0, ['Fwd IAT Min']] = 0.
        pert.loc[pert['Fwd Pkt Len Max'] < 0, ['Fwd Pkt Len Max']] = 0.
        pert.loc[pert['Fwd Pkt Len Min'] > 0, ['Fwd Pkt Len Min']] = 0.

        adv_flows += pert.values
        rectify_adv_flows(adv_flows, flows, pert)

    # Debug: đếm dòng không đổi
    # are_equal = (flows.reset_index(drop=True) == adv_flows.reset_index(drop=True)).all(axis=1)
    # count_equal_rows = are_equal.sum()
    t2 = time.perf_counter()
    res_time = t2 - t1

    res_payload = (adv_flows['Fwd Pkt Len Max'] - flows['Fwd Pkt Len Max']).sum()
    res_iat = (adv_flows['Fwd IAT Max'] - flows['Fwd IAT Max']).sum()
        
    return adv_flows, (res_time, res_payload, res_iat)
