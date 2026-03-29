import torch
import pandas as pd
import numpy as np
from utils.utils import rectify_adv_flows
import time

def DGM(model, lossfn, flows, labels, mask, step, step_length, device, min_val, max_val, nums_of_noise=5, dropoutp=0.2):
    t1 = time.perf_counter()
    momentum = 0.
    mu = 1.
    adv_flows = flows.copy(deep=True)
    
    # ✅ VÒNG LẶP TỪNG BƯỚC (Từng level của SPTS)
    for _ in range(step):
        adv_df = adv_flows.copy(deep=True)
        # Chuẩn hóa dữ liệu (0-1 range)
        adv_df = (adv_df - min_val) / (max_val - min_val)
        adv_df = adv_df.fillna(0)
        adv_tensor = torch.from_numpy(adv_df.values).float().to(device)
        
        # ✅ Target label = 0 (Benign - NIDS dự đoán sai)
        labels_tensor = torch.ones(adv_tensor.size(0), dtype=torch.long).to(device)

        # ✅ BƯỚC 1: Tính gradient ban đầu
        adv_tensor.requires_grad_(True)
        loss = lossfn(model(adv_tensor), labels_tensor)
        loss.backward()
        
        # ✅ BƯỚC 2: Tính TRUNG BÌNH GRADIENT từ nhiều nhiễu (Gradient Averaging)
        agg_grad = 0.
        for j in range(nums_of_noise):  # nums_of_noise=5 (mặc định)
            noise1 = torch.rand(adv_tensor.shape).to(device)
            
            # Dropout mask: loại bỏ 20% đặc trưng (dropoutp=0.2)
            mask_loc1 = torch.from_numpy(
                np.random.choice([0, 1], size=adv_tensor.shape, p=[dropoutp, 1-dropoutp])
            ).to(device)
            
            adv_temp = adv_tensor.data.clone().requires_grad_(True)
            
            # Áp dụng 4 biến thể khác nhau (mỗi cái 25% xác suất)
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
            # ✅ Tích lũy gradient
            agg_grad += adv_temp.grad
        
        # ✅ BƯỚC 3: Tính TRUNG BÌNH GRADIENT
        g = agg_grad / nums_of_noise

        # ✅ BƯỚC 4: Momentum + Chuẩn hóa L1
        momentum = mu * momentum + g / torch.norm(g, p=1)
        
        # ✅ BƯỚC 5: Tính hướng perturb từ momentum
        perturbation_direction = torch.sign(momentum) * mask

        # ✅ BƯỚC 6: Tạo nhiễu và áp dụng lên dữ liệu
        pert = step_length * perturbation_direction
        pert = pd.DataFrame(pert.to("cpu").numpy(), columns=adv_flows.columns)

        # ✅ Ràng buộc SPTS (chỉ 4 đặc trưng Level-1):
        # - Fwd IAT Max: không giảm (< 0 → 0)
        # - Fwd IAT Min: không tăng (> 0 → 0)
        # - Fwd Pkt Len Max: không giảm (< 0 → 0)
        # - Fwd Pkt Len Min: không tăng (> 0 → 0)
        pert.loc[pert['Fwd IAT Max'] < 0, ['Fwd IAT Max']] = 0.
        pert.loc[pert['Fwd IAT Min'] > 0, ['Fwd IAT Min']] = 0.
        pert.loc[pert['Fwd Pkt Len Max'] < 0, ['Fwd Pkt Len Max']] = 0.
        pert.loc[pert['Fwd Pkt Len Min'] > 0, ['Fwd Pkt Len Min']] = 0.

        # ✅ Cập nhật luồng tấn công
        adv_flows += pert.values
        rectify_adv_flows(adv_flows, flows, pert)

    t2 = time.perf_counter()
    return adv_flows, (t2 - t1, ...)