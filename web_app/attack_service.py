"""
Attack Service - Business Logic Layer for SPTS NIDS Simulation.

Module này đóng vai trò cầu nối giữa Streamlit UI và các thuật toán tấn công trong utils/.
Chịu trách nhiệm:
- Quét và liệt kê models có sẵn
- Tạo AAT (Adversarial Attack Traffic) từ raw attack CSV
- Tính Evasion Rate bằng cách đánh giá AAT trên target model
- Thu thập thống kê 4 đặc trưng Level-1 (SPTS)

Không chứa logic gradient; toàn bộ gọi từ utils.MIFGSM, utils.SIM, utils.VMIFGSM, utils.DGM.
"""

import os
import sys
import time
from typing import Callable, Optional

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from utils.utils import load_net, normalize_df, STORAGE_DIR, CustomDataset
from utils.MIFGSM import MIFGSM
from utils.SIM import SIM
from utils.VMIFGSM import VMIFGSM
from utils.DGM import DGM

# Level-1 feature names for SPTS
LEVEL1_FEATURES = ["Fwd Pkt Len Max", "Fwd Pkt Len Min", "Fwd IAT Max", "Fwd IAT Min"]


def get_mask(list_col: list, batch_size: int) -> torch.Tensor:
    """
    Tạo mask cho 4 đặc trưng Level-1 (ràng buộc SPTS).

    Chỉ các cột Fwd Pkt Len Max/Min, Fwd IAT Max/Min được phép thay đổi.
    Các đặc trưng khác có mask=0, không bị perturb.

    Args:
        list_col: Danh sách tên cột features.
        batch_size: Số mẫu trong batch.

    Returns:
        Tensor mask shape (batch_size, n_features), 1.0 cho 4 Level-1, 0.0 cho còn lại.
    """
    df_temp = pd.DataFrame([[0.0] * len(list_col)], columns=list_col)
    for col in LEVEL1_FEATURES:
        if col in df_temp.columns:
            df_temp[col] = 1.0
    return torch.from_numpy(df_temp.loc[0].values).repeat(batch_size, 1)


def get_available_models() -> tuple[list[str], list[str]]:
    """
    Quét thư mục pre-trained_models để lấy danh sách models có sẵn.

    Phân loại theo hậu tố: _t (target) và _s (surrogate).
    Ví dụ: ton_mlp_t.pth → target, ton_mlp_s.pth → surrogate.

    Returns:
        (target_models, surrogate_models): Hai list tên model (không có .pth).
    """
    models_dir = os.path.join(STORAGE_DIR, "custom", "pre-trained_models")
    if not os.path.isdir(models_dir):
        return [], []

    target_models = []
    surrogate_models = []
    for f in os.listdir(models_dir):
        if f.endswith(".pth"):
            name = f[:-4]  # remove .pth
            if name.endswith("_t"):
                target_models.append(name)
            elif name.endswith("_s"):
                surrogate_models.append(name)
    return sorted(target_models), sorted(surrogate_models)


def build_paths_from_models(
    target_model: str,
    surrogate_model: str,
) -> dict:
    """
    Xây dựng đường dẫn file từ tên target và surrogate model.

    Dataset được suy từ surrogate (vd: ton_mlp_s → ton).
    Raw attack CSV: {dataset}_raw_att.csv.

    Args:
        target_model: Tên target model (vd: ton_mlp_t).
        surrogate_model: Tên surrogate model (vd: ton_mlp_s).

    Returns:
        Dict với keys: fp_raw_att, fp_fea_s, fp_minmax_s, fp_model_s,
        fp_fea_t, fp_minmax_t, fp_model_t.
    """
    dataset = surrogate_model.rsplit("_", 2)[0] if surrogate_model else "ton"
    models_dir = os.path.join(STORAGE_DIR, "custom", "pre-trained_models")
    dataset_dir = os.path.join(STORAGE_DIR, "dataset")
    return {
        "fp_raw_att": os.path.join(dataset_dir, f"{dataset}_raw_att.csv"),
        "fp_fea_s": os.path.join(dataset_dir, "fea_s.csv"),
        "fp_minmax_s": os.path.join(dataset_dir, f"{dataset}_minmax_s.csv"),
        "fp_model_s": os.path.join(models_dir, f"{surrogate_model}.pth"),
        "fp_fea_t": os.path.join(dataset_dir, "fea_t.csv"),
        "fp_minmax_t": os.path.join(dataset_dir, f"{dataset}_minmax_t.csv"),
        "fp_model_t": os.path.join(models_dir, f"{target_model}.pth"),
    }


def _count_csv_chunks(fp: str, batch_size: int) -> int:
    """Đếm số dòng trong CSV (trừ header) và trả về số chunk tương ứng."""
    with open(fp, "r") as f:
        total_rows = sum(1 for _ in f) - 1
    return max(1, (total_rows + batch_size - 1) // batch_size)


def generate_aat_from_data(
    fp_raw_att: str,
    fp_fea_s: str,
    fp_minmax_s: str,
    fp_model_s: str,
    fp_model_t: str,
    fp_fea_t: str,
    fp_minmax_t: str,
    algorithm: str,
    iterations: int = 7,
    step_size: float = 140,
    copies: int = 5,
    dropout_rate: float = 0.2,
    batch_size: int = 128,
    progress_callback: Optional[Callable[[int, int, int, Optional[float]], None]] = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Tạo AAT từ raw attack CSV và tính các metrics.

    Luồng: Load surrogate → Đọc raw CSV theo batch → Chạy attack (MIFGSM/SIM/VMIFGSM/DGM)
    → Rectify flows → Đánh giá trên target model → Tính evasion rate.

    Args:
        fp_raw_att: Đường dẫn raw attack CSV.
        fp_fea_s, fp_minmax_s: Features và minmax cho surrogate (60 features).
        fp_model_s, fp_model_t: Đường dẫn .pth của surrogate và target.
        fp_fea_t, fp_minmax_t: Features và minmax cho target (66 features).
        algorithm: MIFGSM | SIM | VMIFGSM | DGM.
        iterations: Số bước lặp (mặc định 7).
        step_size: Kích thước bước (mặc định 140).
        copies, dropout_rate: Tham số riêng DGM.
        progress_callback: Optional. Gọi với (current_chunk, total_chunks, processed_rows, eta_seconds).

    Returns:
        (adv_flows_df, metrics_dict):
        - adv_flows_df: DataFrame adversarial flows.
        - metrics_dict: {evasion_rate: float, level1_stats: dict}.
    """
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lossfn = CrossEntropyLoss()

    list_sm_col = pd.read_csv(fp_fea_s, header=0, index_col=None).columns.tolist()
    df_minmax_s = pd.read_csv(fp_minmax_s, header=0, index_col=None)

    # Load surrogate model
    surrogate_model = load_net(
        os.path.basename(fp_model_s).replace(".pth", ""),
        fp_model_s,
    )
    surrogate_model.to(dev)
    if "lstm" in fp_model_s.lower():
        surrogate_model.train()
    else:
        surrogate_model.eval()

    # Select attack
    attack_map = {
        "MIFGSM": MIFGSM,
        "SIM": SIM,
        "VMIFGSM": VMIFGSM,
        "DGM": DGM,
    }
    att_fn = attack_map.get(algorithm.upper(), MIFGSM)

    # Collect all adv flows and level1 stats
    all_adv_rows = []
    level1_before = {f: [] for f in LEVEL1_FEATURES}
    level1_after = {f: [] for f in LEVEL1_FEATURES}

    total_chunks = _count_csv_chunks(fp_raw_att, batch_size)
    start_time = time.perf_counter()
    chunk_idx = 0

    for raw_flow in pd.read_csv(fp_raw_att, header=0, index_col=None, chunksize=batch_size):
        if len(raw_flow) == 0:
            continue
        mask = get_mask(list_sm_col, len(raw_flow)).to(dev)
        df_flow = raw_flow[list_sm_col].copy()

        # Store before stats for Level-1
        for f in LEVEL1_FEATURES:
            if f in df_flow.columns:
                level1_before[f].extend(df_flow[f].tolist())

        # Call attack
        if algorithm.upper() == "DGM":
            df_adv_flow, _ = att_fn(
                surrogate_model,
                lossfn,
                df_flow,
                None,
                mask,
                iterations,
                step_size,
                dev,
                df_minmax_s.loc[0],
                df_minmax_s.loc[1],
                nums_of_noise=copies,
                dropoutp=dropout_rate,
            )
        else:
            df_adv_flow, _ = att_fn(
                surrogate_model,
                lossfn,
                df_flow,
                None,
                mask,
                iterations,
                step_size,
                dev,
                df_minmax_s.loc[0],
                df_minmax_s.loc[1],
            )

        for f in LEVEL1_FEATURES:
            if f in df_adv_flow.columns:
                level1_after[f].extend(df_adv_flow[f].tolist())

        raw_flow = raw_flow.copy()
        raw_flow[list_sm_col] = df_adv_flow
        all_adv_rows.append(raw_flow)

        chunk_idx += 1
        processed_rows = chunk_idx * batch_size
        if progress_callback:
            elapsed = time.perf_counter() - start_time
            eta = (elapsed / chunk_idx) * (total_chunks - chunk_idx) if chunk_idx > 0 else None
            progress_callback(chunk_idx, total_chunks, processed_rows, eta)

    adv_flows_df = pd.concat(all_adv_rows, ignore_index=True)

    # Save AAT temporarily for evaluation (target model expects 66 features from raw)
    fp_aat_temp = os.path.join(STORAGE_DIR, "custom", "output", "aat_simulation_temp.csv")
    adv_flows_df.to_csv(fp_aat_temp, index=False)

    # Compute evasion rate using target model
    if progress_callback:
        progress_callback(total_chunks + 1, total_chunks + 1, len(adv_flows_df), 0)
    evasion_rate = _compute_evasion_rate(
        fp_aat_temp, fp_fea_t, fp_minmax_t, fp_model_t, dev, batch_size
    )

    # Remove temp file
    if os.path.exists(fp_aat_temp):
        os.remove(fp_aat_temp)

    # Level1 stats
    level1_stats = {}
    for f in LEVEL1_FEATURES:
        if level1_before[f] and level1_after[f]:
            level1_stats[f] = {
                "before": {"mean": sum(level1_before[f]) / len(level1_before[f]), "count": len(level1_before[f])},
                "after": {"mean": sum(level1_after[f]) / len(level1_after[f]), "count": len(level1_after[f])},
            }

    metrics = {
        "evasion_rate": evasion_rate,
        "level1_stats": level1_stats,
    }
    return adv_flows_df, metrics


def _compute_evasion_rate(
    fp_att: str,
    fp_fea: str,
    fp_minmax: str,
    fp_model: str,
    dev: torch.device,
    batch_size: int = 128,
) -> float:
    """
    Đánh giá AAT trên target model và tính Evasion Rate.

    Evasion Rate = % luồng có label=Attack nhưng model dự đoán Benign.
    Công thức: 100 - (TP / (TP+FN)) với TP=đúng Attack, FN=nhầm Attack thành Benign.

    Returns:
        Evasion rate (0-100), làm tròn 2 chữ số.
    """
    dataset = CustomDataset(fp_att, fp_minmax, fp_fea)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    model_name = os.path.basename(fp_model).replace(".pth", "")
    net = load_net(model_name, fp_model)
    net.to(dev)
    net.eval()

    TP, FN = 0, 0
    for flows, labels in dataloader:
        flows, labels = flows.to(dev), labels.to(dev)
        with torch.no_grad():
            pred = net(flows).argmax(1)
        TP += ((pred == 1) & (labels == 1)).sum().item()
        FN += ((pred == 0) & (labels == 1)).sum().item()

    total = TP + FN
    if total == 0:
        return 0.0
    detection_rate = (TP / total) * 100
    evasion_rate = 100 - detection_rate
    return round(evasion_rate, 2)


def get_raw_att_paths() -> list[str]:
    """
    Trả về danh sách file raw attack CSV có sẵn trong STORAGE_DIR/dataset.

    Tìm các file kết thúc bằng _raw_att.csv (vd: ton_raw_att.csv, ids18_raw_att.csv).
    """
    dataset_dir = os.path.join(STORAGE_DIR, "dataset")
    if not os.path.isdir(dataset_dir):
        return []
    candidates = [f for f in os.listdir(dataset_dir) if f.endswith("_raw_att.csv")]
    return sorted(candidates)
