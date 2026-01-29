# Tổng quan về TransAdvAttForNIDS

## Mục đích dự án

Dự án này nghiên cứu về **Transferable Adversarial Attacks** (Tấn công đối kháng có thể chuyển giao) cho **Network Intrusion Detection Systems** (Hệ thống phát hiện xâm nhập mạng - NIDS).

## Kiến trúc tổng thể

Dự án được tổ chức thành 3 nhóm chính:

### 1. Reproducing Results (Tái tạo kết quả)
- **Thư mục**: `reproduce_experiments_results/`
- **Mục đích**: Tái tạo các bảng và hình ảnh trong bài báo nghiên cứu
- **Dữ liệu**: Sử dụng dataset đã được pre-process và AAT đã được generate sẵn

### 2. Custom Training & AAT Generation (Training tùy chỉnh và tạo AAT)
- **Thư mục**: `train_NIDS/`, `generate_AAT/`
- **Mục đích**: Cho phép người dùng train model riêng và generate Adversarial Attack Traffic (AAT)
- **Workflow**: Training → Generate AAT → Test AAT

### 3. Mapping AAT to Packets (Ánh xạ AAT về gói tin thực tế)
- **Thư mục**: `map_AAT_to_pkts/`, `TANTRA/`
- **Mục đích**: Chuyển đổi AAT đã generate về các gói tin mạng thực tế
- **Công cụ**: Sử dụng CICFlowMeter (CFM) để extract features từ PCAP files

## Các thành phần chính

### Models (Mô hình)
- **Target Models** (`utils/target_models.py`): Các NIDS được tấn công (66 hoặc 78 features)
- **Surrogate Models** (`utils/surrogate_models.py`): Mô hình giả lập của attacker (60 features)
- **Kiến trúc hỗ trợ**: MLP, CNN, ResCNN, LSTM, Self-Attention

### Attack Methods (Phương pháp tấn công)
- **MIFGSM** (`utils/MIFGSM.py`): Momentum Iterative Fast Gradient Sign Method
- **SIM** (`utils/SIM.py`): Scale-Invariant Method
- **VMIFGSM** (`utils/VMIFGSM.py`): Variant MIFGSM
- **DGM** (`utils/DGM.py`): Diverse Gradient Method

### Core Utilities (Tiện ích cốt lõi)
- **utils.py**: Dataset loading, normalization, model initialization, flow rectification
- **rectify_adv_flows()**: Điều chỉnh các features phụ thuộc sau khi modify 4 features cấp 1

## Workflow chính

```
1. Dataset Preprocessing
   ↓
2. Training NIDS Models
   ↓
3. Generate AAT (Adversarial Attack Traffic)
   ↓
4. Map AAT to Real Packets
   ↓
5. Re-extract Features & Test
```

## Cấu hình quan trọng

- **STORAGE_DIR**: Đường dẫn đến thư mục chứa datasets và models (cấu hình trong `utils/utils.py`)
- **Device**: Mặc định sử dụng CUDA (`torch.device("cuda")`)
- **Batch Size**: Thường là 128
- **Epochs**: 10 epochs cho training
- **Attack Parameters**: 
  - Iterations: 7
  - Step size: 140

## Dependencies chính

- PyTorch 2.5.1+cu121
- Pandas 2.2.3
- NumPy 2.0.2
- Dpkt 1.9.8 (cho PCAP processing)
- CICFlowMeter (cho feature extraction từ PCAP)

## Cấu trúc thư mục

```
TransAdvAttForNIDS/
├── dataset_preprocess/     # Tiền xử lý dataset
├── train_NIDS/             # Training các NIDS models
├── generate_AAT/           # Generate Adversarial Attack Traffic
├── map_AAT_to_pkts/        # Map AAT về packets thực tế (SPTS)
├── TANTRA/                 # Implementation của TANTRA method
├── reproduce_experiments_results/  # Scripts tái tạo kết quả
└── utils/                  # Utilities và core functions
```
