# Các Workflow chính

## Workflow 1: Reproducing Paper Results

**Mục đích**: Tái tạo các kết quả trong bài báo nghiên cứu.

### Bước 1: Setup
1. Download pre-processed datasets và pre-trained models từ các links trong README
2. Extract và đặt vào `STORAGE_DIR`
3. Cấu hình `STORAGE_DIR` trong `utils/utils.py`
4. Tạo các thư mục: `output`, `output2`, `output3`

### Bước 2: Chạy Scripts
```bash
cd reproduce_experiments_results
python 5_2-Table_5.py        # Table 5
python 5_3_1-Table_6.py      # Table 6
# ... các scripts khác
```

**Lưu ý**: Tất cả scripts có thể chạy trực tiếp không cần parameters.

## Workflow 2: Custom Training và AAT Generation

**Mục đích**: Train model riêng và generate AAT tùy chỉnh.

### Bước 1: Dataset Preprocessing (Nếu cần)
```bash
cd dataset_preprocess
# Chỉnh sửa các scripts để set input/output paths
python build_input_features.py
python build_minmax.py
python divide_dataset_into_target_and_surrogate.py
python split_dataset_into_train_and_test.py
python sampling_training_dataset.py
```

### Bước 2: Training Target NIDS hoặc Surrogate Model
```bash
cd train_NIDS
# Chỉnh sửa parameters trong training.py:
# - dataset_name: 'ton' hoặc 'ids18'
# - model_name: 'mlp', 'cnn', 'rescnn', 'lstm', 'Selfattention'
# - model_type: 't' (target) hoặc 's' (surrogate)

python training.py
```

**Output**: Model được lưu tại `STORAGE_DIR/custom/pre-trained_models/{dataset_name}_{model_name}_{model_type}.pth`

### Bước 3: Verify Model
```bash
python verifying.py
```

### Bước 4: Generate AAT
```bash
cd ../generate_AAT
# Chỉnh sửa parameters trong generate_aat.py:
# - dataset_name, model_name, model_type
# - attack_name: 'MIFGSM', 'SIM', 'VMIFGSM', 'DGM'
# - iteration: 7 (default)
# - step_size: 140 (default)

python generate_aat.py
```

**Output**: `aat.csv` tại `STORAGE_DIR/custom/output/`

### Bước 5: Test AAT
```bash
python test_aat.py
```

## Workflow 3: Adversarial Training

**Mục đích**: Train model với adversarial examples để tăng robustness.

### Option A: Normal Adversarial Training
```bash
cd train_NIDS
# Chỉnh sửa parameters trong normal_adv_training.py
python normal_adv_training.py
```

**Cơ chế**:
- Generate adversarial examples trong quá trình training
- Modify tất cả features có thể (trong normalized space)
- Loss = 0.9 * normal_loss + 0.1 * adv_loss

### Option B: Adversarial Training với SPTS
```bash
cd train_NIDS
# Chỉnh sửa parameters trong adv_training_with_SPTS.py
python adv_training_with_SPTS.py
```

**Cơ chế**:
- Chỉ modify 4 features cấp 1
- Tự động điều chỉnh features phụ thuộc
- Loss = 0.8 * normal_loss + 0.8 * adv_loss

**Output**: Model được lưu với prefix `advtrain_withSPTS_` hoặc `normal_advtrain_`

## Workflow 4: Mapping AAT to Real Packets (SPTS)

**Mục đích**: Convert AAT về các gói tin mạng thực tế.

**Yêu cầu**: CICFlowMeter (CFM) phải được cài đặt và cấu hình.

### Bước 1: Setup CFM Path
Chỉnh sửa `fp_cfm` trong:
- `map_AAT_to_pkts/0_built_features_with_cfm_over_raw_att_pcap.py`
- `map_AAT_to_pkts/4_re-extract_features_with_cfm.py`

### Bước 2: Extract Features từ Raw Attack PCAP
```bash
cd map_AAT_to_pkts
sudo python 0_built_features_with_cfm_over_raw_att_pcap.py
```

**Lưu ý**: Cần chạy với sudo vì CFM yêu cầu admin privileges.

**Output**: `raw_att.csv` tại `output/`

### Bước 3: Generate AAT
```bash
python 1_generate_aat.py
```

**Output**: `raw_aat.csv` tại `output/` (chỉ chứa Flow ID, IPs, Ports, Protocol, và 4 features cấp 1)

### Bước 4: Process AAT
```bash
python 2_process_aat.py
```

**Mục đích**: Tính toán differences giữa original và adversarial traffic.

### Bước 5: Modify PCAP Files
```bash
python 3_modify_pcap.py
```

**Mục đích**: Modify packets trong PCAP files dựa trên AAT.

### Bước 6: Re-extract Features
```bash
sudo python 4_re-extract_features_with_cfm.py
```

**Output**: Features mới sau khi modify packets.

### Bước 7: Test
```bash
python 5_test_aat.py
```

**Mục đích**: Đánh giá attack success rate với features đã re-extract.

## Workflow 5: TANTRA Method

**Mục đích**: Generate AAT bằng cách modify timestamps sử dụng TANTRA.

### Bước 1: Modify Packets
```bash
cd TANTRA
python 0_modify_pkts.py
```

**Cơ chế**:
- Load pre-trained TANTRA LSTM model
- Predict timestamps cho attack packets
- Modify timestamps trong PCAP files

### Bước 2: Re-extract Features
```bash
sudo python 1_re-extract_features_with_cfm.py
```

**Lưu ý**: Cần cấu hình `fp_cfm` trong script này.

### Bước 3: Test
```bash
python 2_test_aat.py
```

## Workflow 6: Complete Pipeline (Từ đầu đến cuối)

### Scenario: Train model mới và test với real packets

1. **Preprocess Dataset**
   ```bash
   cd dataset_preprocess
   # Chỉnh sửa và chạy các preprocessing scripts
   ```

2. **Train Surrogate Model**
   ```bash
   cd ../train_NIDS
   python training.py  # Set model_type='s'
   ```

3. **Generate AAT**
   ```bash
   cd ../generate_AAT
   python generate_aat.py
   python test_aat.py
   ```

4. **Map to Real Packets**
   ```bash
   cd ../map_AAT_to_pkts
   # Chạy các bước từ 0 đến 5
   ```

5. **Evaluate**
   - Kiểm tra attack success rate
   - So sánh với results từ paper

## Common Parameters

### Model Parameters
- `dataset_name`: 'ton' hoặc 'ids18'
- `model_name`: 'mlp', 'cnn', 'rescnn', 'lstm', 'Selfattention'
- `model_type`: 't' (target) hoặc 's' (surrogate)

### Attack Parameters
- `attack_name`: 'MIFGSM', 'SIM', 'VMIFGSM', 'DGM'
- `iteration`: Số iterations (default: 7)
- `step_size`: Step size (default: 140)

### Training Parameters
- `lr`: Learning rate (default: 0.001)
- `epoch`: Số epochs (default: 10)
- `batch_size`: Batch size (default: 128)

### Paths
- `STORAGE_DIR`: Đường dẫn đến datasets và models (cấu hình trong `utils/utils.py`)
- `fp_fea`: Đường dẫn đến file features: `{STORAGE_DIR}/dataset/fea_{model_type}.csv`
- `fp_minmax`: Đường dẫn đến min-max values: `{STORAGE_DIR}/dataset/{dataset_name}_minmax_{model_type}.csv`
- `fp_dataset`: Đường dẫn đến dataset: `{STORAGE_DIR}/dataset/{dataset_name}_sam_train_{model_type}.csv`
- `fp_model`: Đường dẫn đến model: `{STORAGE_DIR}/custom/pre-trained_models/{dataset_name}_{model_name}_{model_type}.pth`
