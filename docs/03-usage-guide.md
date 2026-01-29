# Hướng dẫn Sử dụng

## Setup Ban đầu

### 1. Cài đặt Dependencies
```bash
# Python 3.9.13
# PyTorch 2.5.1+cu121
pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install pandas==2.2.3 numpy==2.0.2 dpkt==1.9.8 matplotlib==3.9.4 seaborn==0.13.2
```

### 2. Cài đặt CICFlowMeter
- Download từ: https://github.com/UNBCIC/CICFlowMeter
- Cài đặt theo hướng dẫn
- Cấu hình `fp_cfm` trong các scripts:
  - `map_AAT_to_pkts/0_built_features_with_cfm_over_raw_att_pcap.py`
  - `map_AAT_to_pkts/4_re-extract_features_with_cfm.py`
  - `TANTRA/1_re-extract_features_with_cfm.py`

### 3. Download Datasets và Models
- Download từ các links trong README.md
- Extract vào một thư mục
- Cấu hình `STORAGE_DIR` trong `utils/utils.py` (line 9)

### 4. Tạo Thư mục Cần thiết
```bash
cd TransAdvAttForNIDS
mkdir output output2 output3
mkdir -p $STORAGE_DIR/adv_pcap
```

## Quick Start: Reproducing Results

### Chạy một Table/Figure
```bash
cd reproduce_experiments_results
python 5_2-Table_5.py
```

**Lưu ý**: Tất cả scripts trong thư mục này có thể chạy trực tiếp không cần parameters.

## Custom Training Workflow

### Bước 1: Preprocess Dataset (Nếu cần)

**Chỉnh sửa scripts trong `dataset_preprocess/`**:
```python
# Ví dụ: build_minmax.py
fp_input = '/path/to/training_dataset.csv'
fp_output = '/path/to/output_minmax.csv'
```

Chạy các scripts theo thứ tự:
```bash
cd dataset_preprocess
python build_input_features.py
python build_minmax.py
python divide_dataset_into_target_and_surrogate.py
python split_dataset_into_train_and_test.py
python sampling_training_dataset.py
```

### Bước 2: Training Model

**Chỉnh sửa `train_NIDS/training.py`**:
```python
dataset_name = 'ton'  # hoặc 'ids18'
model_name = 'mlp'    # hoặc 'cnn', 'rescnn', 'lstm', 'Selfattention'
model_type = 't'      # 't' cho target, 's' cho surrogate
```

**Chạy training**:
```bash
cd train_NIDS
python training.py
```

**Output**: Model được lưu tại `STORAGE_DIR/custom/pre-trained_models/{dataset_name}_{model_name}_{model_type}.pth`

### Bước 3: Verify Model
```bash
python verifying.py
```

### Bước 4: Generate AAT

**Chỉnh sửa `generate_AAT/generate_aat.py`**:
```python
dataset_name = 'ton'
model_name = 'mlp'
model_type = 's'
attack_name = 'MIFGSM'  # hoặc 'SIM', 'VMIFGSM', 'DGM'
iteration = 7
step_size = 140
```

**Chạy generate**:
```bash
cd generate_AAT
python generate_aat.py
```

**Output**: `aat.csv` tại `STORAGE_DIR/custom/output/`

### Bước 5: Test AAT
```bash
python test_aat.py
```

## Adversarial Training

### Normal Adversarial Training

**Chỉnh sửa `train_NIDS/normal_adv_training.py`**:
```python
dataset_name = 'ton'
model_name = 'mlp'
model_type = 't'
```

**Chạy**:
```bash
cd train_NIDS
python normal_adv_training.py
```

### Adversarial Training với SPTS

**Chỉnh sửa `train_NIDS/adv_training_with_SPTS.py`**:
```python
dataset_name = 'ton'
model_name = 'mlp'
```

**Chạy**:
```bash
python adv_training_with_SPTS.py
```

## Mapping AAT to Real Packets

### Yêu cầu
- CICFlowMeter đã được cài đặt
- Có raw attack PCAP files
- Có quyền sudo

### Workflow

#### Bước 1: Extract Features từ Raw PCAP
**Chỉnh sửa `map_AAT_to_pkts/0_built_features_with_cfm_over_raw_att_pcap.py`**:
```python
fp_cfm = '/path/to/CICFlowMeter'
fp_pcap_dir = '/path/to/attack_pcap_files'
fp_output = 'output/raw_att.csv'
```

**Chạy**:
```bash
cd map_AAT_to_pkts
sudo python 0_built_features_with_cfm_over_raw_att_pcap.py
```

#### Bước 2: Generate AAT
**Chỉnh sửa `map_AAT_to_pkts/1_generate_aat.py`**:
```python
dataset_name = 'ton'
model_name = 'mlp'
model_type = 's'
attack_name = 'MIFGSM'
```

**Chạy**:
```bash
python 1_generate_aat.py
```

#### Bước 3: Process AAT
```bash
python 2_process_aat.py
```

#### Bước 4: Modify PCAP Files
```bash
python 3_modify_pcap.py
```

#### Bước 5: Re-extract Features
**Chỉnh sửa `map_AAT_to_pkts/4_re-extract_features_with_cfm.py`**:
```python
fp_cfm = '/path/to/CICFlowMeter'
```

**Chạy**:
```bash
sudo python 4_re-extract_features_with_cfm.py
```

#### Bước 6: Test
```bash
python 5_test_aat.py
```

## TANTRA Workflow

### Bước 1: Modify Packets
```bash
cd TANTRA
python 0_modify_pkts.py
```

### Bước 2: Re-extract Features
**Chỉnh sửa `TANTRA/1_re-extract_features_with_cfm.py`**:
```python
fp_cfm = '/path/to/CICFlowMeter'
```

**Chạy**:
```bash
sudo python 1_re-extract_features_with_cfm.py
```

### Bước 3: Test
```bash
python 2_test_aat.py
```

## Common Parameters

### Model Selection
```python
# Dataset
dataset_name = 'ton'      # hoặc 'ids18'

# Model Architecture
model_name = 'mlp'        # 'mlp', 'cnn', 'rescnn', 'lstm', 'Selfattention'

# Model Type
model_type = 't'          # 't' (target) hoặc 's' (surrogate)
```

### Attack Parameters
```python
# Attack Method
attack_name = 'MIFGSM'    # 'MIFGSM', 'SIM', 'VMIFGSM', 'DGM'

# Attack Settings
iteration = 7             # Số iterations
step_size = 140           # Step size
```

### Training Parameters
```python
lr = 0.001                # Learning rate
epoch = 10                # Số epochs
batch_size = 128          # Batch size
```

## Troubleshooting

### Lỗi: CUDA out of memory
**Giải pháp**:
- Giảm `batch_size` trong scripts
- Hoặc sử dụng CPU: `device = torch.device("cpu")`

### Lỗi: CICFlowMeter không chạy
**Giải pháp**:
- Kiểm tra đường dẫn `fp_cfm`
- Đảm bảo chạy với sudo
- Hoặc cấu hình sudo không cần password cho CFM

### Lỗi: File not found
**Giải pháp**:
- Kiểm tra `STORAGE_DIR` trong `utils/utils.py`
- Kiểm tra các đường dẫn trong scripts
- Đảm bảo datasets và models đã được download

### Lỗi: NaN values
**Giải pháp**:
- Kiểm tra min-max values (có thể có min == max)
- Kiểm tra data quality
- Sử dụng `.fillna(0)` sau normalization

## Best Practices

### 1. Backup Models
Luôn backup models sau khi training:
```bash
cp $STORAGE_DIR/custom/pre-trained_models/*.pth /backup/location/
```

### 2. Logging
Thêm logging để track progress:
```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info(f"Processing batch {i}/{total}")
```

### 3. Validation
Luôn verify model sau khi training:
```bash
python verifying.py
```

### 4. Experiment Tracking
Ghi lại các parameters đã sử dụng:
- Dataset name
- Model architecture
- Attack method và parameters
- Results

### 5. Memory Management
- Sử dụng chunked reading cho large datasets
- Clear cache sau mỗi batch
- Monitor GPU memory usage

## Example Scripts

### Complete Training và Testing Pipeline
```python
# train_and_test.py
import os
import sys
sys.path.append('train_NIDS')
sys.path.append('generate_AAT')

from train_NIDS.training import main as train
from generate_AAT.generate_aat import main as generate_aat

# Training
train('ton', 'mlp', 's', ...)

# Generate AAT
generate_aat('ton', 'mlp', 's', 'MIFGSM', 7, 140, ...)
```

## Performance Tips

1. **Use GPU**: Luôn sử dụng CUDA nếu có
2. **Batch Processing**: Sử dụng batch size phù hợp (128 thường tốt)
3. **Chunked Reading**: Đọc large files theo chunks
4. **Pre-compute**: Pre-compute min-max values và features
5. **Cache**: Cache intermediate results nếu có thể
