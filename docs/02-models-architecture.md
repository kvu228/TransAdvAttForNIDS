# Kiến trúc Models

## Target Models (66 features)

**File**: `utils/target_models.py`

### MLP-t (Multi-Layer Perceptron)
```
Input (66) → Linear(256) → ReLU → Linear(256) → ReLU → 
Linear(256) → ReLU → Linear(256) → ReLU → Linear(2)
```

### CNN-t (Convolutional Neural Network)
```
Input (66) → Conv1d(1→64) → BatchNorm → LeakyReLU →
[Conv1d(64→64) → BatchNorm → LeakyReLU] × 5 →
Flatten → Linear(4224 → 2)
```

### ResCNN-t (Residual CNN)
```
Input (66) → Conv1d(1→64) → BatchNorm → LeakyReLU →
[Conv1d(64→64) → BatchNorm → LeakyReLU + Residual] × 5 →
Flatten → Linear(4224 → 2)
```

### LSTM-t
```
Input (66) → Pad to 80 → Reshape to (batch, 16, 5) →
LSTM(5→256, 3 layers) → Flatten → Linear(4096 → 2)
```

### SelfAttention-t
```
Input (66) → Pad to 80 → Reshape to (batch, 8, 10) →
Add Positional Encoding →
SelfAttention(10→10) →
Flatten → Linear(80→100) → LeakyReLU →
Linear(100→200) → LeakyReLU → Linear(200→2)
```

## Target Models (78 features)

**File**: `utils/target_models_with_78_fea.py`

Các models tương tự như trên nhưng với input size = 78 thay vì 66.

## Surrogate Models (60 features)

**File**: `utils/surrogate_models.py`

### MLP-s
```
Input (60) → Linear(256) → ReLU → Linear(256) → ReLU →
Linear(256) → ReLU → Linear(256) → ReLU → Linear(2)
```

### CNN-s
```
Input (60) → Conv1d(1→64) → BatchNorm → LeakyReLU →
[Conv1d(64→64) → BatchNorm → LeakyReLU] × 5 →
Flatten → Linear(3840 → 2)
```

### ResCNN-s
```
Input (60) → Conv1d(1→64) → BatchNorm → LeakyReLU →
[Conv1d(64→64) → BatchNorm → LeakyReLU + Residual] × 5 →
Flatten → Linear(3840 → 2)
```

### LSTM-s
```
Input (60) → Reshape to (batch, 12, 5) →
LSTM(5→256, 3 layers) → Flatten → Linear(3072 → 2)
```

### SelfAttention-s
```
Input (60) → Reshape to (batch, 6, 10) →
Add Positional Encoding →
SelfAttention(10→10) →
Flatten → Linear(60→100) → LeakyReLU →
Linear(100→200) → LeakyReLU → Linear(200→2)
```

## Surrogate Models với Variable Input Features

**File**: `utils/surrogate_model_with_var_input_fea.py`

### MLP-s_varfea
MLP với số features đầu vào có thể thay đổi (để test với các số features khác nhau).

## TANTRA LSTM Model

**File**: `utils/tantra.py`

### TantraLSTM
```
Input: Sequence of timestamps (window_size=150, feat_dim=2)
→ LSTM(input_size=2, hidden_size=32, num_layers=1)
→ Linear(32→8) → ReLU → Linear(8→1)
→ Output: Predicted next timestamp
```

**Parameters**:
- `ws`: Window size (default: 150)
- `feat_dim`: Feature dimension (default: 2)
- `seq_len`: ws + 1 = 151

## Model Initialization

### Function: `init_net()`
**File**: `utils/utils.py`

```python
init_net(model_type, model_name)
```

**Parameters**:
- `model_type`: 't' (target) hoặc 's' (surrogate)
- `model_name`: 'mlp_t', 'cnn_t', 'rescnn_t', 'lstm_t', 'Selfattention_t' (cho target)
                  'mlp_s', 'cnn_s', 'rescnn_s', 'lstm_s', 'Selfattention_s' (cho surrogate)

**Returns**: Initialized model (chưa load weights)

### Function: `load_net()`
**File**: `utils/utils.py`

```python
load_net(model_name, fp_model)
```

**Parameters**:
- `model_name`: Tên model (ví dụ: 'ton_mlp_s')
- `fp_model`: Đường dẫn đến file .pth

**Returns**: Model đã được load weights

**Lưu ý**: Function tự động detect model type từ tên và load đúng architecture.

## Model Training

### Standard Training
**File**: `train_NIDS/training.py`

- **Optimizer**: Adam (lr=0.001, betas=(0.99, 0.99))
- **Loss**: CrossEntropyLoss
- **Epochs**: 10
- **Batch Size**: 128
- **Device**: CUDA

### Adversarial Training
**File**: `train_NIDS/normal_adv_training.py`, `train_NIDS/adv_training_with_SPTS.py`

- **Optimizer**: Adam (lr=0.001, betas=(0.99, 0.99))
- **Loss**: Combined loss (normal + adversarial)
- **Epochs**: 10
- **Batch Size**: 128
- **Device**: CUDA

## Model Evaluation

### Function: `verifying.py`
**File**: `train_NIDS/verifying.py`

Đánh giá model trên test set:
- Load model
- Load test dataset
- Calculate accuracy
- Print results

## Model Storage

### Naming Convention
```
{dataset_name}_{model_name}_{model_type}.pth
```

**Examples**:
- `ton_mlp_t.pth`: TON_IoT dataset, MLP target model
- `ids18_cnn_s.pth`: CIC-IDS-2018 dataset, CNN surrogate model
- `advtrain_withSPTS_ton_mlp.pth`: Adversarial trained model với SPTS

### Storage Location
```
STORAGE_DIR/custom/pre-trained_models/
```

## Model Usage trong Attacks

### Trong Attack Generation
1. Load surrogate model đã được train
2. Set model to eval mode (trừ LSTM)
3. Generate adversarial examples
4. Test trên target models

### Trong Adversarial Training
1. Initialize model
2. Generate adversarial examples trong training loop
3. Update model với combined loss

## Device Management

- **Default**: CUDA (`torch.device("cuda")`)
- Models được move to device: `model.to(device)`
- Data được move to device: `data.to(device)`

## Model State

### Training Mode
- `model.train()`: Cho training
- BatchNorm và Dropout hoạt động

### Evaluation Mode
- `model.eval()`: Cho inference
- BatchNorm và Dropout không hoạt động
- **Lưu ý**: LSTM models thường được set to train mode ngay cả khi inference (để enable dropout nếu có)
