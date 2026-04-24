# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research framework for the paper "Exploring Transferable Adversarial Attacks for Deep Learning-based Network Intrusion Detection" (JNCA, 2025). Generates adversarial attack traffic (AAT) against deep learning NIDS models and evaluates cross-model transferability.

Datasets: TON_IoT and CIC-IDS-2018. External data/models stored outside repo at `STORAGE_DIR` (configured in `utils/utils.py` line 27).

## Setup

```bash
uv sync                          # install dependencies (Python 3.12+, PyTorch, Pandas, etc.)
mkdir output output2 output3     # required intermediate output dirs in project root
```

`STORAGE_DIR` must point to a directory containing `dataset/`, `custom/pre-trained_models/`, `custom/output/`, and `adv_pcap/`. Pre-trained models and datasets are downloaded separately (see README.md Section 2).

## Running Scripts

No test suite or linter. All scripts are run directly with `python <script>.py`. Parameters are configured by editing variables at the top of each script (dataset name, model name, attack method, etc.).

```bash
# Reproduce paper results (Tables 5-18, Figures 2-9) -- no params needed
python reproduce_experiments_results/5_2-Table_5.py

# Train a model (edit training.py lines 42-43 for model_type/dataset)
python train_NIDS/training.py
python train_NIDS/verifying.py

# Generate adversarial traffic (edit lines 74-86 for params)
python generate_AAT/generate_aat.py
python generate_AAT/test_aat.py

# Adversarial training variants
python train_NIDS/normal_adv_training.py
python train_NIDS/adv_training_with_SPTS.py

# Web dashboard
streamlit run web_app/app.py
```

SPTS packet mapping (`map_AAT_to_pkts/`) and TANTRA (`TANTRA/`) workflows require CICFlowMeter installed externally and sudo privileges. Scripts in those dirs are numbered 0-5 and must run in order. CFM path is set via `fp_cfm` in each script.

## Architecture

### Feature Hierarchy (core concept)

Attacks only modify 4 **Level-1** features directly: `Fwd Pkt Len Max`, `Fwd Pkt Len Min`, `Fwd IAT Max`, `Fwd IAT Min`. **Level-2** features (means, stds, totals like `Flow Duration`, `TotLen Fwd Pkts`, `Fwd Pkt Len Mean/Std`) and **Level-3** features (`Flow Byts/s`, `Flow Pkts/s`) are automatically recalculated by `rectify_adv_flows()` in `utils/utils.py`.

### Model Types

- **Target models** (`utils/target_models.py`): 66-feature input. Architectures: MLP, CNN, ResCNN, LSTM, SelfAttention. 78-feature variants in `target_models_with_78_fea.py`.
- **Surrogate models** (`utils/surrogate_models.py`): 60-feature input (simulates attacker's limited knowledge). Same architectures. Variable-feature variant in `surrogate_model_with_var_input_fea.py`.
- Naming convention: suffix `_t` = target, `_s` = surrogate (e.g., `mlp_t`, `cnn_s`).

### Attack Methods

All in `utils/`: `MIFGSM.py`, `SIM.py`, `VMIFGSM.py`, `DGM.py`. Momentum-based iterative perturbation methods. Default: 7 iterations, step_size=140. All return `(adv_flows, (time, payload_delta, iat_delta))`.

### Key Functions in `utils/utils.py`

- `STORAGE_DIR`: global path to external data (line 27)
- `CustomDataset(fp_data, fp_minmax, fp_fea)`: PyTorch Dataset. Loads CSV, normalizes, converts labels (Benign=0, Attack=1)
- `normalize_df(df, df_minmax)`: min-max normalization to [0,1]
- `init_net(model_type, model_name)`: creates uninitialized model (`'t'` or `'s'`)
- `load_net(model_name, fp_model)`: loads pre-trained weights. Overloaded: `load_net(78, model_name, fp_model)` for 78-feature targets
- `rectify_adv_flows(adv_flows, flows, pert)`: recalculates Level-2/3 features after Level-1 modification. Called by all attack methods

### Data Pipeline

Raw PCAP -> CICFlowMeter -> CSV features -> min-max normalize -> PyTorch tensor -> model inference. Feature lists and min-max values stored as CSVs in `STORAGE_DIR/dataset/` (`fea_t.csv`, `fea_s.csv`, `*_minmax_*.csv`).

### Directory Layout

- `utils/` - Models, attacks, plotting, core utilities
- `train_NIDS/` - Training scripts (standard, adversarial, SPTS adversarial)
- `generate_AAT/` - AAT generation and evaluation
- `reproduce_experiments_results/` - Paper result reproduction (run directly, no params)
- `map_AAT_to_pkts/` - SPTS: map AAT back to PCAP packets (numbered steps 0-5)
- `TANTRA/` - Timestamp-based attack via LSTM (numbered steps 0-2)
- `dataset_preprocess/` - Feature extraction, min-max computation, dataset splitting, oversampling
- `web_app/` - Streamlit dashboard (`app.py` + `attack_service.py`)
- `database/` - SQLite logging (`db_manager.py`, creates `simulation_logs.db`)

## Code Conventions

- Script configuration via top-of-file variables, not CLI args
- Bilingual comments (Vietnamese and English)
- Model files saved as `.pth` in `STORAGE_DIR/custom/pre-trained_models/`
- No fixed random seeds (intentional -- see README)
- CUDA device assumed available; scripts default to GPU
