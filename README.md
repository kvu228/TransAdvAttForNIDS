# TransAdvAttForNIDS

This repository contains the official implementation of our paper:

> Zhongshu Mao, Yiqin Lu, Zhe Cheng, and Kaiqiong Chen.  
> **Exploring Transferable Adversarial Attacks for Deep Learning-based Network Intrusion Detection**  
> Journal of Network and Computer Applications (JNCA), 2025, 104255.

# 1 Introduction

This project focuses on adversarial attack traffic (AAT) generation methods for Network Intrusion Detection Systems (NIDS). It aims to evaluate the robustness and generalization ability of deep learning models under adversarial perturbations in structured data scenarios. To make the repository easier to navigate and to help readers reproduce the experiments in the paper, we grouped all experiments into three categories:
1. Reproducing the manuscript’s results.
    
    We provide pre-processed datasets and the corresponding generated AAT, enabling users to reproduce almost all results reported in the manuscript.
    
2. Custom training and AAT generation.
    
    To address concerns about our prepared data, we provide scripts that enable users to train their own models and generate AAT. For batch training across architectures and datasets, use the **`reimplemented_models`** directory (Section 4.4). Although we did not fix random seeds, repeated experiments confirm that omitting a seed does not affect the manuscript’s core findings or conclusions.
    
3. Mapping AAT to practical packets (including TANTRA).
    
    This set of experiments shows how to convert the generated AAT back into actual network packets and also includes scripts for generating AAT with TANTRA.
    
In light of the considerations above, this README is organized as follows: Section 2 covers the preliminaries—environment setup, dataset downloads, and auxiliary tools; Section 3 explains how to reproduce the manuscript’s results; Sections 4 and 5 describe custom model training and AAT generation, respectively; Section 6 describes the original manuscript workflow involving CICFlowMeter (**not reimplemented in this codebase**—see item 3 in Section 2); Section 7 offers supplementary notes; and Section 8 documents the Streamlit dashboard and how to interpret its outputs.

# 2 Preliminaries

1. Environment setup
    - Ubuntu 18.04.6 LTS
    - Python 3.9.13
    - PyTorch 2.5.1+cu121
    - Pandas 2.2.3
    - Numpy 2.0.2
    - Dpkt 1.9.8
    - Matplotlib 3.9.4
    - Seaborn 0.13.2
2. Dataset：
    - [CIC-IDS-2018](https://www.unb.ca/cic/datasets/ids-2018.html)
    - [TON_IoT](https://research.unsw.edu.au/projects/toniot-datasets)
3. **CICFlowMeter (CFM) — reimplementation status.** The **[CICFlowMeter](https://github.com/UNBCIC/CICFlowMeter)**-based pipeline (PCAP feature extraction, mapping AAT to packets, re-extraction after edits—corresponding to Section 6 and related steps in the paper) **is not reimplemented in the current codebase.** You can skip CFM installation entirely if you only work with CSV features (training, AAT generation on tabular data, Streamlit in Section 8).

   Section 6 below is kept as a **description of the original manuscript workflow** for reference; do not expect those steps to run end-to-end on this repository without additional environment work and code changes.

   **When CFM is unnecessary:** training (Section 4, including `reimplemented_models`), CSV-based AAT generation (Section 5), and the Streamlit app (Section 8)—as long as the required CSVs exist under `STORAGE_DIR`.

   **Technical reference (upstream project):** if you extend the PCAP pipeline yourself, install CFM on Linux/WSL2, set `fp_cfm` in scripts under `map_AAT_to_pkts/` and `TANTRA/`, and handle elevated privileges as required. Detailed sudo/CFM setup from the original project README is omitted here because the maintained reproduction path does not depend on CFM.

4. The pre-trained models and pre-processed dataset can be downloaded from the following 4 addresses:

   1. [address_1](https://zenodo.org/records/15597259?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjA3MTgyNzgzLWI5ZjAtNGIwYi04NGZkLTRlNzhiZDg0NGE4YiIsImRhdGEiOnt9LCJyYW5kb20iOiJlMjRmY2JmNTY5MzQwNTdmZmVmZjY2M2NkOGE3ODQ5MiJ9.W0ax17EhKsZdmX-OMkZy0xczh--MgRQn0V9KRPckV0D_SsuHK04R6mewrOJ3uZHc93woNOL9G1Ock3_9_SUsCw)
   2. [address_2](https://drive.google.com/drive/folders/1Ne3s40AgGe0H6tMP0gug-tsCc_zw7qpb?usp=sharing)
   3. [address_3](https://drive.google.com/drive/folders/1za3qA1g1WNlkt5MO-PDICDARrKjtZZqr?usp=sharing)
   4. [address_4](https://drive.google.com/drive/folders/1HYvxwKOdRaHRy9dmaU7goXonHNyZAVo9?usp=sharing)
   5. There are 11 archive files in total and extract each one individually. Note that **5_4_4.part1.rar** and **5_4_4.part2.rar** belong to the same archive; you can extract them with: `unrar x 5_4_4.rar`. 
   6. After extraction, create a new, empty directory anywhere (you can choose any name), then move the 10 extracted folders into this directory. Then, open **TransAdvAttForNIDS/utils/utils.py** and update the **STORAGE_DIR** variable (line 9) to the path of that new directory. 
   7. Within the **STORAGE_DIR** directory, create an empty folder named **adv_pcap**.
   8. Before running the code, create three empty directories—named `output`, `output2`, and `output3`—inside the `TransAdvAttForNIDS/` directory. These empty directories will store intermediate files generated during execution.
   9. We provide the randomly oversampled TON_IoT datasets for both the target NIDSs and the surrogate models. However, the CIC-IDS-2018 oversampled files are not included. If you wish to train on CIC-IDS-2018, be sure to apply random oversampling to **TransAdvAttForNIDS/dataset/ids18_train_s.csv** and **TransAdvAttForNIDS/dataset/ids18_train_t.csv** using the script at **TransAdvAttForNIDS/dataset_preprocess/sampling_training_dataset.py** (see Section 7, **Supplementary Notes**). The oversampled outputs need to be saved in the `STORAGE_DIR/dataset` directory as **ids18_sam_train_s.csv** and **ids18_sam_train_t.csv**. 

# 3 Reproducing the manuscript’s results

We have included all data generated as well as the corresponding scripts. Simply running these scripts will recreate every table and figure reported, including Tables 5–18 and Figures 2–9.

1. Taking Table 5 as an example, the detailed steps are as follows:
    1. Change to the target directory.
        
        ```bash
        cd TransAdvAttForNIDS/reproduce_experiments_results
        ```
        
    2. Run the script.
        
        ```bash
        python 5_2-Table_5.py
        ```
2. All scripts in the `reproduce_experiments_results` directory can be run directly without any additional parameters.
3. Please note that the results for Tables 15 and 16 may be similar to, but not identical to, those presented in the paper. These tables require real-time measurement of AAT generation time, and the generation process itself involves inherent randomness. So, each runtime can produce slightly different outcomes.
   
# 4 Training NIDSs
We do not introduce new theories or methods for improving NIDS performance, so training the NIDS is not a contribution of our paper. Nevertheless, to eliminate potential concerns, we have supplied training scripts. Unlike “Reproducing the manuscript’s results”, we have not written a separate script for every target NIDS or surrogate model. Because the training procedure is the same, we provide a demo script that trains MLP-t and MLP-s on the TON_IoT dataset. Complete implementations of the other model architectures are available in `TransAdvAttForNIDS/utils/surrogate_model_with_var_input_fea.py` and `TransAdvAttForNIDS/utils/target_models_with_78_fea.py`. If you wish to train models with different architectures, modify the script accordingly. For batch training of all architectures on TON_IoT and CIC-IDS-2018, prefer Section 4.4 (`reimplemented_models`).

## 4.1 Training target NIDSs and surrogate models

1. Change to the target directory.
    
    ```bash
    cd TransAdvAttForNIDS/train_NIDS
    ```
    
2. Run the script.
    
    ```bash
    python training.py
    ```
    
3. Test the trained model.
    
    ```bash
    python verifying.py
    ```
    
4. The default dataset is TON_IoT, and the default model is the MLP-t with 66 input features. To switch to an MLP-s, set `model_type = 's'` in **training.py** (line 43). By default, the MLP-s uses 60 input features, matching the configuration described in Section 5.3.1 of the manuscript.
5. After 10 epochs, the model will be saved to `STORAGE_DIR/custom/pre-trained_models`.
6. If you want to train an MLP-t with 78 input features or a surrogate model with fewer input features, you must adjust the model architecture. The corresponding definitions are located in **TransAdvAttForNIDS/utils/target_models_with_78_fea.py** and **TransAdvAttForNIDS/utils/surrogate_model_with_var_input_fea.py**, respectively.
   
## 4.2 Normal adversarial training

1. Change to the target directory.
    
    ```bash
    cd TransAdvAttForNIDS/train_NIDS
    ```
    
2. Run the script.
    
    ```bash
    python normal_adv_training.py
    ```
    
3. After 10 training epochs, the model will be saved to `STORAGE_DIR/custom/pre-trained_models`.
4. The default configurations are training an MLP-t with 66 input features on the TON_IoT. If you want to train a different model or switch to the CIC-IDS-2018, adjust the model name, dataset path, min–max scaling values, and input features. These parameters are clearly identified in lines 62–69 of the script.
   
## 4.3 Adversarial training with SPTS

1. Change to the target directory.
    
    ```bash
    cd TransAdvAttForNIDS/train_NIDS
    ```
    
2. Run the script.
    
    ```bash
    python adv_training_with_SPTS.py
    ```
    
3. After 10 training epochs, the model will be saved to `STORAGE_DIR/custom/pre-trained_models`.

4. The default configurations are training an MLP-t with 66 input features on the TON_IoT. If you want to train a different model or switch to the CIC-IDS-2018, adjust the model name, dataset path, min–max scaling values, and input features. These parameters are clearly identified in lines 84–91 of the script.

## 4.4 Batch training with `reimplemented_models`

The directory `TransAdvAttForNIDS/reimplemented_models/` contains scripts to retrain all architectures (MLP, CNN, ResCNN, LSTM, Self-Attention) for both target (`t`) and surrogate (`s`) roles on **TON_IoT** (`ton`) and **CIC-IDS-2018** (`ids18`), compatible with `init_net` / `load_net` in `utils/`.

**Data requirements** (under `STORAGE_DIR/dataset/` after setting `STORAGE_DIR` in `utils/utils.py`):

- `fea_t.csv`, `fea_s.csv`
- `{ton|ids18}_minmax_t.csv`, `{ton|ids18}_minmax_s.csv`
- `{ton|ids18}_sam_train_t.csv`, `{ton|ids18}_sam_train_s.csv` (for IDS-2018, oversample and name as in Section 2, item 9)

**Run from the repository root** (`TransAdvAttForNIDS/`):

1. **Standard (cross-entropy) training only:**
   ```bash
   python reimplemented_models/train_all_standard_models.py
   ```
   Common options: `--datasets ton ids18`, `--epochs 10`, `--architectures mlp cnn ...`, `--model-types t s`, `--output-dir /path/to/save`.

2. **Adversarial training with SPTS** (MI-FGSM + Level-1 mask):
   ```bash
   python reimplemented_models/train_all_adv_with_spts.py
   ```

3. **Standard adversarial training** (without SPTS):
   ```bash
   python reimplemented_models/train_all_adv_normal.py
   ```

4. **All three stages in sequence** (standard → SPTS adv → normal adv):
   ```bash
   ./reimplemented_models/run_train_all.sh
   ```
   Pass-through args example: `./reimplemented_models/run_train_all.sh --epochs 5`.

5. **Same as (4), with each stage parallelized across GPUs:**
   ```bash
   ./reimplemented_models/run_train_all_parallel.sh 0 1 2 3
   ```

**Multi-GPU sharding** (one process per GPU, logs under `reimplemented_models/parallel_logs/`):

```bash
./reimplemented_models/run_train_standard_parallel.sh 0 1 2 3
./reimplemented_models/run_train_adv_spts_parallel.sh 0 1 2 3
./reimplemented_models/run_train_adv_normal_parallel.sh 0 1 2 3
```

Or call `./reimplemented_models/parallel_train_on_gpus.sh <script_name.py> <gpu_ids...> -- <python_args>`.

**Checkpoint naming** (default output directory is `reimplemented_models/` unless you set `--output-dir`):

- Standard: `{dataset}_{arch}_{t|s}.pth` (e.g. `ton_mlp_t.pth`)
- Adv + SPTS: `advtrain_withSPTS_{dataset}_{arch}_{t|s}.pth`
- Normal adv: `normal_advtrain_{dataset}_{arch}_{t|s}.pth`

**Using checkpoints with Streamlit (Section 8):** the app scans `STORAGE_DIR/custom/pre-trained_models/` and classifies models by the `_t` / `_s` suffix in the filename (without `.pth`). Copy or symlink the `.pth` files you need there (or set `--output-dir` to that folder when training). You also need `{dataset}_raw_att.csv` in `STORAGE_DIR/dataset/` for the selected dataset.

# 5 Generating AAT

1. Change to the target directory.
    
    ```bash
    cd TransAdvAttForNIDS/generate_AAT
    ```
    
2. Run the script.
    
    ```bash
    python generate_aat.py
    ```
    
3. Test the AAT.
    
    ```bash
    python test_aat.py
    ```
    
4. By default, the script uses the TON_IoT, an MLP-s surrogate model, the MI-FGSM attack with 7 iterations, and a step size of 140. These parameters are defined in lines 74–86 and can be modified to generate a custom AAT.
5. The generated AAT is saved as **aat.csv** in `STORAGE_DIR/custom/output`.

# 6 Mapping AAT to practical packets (manuscript workflow — CFM not reimplemented here)

**Note:** This section describes the **original manuscript** procedure (map AAT to PCAP, use CICFlowMeter for feature extraction/re-extraction). As stated in Section 2, item 3, the **CICFlowMeter pipeline is not reimplemented** in this codebase; do not expect these steps to run end-to-end without extra setup and modifications.

This set of experiments maps the generated AAT back to practical packets and then re-extracts features with CFM. Note that the attack flows were further filtered, leaving a total of 36,980 flows (see the first paragraph of Section 5.3.2 in the manuscript). Because the procedure involves multiple steps, the scripts are numbered to indicate the correct execution order.

## 6.1 For SPTS

1. Change to the target directory.
    
    ```bash
    cd TransAdvAttForNIDS/map_AAT_to_pkts
    ```
    
2. Extract features with CFM. While CFM yields the complete feature set used by the target NIDS, we intentionally employ a subset to simulate the attacker’s limited knowledge.
    
    ```bash
    python 0_built_features_with_cfm_over_raw_att_pcap.py
    ```
    
3. Generate the AAT. Unlike the procedure in Section 5 (Generating AAT), the AAT produced here retains only the essential fields—‘Flow ID’, ‘Src IP’, ‘Src Port’, ‘Dst IP’, ‘Dst Port’, ‘Protocol’, 'Fwd Pkt Len Max', 'Fwd Pkt Len Min', 'Fwd IAT Max', and 'Fwd IAT Min’.
    
    ```bash
    python 1_generate_aat.py
    ```
    
4. Process the generated AAT by computing the differences between the original traffic and the AAT.
    
    ```bash
    python 2_process_aat.py
    ```
    
5. Modify packets.
    
    ```bash
    python 3_modify_pcap.py
    ```
    
6. Re-extract features with CFM.
    
    ```bash
    python 4_re-extract_features_with_cfm.py
    ```
    
7. Test the re-extracted features.
    
    ```bash
    python 5_test_aat.py
    ```
    
8. By default, the scripts use the MLP-s (trained in Section 4) surrogate model and MI-FGSM attack with 7 iterations, and a step size of 140. To customize these settings, modify lines 93–105 in **1_generate_aat.py**.

## 6.2 For TANTRA

TANTRA trains an LSTM to learn normal traffic patterns. We adopt exactly the architecture and hyperparameters specified in the original paper, so no additional training script is included; instead, we provide the pre-trained model.

1. Change to the target directory.
    
    ```bash
    cd TransAdvAttForNIDS/TANTRA
    ```
    
2. Modify the *Timestamp* of each attack packet according to the trained LSTM model.
    
    ```bash
    python 0_modify_pkts.py
    ```
    
3. Re-extract features with CFM.
    
    ```bash
    python 1_re-extract_features_with_cfm.py
    ```
    
4. Test the re-extracted features.
    
    ```bash
    python 2_test_aat.py
    ```

# 7 Supplementary Notes

We also supply dataset-preprocessing scripts located in the `TransAdvAttForNIDS/dataset_preprocess` directory.

1. The `divide_dataset_into_target_and_surrogate.py` script divides the dataset into two subsets—one for the target NIDS and one for the surrogate model.
2. The `split_dataset_into_train_and_test.py` script splits each subset into training and test sets according to a specified ratio.
3. The `sampling_training_dataset.py` script performs random oversampling on the training dataset.
4. The script `build_input_features.py` extracts the model’s input features.
5. The `build_minmax.py` script extracts the maximum and minimum values from the training dataset; these values are then used for normalization.
6. Please note that the above scripts cannot be run directly. To execute them, you must specify the input and output file paths within each script. Their code and logic are straightforward, so no further explanation is needed.
7. For the TON_IoT, we recommend using the pre‐labeled CSV files we provide, since extracting features from the raw PCAP files with CFM and labeling them are not trivial.

# 8 Streamlit dashboard (`web_app`)

The web UI simulates AAT generation on CSV data, reports **Evasion Rate** on the target model, plots Level-1 (SPTS) feature shifts, compares raw vs. adversarial flows, and stores a run history.

## 8.1 Setup and launch

1. Install Streamlit if needed:
   ```bash
   pip install streamlit
   ```
2. Ensure `STORAGE_DIR` is set in `utils/utils.py` and required files exist under `STORAGE_DIR/dataset/` (notably `{ton|ids18}_raw_att.csv`, `fea_t.csv`, `fea_s.csv`, and the matching min–max files).
3. Place `.pth` checkpoints (target names ending in `_t`, surrogate in `_s`) under `STORAGE_DIR/custom/pre-trained_models/`—e.g. copy or symlink from Section 4.4, or pass `--output-dir` when training.
4. From the repository root:
   ```bash
   streamlit run web_app/app.py
   ```
5. Open the URL printed by Streamlit (typically `http://localhost:8501`).

## 8.2 Sidebar workflow

1. Select **Target Model** (NIDS under test) and **Surrogate Model** (used for attack gradients). Prefer pairs from the same dataset, e.g. `ton_mlp_t` and `ton_mlp_s`.
2. Choose the **algorithm** (MIFGSM, SIM, VMIFGSM, DGM) and tune **Iterations** and **Step size**; for DGM, set **Copies** and **Dropout rate**.
3. Click **Generate AAT** and wait for the progress bar (batched AAT generation, then evaluation on the target).

## 8.3 Reading the main results

- **Evasion Rate (Bypass NIDS %):** percentage of attack flows classified as **Benign** by the target. **Higher** means a stronger attack in this simulation (easier NIDS bypass).
- **Bar chart (four Level-1 features):** mean values **before** vs. **after** perturbation for SPTS columns (`Fwd Pkt Len Max/Min`, `Fwd IAT Max/Min`), showing how much those features move.

## 8.4 Extra tabs

- **Compare AAT:** side-by-side preview (first 50 rows) of raw vs. adversarial tables; download **raw** and **adv** CSVs. **Change statistics** list numeric columns with `mean(|Δ|)`, `mean(Δ)`, `std(Δ)`, and `changed_%` (share of rows where the feature changed).
- **ExplainAI:** ranks features by `mean(|Δ|)`; adjust **Top-K** and use **Drill-down for one flow** (row index) to inspect raw/adv/Δ per feature for a single flow.
- **History:** markdown table of past **Generate AAT** runs (algorithm, hyperparameters, model pair, evasion rate, timestamp) for quick comparisons.

If the sidebar reports missing models, check `STORAGE_DIR/custom/pre-trained_models/` and the `_t` / `_s` naming. If raw attack data is missing, verify `{dataset}_raw_att.csv` under `STORAGE_DIR/dataset/`.
