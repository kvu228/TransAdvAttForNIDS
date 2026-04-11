# 1 Giới thiệu

Để giúp repository dễ điều hướng hơn và hỗ trợ người đọc tái tạo các thí nghiệm trong bài báo, chúng tôi đã nhóm tất cả các thí nghiệm thành ba loại:

1. **Tái tạo kết quả trong bài báo**.
    
    Chúng tôi cung cấp các dataset đã được tiền xử lý và AAT đã được tạo tương ứng, cho phép người dùng tái tạo gần như tất cả các kết quả được báo cáo trong bài báo.
    
2. **Training tùy chỉnh và tạo AAT**.
    
    Để giải quyết các lo ngại về dữ liệu đã chuẩn bị, chúng tôi cung cấp các script cho phép người dùng train các model riêng và tạo AAT. Mặc dù chúng tôi không cố định random seeds, các thí nghiệm lặp lại xác nhận rằng việc bỏ qua seed không ảnh hưởng đến các phát hiện hoặc kết luận cốt lõi của bài báo.
    
3. **Ánh xạ AAT về các gói tin thực tế (bao gồm TANTRA)**.
    
    Nhóm thí nghiệm này cho thấy cách chuyển đổi AAT đã tạo về các gói tin mạng thực tế và cũng bao gồm các script để tạo AAT với TANTRA.

Dựa trên các cân nhắc trên, README này được tổ chức như sau: Mục 2 bao gồm các điều kiện tiên quyết—thiết lập môi trường, tải dataset và các công cụ phụ trợ; Mục 3 giải thích cách tái tạo kết quả trong bài báo; Mục 4 và 5 mô tả training model tùy chỉnh và tạo AAT, tương ứng; Mục 6 chi tiết quy trình ánh xạ AAT đã tạo về các gói tin mạng thực tế; và Mục 7 cung cấp các ghi chú bổ sung.

# 2 Điều kiện tiên quyết

## 2.1 Thiết lập môi trường

### Hệ điều hành

**Ubuntu/Linux (Khuyến nghị)**:
- Ubuntu 18.04.6 LTS hoặc các phiên bản Linux khác
- Đây là môi trường được test và khuyến nghị

**Windows**:
- Code có thể chạy trên Windows, nhưng cần lưu ý:
  - **WSL2 (Windows Subsystem for Linux)** là giải pháp tốt nhất - cho phép chạy Ubuntu trên Windows
  - Nếu chạy trực tiếp trên Windows:
    - CICFlowMeter có thể không hoạt động (yêu cầu Linux)
    - Đường dẫn file cần điều chỉnh (sử dụng `\\` thay vì `/`)
    - Một số scripts sử dụng `subprocess.run` với sudo có thể không hoạt động
  - **Khuyến nghị**: Sử dụng WSL2 để có trải nghiệm tương tự Ubuntu

**macOS**:
- Có thể chạy được nhưng cần điều chỉnh một số đường dẫn và dependencies

### Python và Dependencies

- **Python 3.9.13** (hoặc tương thích)
- **PyTorch 2.5.1+cu121** (hoặc phiên bản tương thích)
- **Pandas 2.2.3**
- **NumPy 2.0.2**
- **Dpkt 1.9.8**
- **Matplotlib 3.9.4**
- **Seaborn 0.13.2**

**Cài đặt dependencies**:
```bash
pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install pandas==2.2.3 numpy==2.0.2 dpkt==1.9.8 matplotlib==3.9.4 seaborn==0.13.2
```

**Lưu ý cho Windows**:
- Nếu không có GPU NVIDIA, cài đặt PyTorch CPU version:
  ```bash
  pip install torch torchvision
  ```
- Đảm bảo Python được thêm vào PATH

## 2.2 Dataset

- **[CIC-IDS-2018](https://www.unb.ca/cic/datasets/ids-2018.html)**
- **[TON_IoT](https://research.unsw.edu.au/projects/toniot-datasets)**

## 2.3 CICFlowMeter (CFM)

**[CICFlowMeter](https://github.com/UNBCIC/CICFlowMeter)**: Nếu CICFlowMeter không được cài đặt đúng cách, các thí nghiệm trong Mục 6 (**Ánh xạ AAT về các gói tin thực tế**) không thể thực thi.

**Lưu ý quan trọng về hệ điều hành**:
- **Linux/Ubuntu**: CFM hoạt động tốt, yêu cầu quyền administrator
- **Windows**: CFM được thiết kế cho Linux và có thể không hoạt động trên Windows. Các giải pháp:
  1. **Sử dụng WSL2**: Cài đặt CFM trong WSL2 Ubuntu
  2. **Sử dụng Docker**: Chạy CFM trong container Linux
  3. **Bỏ qua Section 6**: Nếu chỉ cần training và generate AAT (Section 3-5), không cần CFM

**Cài đặt CFM** (Linux/WSL2):

1. Sau khi cài đặt CFM, bạn cần cập nhật đường dẫn CFM trong ba script:
   - `TransAdvAttForNIDS/map_AAT_to_pkts/0_built_features_with_cfm_over_raw_att_pcap.py`
   - `TransAdvAttForNIDS/map_AAT_to_pkts/4_re-extract_features_with_cfm.py`
   - `TransAdvAttForNIDS/TANTRA/1_re-extract_features_with_cfm.py`
   
   Biến liên quan là `fp_cfm`.

2. Chạy CFM yêu cầu quyền administrator, vì vậy bất kỳ script nào gọi CFM phải chạy với quyền root. Tuy nhiên, chúng tôi nhận thấy rằng chỉ đơn giản sử dụng `sudo python script.py` không hoạt động. Do đó, chúng tôi cung cấp hai phương pháp thay thế:
   - **Phương pháp 1**: Nhúng mật khẩu sudo trực tiếp vào lệnh `subprocess.run` của script (hy sinh bảo mật)
   - **Phương pháp 2**: Cấu hình CFM để "chạy dưới sudo mà không yêu cầu mật khẩu"

**Cấu hình sudo không cần mật khẩu** (Linux/WSL2):
```bash
# Thêm vào /etc/sudoers (sử dụng visudo)
username ALL=(ALL) NOPASSWD: /path/to/CICFlowMeter
```

## 2.4 Tải Models và Datasets đã xử lý

Các model đã train và dataset đã tiền xử lý có thể được tải từ 4 địa chỉ sau:

1. [address_1](https://zenodo.org/records/15597259?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjA3MTgyNzgzLWI5ZjAtNGIwYi04NGZkLTRlNzhiZDg0NGE4YiIsImRhdGEiOnt9LCJyYW5kb20iOiJlMjRmY2JmNTY5MzQwNTdmZmVmZjY2M2NkOGE3ODQ5MiJ9.W0ax17EhKsZdmX-OMkZy0xczh--MgRQn0V9KRPckV0D_SsuHK04R6mewrOJ3uZHc93woNOL9G1Ock3_9_SUsCw)
2. [address_2](https://drive.google.com/drive/folders/1Ne3s40AgGe0H6tMP0gug-tsCc_zw7qpb?usp=sharing)
3. [address_3](https://drive.google.com/drive/folders/1za3qA1g1WNlkt5MO-PDICDARrKjtZZqr?usp=sharing)
4. [address_4](https://drive.google.com/drive/folders/1HYvxwKOdRaHRy9dmaU7goXonHNyZAVo9?usp=sharing)

**Hướng dẫn setup**:

5. Có tổng cộng 11 file archive và giải nén từng file riêng lẻ. Lưu ý rằng **5_4_4.part1.rar** và **5_4_4.part2.rar** thuộc cùng một archive; bạn có thể giải nén chúng bằng: `unrar x 5_4_4.rar`.

6. Sau khi giải nén, tạo một thư mục trống mới ở bất kỳ đâu (bạn có thể chọn bất kỳ tên nào), sau đó di chuyển 10 thư mục đã giải nén vào thư mục này. Sau đó, mở **TransAdvAttForNIDS/utils/utils.py** và cập nhật biến **STORAGE_DIR** (dòng 9) thành đường dẫn của thư mục mới đó.

   **Lưu ý cho Windows**:
   - Sử dụng đường dẫn Windows format: `C:\\Users\\YourName\\path\\to\\storage`
   - Hoặc sử dụng raw string: `r"C:\Users\YourName\path\to\storage"`
   - Trong WSL2, sử dụng đường dẫn Linux: `/mnt/c/Users/YourName/path/to/storage`

7. Trong thư mục **STORAGE_DIR**, tạo một thư mục trống có tên **adv_pcap**.

8. Trước khi chạy code, tạo ba thư mục trống—có tên `output`, `output2`, và `output3`—bên trong thư mục `TransAdvAttForNIDS/`. Các thư mục trống này sẽ lưu trữ các file trung gian được tạo trong quá trình thực thi.

9. Chúng tôi cung cấp các dataset TON_IoT đã được oversample ngẫu nhiên cho cả target NIDSs và surrogate models. Tuy nhiên, các file oversampled của CIC-IDS-2018 không được bao gồm. Nếu bạn muốn train trên CIC-IDS-2018, hãy đảm bảo áp dụng random oversampling cho **TransAdvAttForNIDS/dataset/ids18_train_s.csv** và **TransAdvAttForNIDS/dataset/ids18_train_t.csv** bằng script tại **TransAdvAttForNIDS/dataset_preprocess/sampling_training_dataset.py** (xem Mục 7, **Ghi chú bổ sung**). Các output đã oversample cần được lưu trong thư mục `STORAGE_DIR/dataset` dưới dạng **ids18_sam_train_s.csv** và **ids18_sam_train_t.csv**.

# 3 Tái tạo kết quả trong bài báo

Chúng tôi đã bao gồm tất cả dữ liệu đã được tạo cũng như các script tương ứng. Chỉ cần chạy các script này sẽ tái tạo mọi bảng và hình ảnh được báo cáo, bao gồm Tables 5–18 và Figures 2–9.

1. Lấy Table 5 làm ví dụ, các bước chi tiết như sau:
    1. Chuyển đến thư mục đích.
        
        ```bash
        cd TransAdvAttForNIDS/reproduce_experiments_results
        ```
        
    2. Chạy script.
        
        ```bash
        python 5_2-Table_5.py
        ```

2. Tất cả các script trong thư mục `reproduce_experiments_results` có thể chạy trực tiếp mà không cần thêm tham số nào.

3. Lưu ý rằng kết quả cho Tables 15 và 16 có thể tương tự nhưng không giống hệt với những gì được trình bày trong bài báo. Các bảng này yêu cầu đo thời gian tạo AAT theo thời gian thực, và quá trình tạo chính nó liên quan đến tính ngẫu nhiên vốn có. Vì vậy, mỗi lần chạy có thể tạo ra kết quả hơi khác nhau.

# 4 Training NIDSs

Chúng tôi không giới thiệu các lý thuyết hoặc phương pháp mới để cải thiện hiệu suất NIDS, vì vậy việc training NIDS không phải là đóng góp của bài báo chúng tôi. Tuy nhiên, để loại bỏ các lo ngại tiềm ẩn, chúng tôi đã cung cấp các script training. Không giống như "Tái tạo kết quả trong bài báo", chúng tôi không viết một script riêng cho mỗi target NIDS hoặc surrogate model. Vì quy trình training giống nhau, chúng tôi cung cấp một script demo để train MLP-t và MLP-s trên dataset TON_IoT. Các implementation đầy đủ của các kiến trúc model khác có sẵn trong các thư mục `TransAdvAttForNIDS/utils/surrogate_model_with_var_input_fea.py` và `TransAdvAttForNIDS/utils/target_models_with_78_fea.py`. Nếu bạn muốn train các model với kiến trúc khác nhau, hãy chỉnh sửa script cho phù hợp.

## 4.1 Training target NIDSs và surrogate models

1. Chuyển đến thư mục đích.
    
    ```bash
    cd TransAdvAttForNIDS/train_NIDS
    ```
    
2. Chạy script.
    
    ```bash
    python training.py
    ```
    
3. Test model đã được train.
    
    ```bash
    python verifying.py
    ```
    
4. Dataset mặc định là TON_IoT, và model mặc định là MLP-t với 66 input features. Để chuyển sang MLP-s, đặt `model_type = 's'` trong **training.py** (dòng 43). Theo mặc định, MLP-s sử dụng 60 input features, khớp với cấu hình được mô tả trong Mục 5.3.1 của bài báo.

5. Sau 10 epochs, model sẽ được lưu vào `STORAGE_DIR/custom/pre-trained_models`.

6. Nếu bạn muốn train một MLP-t với 78 input features hoặc một surrogate model với ít input features hơn, bạn phải điều chỉnh kiến trúc model. Các định nghĩa tương ứng nằm trong **TransAdvAttForNIDS/utils/target_models_with_78_fea.py** và **TransAdvAttForNIDS/utils/surrogate_model_with_var_input_fea.py**, tương ứng.

## 4.2 Normal adversarial training

1. Chuyển đến thư mục đích.
    
    ```bash
    cd TransAdvAttForNIDS/train_NIDS
    ```
    
2. Chạy script.
    
    ```bash
    python normal_adv_training.py
    ```
    
3. Sau 10 epochs training, model sẽ được lưu vào `STORAGE_DIR/custom/pre-trained_models`.

4. Các cấu hình mặc định là training một MLP-t với 66 input features trên TON_IoT. Nếu bạn muốn train một model khác hoặc chuyển sang CIC-IDS-2018, hãy điều chỉnh tên model, đường dẫn dataset, giá trị min–max scaling và input features. Các tham số này được xác định rõ ràng trong các dòng 62–69 của script.

## 4.3 Adversarial training với SPTS

1. Chuyển đến thư mục đích.
    
    ```bash
    cd TransAdvAttForNIDS/train_NIDS
    ```
    
2. Chạy script.
    
    ```bash
    python adv_training_with_SPTS.py
    ```
    
3. Sau 10 epochs training, model sẽ được lưu vào `STORAGE_DIR/custom/pre-trained_models`.

4. Các cấu hình mặc định là training một MLP-t với 66 input features trên TON_IoT. Nếu bạn muốn train một model khác hoặc chuyển sang CIC-IDS-2018, hãy điều chỉnh tên model, đường dẫn dataset, giá trị min–max scaling và input features. Các tham số này được xác định rõ ràng trong các dòng 84–91 của script.

# 5 Tạo AAT

1. Chuyển đến thư mục đích.
    
    ```bash
    cd TransAdvAttForNIDS/generate_AAT
    ```
    
2. Chạy script.
    
    ```bash
    python generate_aat.py
    ```
    
3. Test AAT.
    
    ```bash
    python test_aat.py
    ```
    
4. Theo mặc định, script sử dụng TON_IoT, một surrogate model MLP-s, tấn công MI-FGSM với 7 iterations và step size là 140. Các tham số này được định nghĩa trong các dòng 74–86 và có thể được chỉnh sửa để tạo một AAT tùy chỉnh.

5. AAT đã được tạo được lưu dưới dạng **aat.csv** trong `STORAGE_DIR/custom/output`.

# 6 Ánh xạ AAT về các gói tin thực tế

Nhóm thí nghiệm này ánh xạ AAT đã được tạo về các gói tin thực tế và sau đó re-extract features với CFM. Lưu ý rằng các attack flows đã được lọc thêm, để lại tổng cộng 36,980 flows (xem đoạn đầu tiên của Mục 5.3.2 trong bài báo). Vì quy trình liên quan đến nhiều bước, các script được đánh số để chỉ ra thứ tự thực thi đúng.

**Lưu ý cho Windows**: Section này yêu cầu CICFlowMeter và có thể không hoạt động trên Windows native. Khuyến nghị sử dụng WSL2.

## 6.1 Cho SPTS

1. Chuyển đến thư mục đích.
    
    ```bash
    cd TransAdvAttForNIDS/map_AAT_to_pkts
    ```
    
2. Extract features với CFM. Trong khi CFM tạo ra tập features đầy đủ được sử dụng bởi target NIDS, chúng tôi cố ý sử dụng một tập con để mô phỏng kiến thức hạn chế của attacker.
    
    ```bash
    python 0_built_features_with_cfm_over_raw_att_pcap.py
    ```
    
    **Lưu ý**: Trên Linux/WSL2, có thể cần chạy với `sudo`:
    ```bash
    sudo python 0_built_features_with_cfm_over_raw_att_pcap.py
    ```
    
3. Tạo AAT. Không giống như quy trình trong Mục 5 (Tạo AAT), AAT được tạo ở đây chỉ giữ lại các trường cần thiết—'Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Protocol', 'Fwd Pkt Len Max', 'Fwd Pkt Len Min', 'Fwd IAT Max', và 'Fwd IAT Min'.
    
    ```bash
    python 1_generate_aat.py
    ```
    
4. Xử lý AAT đã được tạo bằng cách tính toán sự khác biệt giữa traffic gốc và AAT.
    
    ```bash
    python 2_process_aat.py
    ```
    
5. Modify packets.
    
    ```bash
    python 3_modify_pcap.py
    ```
    
6. Re-extract features với CFM.
    
    ```bash
    python 4_re-extract_features_with_cfm.py
    ```
    
    **Lưu ý**: Trên Linux/WSL2, có thể cần chạy với `sudo`:
    ```bash
    sudo python 4_re-extract_features_with_cfm.py
    ```
    
7. Test các features đã re-extract.
    
    ```bash
    python 5_test_aat.py
    ```
    
8. Theo mặc định, các script sử dụng surrogate model MLP-s (đã được train trong Mục 4) và tấn công MI-FGSM với 7 iterations và step size là 140. Để tùy chỉnh các cài đặt này, hãy chỉnh sửa các dòng 93–105 trong **1_generate_aat.py**.

## 6.2 Cho TANTRA

TANTRA train một LSTM để học các pattern của normal traffic. Chúng tôi áp dụng chính xác kiến trúc và hyperparameters được chỉ định trong bài báo gốc, vì vậy không có script training bổ sung nào được bao gồm; thay vào đó, chúng tôi cung cấp model đã được pre-train.

1. Chuyển đến thư mục đích.
    
    ```bash
    cd TransAdvAttForNIDS/TANTRA
    ```
    
2. Modify *Timestamp* của mỗi attack packet theo model LSTM đã được train.
    
    ```bash
    python 0_modify_pkts.py
    ```
    
3. Re-extract features với CFM.
    
    ```bash
    python 1_re-extract_features_with_cfm.py
    ```
    
    **Lưu ý**: Trên Linux/WSL2, có thể cần chạy với `sudo`:
    ```bash
    sudo python 1_re-extract_features_with_cfm.py
    ```
    
4. Test các features đã re-extract.
    
    ```bash
    python 2_test_aat.py
    ```

# 7 Ghi chú bổ sung

Chúng tôi cũng cung cấp các script tiền xử lý dataset nằm trong thư mục `TransAdvAttForNIDS/dataset_preprocess`.

1. Script `divide_dataset_into_target_and_surrogate.py` chia dataset thành hai tập con—một cho target NIDS và một cho surrogate model.

2. Script `split_dataset_into_train_and_test.py` chia mỗi tập con thành training và test sets theo một tỷ lệ được chỉ định.

3. Script `sampling_training_dataset.py` thực hiện random oversampling trên training dataset.

4. Script `build_input_features.py` extract các input features của model.

5. Script `build_minmax.py` extract các giá trị maximum và minimum từ training dataset; các giá trị này sau đó được sử dụng để normalization.

6. Lưu ý rằng các script trên không thể chạy trực tiếp. Để thực thi chúng, bạn phải chỉ định các đường dẫn file input và output trong mỗi script. Code và logic của chúng khá đơn giản, vì vậy không cần giải thích thêm.

7. Đối với TON_IoT, chúng tôi khuyến nghị sử dụng các file CSV đã được pre-label mà chúng tôi cung cấp, vì việc extract features từ các file PCAP thô với CFM và label chúng không phải là việc đơn giản.

## 7.1 Hướng dẫn cho Windows Users

### Sử dụng WSL2 (Khuyến nghị)

1. **Cài đặt WSL2**:
   ```powershell
   wsl --install
   ```
   Hoặc theo hướng dẫn: https://docs.microsoft.com/en-us/windows/wsl/install

2. **Cài đặt Ubuntu trong WSL2**:
   - Chọn Ubuntu từ Microsoft Store
   - Setup user và password

3. **Truy cập files từ Windows**:
   - Windows files có thể truy cập từ WSL2 tại `/mnt/c/Users/...`
   - WSL2 files có thể truy cập từ Windows tại `\\wsl$\Ubuntu\home\...`

4. **Cài đặt dependencies trong WSL2**:
   ```bash
   # Trong WSL2 Ubuntu terminal
   sudo apt update
   sudo apt install python3.9 python3-pip
   pip install torch pandas numpy dpkt matplotlib seaborn
   ```

5. **Chạy code từ WSL2**:
   ```bash
   cd /mnt/c/Users/YourName/Testplace/TransAdvAttForNIDS
   python training.py
   ```

### Chạy trực tiếp trên Windows (Không khuyến nghị)

Nếu bạn muốn chạy trực tiếp trên Windows mà không dùng WSL2:

1. **Cài đặt Python và dependencies**:
   ```powershell
   # Cài đặt Python từ python.org
   pip install torch pandas numpy dpkt matplotlib seaborn
   ```

2. **Điều chỉnh đường dẫn**:
   - Sửa các đường dẫn trong code từ `/path/to/file` sang `C:\\path\\to\\file`
   - Hoặc sử dụng `os.path.join()` và `pathlib.Path` để cross-platform

3. **Bỏ qua Section 6**:
   - Section 6 yêu cầu CICFlowMeter và không hoạt động trên Windows
   - Các Section 3-5 (training và generate AAT) vẫn hoạt động tốt

4. **Sửa các lệnh subprocess**:
   - Các lệnh `subprocess.run(['rm', ...])` cần đổi thành Windows equivalent
   - Hoặc sử dụng `os.remove()` và `shutil.rmtree()` cho cross-platform

### Troubleshooting cho Windows

- **Lỗi đường dẫn**: Sử dụng `os.path.join()` thay vì nối chuỗi trực tiếp
- **Lỗi permissions**: Chạy PowerShell/CMD với quyền Administrator
- **Lỗi CUDA**: Nếu không có GPU NVIDIA, cài đặt PyTorch CPU version
- **Lỗi CICFlowMeter**: Sử dụng WSL2 hoặc bỏ qua Section 6
