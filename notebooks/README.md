# Notebooks — Minh họa nghiên cứu TransAdvAttForNIDS

Các notebook Jupyter để minh họa và chạy thử: setup/verify model, training, generate AAT, reproduce bảng/figure.

## Cách chạy

**Luôn mở Jupyter từ thư mục gốc repo:**

```bash
cd /mmlab_students/storageStudents/nguyenvd/nids/TransAdvAttForNIDS
source .venv/bin/activate   # nếu dùng uv .venv
jupyter notebook
# hoặc: jupyter lab
```

Sau đó mở từng file `.ipynb` từ tab File Browser. Chạy các cell theo thứ tự từ trên xuống.

## Danh sách notebook

| File | Nội dung |
|------|----------|
| `01_setup_and_verify.ipynb` | Thiết lập path, STORAGE_DIR; verify model (Accuracy, Precision, Recall, F1) |
| `02_generate_aat.ipynb` | Sinh AAT (MIFGSM/SIM/VMIFGSM/DGM), xem mẫu, test trên target model |
| `03_training.ipynb` | Train NIDS (target/surrogate) trên TON_IoT hoặc CIC-IDS-2018 |
| `04_reproduce_figures.ipynb` | Reproduce Table 5 (demo 1 dataset + vài model), Figure 2 (heatmap) với `plot_hm` |

## Lưu ý

- Trong mỗi notebook, cell đầu tiên set `STORAGE_DIR`; có thể sửa đường dẫn cho đúng với máy bạn.
- Cần có dữ liệu và model trong `STORAGE_DIR` (dataset, pre-trained_models, v.v.). Xem README gốc để tải dữ liệu.
- Notebook 04 (reproduce) cần AAT đã sinh trong `STORAGE_DIR/AAT/...` cho Figure 2 đầy đủ.

## Troubleshooting: Authentication at http://localhost:8888 failed

Lỗi `JupyterLoginException: Authentication at http://localhost:8888 failed` xảy ra khi Cursor/IntelliJ cố kết nối tới Jupyter server tại `localhost:8888` nhưng không có server đang chạy hoặc IDE không có token đúng.

**Cách 1 — Dùng kernel Python trực tiếp (khuyến nghị trong Cursor/VS Code):**

- Không chọn "Connect to Jupyter Server".
- Trong notebook: **Select Kernel** → **Python Environments** → chọn interpreter của project (ví dụ `.venv` do `uv` tạo).
- Notebook chạy bằng kernel local, không cần Jupyter server riêng.

**Cách 2 — Chạy Jupyter server rồi kết nối từ IDE:**

1. Từ thư mục gốc repo (đã activate venv), chạy **một** trong hai cách:
   - **Có token in ra (để dán vào IDE):**
     ```bash
     cd /mmlab_students/storageStudents/nguyenvd/nids/TransAdvAttForNIDS
     source .venv/bin/activate
     jupyter notebook --ServerApp.token='jupyter-local' --ServerApp.password=''
     ```
     Terminal sẽ in URL dạng `http://127.0.0.1:8888/?token=jupyter-local` (hoặc port khác nếu 8888 bận). Copy **toàn bộ URL**.
   - **Hoặc chạy bình thường:** `jupyter notebook` — nếu thấy "there is no token information" hoặc không thấy `?token=...`, dùng lệnh có `--ServerApp.token` ở trên.
2. **Đúng port:** Nếu log báo "The port 8888 is already in use", Jupyter chạy ở **8889**. Trong IDE dùng `http://127.0.0.1:8889/?token=...` (không phải 8888).
3. Trong Cursor/IntelliJ/PyCharm: **Jupyter** / **Add Jupyter Server** → dán URL (gồm cả `?token=...`) làm server URL.
4. Chọn kernel từ server đó và chạy notebook.

Nếu bạn mở notebook bằng trình duyệt (sau khi chạy `jupyter notebook`), chỉ cần mở file `.ipynb` từ tab File Browser trong Jupyter — không cần cấu hình token trong IDE.
