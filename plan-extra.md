Plan: Phát triển thêm đề tài TransAdvAttForNIDS

Nhóm đang làm đồ án cuối kì (IT3004 - Chuyên đề nghiên cứu NIDS) tái thực nghiệm bài báo Mao et al. (2025) về SPTS + DGM. Nhóm đã reproduce thành công ở mức feature-level nhưng thất bại ở packet-level.
 Cần thêm đóng góp kỹ thuật trong 2 ngày để tăng điểm học thuật.

 Đóng góp đề xuất (ưu tiên theo impact/effort)

 1. Thêm thuật toán NI-FGSM vào SPTS (3-4 giờ code + chạy thí nghiệm)

 Lý do: Đây là đóng góp kỹ thuật rõ ràng nhất - paper chỉ có 4 thuật toán (MI-FGSM, SIM, VMI-FGSM, DGM), nhóm thêm thuật toán thứ 5.

 NI-FGSM (Nesterov Iterative FGSM): Sử dụng Nesterov momentum thay vì momentum thông thường. Nesterov "nhìn trước" (look-ahead) vào vị trí tiếp theo trước khi tính gradient, giúp hội tụ tốt hơn.

 Files cần tạo/sửa:
 - Tạo: utils/NIFGSM.py (copy từ MIFGSM.py, sửa momentum update)
 - Sửa: generate_AAT/generate_aat.py (thêm dispatch cho NIFGSM)
 - Tạo: utils/NIFGSM_forAdvTrain.py (cho adversarial training)

 Thay đổi core (so với MI-FGSM):
 # MI-FGSM (hiện tại):
 momentum = mu * momentum + grad / norm(grad)
 pert = sign(momentum) * mask

 # NI-FGSM (Nesterov):
 # Bước 1: look-ahead gradient tại vị trí "tương lai"
 nes_adv = adv_tensor + step_length * mu * momentum_tensor * mask
 loss = lossfn(model(nes_adv), labels)
 loss.backward()
 grad = nes_adv.grad
 # Bước 2: cập nhật momentum bình thường
 momentum = mu * momentum + grad / norm(grad)
 pert = sign(momentum) * mask

 2. Cross-attack defense evaluation (2-3 giờ chạy thí nghiệm)

 Lý do: Paper chỉ train phòng thủ bằng MI-FGSM rồi test bằng tất cả attacks. Nhóm thử train bằng DGM và test bằng MI-FGSM (và ngược lại) -> câu hỏi: phòng thủ có tổng quát không?

 Files cần sửa:
 - train_NIDS/adv_training_with_SPTS.py: Thay MIFGSM bằng DGM/NI-FGSM
 - Tạo script evaluate tương tự reproduce_experiments_results/5_6-Table_17.py

 Ma trận thí nghiệm:
┌────────────────────┬───────────────────┬───────────────┬───────────────────┐
│ Train defense bằng │ Test bằng MI-FGSM │ Test bằng DGM │ Test bằng NI-FGSM │
├────────────────────┼───────────────────┼───────────────┼───────────────────┤
│ MI-FGSM (paper)    │ Có rồi            │ Có rồi        │ MỚI               │
├────────────────────┼───────────────────┼───────────────┼───────────────────┤
│ DGM                │ MỚI               │ MỚI           │ MỚI               │
├────────────────────┼───────────────────┼───────────────┼───────────────────┤
│ NI-FGSM            │ MỚI               │ MỚI           │ MỚI               │
└────────────────────┴───────────────────┴───────────────┴───────────────────┘

3. Fix bug truncate_packet (30 phút)

Lý do: Phát hiện bug tiềm ẩn trong map_AAT_to_pkts/3_modify_pcap.py dòng 61.

File: map_AAT_to_pkts/3_modify_pcap.py
# Bug hiện tại (dòng 61):
l4.data = l4.data[:len(l4.data)+ 1 - need_len if len(l4.data)+ 1 - need_len > 0 else 1]

# Sửa thành:
l4.data = l4.data[:max(1, len(l4.data) - need_len)]

4. Chạy transferability NI-FGSM (phụ thuộc GPU time)

Tạo script reproduce tương tự các script trong reproduce_experiments_results/ để sinh bảng kết quả transferability cho NI-FGSM trên cả 2 dataset.

Thứ tự thực hiện

Ngày 1:
1. Tạo utils/NIFGSM.py + utils/NIFGSM_forAdvTrain.py
2. Cập nhật generate_AAT/generate_aat.py để dispatch NI-FGSM
3. Fix bug truncate_packet()
4. Chạy generate AAT với NI-FGSM trên TON_IoT (5 surrogate models)
5. Chạy test transferability NI-FGSM

Ngày 2:
6. Train defense bằng DGM + NI-FGSM (adv_training_with_SPTS)
7. Evaluate cross-attack defense
8. Tổng hợp kết quả thành bảng/hình cho báo cáo
9. Cập nhật báo cáo Word

Verification

- Chạy generate_AAT/generate_aat.py với attack_name='NIFGSM' -> kiểm tra output aat.csv
- Chạy generate_AAT/test_aat.py -> so sánh detection rate với MI-FGSM, DGM
- So sánh NI-FGSM transferability matrix với Table 8 của paper
- Cross-defense: so sánh detection rate khi train/test bằng attack khác nhau

Files chính cần thay đổi

- utils/NIFGSM.py (MỚI)
- utils/NIFGSM_forAdvTrain.py (MỚI)
- generate_AAT/generate_aat.py (thêm dispatch)
- map_AAT_to_pkts/3_modify_pcap.py (fix bug truncate)
- train_NIDS/adv_training_with_SPTS.py (thêm option attack method)
