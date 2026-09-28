# TT-26 — CNN: Sàng lọc viêm phổi trên ảnh X-quang ngực

> Khoá: Học máy · Buổi 10 (CNN & Computer Vision) · Thuật toán: CNN + Transfer Learning (EfficientNetB0)

## ⚖️ CẢNH BÁO Y TẾ (BẮT BUỘC ĐỌC)

- Đây là công cụ **sắp thứ tự ưu tiên đọc phim (sàng lọc)**, **KHÔNG PHẢI công cụ chẩn đoán**.
- Dữ liệu huấn luyện thu thập tại **Quảng Châu (Trung Quốc)**, trên **bệnh nhi 1–5 tuổi** → kết quả **không tổng quát hoá** cho người lớn hoặc dân số khác.
- **Mọi ca đều phải có bác sĩ đọc lại.** Model chỉ giúp gắn cờ ca nghi ngờ để bác sĩ trực xem trước.
- Trong thử nghiệm này model **bỏ sót 6 / 390 ca viêm phổi (1,5%)** trên tập test (xem mục 5) — nghĩa là không thể tin tưởng hoàn toàn vào kết quả "bình thường" của model. Đồng thời model **báo động giả 118 / 234 ca bình thường (50,4%)**, nên chỉ phù hợp để ưu tiên thứ tự đọc phim, không dùng để loại trừ hay xác nhận bệnh.

---

## 1. Bài toán

Bệnh viện tuyến huyện thiếu bác sĩ chẩn đoán hình ảnh trực đêm, ảnh X-quang chụp lúc 2h sáng phải chờ đến sáng mới có người đọc. Mục tiêu: công cụ tự động **gắn cờ ca nghi viêm phổi** để bác sĩ ưu tiên xem trước.

**Metric chính là RECALL (mục tiêu ≥ 0,97 khi chọn ngưỡng)**: bỏ sót viêm phổi ở trẻ em có thể dẫn tới tử vong, còn báo động giả chỉ tốn thêm một lần bác sĩ xem lại. Vì vậy không dùng accuracy làm metric chính (accuracy che giấu ca bỏ sót do dữ liệu mất cân bằng).

## 2. Dữ liệu

Bộ **Chest X-Ray Images (Pneumonia)** – Kaggle (paultimothymooney), 5.863 ảnh JPEG, 2 lớp NORMAL / PNEUMONIA.

| Tập | NORMAL | PNEUMONIA | Tổng |
|---|---|---|---|
| train | 1.341 | 3.875 | 5.216 |
| val (gốc) | 8 | 8 | 16 |
| test | 234 | 390 | 624 |

Kết quả đếm ảnh thực tế trên máy khớp hoàn toàn với bảng trên, và làm lộ **3 vấn đề** của bộ dữ liệu:

1. **Val gốc chỉ có 16 ảnh** → quá nhỏ, kết quả nhiễu, không thể dùng để chọn model. **Cách xử lý:** bỏ thư mục `val/` gốc, tự tách lại **15% từ tập train** (`validation_split=0.15`, seed cố định) → **train 4.434 ảnh, validation 782 ảnh (190 NORMAL / 592 PNEUMONIA)**. Việc tách là ngẫu nhiên có seed (không phân tầng chính xác), nên tỉ lệ PNEUMONIA của validation (75,7%) xấp xỉ tập train.
2. **Mất cân bằng lớp:** train có 74,3% PNEUMONIA. **Cách xử lý:** dùng `class_weight` tính từ nhãn thật của tập train.
3. **Phân phối train ≠ test:** PNEUMONIA chiếm 74,3% ở train nhưng 62,5% ở test. Đây là đặc điểm thật của dữ liệu, không sửa được; hệ quả là accuracy trên train/val thường cao hơn test một cách có hệ thống, và cần nêu rõ trong báo cáo.

## 3. Phương pháp

### 3.1. Ba mô hình được so sánh

| # | Mô hình | Mô tả |
|---|---|---|
| 0 | Baseline CNN | 3 khối Conv (32-64-128) + BatchNorm + MaxPool, train từ đầu |
| 1 | Transfer Learning | EfficientNetB0 pretrained ImageNet, **đóng băng** base, chỉ train đầu phân loại (lr = 1e-3) |
| 2 | Fine-tuning | Mở khoá 30 layer cuối của base, train tiếp với **lr = 1e-5** (nhỏ hơn 100 lần) |

Lý do dùng Transfer Learning: train CNN từ đầu trên ~5.000 ảnh gần như chắc chắn overfit; mạng đã học trên 1,2 triệu ảnh ImageNet cho đặc trưng tổng quát tốt hơn nhiều. Lý do lr nhỏ khi fine-tune: lr lớn sẽ phá huỷ trọng số ImageNet chỉ sau vài bước đầu.

### 3.2. Augmentation hợp lý với y tế

-  Dùng: xoay ±10°, dịch ±10%, zoom ±10%, đổi tương phản nhẹ (chỉ áp dụng cho tập train).
-  **Không lật ngang:** tim nằm lệch bên trái lồng ngực, lật ngang tạo ảnh sai giải phẫu, model sẽ học sai cấu trúc. Không lật dọc, không biến dạng mạnh.

### 3.3. Chọn ngưỡng quyết định

Ngưỡng được chọn **trên tập validation** (không dùng test) sao cho recall ≥ 0,97, sau đó **đánh giá tập test đúng 1 lần** ở ngưỡng đó để tránh rò rỉ thông tin từ test.

## 4. Kết quả huấn luyện

Ba giai đoạn huấn luyện chạy trên CPU (Windows, không GPU), early stopping theo `val_auc` (patience = 4, khôi phục trọng số tốt nhất). Số liệu validation dưới đây tính ở ngưỡng mặc định 0,5 (metric của Keras), trên tập validation tự tách 782 ảnh.

| Mô hình | Epoch chạy | Recall (val) | AUC (val) | Ghi chú |
|---|---|---|---|---|
| Baseline CNN (từ đầu) | 8 (khôi phục epoch 4) | 1,0000 | 0,9641 | Không ổn định: val_accuracy = 0,7724 (gần bằng tỉ lệ PNEUMONIA 75,7%) cho thấy model gần như đoán toàn bộ là viêm phổi; train accuracy chỉ ~0,88 |
| Transfer Learning (đóng băng) | 10 | 0,9155 | 0,9852 | val accuracy 0,9297; train accuracy 0,9272 |
| Fine-tuning (lr = 1e-5) | 10 | 0,9223 | **0,9907** | val accuracy 0,9322; train accuracy 0,9418 |

Learning curves: `reports/learning_curves.png`. **Ngưỡng chọn được (recall ≥ 0,97 trên val): `0,19`.**

**Nhận xét:**
- **Baseline (từ đầu) kém và bất ổn:** val AUC dao động mạnh giữa các epoch (0,88 → 0,51 → 0,50 → 0,96 → 0,87 → 0,69…), trong khi train AUC tăng đều. Recall val = 1,0 ở 4 epoch đầu là giả tạo vì val_accuracy đúng bằng tỉ lệ lớp PNEUMONIA (model đoán toàn bộ là viêm phổi). Điều này khớp với dự đoán của đề: train từ đầu với ~5.000 ảnh không đủ ổn định.
- **Transfer Learning cải thiện rõ rệt:** val AUC 0,9852 so với 0,9641, và đường cong val ổn định, tăng đều qua 10 epoch.
- **Fine-tuning cải thiện thêm:** val AUC 0,9907 (từ 0,9852), val recall 0,9223 (từ 0,9155). Chênh lệch nhỏ nhưng nhất quán, và lr = 1e-5 không phá hỏng trọng số ImageNet (epoch đầu của fine-tune có tụt nhẹ val AUC 0,9823 rồi tăng lại).
- **Không thấy dấu hiệu overfit:** train và val sát nhau ở cả hai giai đoạn cuối (train accuracy 0,94 / val accuracy 0,93 ở fine-tuning), và val AUC vẫn đang tăng ở epoch cuối → model có thể chưa hội tụ hoàn toàn; huấn luyện thêm epoch có thể cải thiện thêm (chưa thử trong bài này).
- Recall val ở ngưỡng 0,5 chỉ đạt ~0,92, thấp hơn mục tiêu 0,97, vì vậy ngưỡng được hạ xuống 0,19 để đạt recall ≥ 0,97 trên validation.

## 5. Đánh giá trên tập test

Ngưỡng dùng: **0,19** (chọn trên validation, đánh giá test đúng 1 lần).

| Chỉ số (test, n = 624) | Giá trị |
|---|---|
| Recall / Sensitivity (PNEUMONIA) | **0,9846** (384 / 390) |
| Precision (PNEUMONIA) | 0,7649 |
| F1 (PNEUMONIA) | 0,8610 |
| Accuracy | 0,8013 (500 / 624) |
| Specificity / Recall (NORMAL) | 0,4957 (116 / 234) |
| Precision (NORMAL) | 0,9508 |
| **Số ca viêm phổi BỊ BỎ SÓT (False Negative)** | **6 / 390** |
| Số báo động giả (False Positive) | 118 / 234 (50,4%) |

Ma trận nhầm lẫn (hàng = thực tế, cột = dự đoán):

| | Dự đoán NORMAL | Dự đoán PNEUMONIA |
|---|---|---|
| **Thực tế NORMAL** | 116 (TN) | 118 (FP) |
| **Thực tế PNEUMONIA** | 6 (FN) | 384 (TP) |

Ma trận nhầm lẫn: `reports/confusion_matrix_test.png`

**Đối chiếu tiêu chí:** Recall test = 0,9846 ≥ 0,96 → **đạt** tiêu chí recall. Accuracy = 0,8013 **thấp hơn** mức tham chiếu của đề (~0,88–0,93).

**Nhận xét:**
- Model làm đúng mục tiêu sàng lọc: chỉ bỏ sót 6 trong 390 ca viêm phổi (1,5%).
- Cái giá là đặc hiệu thấp (0,4957): khoảng một nửa số ca bình thường bị gắn cờ nhầm, nên mức giảm tải cho bác sĩ chỉ ở mức vừa phải. Ngược lại, khi model nói "bình thường" thì khá đáng tin (precision NORMAL = 0,9508).
- Accuracy thấp hơn recall và thấp hơn accuracy trên validation (~0,93 ở ngưỡng 0,5) là điều mà đề đã dự báo do phân phối test khác train/val (test có 62,5% PNEUMONIA, train/val ~74–76%). Lưu ý accuracy validation và test ở đây tính ở hai ngưỡng khác nhau (0,5 và 0,19) nên không so sánh trực tiếp được.
- Ngưỡng phải hạ xuống 0,19 để đạt recall ≥ 0,97 trên validation cho thấy phân tách hai lớp của model còn hạn chế. Các hướng cải thiện (chưa thử): huấn luyện thêm epoch (val AUC vẫn đang tăng), hiệu chuẩn xác suất, thử ResNet50 / DenseNet121.

## 7. Hạn chế

- Dữ liệu từ **một cơ sở** (Quảng Châu), **bệnh nhi 1–5 tuổi** → không tổng quát hoá cho người lớn hoặc quần thể khác.
- Phân phối train khác test (74,3% vs 62,5% PNEUMONIA).
- Tỉ lệ báo động giả cao (50,4% ca bình thường bị gắn cờ) và accuracy test 0,80 thấp hơn mức tham chiếu của đề.
- Chỉ huấn luyện 10 epoch mỗi giai đoạn (giới hạn thời gian CPU); val AUC vẫn tăng ở epoch cuối.
- Ảnh bị nén xuống 224×224 có thể làm mất chi tiết tổn thương nhỏ.
- Chưa hiệu chuẩn xác suất (calibration) và chưa ước lượng độ bất định → xác suất model đưa ra chưa chắc phản ánh mức chắc chắn thật.
- Huấn luyện chạy trên CPU (Windows, không có GPU) nên số epoch được giữ ở mức vừa phải.

## 9. Cấu trúc project & cách chạy

```
TT-26-CNN-<HoTen>/
├── README.md
├── requirements.txt
├── notebooks/cnn_chest_xray.ipynb
├── src/
│   ├── data_pipeline.py    # đếm ảnh, tách val 15%, class_weight, augmentation
│   ├── model.py            # baseline CNN, EfficientNetB0, fine-tune
│   ├── train.py            # huấn luyện 3 giai đoạn, chọn ngưỡng, đánh giá test
│   ├── gradcam.py          # ảnh Grad-CAM
│   └── evaluate.py         # phân tích ca dự đoán sai
├── models/best_model.keras
├── data/chest_xray/{train,val,test}/   # dữ liệu tải từ Kaggle 
└── reports/{learning_curves.png, confusion_matrix_test.png, gradcam_examples.png, ca_du_doan_sai.png, final_results.json}
```

Chạy từ thư mục gốc project:

```bash
pip install -r requirements.txt
python -m src.train --data_dir data/chest_xray --out_dir reports --models_dir models
python -m src.gradcam --model models/best_model.keras --data_dir data/chest_xray --out_dir reports
python -m src.evaluate --model models/best_model.keras --data_dir data/chest_xray --threshold <ngưỡng trong final_results.json> --out_dir reports
```