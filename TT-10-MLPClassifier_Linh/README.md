# TT-10 — MLP Classifier: Đọc số viết tay trên séc / phiếu chuyển khoản

**Dữ liệu dùng:** MNIST thật qua `sklearn.datasets.fetch_openml('mnist_784', ...)` — 70.000 ảnh 28×28, 10 lớp.
Train: 50.400 | Validation: 5.600 | Test: 14.000
*(Validation được tách riêng từ tập train, chỉ dùng để CHỌN kiến trúc/activation/learning rate — không dùng để chọn hyperparameter trên test nhằm tránh rò rỉ dữ liệu. Test chỉ dùng đúng một lần để báo cáo số liệu cuối.)*

---

## 1. Bảng so sánh CÓ / KHÔNG chuẩn hoá

Ở bước này dùng cố định kiến trúc (128, 64) để so sánh riêng tác động của việc chuẩn hoá dữ liệu.

| Model | Accuracy | Ghi chú |
|---|---|---|
| Baseline — Logistic Regression | 0.9211 | dùng dữ liệu đã chuẩn hoá |
| MLP (128,64) — **không** chuẩn hoá | 0.9709 | pixel 0–255 thô, iters=58, time=119.6s |
| MLP (128,64) — **có** chuẩn hoá | **0.9796** | iters=64, time=123.3s, tốt nhất |

**Nhận xét:** chuẩn hoá giúp accuracy tăng ~0,87 điểm % so với không chuẩn hoá, và vượt baseline Logistic Regression tới ~5,85 điểm %. Việc scale dữ liệu về cùng khoảng giúp mạng hội tụ ổn định hơn (không chuẩn hoá dễ khiến gradient dao động mạnh do input scale lớn — pixel 0–255 làm sai số ban đầu lớn hơn nhiều so với input đã chuẩn hoá).

---

## 2. Bảng so sánh kiến trúc (≥ 4 kiến trúc)

Accuracy dưới đây đo trên tập **validation** (dùng để chọn kiến trúc, không phải test) — tránh rò rỉ dữ liệu khi chọn hyperparameter.

| Kiến trúc | Accuracy (validation) | Số tham số | Thời gian train |
|---|---|---|---|
| (64,) | 0.9730 | 50.890 | 15.7s |
| (128,) | 0.9762 | 101.770 | 40.7s |
| (128, 64) | 0.9782 | 109.386 | 51.6s |
| **(256, 128, 64)** | **0.9793** | 242.762 | 190.9s |

→ Kiến trúc tốt nhất trên validation: **(256, 128, 64)**, acc_val = 0.9793.
Đánh giá lại kiến trúc này trên tập **test** (chỉ một lần, dùng cho số báo cáo cuối): **acc = 0.9766**.

**Nhận xét:** (256,128,64) đạt accuracy validation cao nhất, nhỉnh hơn (128,64) khoảng 0,11 điểm % nhưng đổi lại thời gian train tăng gần gấp 4 lần (190.9s so với 51.6s) và số tham số tăng hơn gấp đôi. Khi đo trên test, kiến trúc này đạt 0.9766 — thấp hơn một chút so với accuracy 0.9796 của (128,64) đo được ở Mục 1 (do khác tập đánh giá: validation dùng để chọn, còn 0.9796 ở Mục 1 là kết quả trên test của một lần fit riêng với toàn bộ tập train). Đây là khác biệt bình thường giữa các lần đánh giá trên các tập/khởi tạo khác nhau chứ không phải sai số hệ thống. → Chọn **(256, 128, 64)** làm kiến trúc để lưu vào pipeline triển khai (`models/mlp_pipeline.joblib`) vì nó là lựa chọn tốt nhất theo quy trình chọn mô hình hợp lệ (chọn trên validation, không chọn trên test).

---

## 3. Loss curve

Loss curve minh hoạ cho MLP (128,64) chuẩn hoá (mô hình dùng xuyên suốt Mục 3, 6–8, 11, 12 để minh hoạ các bước phân tích — khác với kiến trúc (256,128,64) được chọn để lưu triển khai ở Mục 2, xem giải thích ở Mục 10).

Loss giảm đều và mượt, không có dấu hiệu dao động/NaN → learning rate và cấu hình phù hợp. `early_stopping=True` giúp dừng đúng lúc (hội tụ sau 64 epoch), tránh train quá lâu không cần thiết.

---

## 4. So sánh 3 activation function

Accuracy đo trên tập **validation**.

| Activation | Accuracy (validation) | Số epoch hội tụ |
|---|---|---|
| **relu** | **0.9782** | 44 |
| tanh | 0.9771 | 37 |
| logistic | 0.9741 | 37 |

**Giải thích vanishing gradient:** hàm `logistic` (sigmoid) có đạo hàm tối đa chỉ 0.25 và bão hoà (đạo hàm ≈ 0) khi input lớn/nhỏ ở hai đầu. Qua nhiều tầng, đạo hàm bị nhân dồn liên tiếp → gradient lan truyền về các tầng đầu ngày càng nhỏ dần ("vanishing") → trọng số cập nhật rất chậm và bị kẹt ở nghiệm kém tối ưu hơn, dẫn tới accuracy thấp nhất (0.9741) dù chỉ dừng sau 37 epoch (do early stopping thấy loss không cải thiện thêm chứ không phải vì đã hội tụ tốt). ReLU (đạo hàm = 1 khi z>0, không bão hoà phía dương) cho phép gradient lan truyền hiệu quả hơn qua nhiều tầng, nên đạt accuracy cao nhất dù cần nhiều epoch nhất (44) để khai thác hết khả năng học của mạng.

---

## 5. So sánh 3 learning rate

Accuracy đo trên tập **validation**.

| Learning rate | Accuracy (validation) | Ghi chú |
|---|---|---|
| 1e-2 | 0.9725 | học nhanh nhưng kém ổn định, dễ "nhảy" qua điểm tối ưu |
| **1e-3** | **0.9782** | tối ưu |
| 1e-4 | 0.9739 | học chậm hơn, hội tụ chưa hết trong ngân sách epoch cho phép |

**Nhận xét:** learning rate 1e-3 cho kết quả tốt nhất — đủ lớn để hội tụ nhanh, đủ nhỏ để không dao động quá mức quanh điểm tối ưu. 1e-2 tuy nhanh nhưng bước nhảy lớn khiến mô hình khó ổn định ở gần cực tiểu, còn 1e-4 quá thận trọng nên chưa khai thác hết khả năng học trong cùng số epoch.

---

## 6. Ma trận nhầm lẫn 10×10

**Cặp số hay bị nhầm nhất:** số thật `4` bị đoán thành `9` (26 lần trong tập test). Đây là cặp số kinh điển hay bị nhầm trên MNIST vì nét viết tay của "4" và "9" có phần thân trên khá giống nhau, đặc biệt khi nét vòng của số 9 viết không khép kín hoặc nét số 4 viết theo kiểu có gạch nối phía trên.

---

## 7. Ảnh dự đoán sai

Có 20 ảnh dự đoán sai trong batch đầu được lưu lại để minh hoạ. Phần lớn là những chữ số viết mơ hồ, nét chưa rõ ràng hoặc bị biến dạng — ngay cả mắt người cũng dễ nhầm ở một vài trường hợp, cho thấy phần lỗi còn lại không hoàn toàn do model yếu mà một phần do chất lượng/độ mơ hồ của dữ liệu gốc.

---

## 8. Bảng Human-in-the-loop (ngưỡng tin cậy 99%)

| Nhóm | Số lượng | Tỉ lệ | Accuracy trong nhóm |
|---|---|---|---|
| Tự động xử lý (confidence ≥ 99%) | 13.540 / 14.000 | **96,7%** | **99,35%** |
| Chuyển người kiểm tra (confidence < 99%) | 460 / 14.000 | 3,3% | — |
| **Tổng thể (không lọc ngưỡng)** | 14.000 / 14.000 | 100% | 97,96% |

**Ý nghĩa nghiệp vụ:** với ngưỡng 99%, hệ thống tự động duyệt tới 96,7% số séc — và trong nhóm đó độ chính xác đạt 99,35%, gần như không có sai sót lọt qua ở nhóm tự động. Chỉ 3,3% còn lại (những trường hợp máy "lưỡng lự") cần chuyển cho nhân viên kiểm tra thủ công — đúng tinh thần human-in-the-loop bắt buộc trong nghiệp vụ tài chính. So với việc nhân viên phải nhập tay 100% séc, hệ thống này giúp giảm khối lượng công việc thủ công xuống chỉ còn ~3,3%, một mức cải thiện hiệu quả rất lớn so với xử lý thủ công toàn bộ.

---

## 9. Model đã lưu (deploy)

| | |
|---|---|
| Kiến trúc | (256, 128, 64) — chọn theo Mục 2 dựa trên validation |
| Tiền xử lý | chia pixel cho 255.0 (`FunctionTransformer`), **đồng nhất** với tiền xử lý dùng để đo accuracy trong toàn bộ báo cáo — không dùng `StandardScaler` |
| Accuracy trên test | 0.9766 |
| File | `models/mlp_pipeline.joblib` |

Script `train.py` có bước `assert` tự kiểm tra accuracy của pipeline đã lưu khớp với accuracy đã báo cáo ở Mục 2 trước khi ghi file, để đảm bảo model deploy đúng là model đã được đánh giá trong báo cáo này.

---

## 10. Hạn chế

- **MLP làm mất cấu trúc không gian của ảnh** — đây là hạn chế lớn nhất so với CNN (không nằm trong phạm vi bài này) cho các bài toán ảnh thực tế có độ phức tạp cao hơn (chữ viết tay đa dạng, ảnh chụp nghiêng, nhiễu...).
- Báo cáo dùng hai mô hình cho hai mục đích khác nhau, cần đọc đúng ngữ cảnh: MLP (128,64) cố định dùng để minh hoạ loss curve / so sánh activation / learning rate / confusion matrix / human-in-the-loop (acc test = 0.9796), còn kiến trúc (256,128,64) là kiến trúc được **chọn qua quy trình validation hợp lệ** ở Mục 2 và là kiến trúc thực sự được lưu vào `models/mlp_pipeline.joblib` để triển khai (acc test = 0.9766). Đây không phải là sai lệch/bug — nhưng cần nêu rõ để người đọc không nhầm là cùng một model.
- Ngưỡng 99% ở Mục 8 là một lựa chọn — trong triển khai thực tế cần tinh chỉnh theo khẩu vị rủi ro và chi phí nhân sự kiểm tra thủ công của ngân hàng.
- Toàn bộ số liệu trong README này được sinh ra từ `reports/metrics.json` sau khi chạy `src/train.py` — nếu chạy lại (máy khác, hoặc set `RANDOM_STATE` khác) số liệu có thể lệch nhẹ so với bảng trên; nên đối chiếu lại `metrics.json` trước khi nộp bài.

---

## Cấu trúc thư mục

```
TT-10-MLP-HoTen/
├── README.md                     ← file này
├── src/train.py                  ← script chạy toàn bộ pipeline + lưu model
├── models/mlp_pipeline.joblib    ← model đã train (chia /255 + MLP (256,128,64))
├── reports/
│   ├── metrics.json              ← toàn bộ số liệu gốc, nguồn duy nhất cho README
│   ├── loss_curves.png
│   ├── kien_truc_comparison.png
│   ├── activation_comparison.png
│   ├── learning_rate_comparison.png
│   ├── confusion_10x10.png
│   ├── anh_sai.png
└── requirements.txt
```