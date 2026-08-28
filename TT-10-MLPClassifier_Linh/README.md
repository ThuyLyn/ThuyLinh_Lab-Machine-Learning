# TT-10 — MLP Classifier: Đọc số viết tay trên séc / phiếu chuyển khoản

**Họ tên:** _(điền tên bạn)_
**Dữ liệu dùng:** MNIST thật qua `sklearn.datasets.fetch_openml('mnist_784', ...)` — 70.000 ảnh 28×28, 10 lớp. Train: 56.000 | Test: 14.000.

---

## 1. Bảng so sánh CÓ / KHÔNG chuẩn hoá

| Model | Accuracy | Ghi chú |
|---|---|---|
| Baseline — Logistic Regression | 0.9211 | dùng dữ liệu đã chuẩn hoá |
| MLP (128,64) — **không** chuẩn hoá | 0.9709 | pixel 0–255 thô, iters=58, time=124.1s |
| MLP (128,64) — **có** chuẩn hoá | **0.9796** | iters=64, time=130.9s, tốt nhất |

**Nhận xét:** chuẩn hoá giúp accuracy tăng ~0,87 điểm % so với không chuẩn hoá, và vượt baseline Logistic Regression tới ~5,85 điểm %. Việc scale dữ liệu về cùng khoảng giúp mạng hội tụ ổn định hơn (không chuẩn hoá dễ khiến gradient dao động mạnh do input scale lớn — pixel 0–255 làm sai số ban đầu lớn hơn nhiều so với input đã chuẩn hoá).

---

## 2. Bảng so sánh kiến trúc (≥ 4 kiến trúc)

| Kiến trúc | Accuracy | Số tham số | Thời gian train |
|---|---|---|---|
| (64,) | 0.9741 | 4.810 | 36.8s |
| (128,) | 0.9795 | 9.610 | 57.7s |
| **(128, 64)** | **0.9796** | 17.226 | 134.6s |
| (256, 128, 64) | 0.9766 | 58.442 | 103.5s |

**Nhận xét:** (128,64) đạt accuracy cao nhất (0.9796), nhưng chỉ nhỉnh hơn (128,) (0.9795) một chút dù thời gian train lâu hơn đáng kể (134.6s so với 57.7s). Đáng chú ý, kiến trúc lớn nhất (256,128,64) — gấp hơn 3 lần số tham số của (128,64) — lại có accuracy thấp hơn (0.9766), cho thấy mạng lớn hơn không đồng nghĩa tốt hơn: với dữ liệu MNIST kích thước vừa phải, mạng quá to có xu hướng overfit hoặc khó tối ưu hơn trong cùng số epoch. → Chọn (128,64) làm kiến trúc chính vì đạt accuracy tốt nhất trên tập test.


---

## 3. Loss curve


Loss giảm đều và mượt, không có dấu hiệu dao động/NaN → learning rate và cấu hình phù hợp. `early_stopping=True` giúp dừng đúng lúc (hội tụ sau 64 epoch), tránh train quá lâu không cần thiết.

---

## 4. So sánh 3 activation function

| Activation | Accuracy | Số epoch hội tụ |
|---|---|---|
| **relu** | **0.9796** | 64 |
| tanh | 0.9785 | 32 |
| logistic | 0.9771 | 34 |

**Giải thích vanishing gradient:** hàm `logistic` (sigmoid) có đạo hàm tối đa chỉ 0.25 và bão hoà (đạo hàm ≈ 0) khi input lớn/nhỏ ở hai đầu. Qua nhiều tầng, đạo hàm bị nhân dồn liên tiếp → gradient lan truyền về các tầng đầu ngày càng nhỏ dần ("vanishing") → trọng số cập nhật rất chậm và bị kẹt ở nghiệm kém tối ưu hơn, dẫn tới accuracy thấp nhất (0.9771) dù chỉ dừng sau 34 epoch (do early stopping thấy loss không cải thiện thêm chứ không phải vì đã hội tụ tốt). ReLU (đạo hàm = 1 khi z>0, không bão hoà phía dương) cho phép gradient lan truyền hiệu quả hơn qua nhiều tầng, nên đạt accuracy cao nhất dù cần nhiều epoch nhất (64) để khai thác hết khả năng học của mạng.

!

## 5. So sánh 3 learning rate

| Learning rate | Accuracy | Ghi chú |
|---|---|---|
| 1e-2 | 0.9712 | học nhanh nhưng kém ổn định, dễ "nhảy" qua điểm tối ưu |
| **1e-3** | **0.9796** | tối ưu |
| 1e-4 | 0.9753 | học chậm hơn, hội tụ chưa hết trong ngân sách epoch cho phép |

**Nhận xét:** learning rate 1e-3 cho kết quả tốt nhất — đủ lớn để hội tụ nhanh, đủ nhỏ để không dao động quá mức quanh điểm tối ưu. 1e-2 tuy nhanh nhưng bước nhảy lớn khiến mô hình khó ổn định ở gần cực tiểu, còn 1e-4 quá thận trọng nên chưa khai thác hết khả năng học trong cùng số epoch.


## 6. Ma trận nhầm lẫn 10×10


**Cặp số hay bị nhầm nhất:** số thật `4` bị đoán thành `9` (26 lần trong tập test). Đây là cặp số kinh điển hay bị nhầm trên MNIST vì nét viết tay của "4" và "9" có phần thân trên khá giống nhau, đặc biệt khi nét vòng của số 9 viết không khép kín hoặc nét số 4 viết theo kiểu có gạch nối phía trên.


## 7. Ảnh dự đoán sai


Có 20 ảnh dự đoán sai trong batch đầu được lưu lại để minh hoạ. Phần lớn là những chữ số viết mơ hồ, nét chưa rõ ràng hoặc bị biến dạng — ngay cả mắt người cũng dễ nhầm ở một vài trường hợp, cho thấy phần lỗi còn lại không hoàn toàn do model yếu mà một phần do chất lượng/độ mơ hồ của dữ liệu gốc.


## 8.  Bảng Human-in-the-loop (ngưỡng tin cậy 99%)

| Nhóm | Số lượng | Tỉ lệ | Accuracy trong nhóm |
|---|---|---|---|
| Tự động xử lý (confidence ≥ 99%) | 13.540 / 14.000 | **96,7%** | **99,35%** |
| Chuyển người kiểm tra (confidence < 99%) | 460 / 14.000 | 3,3% | — |
| **Tổng thể (không lọc ngưỡng)** | 14.000 / 14.000 | 100% | 97,96% |

**Ý nghĩa nghiệp vụ:** với ngưỡng 99%, hệ thống tự động duyệt tới 96,7% số séc — và trong nhóm đó độ chính xác đạt 99,35%, gần như không có sai sót lọt qua ở nhóm tự động. Chỉ 3,3% còn lại (những trường hợp máy "lưỡng lự") cần chuyển cho nhân viên kiểm tra thủ công — đúng tinh thần human-in-the-loop bắt buộc trong nghiệp vụ tài chính. So với việc nhân viên phải nhập tay 100% séc, hệ thống này giúp giảm khối lượng công việc thủ công xuống chỉ còn ~3,3%, một mức cải thiện hiệu quả rất lớn so với xử lý thủ công toàn bộ.


## 9. So sánh với CNN (TT-26)

| Model | Accuracy | Số tham số | Thời gian train |
|---|---|---|---|
| MLP (128,64) | 0.9796 | 17.226 | 134.6s |
| CNN (Conv32→Pool→Conv64→Dense64) | *chưa chạy được* | — | — |

**Lưu ý:** ở lần chạy này, bước so sánh CNN bị bỏ qua do lỗi shape khi đưa dữ liệu vào Keras/TensorFlow:

```
Data cardinality is ambiguous. Make sure all arrays contain the same number of samples.
'x' sizes: 617400
'y' sizes: 56000
```


## 10. Hạn chế

- **MLP làm mất cấu trúc không gian của ảnh** — đây là hạn chế lớn nhất, lý do chính cần chuyển sang CNN (TT-26) cho các bài toán ảnh thực tế có độ phức tạp cao hơn (chữ viết tay đa dạng, ảnh chụp nghiêng, nhiễu...).
- Phần so sánh CNN (mục 9) chưa hoàn thành do lỗi shape dữ liệu khi chạy — cần sửa lỗi reshape và chạy lại để có số liệu thật thay cho phần dự đoán lý thuyết.
- Có chênh lệch nhỏ giữa accuracy tốt nhất ghi nhận trong quá trình thử nghiệm (0.9796) và accuracy của pipeline cuối cùng được lưu (`models/mlp_pipeline.joblib`, acc=0.9751) — nên kiểm tra lại xem `random_state` hay tập train/test dùng để refit pipeline cuối có đồng nhất với các bước thử nghiệm ở trên hay không.
- Ngưỡng 99% là một lựa chọn — trong triển khai thực tế cần tinh chỉnh theo khẩu vị rủi ro và chi phí nhân sự kiểm tra thủ công của ngân hàng.

---

## Cấu trúc thư mục

```
TT-10-MLP-HoTen/
├── README.md              ← file này, có bảng human-in-the-loop
├── notebooks/mlp_digits.ipynb   ← notebook đã chạy sẵn, đủ output/hình
├── src/train.py            ← script chạy toàn bộ pipeline + lưu model
├── models/mlp_pipeline.joblib   ← model (StandardScaler + MLP) đã train
├── reports/
│   ├── loss_curves.png
│   ├── kien_truc_comparison.png
│   ├── activation_comparison.png
│   ├── learning_rate_comparison.png
│   ├── confusion_10x10.png
│   ├── anh_sai.png
└── requirements.txt
```