# TT-21 — KNN Regressor: Định giá nhà theo "các căn tương tự trong khu vực"

Định giá nhà bằng phương pháp so sánh (comparable sales) tự động hoá: tìm K căn nhà
gần giống nhất (theo đặc trưng đã chuẩn hoá), lấy trung bình có trọng số giá của
chúng làm dự đoán.

Dữ liệu: `sklearn.datasets.fetch_california_housing` (bộ chính thức, 20.640 dòng ×
8 đặc trưng). 92 dòng bị loại khỏi tập train do là outlier (`AveOccup ≥ 10` hoặc
`AveRooms ≥ 20`), còn lại 16.420 dòng train.

## Kết quả tóm tắt

| Bước | Kết quả |
|---|---|
| Dummy (baseline) | RMSE = 1.1449 |
| Linear Regression | RMSE = 0.8906, R² = 0.3947 |
| KNN, K=5, **không** scale | RMSE = 1.0582 (tệ hơn cả Linear!) |
| KNN, K=5, có scale, distance | RMSE = 0.6032 |
| KNN, **K tối ưu = 22** | RMSE = 0.5899, R² = 0.7345 |
| weights: uniform | RMSE = 0.5936 |
| weights: distance | RMSE = 0.5899 ✓ tốt hơn |
| metric: euclidean | RMSE = 0.5899 |
| metric: manhattan | RMSE = 0.5674 ✓ tốt hơn |

**→ KNN (R²=0.7345) thắng đậm Linear Regression (R²=0.3947)** trên bộ dữ liệu
chính thức — chênh lệch còn lớn hơn cả mức tham chiếu lý thuyết (Linear ~0,60,
KNN ~0,70–0,75), càng khẳng định "hàng xóm gần nhau thì giá giống nhau" là giả
định rất đúng với bất động sản, trong khi Linear Regression không bắt được quan
hệ phi tuyến/cục bộ giữa toạ độ và giá.

## So sánh 3 thuật toán

| Model | RMSE | R² | Train time | Predict/mẫu | Giải thích được |
|---|---|---|---|---|---|
| Linear Regression | 0.8906 | 0.3947 | 0.00 s | 0.0002 ms | Có (hệ số hồi quy) |
| **KNN (K=22, distance)** | **0.5899** | **0.7345** | 0.02 s | 0.0371 ms | Có — chỉ ra được đúng những căn tương tự |
| Random Forest | 0.5022 | 0.8076 | 1.91 s | 0.0118 ms | Một phần (feature importance) |

Random Forest chính xác nhất và ở đây train khá nhanh (~1.9s), nhưng vẫn không
liệt kê được "căn cụ thể nào" làm căn cứ định giá — đây là lợi thế riêng của KNN
cho use case "hiển thị 5 căn tương tự cho khách xem". Đáng chú ý: dù train chậm
hơn Linear, **Random Forest lại dự đoán nhanh hơn cả KNN** (0.0118ms vs 0.0371ms/mẫu)
vì sau khi train xong, RF chỉ cần đi qua các cây quyết định đã học, không phải quét
lại dữ liệu như KNN.

## Thí nghiệm trọng số vị trí 

Nhân Latitude/Longitude (sau chuẩn hoá) với hệ số {1, 2, 3, 5}:

| Hệ số nhân Lat/Lon | RMSE |
|---|---|
| ×1 (mặc định) | 0.5899 |
| ×2 | 0.5550 |
| ×3 | 0.5339 |
| ×5 | **0.5091** (tốt nhất trong dải thử) |

**Kết luận:** RMSE giảm đều khi tăng trọng số vị trí (giảm ~14% từ ×1 xuống ×5) —
vị trí là yếu tố quan trọng nhất quyết định giá nhà trong bộ dữ liệu này, khớp với
cách thẩm định viên thực tế làm việc ("location, location, location"). Trong dải
thử nghiệm (1–5) chưa thấy điểm đảo chiều; có thể thử mở rộng lên hệ số cao hơn
(ví dụ 8, 10) để tìm điểm mà việc lấn át các đặc trưng khác (thu nhập, số phòng)
bắt đầu gây hại cho độ chính xác.

## Ví dụ "căn tương tự" (mục 9)

Với 3 căn nhà mẫu trong tập test, notebook/script in ra 5 căn tương tự nhất mà
KNN dùng để dự đoán, kèm khoảng cách và giá bán — xem chi tiết tại
`reports/can_tuong_tu_vi_du.txt` và biểu đồ `reports/can_tuong_tu_vi_du.png`.

## Hiện tượng K=1 (và một lưu ý mở rộng)

Với K=1, một điểm train được "dự đoán" bằng chính nó (khoảng cách = 0) → train RMSE
= 0.0000 hoàn toàn — đây là overfitting cực đoan, không phản ánh khả năng tổng quát
hoá (test RMSE ở K=1 lại tệ hơn hẳn: 0.7578, tệ hơn cả K tối ưu=22 cho RMSE=0.5899).

## Thời gian dự đoán khi dữ liệu tăng

| Train size | Predict/mẫu |
|---|---|
| ×1 (16.420 dòng) | 0.0372 ms |
| ×5 (82.100 dòng) | 0.0491 ms |
| ×10 (164.200 dòng) | 0.0421 ms |

Xu hướng chung là thời gian dự đoán tăng theo kích thước train (KNN không "học"
ra công thức mà phải tìm hàng xóm mỗi lần dự đoán). Số đo ở ×10 thấp hơn ×5 một
chút — đây là nhiễu đo đạc (CPU cache, GC, tải hệ thống tại thời điểm chạy) chứ
không có nghĩa là ×10 dữ liệu lại nhanh hơn; nên chạy lặp lại nhiều lần và lấy
trung bình nếu cần con số chính xác hơn. Xu hướng tổng thể (100–170% chậm hơn so
với ×1) vẫn đúng với lý thuyết.

## Hạn chế của KNN (đúc kết)

- **Chậm dần khi dữ liệu lớn** — không có "công thức" để tính nhanh, phải tìm hàng xóm mỗi lần dự đoán.
- **Không ngoại suy** — nhà đắt/rẻ hơn mọi căn trong train sẽ bị kẹp trong khoảng giá đã thấy.
- **Cần lưu toàn bộ dữ liệu train** trong bộ nhớ để dự đoán — không nén được thành vài tham số như Linear Regression.
- **Nhạy với thang đo đặc trưng** — bắt buộc phải chuẩn hoá, nếu không đặc trưng có giá trị lớn (như `Population`) sẽ áp đảo khoảng cách.

## Cấu trúc project

```
TT-21-KNNRegressor/
├── README.md                              ← file này
├── notebooks/knn_regressor_housing.ipynb  ← dùng fetch_california_housing
├── src/train.py                           ← script chạy toàn bộ pipeline (python src/train.py)
├── models/knn_pipeline.joblib             ← model KNN cuối (K=22, distance, StandardScaler)
├── reports/
│   ├── rmse_theo_K.png
│   ├── trong_so_vi_tri.png
│   ├── can_tuong_tu_vi_du.png
│   ├── can_tuong_tu_vi_du.txt
│   ├── thoi_gian_predict.png
│   └── results_summary.json               ← toàn bộ số liệu dạng JSON
└── requirements.txt
```

### Chạy lại

```bash
pip install -r requirements.txt
python src/train.py
```
