# TT-22 — MLP Regressor: Dự đoán mức tiêu hao nhiên liệu (Auto MPG)

Dự án dự đoán `mpg` (miles per gallon) từ các thông số kỹ thuật của xe, dùng
`MLPRegressor` (sklearn), có so sánh với Dummy, Linear Regression, Random Forest, SVR.

## Cấu trúc

```
TT-22-MLPRegressor/
├── README.md
├── requirements.txt
├── dataset/                     
├── notebooks/
│   └── mlp_regressor_mpg.ipynb  
├── src/
│   └── train.py              
├── models/
│   └── mlp_reg.joblib     
└── reports/                 
```

## Dữ liệu
- `horsepower` thiếu 6 giá trị (`?`) → điền median.
- Bỏ `car_name` (chỉ là định danh); `origin` (mã vùng 1/2/3) → one-hot.
- Kết quả: 398 dòng, 9 đặc trưng + `mpg`. Chia train/test = 318/80 (`random_state=42`).

## Cách chạy

```bash
pip install -r requirements.txt

# Cách 1: chạy toàn bộ pipeline, tự sinh hết report/model
python src/train.py
# (hoặc: python src/train.py --data "đường/dẫn/auto-mpg.data")

# Cách 2: mở notebook, chạy từng cell để xem trực quan
jupyter notebook notebooks/mlp_regressor_mpg.ipynb
```

## Phương pháp (quan trọng để đọc đúng kết quả)

- **Tập test chỉ dùng một lần**, ở bước cuối. Mọi lựa chọn (scaling, kiến trúc, alpha,
  activation) được chọn bằng cross-validation `RepeatedKFold` (5 fold × 3 lần) **chỉ trên tập train**.
- Các lựa chọn chốt lần lượt: kiến trúc → alpha → activation (tìm từng yếu tố một, không phải lưới đầy đủ).
- Mọi MLP dùng `early_stopping=True`, `n_iter_no_change=30`, `max_iter=2000`.
- **Overfit gap = RMSE val − RMSE train**, lấy trung bình 15 fold CV (kèm độ lệch chuẩn),
  thay vì một lần chia train/test 80 mẫu vốn rất nhiễu.
- MLP vs Random Forest trên test được so bằng bootstrap (2000 lần) khoảng tin cậy 95% của hiệu RMSE.

## Những gì `train.py` tạo ra trong `reports/`

| File | Nội dung |
|---|---|
| `eda.png` | Scatter weight-vs-mpg và model_year-vs-mpg |
| `scale_comparison.csv/png` | So sánh 3 trường hợp: không scale / chỉ scale X / scale cả X&y (CV) |
| `architecture_comparison.csv`, `kien_truc_overfit.png` | So sánh 4 kiến trúc (16), (64), (64,32), (256,128,64) + số tham số + overfit gap (CV) |
| `alpha_sweep.csv/png` | RMSE theo alpha ∈ {1e-4, 1e-3, 1e-2, 1e-1, 1} (CV) |
| `activation_comparison.csv` | So sánh relu / tanh / logistic (CV) |
| `loss_curve.png` | `loss_curve_` và `validation_scores_` (R²) của MLP cuối, hai trục riêng |
| `final_comparison.csv` | Bảng tổng hợp CV + test của Dummy/Linear/SVR/RandomForest/MLP |
| `metrics.json` | Cấu hình được chọn, toàn bộ số liệu cuối, CI bootstrap MLP − RF |

## Bảng so sánh cuối

| Thuật toán | RMSE val (CV) | RMSE test | R² test | Thời gian train | Giải thích được? |
|---|---|---|---|---|---|
| DummyRegressor | 7.953 ± 0.524 | 7.347 | −0.0040 | 0.03s | Có (baseline) |
| LinearRegression | 3.470 ± 0.274 | 2.888 | 0.8449 | 0.001s | Có |
| SVR | 3.535 ± 0.384 | 2.680 | 0.8665 | 0.006s | Thấp |
| RandomForest | 2.964 ± 0.353 | 2.130 | 0.9157 | 0.11s | Trung bình |
| **MLPRegressor (64,32), alpha=0.1, relu, scale X&y** | **2.953 ± 0.313** | **2.105** | **0.9176** | 0.18s | Thấp |

Thời gian train chỉ mang tính tham khảo (đo một lần, phụ thuộc máy).
Khoảng tin cậy 95% của RMSE(MLP) − RMSE(RF) trên test: **[−0.404, +0.322]**, chứa 0 →
không phân biệt được hai mô hình.

## Kết quả chi tiết theo từng thí nghiệm (CV trên tập train)

**Ảnh hưởng của scaling:**

| Cách scale | RMSE val | RMSE train |
|---|---|---|
| Không scale | 5.086 ± 4.949 | 4.849 |
| Chỉ scale X | 3.161 ± 0.300 | 2.785 |
| Scale cả X và y | **2.954 ± 0.292** | 2.537 |

Không scale làm MLP rất không ổn định (độ lệch chuẩn gần bằng giá trị trung bình) vì các
đặc trưng khác thang đo (`weight` hàng nghìn, `acceleration` hàng chục). Scale X giúp rõ
rệt; scale thêm y cho kết quả tốt nhất. Đây là yếu tố ảnh hưởng mạnh nhất đối với MLP.

**So sánh kiến trúc:**

| Kiến trúc | RMSE val | RMSE train | Overfit gap |
|---|---|---|---|
| (16,) | 3.142 ± 0.337 | 2.838 | +0.304 |
| (64,) | 3.092 ± 0.379 | 2.808 | +0.284 |
| **(64, 32)** | **2.954 ± 0.292** | 2.537 | +0.418 |
| (256, 128, 64) | 2.967 ± 0.396 | 2.481 | +0.486 |

Mạng lớn nhất không tốt hơn mạng vừa, và gap tăng nhẹ theo độ phức tạp. Tuy nhiên mọi gap
đều nhỏ (+0.3 đến +0.5) và độ lệch chuẩn của gap (~0.4) gần bằng chính nó, nên không nên
diễn giải chi tiết từng chênh lệch. Cũng không nên quy "mạng không overfit nặng" cho riêng
`early_stopping`, vì thí nghiệm này không có nhóm đối chứng tắt early stopping.

**Dò alpha (kiến trúc (64, 32)):**

| alpha | RMSE val | RMSE train |
|---|---|---|
| 1e-4 | 2.954 ± 0.292 | 2.537 |
| 1e-3 | 2.958 ± 0.295 | 2.525 |
| 1e-2 | 2.956 ± 0.281 | 2.539 |
| **1e-1** | **2.953 ± 0.313** | 2.550 |
| 1 | 2.991 ± 0.355 | 2.715 |

RMSE gần như không đổi từ 1e-4 đến 1e-1 (chênh < 0.01, nhỏ hơn nhiều so với độ lệch chuẩn):
trên bài toán nhỏ này, regularization L2 thêm vào gần như không ảnh hưởng. `alpha=0.1` chỉ
là giá trị có trung bình thấp nhất, không phải lựa chọn có ý nghĩa thống kê.

**So sánh activation:**

| Activation | RMSE val | RMSE train |
|---|---|---|
| **relu** | **2.953 ± 0.313** | 2.550 |
| tanh | 3.024 ± 0.320 | 2.823 |
| logistic | 3.170 ± 0.317 | 3.080 |

`relu` có trung bình tốt nhất; chênh lệch với tanh nhỏ hơn độ lệch chuẩn, còn logistic kém
hơn rõ hơn một chút (logistic dễ bão hoà gradient, nhưng thí nghiệm này không kiểm chứng
nguyên nhân).

**Quy đổi sang chi phí thực tế:** công thức `L/100km = 235.215 / mpg`. Giả định minh hoạ:
15.000 km/năm, 23.000 VNĐ/lít (nên cập nhật theo giá xăng hiện tại).

| mpg dự đoán | L/100km | Tiền xăng/năm |
|---|---|---|
| 33.7 | 6.97 | ≈ 24,0 triệu VNĐ |
| 30.0 | 7.84 | ≈ 27,0 triệu VNĐ |
| 18.9 | 12.42 | ≈ 42,8 triệu VNĐ |

Chênh lệch giữa xe tiết kiệm nhất và tốn nhất trong 3 xe mẫu gần gấp đôi (~24 so với ~43
triệu VNĐ/năm). Sai số RMSE ≈ 2,1 mpg tương ứng khoảng 1 L/100km ở mức tiêu thụ trung bình.

## Kết luận

- **MLP (R² test 0.9176) ngang Random Forest (0.9157)**, cả hai vượt rõ Linear Regression
  (0.8449) và SVR (0.8665). Điều này cho thấy quan hệ giữa đặc trưng và mpg có tính phi tuyến.
- **MLP không thắng Random Forest một cách có ý nghĩa.** Chênh lệch RMSE chỉ ~1% (2.105 so
  với 2.130 trên test; 2.953 so với 2.964 trên CV), và khoảng tin cậy bootstrap chứa 0. Nên
  xem đây là "hai mô hình ngang sức".
- **Random Forest thực dụng hơn** cho dữ liệu bảng nhỏ này: không cần scale X/y, ít siêu
  tham số, ổn định, trong khi MLP chỉ đạt mức ngang RF sau khi scale cả X và y và dò cấu
  hình. RF overfit nhiều hơn (gap +1.84) nhưng vẫn tổng quát hoá ngang MLP.
- **Scaling là yếu tố then chốt của MLP**; kiến trúc, alpha, activation ảnh hưởng nhỏ trong
  phạm vi đã thử.

## Hạn chế

- Chỉ 398 mẫu, tập test 80 mẫu nên số liệu test nhiễu. RMSE test (≈2.1) thấp hơn hẳn RMSE
  CV (≈2.95) và R² test (0.917) cao hơn R² CV (0.853) cho cùng mô hình → lần chia này tương
  đối "dễ"; không nên dùng số test để xếp hạng chi li.
- Một lần chia train/test, chưa có CV lồng (nested CV).
- Median `horsepower` tính trên toàn bộ dữ liệu trước khi chia (rò rỉ rất nhỏ, 6 dòng);
  cách chuẩn là đưa `SimpleImputer` vào Pipeline.
- Tìm siêu tham số từng yếu tố một trên lưới nhỏ; chưa thử `learning_rate`, `solver`, `batch_size`.
- Mô hình lưu chỉ huấn luyện trên 318 mẫu train, chưa huấn luyện lại trên toàn bộ dữ liệu.
- Dữ liệu xe đời 1970–1982, không đại diện cho xe hiện đại; chi phí xăng chỉ là ví dụ.
- Kết quả có thể lệch nhẹ giữa các phiên bản scikit-learn.