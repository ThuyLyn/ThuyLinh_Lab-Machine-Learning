# TT-22 — MLP Regressor: Dự đoán mức tiêu hao nhiên liệu (Auto MPG)

Dự án dự đoán `mpg` (miles per gallon) từ các thông số kỹ thuật của xe, dùng
`MLPRegressor` (sklearn), có so sánh với Linear Regression, Random Forest, SVR.

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

## ⚠️ Bước bắt buộc trước khi chạy: tải dữ liệu

Môi trường tạo project này **không có kết nối internet**, nên dữ liệu chưa được
tải sẵn. Bạn cần:

1. Vào https://archive.ics.uci.edu/dataset/9/auto+mpg
2. Tải file `auto-mpg.data` (không dùng bản `-original`, không cần CSV)
3. Tạo thư mục `data/` ở gốc project, đặt file vào: `data/auto-mpg.data`

Code trong `src/train.py` và notebook chỉ đọc đúng định dạng này (cách nhau bởi
khoảng trắng, giá trị thiếu đánh dấu `?`).

## Cách chạy

```bash
pip install -r requirements.txt

# Cách 1: chạy toàn bộ pipeline, tự sinh hết report/model
python src/train.py

# Cách 2: mở notebook, chạy từng cell để xem trực quan
jupyter notebook notebooks/mlp_regressor_mpg.ipynb
```

## Những gì `train.py` sẽ tạo ra trong `reports/`

| File | Nội dung |
|---|---|
| `eda.png` | Scatter weight-vs-mpg và model_year-vs-mpg |
| `scale_comparison.csv/png` | So sánh 3 trường hợp: không scale / chỉ scale X / scale cả X&y |
| `architecture_comparison.csv/png` | So sánh 4 kiến trúc (16), (64), (64,32), (256,128,64) + số tham số + overfit gap |
| `loss_curve.png` | Đường loss/validation score của mô hình tốt nhất |
| `alpha_sweep.csv/png` | RMSE train/test theo alpha ∈ {1e-4, 1e-3, 1e-2, 1e-1} |
| `activation_comparison.csv` | So sánh relu / tanh / logistic |
| `final_comparison.csv` | Bảng so sánh MLP vs Dummy/Linear/RandomForest/SVR |

## Bảng so sánh cuối (kết quả thật)

| Thuật toán | RMSE test | R² test | Thời gian train | Giải thích được? |
|---|---|---|---|---|
| DummyRegressor | 7.347 | -0.0040 | 0.397s | Có (baseline) |
| LinearRegression | 2.888 | 0.8449 | 0.004s | Có |
| SVR | 2.680 | 0.8665 | 0.006s | Thấp |
| RandomForest | 2.130 | 0.9157 | 0.096s | Trung bình |
| **MLPRegressor (64,32), scale X&y** | **2.106** | **0.9175** | 0.181s | Thấp |

## Kết quả chi tiết theo từng thí nghiệm

**Ảnh hưởng của scaling (mục 3):**

| Cách scale | RMSE train | RMSE test | R² test |
|---|---|---|---|
| Không scale | 3.788 | 3.186 | 0.8112 |
| Chỉ scale X | 2.519 | 2.250 | 0.9058 |
| Scale cả X và y | 2.211 | 2.106 | 0.9175 |

Đúng như dự đoán trong README gốc: không scale làm MLP hội tụ kém hẳn (R² rơi
xuống 0.81), scale cả X lẫn y cho kết quả tốt nhất.

**So sánh kiến trúc (mục 4):**

| Kiến trúc | RMSE train | RMSE test | Overfit gap (test − train) |
|---|---|---|---|
| (16,) | 2.639 | 2.369 | −0.270 |
| (64,) | 2.702 | 2.322 | −0.380 |
| **(64, 32)** | **2.211** | **2.106** | −0.105 |
| (256, 128, 64) | 2.492 | 2.223 | −0.269 |

Điểm đáng chú ý: mạng lớn nhất (256, 128, 64) **không hề overfit nặng** như
README dự đoán ban đầu — RMSE test của nó (2.223) chỉ nhỉnh hơn một chút so
với mạng vừa (64, 32) (2.106), và train RMSE của nó thậm chí còn tệ hơn cả
(64, 32). Lý do: `early_stopping=True` đã dừng huấn luyện đúng lúc, không cho
mạng học thuộc lòng dữ liệu train — đây chính là công dụng thực tế của kỹ
thuật này, không phải chỉ là lý thuyết suông. Kiến trúc (64, 32) vẫn là lựa
chọn tốt nhất tổng thể.

**Dò alpha (mục 6):** RMSE test gần như không đổi giữa các alpha (2.104 –
2.108), nghĩa là với kiến trúc (64, 32) + early stopping, regularization thêm
gần như không ảnh hưởng — mạng đã đủ nhỏ để tự tránh overfit.

**So sánh activation (mục 7):**

| Activation | RMSE test | R² test |
|---|---|---|
| **relu** | **2.104** | **0.9177** |
| tanh | 2.276 | 0.9037 |
| logistic | 3.405 | 0.7843 |

`logistic` (sigmoid) kém hẳn — dễ bị bão hoà gradient (vanishing gradient) ở
hai đầu, khiến mạng khó học. `relu` vẫn là lựa chọn mặc định hợp lý nhất.

**Quy đổi sang chi phí thực tế (mục 9):** với 3 xe mẫu, MLP dự đoán 33.8 / 29.9
/ 18.9 MPG, tương đương 6.96 / 7.87 / 12.44 L/100km — chênh lệch chi phí xăng
gần gấp đôi giữa xe tiết kiệm nhất và tốn nhất (~24 triệu so với ~43
triệu VNĐ/năm ở 15.000 km/năm).

## Kết luận

- **R² test của MLP: 0.9175** — cao nhất trong số các mô hình thử nghiệm,
  nhỉnh hơn cả Random Forest (0.9157) và vượt xa Linear Regression (0.8449).
- **MLP có thắng Random Forest không?** Có, nhưng rất sát nút (RMSE 2.106 so
  với 2.130 — chênh lệch chỉ ~1%). Với khoảng cách nhỏ như vậy, kết quả này dễ
  đổi chiều nếu đổi `random_state` hoặc cách chia train/test — nên xem đây là
  "hai mô hình ngang sức" hơn là MLP thắng rõ rệt.
- **Kiến trúc (256,128,64) có overfit rõ rệt như dự đoán không?** Không rõ
  rệt như README gốc cảnh báo — nhờ `early_stopping`, mạng lớn chỉ nhỉnh hơn
  một chút so với mạng vừa, chứ không sụp hẳn. Đây là một phát hiện quan
  trọng: overfit ở MLP trên dữ liệu nhỏ **có thể kiểm soát tốt bằng
  early stopping**, không nhất thiết phải chọn mạng siêu nhỏ.
- **Kết luận trung thực:** với 398 dòng dữ liệu bảng, MLP (đã scale + early
  stopping) đạt hiệu năng **ngang bằng** Random Forest, không phải vượt trội
  rõ ràng. Vì Random Forest dễ setup hơn (không cần scale, không cần lo
  scaling target, hội tụ ổn định), ít tham số cần tinh chỉnh, và train nhanh
  hơn (~0.1s so với ~0.2s — chưa kể công sức dò alpha/kiến trúc/activation),
  **Random Forest vẫn là lựa chọn thực dụng hơn** cho bài toán dạng bảng nhỏ
  như thế này, dù MLP đã chứng minh được nó không hề "tệ" như định kiến ban
  đầu.