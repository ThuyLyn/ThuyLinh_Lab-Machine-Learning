# TT-11 — Linear Regression: Định giá nhà ở California

## 1. Tổng quan
Bài toán: dự đoán giá nhà trung vị (`MedHouseVal`) từ 8 đặc trưng của bộ dữ liệu
California Housing, dùng Linear Regression (OLS) — model có hệ số diễn giải được,
phù hợp cho nghiệp vụ thẩm định giá cần giải thích quyết định.

## 2. Cách chạy
```bash
pip install -r requirements.txt
python src/train.py
```
Script dùng trực tiếp `sklearn.datasets.fetch_california_housing()` để tải dữ liệu
(cần kết nối mạng ra ngoài, sklearn sẽ tự tải và cache lại ở lần chạy đầu), sau đó
huấn luyện và ghi toàn bộ hình + model vào `reports/` và `models/`. Có thể chạy
tương tác từng bước trong `notebooks/linear_regression_housing.ipynb`.

## 3. Kết quả thực tế (chạy trên máy bạn)

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Baseline (Dummy — đoán trung bình) | 1.1449 | – | – |
| **Linear Regression** | **0.6794** | 0.4978 | **0.6477** |
| + log-transform | 0.7373 | – | 0.5852 |
| + feature engineering (khoảng cách SF/LA) | 0.6684 | – | 0.6590 |

MAPE của Linear Regression cơ bản: **30.39%**.

So với mức tham chiếu trong đề gốc (R² ~0.58–0.61, RMSE ~0.72–0.75), model của bạn
đạt **R² = 0.648, RMSE = 0.679 — tốt hơn mức tham chiếu**, đạt yêu cầu "RMSE tốt hơn
baseline rõ rệt" (0.679 so với 1.145 baseline, giảm ~41%).

## 4. Kiểm tra 4 giả định của hồi quy tuyến tính

| # | Giả định | Cách kiểm tra | Kết luận |
|---|---|---|---|
| ① | Tuyến tính | Scatter plot, residual plot | `MedInc` khá tuyến tính với giá; `Latitude/Longitude` thì **không** — quan hệ toạ độ–giá rõ ràng phi tuyến (xem `ban_do_gia.png`), model tuyến tính bắt kém phần này |
| ② | Độc lập | Dữ liệu theo block dân cư, không có chuỗi thời gian lặp | Chấp nhận được cho bài toán này |
| ③ | Phương sai đều | `reports/residual_plot.png` | **Vi phạm**: phần dư có xu hướng lan rộng khi giá dự đoán tăng → hình phễu — dấu hiệu heteroscedasticity điển hình khi dự đoán giá |
| ④ | Sai số chuẩn | `reports/qq_plot.png` | Phần dư lệch khỏi đường chuẩn ở hai đầu, một phần do nhãn bị cắt ngọn ở 5.0 (965 dòng, 4.68% dữ liệu) |

**Thử log-transform** (`log(1+giá)`) để giảm heteroscedasticity: kết quả thực tế
**không cải thiện** — RMSE tăng từ 0.679 lên 0.737, R² giảm từ 0.648 xuống 0.585.
→ Kết luận: với bộ dữ liệu này, log-transform đánh đổi độ chính xác dự đoán để lấy
phần dư "đều" hơn về mặt lý thuyết, nhưng không đáng — giữ nguyên model gốc (dự đoán
giá trực tiếp) là lựa chọn tốt hơn.

## 5. Đa cộng tuyến (VIF)

| Feature | VIF |
|---|---|
| Latitude | 9.777 |
| Longitude | 9.315 |
| AveRooms | 3.354 |
| MedInc | 2.962 |
| AveBedrms | 1.808 |
| HouseAge | 1.280 |
| Population | 1.206 |
| AveOccup | 1.114 |

`Latitude`/`Longitude` có VIF gần ngưỡng cảnh báo 10 (vì cùng mã hoá vị trí địa lý —
biết một cái gần như biết cả vùng), cần lưu ý khi diễn giải riêng lẻ hai hệ số này.
`AveRooms`/`AveBedrms` (đã cảnh báo trong đề bài gốc) ở mức chấp nhận được (VIF < 5)
sau khi clip outlier ở bước 2 — đa cộng tuyến không phải vấn đề nghiêm trọng ở đây,
ngoại trừ cặp toạ độ.

## 6. Bảng hệ số đã chuẩn hoá

| Feature | Hệ số | Diễn giải |
|---|---|---|
| Latitude | −0.917 | Càng lên phía Bắc California (giữ các yếu tố khác cố định), giá nhà trung vị có xu hướng giảm — vùng phía Nam (LA) đắt hơn |
| Longitude | −0.845 | Càng về phía Đông (xa bờ biển), giá nhà có xu hướng giảm |
| MedInc | +0.825 | Thu nhập khu vực càng cao, giá nhà càng cao — yếu tố ảnh hưởng dương mạnh nhất |
| AveOccup | −0.268 | Mật độ người/hộ càng cao, giá nhà có xu hướng thấp hơn |
| AveBedrms | +0.156 | Càng nhiều phòng ngủ trung bình, giá có xu hướng tăng nhẹ |
| HouseAge | +0.145 | Nhà càng cũ (block lâu đời hơn), giá có xu hướng cao hơn một chút — có thể do các khu trung tâm lâu đời thường đắt hơn |
| AveRooms | −0.142 | Số phòng trung bình càng cao lại có hệ số âm nhẹ — khả năng do đa cộng tuyến với AveBedrms/AveOccup, không nên diễn giải độc lập |
| Population | +0.051 | Ảnh hưởng rất yếu, gần như không đáng kể |

**3 yếu tố ảnh hưởng mạnh nhất:** `Latitude` (−0.917), `Longitude` (−0.845), `MedInc`
(+0.825) — vị trí địa lý và thu nhập khu vực chi phối phần lớn giá nhà.

⚠️ Đây là **tương quan**, không phải quan hệ nhân quả. Không thể kết luận "tăng thu
nhập LÀM giá nhà tăng" — cả hai đều chịu ảnh hưởng của các yếu tố khác (khu vực, cơ sở
hạ tầng...). Tương tự, hệ số âm của `AveRooms` không có nghĩa "nhiều phòng hơn thì
nhà rẻ hơn" — đây là hệ quả của đa cộng tuyến (mục 5), cần cẩn trọng khi trình bày.

## 7. Hạn chế
- **Nhãn bị cắt ngọn ở 5.0** (500k USD): **965 dòng (4.68% dữ liệu)** bị dồn đúng giá
  trần → model **không bao giờ dự đoán được nhà đắt hơn 500k**, cần nêu rõ khi dùng
  thực tế (ví dụ hệ thống cảnh báo tin đăng giá bất thường sẽ luôn báo sai với phân
  khúc siêu cao cấp).
- **Quan hệ toạ độ–giá là phi tuyến**: model tuyến tính bắt kém các cụm giá cao cục bộ
  quanh San Francisco/Los Angeles — R² chỉ đạt 0.648, còn cách khá xa so với Random
  Forest (~0.80, theo README gốc) chính vì lý do này. Feature engineering (khoảng
  cách tới SF/LA) đã giúp cải thiện nhẹ (R² 0.648 → 0.659) nhưng chưa giải quyết
  triệt để.
- Outlier ở `AveRooms`, `AveOccup`, `AveBedrms`, `Population` đã được clip ở phân vị 99
  trước khi huấn luyện (207 dòng bị ảnh hưởng ở mỗi cột).
- Đa cộng tuyến giữa `Latitude`/`Longitude` (VIF ~9.3–9.8) khiến việc diễn giải riêng
  lẻ hai hệ số này kém tin cậy hơn các đặc trưng khác.

## 8. Cấu trúc thư mục
```
TT-11-LinearRegression/
├── README.md
├── requirements.txt
├── notebooks/linear_regression_housing.ipynb
├── src/train.py
├── models/lr_pipeline.joblib
└── reports/
    ├── ban_do_gia.png
    ├── residual_plot.png
    ├── residual_plot_log.png
    ├── qq_plot.png
    ├── he_so.png
    ├── he_so.csv
    └── vif.csv
```