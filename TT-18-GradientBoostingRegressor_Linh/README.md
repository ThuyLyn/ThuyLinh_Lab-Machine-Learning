# TT-18 — Gradient Boosting Regressor: Kết quả thẩm định giá nhà tự động (AVM)

> Chạy trên bộ dữ liệu **Ames Housing** (Kaggle House Prices — Advanced Regression Techniques).

## 1. Xử lý giá trị thiếu (Bước 1–2)

| Chỉ số | Giá trị |
|---|---|
| Tổng số cột có giá trị thiếu | 19 |
| Thiếu vì "không có tiện ích" (điền `'None'`/`0`) | 17 cột |
| Thiếu thật (điền median/mode) | 2 cột |
| Giá trị thiếu còn lại sau xử lý | 0 |

Toàn bộ 17 cột dạng "không có tiện ích" (`PoolQC`, `Alley`, `GarageType`...) được giữ nguyên
thông tin thay vì bị xoá hay hiểu nhầm là lỗi dữ liệu — đúng yêu cầu nghiệp vụ.

## 2. Encoding & phân phối nhãn (Bước 3–5)

| Chỉ số | Giá trị |
|---|---|
| Số đặc trưng sau encoding | 235 |
| Skew `SalePrice` gốc | 1.883 (lệch phải mạnh) |
| Skew `SalePrice` sau `log1p` | 0.121 (gần chuẩn) |

Việc áp dụng `log1p` giúp phân phối nhãn gần với phân phối chuẩn hơn nhiều, đúng như
kỳ vọng khi dữ liệu có một số căn nhà siêu đắt gây lệch.

## 3. Baseline models (Bước 6)

| Model | RMSE (thang log) |
|---|---|
| Dummy (mean) | 0.4332 |
| Linear Regression | 0.1417 |
| Ridge (alpha=10) | 0.1363 |

## 4. Gradient Boosting Regressor (Bước 7)

| Chỉ số | Giá trị |
|---|---|
| Số cây thực tế dùng (early stopping) | 308 / 1000 |
| RMSE (thang log) trên validation | 0.1366 |

GBR đạt RMSE gần tương đương Ridge — cho thấy với bộ đặc trưng hiện tại, phần lớn
tín hiệu đã tuyến tính hoá tốt qua encoding; GBR vẫn có lợi thế ở phần mô hình hoá
khoảng dự báo (xem mục 6).

 Biểu đồ train/validation loss theo số cây: `reports/loss_theo_so_cay.png`

## 5. Hiệu quả Feature Engineering (Bước 9)

| | RMSE (thang log) |
|---|---|
| Trước feature engineering | 0.1397 |
| Sau feature engineering (`TotalSF`, `TuoiNha`, `DaSuaChua`) | 0.1390 |
| **Cải thiện** | **0.50%** |

Mức cải thiện khiêm tốn — hợp lý vì các đặc trưng gốc (diện tích tầng, năm xây)
đã được model học được phần lớn thông tin tương tự qua boosting.

## 6. Hồi quy phân vị & Median APE (Bước 10–11)

| Chỉ số | Giá trị | Mục tiêu / Kỳ vọng |
|---|---|---|
| **Median APE** | **5.38%** | < 12% ✅ **Đạt, tốt hơn mức tham chiếu (~8–10%)** |
| Tỉ lệ phủ thực tế của khoảng 10–90% | 69.9% | ~80% ⚠️ **Thấp hơn kỳ vọng** |

 Biểu đồ: `reports/khoang_gia.png`, `reports/ape_distribution.png`

**Ghi chú:** Median APE rất tốt (5.38%, vượt xa mục tiêu 12%), nhưng tỉ lệ phủ 69.9%
thấp hơn 80% kỳ vọng — nghĩa là khoảng dự báo [P10, P90] đang **hơi hẹp** so với biến
động thực tế của giá nhà, nên nhiều căn nhà nằm ngoài khoảng dự báo hơn mức lý thuyết.
Có thể cải thiện bằng cách nới `alpha` hai đầu (ví dụ dùng phân vị 0.05/0.95) hoặc tăng
`n_estimators` cho 2 model phân vị biên.

## 7. Cơ chế Human-in-the-loop (Bước 12)

| Nhóm | Tỉ lệ |
|---|---|
| Tự động duyệt (khoảng dự báo hẹp ≤ 25%) | **56.8%** |
| Chuyển thẩm định viên (khoảng dự báo rộng > 25%) | **43.2%** |

Hơn một nửa hồ sơ có thể được AVM duyệt tự động, giảm đáng kể khối lượng công việc
thủ công cho thẩm định viên, trong khi các hồ sơ có độ bất định cao vẫn được chuyển
sang con người xử lý — đúng tinh thần "human-in-the-loop" của bài toán.

## 8. So sánh tốc độ train (Bước 13)

| Model | Thời gian train |
|---|---|
| GradientBoostingRegressor | 2.88s |
| HistGradientBoostingRegressor | 0.98s |

`HistGradientBoostingRegressor` nhanh hơn khoảng **2.9 lần** nhờ thuật toán
histogram-based, cùng độ chính xác tương đương — đáng cân nhắc khi triển khai
production ở quy mô lớn.

