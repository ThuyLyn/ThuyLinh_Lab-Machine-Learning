# TT-16 — Decision Tree Regressor: Định giá cước chuyến xe

Bảng giá dạng LUẬT cho tổng đài, dùng cây hồi quy nông (max_depth=5).

## 1. Cấu trúc dự án

```
TT-16-DecisionTreeRegressor-<HoTen>/
├── README.md                       
├── notebooks/tree_regressor_taxi.ipynb
├── src/train.py
├── models/tree_reg.joblib
├── outputs/bang_tra_cuoc.csv        
├── reports/
│   ├── eda_overview.png
│   ├── ham_bac_thang.png
│   ├── mae_theo_depth.png
│   ├── cay_quyet_dinh.png
│   ├── cay_quyet_dinh.txt
│   ├── so_sanh_mo_hinh.png
│   └── log_chay.txt
└── requirements.txt
```
## 2. Kết quả chạy thật (log)

### Bước 1 — Đọc dữ liệu
- Số dòng đọc được từ file gốc: **3,724,889**
- Số dòng sau khi lấy mẫu: **200,000** (đúng `N_SAMPLE = 200_000`)

### Bước 2 — Làm sạch dữ liệu
| Lý do loại | Số dòng bị loại |
|---|---|
| `fare_amount <= 0` | 2,290 |
| `trip_distance <= 0` hoặc `> 100` | 6,540 |
| `passenger_count == 0` | 54,088 |

- Còn lại sau làm sạch: **137,082 / 200,000 dòng (68.5%)**
- **[!] RÒ RỈ DỮ LIỆU:** không đưa `tip_amount`, `tolls_amount`, `total_amount` vào đặc trưng X, vì các giá trị này chỉ biết được **sau khi** chuyến đi kết thúc.

### Bước 3 — Đặc trưng sử dụng
`trip_distance`, `passenger_count`, `PULocationID`, `DOLocationID`, `hour`, `dow`, `is_rush_hour`

### Bước 4 — EDA
Đã lưu `reports/eda_overview.png`

### Bước 5 — Baseline
| Mô hình | MAE |
|---|---|
| DummyRegressor (mean) | 12.05 |
| Công thức thủ công (3 + 2·km) | 9.63 |

### Bước 6 — Cây không giới hạn độ sâu (chứng minh overfit)
- Số lá: **90,316** | Số tầng: **41**
- MAE train = **0.010** (gần 0 → cây "học thuộc" dữ liệu)
- MAE test = **3.602** (cao hơn nhiều → OVERFIT rõ ràng)

### Bước 7 — Quét max_depth từ 1 đến 20
- Độ sâu có MAE test thấp nhất: **15** (MAE = 3.041)
- Đề bài chọn **max_depth=5** để ưu tiên khả năng tra bảng bằng tay, dù đây không phải điểm tối ưu thống kê tuyệt đối.
- Đã lưu `reports/mae_theo_depth.png`

### Bước 8 — Hàm dự đoán theo quãng đường (bậc thang)
- Đã lưu `reports/ham_bac_thang.png`
- Số mức giá (lá) tối đa có thể trả ra: **27**

### Bước 9 — Xuất cây
- `reports/cay_quyet_dinh.txt` (dạng text, `export_text`)
- `reports/cay_quyet_dinh.png` (hình vẽ cây)

### Bước 10 — Bảng tra cước cho tổng đài
- Đã lưu `outputs/bang_tra_cuoc.csv` (**27 mức giá**)

### Bước 11 — Kiểm tra sai số
| Chỉ số | Giá trị |
|---|---|
| MAE | 3.20 |
| MAPE | 20.23% |
| RMSE | 7.51 |
| % chuyến sai số trong ±15% | **53.0%** |

### Bước 12 — So sánh mô hình
| Mô hình | MAE |
|---|---|
| Decision Tree (depth=5) | 3.20 |
| Linear Regression | 3.40 |
| Random Forest | 2.90 |

Đã lưu `reports/so_sanh_mo_hinh.png`. Random Forest cho MAE thấp nhất, nhưng không tra bảng bằng tay được như cây đơn depth=5 — đây là đánh đổi giữa độ chính xác và khả năng diễn giải/vận hành cho tổng đài.

> Toàn bộ log chi tiết được lưu tại `reports/log_chay.txt`.

## 3. Vì sao cây KHÔNG ngoại suy được?

Cây quyết định học bằng cách **so sánh ngưỡng** (ví dụ: "quãng đường > 15km hay
không"), chứ không học một **công thức toán học** liên tục.

Xét một chuyến đi rất xa — xa hơn bất kỳ chuyến nào trong tập huấn luyện:

- Chuyến này vẫn rơi vào nhánh "quãng đường > X km" xa nhất mà cây có,
- Và nhận **chính xác** giá trị trung bình của lá đó — giống hệt chuyến xa nhất
  từng thấy trong lúc huấn luyện,
- Dù thực tế chuyến đó phải đắt tiền hơn rất nhiều.

Khác biệt cơ bản với hồi quy tuyến tính: `y = a + b·x` vẫn tính được cho x lớn
(dù có thể sai vì ngoài phạm vi dữ liệu), còn cây chỉ "tra bảng" nên **không bao
giờ vượt quá giá trị lớn nhất đã thấy trong tập huấn luyện**.

**Hệ quả nghiệp vụ:** nếu hệ thống có thể gặp chuyến siêu dài (liên tỉnh, sân
bay xa...), cần có quy tắc phụ (ví dụ: cộng thêm phí cố định theo km cho phần
vượt ngưỡng tối đa mà cây từng thấy) — không thể tin tưởng hoàn toàn vào bảng
tra cước cho các trường hợp ngoại lệ. Số liệu thực tế minh chứng điều này: chỉ
**53.0%** số chuyến trong tập test đạt sai số trong khoảng ±15%.

## 4. Cạm bẫy cần tránh

| Cạm bẫy | Cách xử lý trong code |
|---|---|
| Dùng `total_amount` làm đặc trưng | Loại khỏi `features`, có ghi chú rõ trong code |
| `max_depth=None` gây overfit | Chứng minh bằng bước 6: MAE train 0.010 vs MAE test 3.602 |
| Kỳ vọng cây ngoại suy | Giải thích ở mục 4 phía trên |
| Quên loại chuyến giá âm | Bước làm sạch loại `fare_amount <= 0` (2,290 dòng), log rõ số dòng bị loại |
| Không lấy mẫu | Luôn giới hạn `N_SAMPLE = 200_000` trước khi train |
| Lá dựa trên quá ít chuyến | `min_samples_leaf=500` bắt buộc ở mọi cây được train |