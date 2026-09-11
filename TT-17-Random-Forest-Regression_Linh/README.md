# TT-17 — Random Forest Regressor: Dự đoán giá vé máy bay

---

## Cách chạy

```bash
pip install -r requirements.txt
```

1. Tải file `Clean_Dataset.csv` từ Kaggle, đặt vào thư mục `data/`.
2. Chạy:
```bash
python src/train.py
```

Script sẽ tự thực hiện toàn bộ 12 bước (EDA, baseline, Random Forest, permutation
importance, PDP, khoảng dự báo, thí nghiệm ngoại suy, so sánh XGBoost), lưu:
- Hình ảnh vào `reports/*.png`
- Model đã train vào `models/rf_reg.joblib`
- Số liệu tổng kết vào `reports/tong_ket_ket_qua.json`

```

Notebook tương đương: `notebooks/rf_regressor_flight.ipynb`.

---

## Kết quả (chạy trên dữ liệu thật — 300.153 dòng)

### So sánh model

| Model | RMSE (Rupee) | R² |
|---|---:|---:|
| Dummy | 22.704 | -0.0000 |
| Linear Regression | 6.762 | 0.9113 |
| Cây đơn (TT-16) | 3.532 | 0.9758 |
| **Random Forest (300 cây)** | **2.694** | **0.9859** |
| XGBoost (chưa tune) | 3.460 | 0.9768 |

`oob_score_ = 0,9864`, sát với R² test — mô hình ổn định, không lệch train/test.

RMSE theo số cây gần như **bão hoà từ khoảng 100 cây** trở đi (2.699 → 2.694 khi tăng
từ 100 lên 500 cây), nên dùng ~100–150 cây là đủ cho triển khai thực tế thay vì 300–500.

### Permutation importance

`class` (Economy/Business) áp đảo hoàn toàn các biến khác — importance gấp ~12 lần
biến đứng thứ 2 (`duration`). Các biến thành phố (`airline`, `source_city`,
`destination_city`) và `days_left` có ảnh hưởng ở mức trung bình-thấp; `stops` ảnh
hưởng ít nhất.

### PDP — `days_left`

Giữ nguyên mọi yếu tố khác, mô hình dự đoán giá trung bình:
- `days_left = 14`: **23.860 Rupee**
- `days_left = 7`: **25.004 Rupee**

**→ Đợi từ 14 xuống còn 7 ngày trước chuyến bay làm giá tăng trung bình 1.144 Rupee
(~4,8%).** Đây là con số dùng trực tiếp để tư vấn khách: mua sớm hơn thường rẻ hơn,
lệch giá rõ nhất trong ~1 tuần cuối trước bay.

### Khoảng dự báo 10–90%

Lấy dự đoán từ tất cả các cây trong rừng, tính phân vị 10% và 90% cho mỗi mẫu →
tỉ lệ giá thật rơi vào đúng khoảng đó trên tập test là **86,5%** (kỳ vọng lý thuyết
~80%). Khoảng hơi rộng hơn cần thiết một chút, nhưng phù hợp cho nghiệp vụ: thay vì
hiển thị 1 con số cứng, hệ thống hiển thị **"giá dự kiến trong khoảng X–Y Rupee"**.

### Thí nghiệm ngoại suy

Dữ liệu train chỉ có `days_left` từ 1–49. Khi ép `days_left` lên 60, 100, rồi 500,
giá dự đoán **giữ nguyên tuyệt đối** (7.304 Rupee, không đổi) — xác nhận Random
Forest không thể ngoại suy: mẫu luôn rơi vào đúng 1 lá (leaf) có ngưỡng chia lớn
nhất từng học, trả về giá trung bình cố định của lá đó bất kể input vượt xa dải
train tới đâu. Trong sản phẩm thực tế, cần chặn hoặc cảnh báo khi dự đoán ngoài
khoảng `days_left` 1–49.

---

## Cạm bẫy đã tránh

| Cạm bẫy | Cách xử lý |
|---|---|
| Giữ cột `flight` | Đã bỏ — tránh one-hot sinh hàng nghìn cột, cây học thuộc mã chuyến |
| Dùng `feature_importances_` mặc định | Dùng permutation importance thay thế |
| Kỳ vọng ngoại suy ngoài dải train | Đã thí nghiệm và xác nhận giá bị kẹp trần |
| Chỉ báo 1 con số giá | Đã bổ sung khoảng dự báo 10–90% |

## Cấu trúc project

```
TT-17-RandomForestRegressor-/
├── README.md
├── notebooks/rf_regressor_flight.ipynb
├── src/train.py
├── src/tao_du_lieu_gia_lap.py   (chỉ để test, không dùng khi nộp bài)
├── models/rf_reg.joblib
├── reports/{*.png, tong_ket_ket_qua.json}
└── requirements.txt
```