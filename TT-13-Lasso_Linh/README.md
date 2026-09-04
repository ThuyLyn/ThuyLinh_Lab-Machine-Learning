# TT-13-Lasso — Kết quả bài tập

## Cách chạy

```bash
pip install -r requirements.txt
python src/train.py
```

---

## BƯỚC 1: Nạp dữ liệu và mở rộng thành 200 cột

| | |
|---|---|
| Kích thước `X_full` | (442, 200) — 10 biến thật + 190 biến nhiễu |
| Tập train | (353, 200) |
| Tập test | (89, 200) |

---

## BƯỚC 2: Baseline Linear Regression (200 cột, không regularization)

| | RMSE |
|---|---|
| Train | 33.99 |
| Test | 84.78 |
| **Chênh lệch train/test** | **50.79** |

→ Chênh lệch rất lớn giữa train và test cho thấy mô hình **overfit rõ rệt**: khi
không có gì kìm hãm, Linear Regression dùng cả 190 cột nhiễu để "khớp" dữ liệu
train, nhưng những cột đó không mang thông tin thật nên dự đoán trên test kém hẳn.

---

## BƯỚC 3: LassoCV (dò alpha bằng 5-fold CV)

| | |
|---|---|
| Alpha tối ưu | **7.92483** |
| RMSE train | 56.28 |
| RMSE test | 53.73 |

→ RMSE train của Lasso cao hơn Linear Regression baseline, nhưng RMSE test lại
**thấp hơn nhiều** (53.73 so với 84.78) — dấu hiệu rõ ràng của việc giảm overfit
nhờ loại bỏ các biến nhiễu.

---

## BƯỚC 4: Chấm điểm chọn biến

| Chỉ số | Giá trị |
|---|---|
| Tổng số biến Lasso giữ lại | **5/200** |
| Giữ **ĐÚNG** (biến thật) | **4/10** → `bmi`, `bp`, `s3`, `s5` |
| Giữ **NHẦM** (biến nhiễu) | **1** → `chi_so_nhieu_009` |
| **BỎ SÓT** (biến thật bị loại) | **6** → `age`, `s1`, `s2`, `s4`, `s6`, `sex` |

> Kết quả 5/200 biến (thấp hơn mức tham chiếu 8–15 nêu trong đề bài) là hợp lý:
> `LassoCV` chọn một alpha khá lớn (7.92) vì bộ `load_diabetes` vốn có tín hiệu
> yếu, khiến chỉ những biến có tương quan mạnh nhất với target mới "sống sót".
> Không đạt 10/10 là bình thường như đề bài đã lưu ý trước.

---

## BƯỚC 5: Coefficient path

Đã lưu `reports/lasso_path.png` (đường đi của từng hệ số khi alpha thay đổi) và
`reports/rmse_vs_alpha.png` (RMSE train/test theo alpha). Nhìn vào hai hình này:
khi alpha tăng dần, các hệ số của biến nhiễu "rơi" về 0 trước, còn các biến thật
có tín hiệu mạnh (`bmi`, `s5`...) trụ lại lâu hơn.

---

## BƯỚC 6: So sánh Ridge vs Lasso

| Model | Số biến giữ lại | RMSE test |
|---|---|---|
| Linear Regression (200 cột) | 200 | 84.78 |
| Ridge (L2) | 200 | 58.05 |
| **Lasso (L1)** | **5** | **53.73** |

→ Đúng như lý thuyết: Ridge **không loại bỏ biến nào** (chỉ co nhỏ hệ số, hình
tròn không có góc nằm trên trục), trong khi Lasso loại được 195/200 biến. Đáng
chú ý là Lasso còn cho RMSE test **tốt nhất** trong 3 mô hình — vì bỏ bớt nhiễu
giúp giảm overfit hiệu quả hơn cả việc chỉ co hệ số như Ridge.

---

## BƯỚC 7: Thí nghiệm biến tương quan (nhân đôi cột `bmi`)

Tạo thêm cột `bmi_ban_sao` gần như trùng với `bmi` (chỉ thêm nhiễu rất nhỏ), chạy
lại Lasso với 5 seed khác nhau:

| seed | Giữ `bmi`? | Giữ `bmi_ban_sao`? | Hệ số `bmi` | Hệ số `bmi_ban_sao` |
|---|---|---|---|---|
| 0 | ✅ | ❌ | 26.1370 | 0.0 |
| 1 | ✅ | ❌ | 24.7174 | 0.0 |
| 42 | ✅ | ❌ | 25.1846 | 0.0 |
| 99 | ✅ | ❌ | 23.8972 | 0.0 |
| 123 | ✅ | ❌ | 22.9498 | 0.0 |

Đã lưu `reports/chon_bien_score.png`.

→ Trong lần chạy này, Lasso luôn nhất quán chọn `bmi` và loại `bmi_ban_sao` — tuy
nhiên đây là biểu hiện của **cơ chế "chọn ngẫu nhiên 1 trong nhóm tương quan"**
đã nêu trong lý thuyết Lasso, không phải sự ổn định thật sự: chỉ cần đổi thứ tự
cột hoặc thay đổi nhẹ dữ liệu, kết quả có thể đảo ngược. Đây chính là lý do
ElasticNet (TT-14) ra đời — nó thêm phạt L2 để giữ **cả nhóm** biến tương quan
thay vì chọn một cách "may rủi".

---

## BƯỚC 8: Debiased Lasso

| | RMSE |
|---|---|
| Train (debiased, 5 biến) | 54.56 |
| Test (debiased, 5 biến) | 53.74 |
| Test (Lasso đầy đủ, có phạt) | 53.73 |

→ RMSE gần như không đổi giữa Lasso đầy đủ và Debiased Lasso — cho thấy 5 biến
Lasso chọn đã mang gần hết thông tin dự đoán được; việc "gỡ co" hệ số ở đây không
cải thiện đáng kể vì tập biến đã rất nhỏ và gọn.

---

## BƯỚC 9: Đề xuất bộ xét nghiệm cuối cùng

**Bộ xét nghiệm đề xuất (5 chỉ số):** `bmi`, `bp`, `s3`, `s5`, `chi_so_nhieu_009`

| | Số chỉ số | Chi phí ước tính (giả định 25.000đ/chỉ số) |
|---|---|---|
| Ban đầu | 200 | 5.000.000 đ/bệnh nhân |
| Sau chọn lọc bằng Lasso | 5 | 125.000 đ/bệnh nhân |
| **Tiết kiệm** | | **97.5%** |
---

## Sản phẩm đã lưu

```
├── reports/lasso_path.png
├── reports/rmse_vs_alpha.png
├── reports/ridge_vs_lasso.png
├── reports/chon_bien_score.png
├── reports/de_xuat_cuoi_cung.json
└── models/lasso_pipeline.joblib
```
