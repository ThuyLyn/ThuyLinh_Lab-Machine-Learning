# TT-20-SVR — Dự đoán cường độ chịu nén bê tông

## 0. TÓM TẮT KẾT QUẢ

| Model | R² (test) | RMSE (MPa) |
|---|---|---|
| DummyRegressor (baseline) | -0.0002 | 16.05 |
| Linear Regression | 0.8305 | 6.61 |
| SVR — KHÔNG scale | 0.2892 | 13.53 |
| SVR — chỉ scale X | 0.8841 | 5.47 |
| **SVR — scale X & y, RBF (tuned)** | **0.8911** | **5.30** |
| Random Forest | 0.8941 | 5.22 |
| **XGBoost** | **0.9320** | **4.19** |

→ Đạt tiêu chí R² > 0,88. XGBoost vẫn tốt hơn SVR một chút, đúng như mức tham chiếu trong đề bài.

---

## 1. ĐẶC TRƯNG TỪ KIẾN THỨC MIỀN 

Ba đặc trưng được thêm dựa trên kiến thức xây dựng, không phải model tự học được:

| Đặc trưng | Công thức | Lý do | Tương quan với target |
|---|---|---|---|
| `water_cement_ratio` | `Water / Cement` | Tỉ lệ nước/xi măng là yếu tố **quyết định** cường độ bê tông trong thực tế xây dựng — tỉ lệ càng thấp, bê tông càng đặc chắc | -0.501 (mạnh hơn `Water` riêng lẻ: -0.290) |
| `tong_chat_ket_dinh` | `Cement + Slag + Fly Ash` | Tổng lượng chất kết dính (không chỉ riêng xi măng — tro bay và xỉ lò cũng góp phần kết dính) | 0.613 (mạnh hơn `Cement` riêng lẻ: 0.498) |
| `log_age` | `log(1 + Age)` | Cường độ bê tông tăng theo **log** của tuổi (tăng nhanh 7 ngày đầu, chậm dần sau 28 ngày), không tuyến tính | 0.549 (mạnh hơn `Age` gốc: 0.329) |

**Đo hiệu quả thực tế (SVR, cùng tham số C=1000, gamma=0.01, epsilon=0.1):**

| | R² | RMSE |
|---|---|---|
| KHÔNG có đặc trưng miền (8 cột gốc) | 0.8584 | 6.041 |
| CÓ đặc trưng miền (11 cột) | 0.8889 | 5.351 |

→ **Cải thiện RMSE 11,4%** (0.690 MPa) chỉ nhờ 3 đặc trưng công thức đơn giản — không cần model phức tạp hơn.

---

## 2. SCALE X VÀ Y — TẠI SAO BẮT BUỘC

`epsilon` mặc định của SVR = 0.1. Với `y` tính bằng MPa (dao động 2–83), 0.1 MPa gần như không có ý nghĩa —
gần như MỌI điểm rơi ra ngoài "ống" ε-insensitive, khiến SVR mất hết lợi thế bỏ qua sai số nhỏ.

| Cấu hình | R² | RMSE |
|---|---|---|
| Không scale gì | 0.289 | 13.53 |
| Chỉ scale X | 0.884 | 5.47 |
| **Scale cả X và y** | **0.894** | **5.22** |

Kết luận: không scale gần như phá hỏng hoàn toàn model. Dùng `TransformedTargetRegressor` để tự động scale/unscale `y`.

---

## 3. SO SÁNH KERNEL

| Kernel | R² | RMSE | Support vectors |
|---|---|---|---|
| linear | 0.829 | 6.63 | 641 |
| **rbf** | **0.894** | **5.22** | 450 |
| poly (degree=2) | 0.392 | 12.52 | 717 |
| poly (degree=3) | 0.816 | 6.89 | 605 |
 
RBF thắng rõ — quan hệ giữa thành phần bê tông và cường độ có tính phi tuyến phức tạp. `poly` nhạy tham số
(`coef0` mặc định chưa phù hợp), không nên dùng nếu không tune thêm.

---

## 4. GRIDSEARCH C × GAMMA × EPSILON

Best params: `C=1000, gamma=0.01, epsilon=0.1` → RMSE (CV 5-fold) = 4.89 MPa, Test R²=0.891.

Xu hướng heatmap RMSE:
- `gamma=1` (quá lớn) → RMSE tệ nhất ở MỌI C → overfit rõ rệt.
- `gamma='scale'` ≈ `gamma=0.1` về mặt số học với bộ dữ liệu này (8 đặc trưng đã chuẩn hoá).
- C càng lớn cần gamma càng nhỏ để tránh overfit.

Model cuối có **504/824 (61,2%)** mẫu train là support vector — khá cao do `C=1000` lớn, đổi lại RMSE thấp nhất.

---

## 5. SVR KHÔNG MỞ RỘNG ĐƯỢC — BẰNG CHỨNG THỜI GIAN TRAIN

| Số mẫu | SVR (giây) | XGBoost (giây) |
|---|---|---|
| 824 (×1) | 0.56 | 0.09 |
| 1,648 (×2) | 1.91 | 0.08 |
| 3,296 (×4) | 7.93 | 0.09 |
| 6,592 (×8) | 42.09 | 0.14 |

Từ ×1 lên ×8 dữ liệu, thời gian train SVR tăng **~75 lần** (nằm giữa xu hướng O(n²) và O(n³): số mũ thực nghiệm ≈ 2,08)
trong khi XGBoost gần như không đổi (dao động trong khoảng nhiễu đo, 0,08–0,14 giây).
→ Xác nhận SVR không phù hợp cho dữ liệu > 50.000 dòng như README cảnh báo.

---

## 6. HẠN CHẾ

- SVR không giải thích được từng dự đoán (không có feature importance trực tiếp như cây quyết định).
- Không mở rộng được cho dữ liệu lớn (độ phức tạp O(n²)–O(n³)).
- Bài toán an toàn kết cấu cần phân tích riêng phần dự báo CAO HƠN thực tế (xem mở rộng #2 trong README gốc) —
  chưa thực hiện trong phiên bản này, nên KHÔNG dùng model này để ra quyết định đổ móng thực tế mà chưa
  kiểm định lại bằng phân vị thấp (quantile regression).

---
