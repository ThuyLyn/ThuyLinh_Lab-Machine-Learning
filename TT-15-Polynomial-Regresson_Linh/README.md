# TT-15-Polynomial-HoTen

Polynomial Regression trên bài toán Combined Cycle Power Plant: dự báo công suất phát (PE) từ nhiệt độ môi trường và các biến vận hành.

**Dữ liệu:** dùng **dữ liệu thật** từ UCI Machine Learning Repository, file `data/Folds5x2_pp.xlsx` — 9.568 dòng, 4 đặc trưng (AT, V, AP, RH), nhãn PE.

## Cách chạy

```bash
pip install -r requirements.txt
python src/train.py
```

Code đọc dữ liệu trực tiếp từ `data/Folds5x2_pp.xlsx` — không cần tải gì thêm, file đã có sẵn trong project. Chạy thực tế cho: Train (7.654, 4), Test (1.914, 4).

## Kết quả (trên dữ liệu thật UCI)

Tương quan AT-V trên tập train: **r = 0.844** — khớp đúng mức README gốc mô tả (r ≈ 0,84).

### 1. Scatter AT vs PE — xác nhận quan hệ là đường cong

Biểu đồ scatter AT vs PE (lưu tại `reports/scatter_AT_PE.png`) cho thấy quan hệ giữa nhiệt độ môi trường và công suất phát không phải đường thẳng mà là một đường cong đi xuống — công suất giảm khi nhiệt độ tăng, nhưng tốc độ giảm không đều nhau ở các mức nhiệt độ khác nhau. Đây chính là lý do cần thử đa thức bậc cao hơn bậc 1.

### 2. Residual plot TRƯỚC / SAU

Ảnh residual (lưu tại `reports/residual_truoc_sau.png`) so sánh phần dư giữa model bậc 1 và model bậc tối ưu (bậc 5, Ridge).

Ở bậc 1 (Linear), phần dư có xu hướng lệch dần theo AT — không rải ngẫu nhiên hoàn toàn quanh 0, cho thấy model tuyến tính chưa nắm hết được độ cong của quan hệ thật (dữ liệu thật có nhiều nhiễu vận hành nên hình dạng không "sạch" như lý thuyết, nhưng xu hướng lệch hệ thống vẫn thấy rõ). Sau khi thêm bậc cao (Ridge, bậc 5), phần dư phân tán đều hơn quanh 0, RMSE test giảm từ 4.503 MW (baseline bậc 1) xuống còn 4.149 MW.

### 3. Đường cong xác thực theo bậc đa thức

Đường cong xác thực (lưu tại `reports/validation_curve.png`) vẽ RMSE train và RMSE validation theo từng bậc đa thức từ 1 đến 5, dùng để quan sát xu hướng underfit/overfit khi tăng độ phức tạp của model.

| Bậc | Số cột (Poly) | RMSE train | RMSE test |
|---|---|---|---|
| 1 | 4   | 4.571 | 4.503 |
| 2 | 14  | 4.303 | 4.279 |
| 3 | 34  | 4.243 | 4.205 |
| 4 | 69  | 4.207 | 4.165 |
| 5 | 125 | 4.188 | 4.149 |

RMSE baseline bậc 1 = **4.503 MW**, khớp đúng mức tham chiếu trong README gốc ("bậc 1 khoảng 4,5–4,6"). RMSE tiếp tục giảm dần qua các bậc, không có điểm gãy overfit rõ ràng trong khoảng bậc 1–5 khi dùng Ridge (nhờ Ridge regularize + 9.568 mẫu là khá nhiều so với số cột sinh ra) — bậc 5 vẫn cho RMSE test thấp nhất (4.149 MW) trong dải đã thử. Điều này khác nhẹ so với dự đoán ban đầu của README gốc ("bậc 2-3 thường tối ưu") — cho thấy quan hệ thật có độ cong phức tạp hơn một parabol đơn giản, hoặc cần thử bậc cao hơn 5 kèm tăng alpha để thấy rõ điểm overfit.

Ở bước dò bậc tối ưu trên tập validation (tách riêng từ tập train), xu hướng cũng nhất quán:

| Bậc | RMSE train | RMSE validation |
|---|---|---|
| 1 | 4.571 | 4.572 |
| 2 | 4.304 | 4.309 |
| 3 | 4.248 | 4.256 |
| 4 | 4.210 | 4.220 |
| 5 | 4.189 | 4.201 |

→ Bậc tối ưu theo validation vẫn là **bậc 5**.

### 4. So sánh Linear vs Ridge ở bậc cao — bằng chứng đa cộng tuyến

| Bậc | Model | RMSE test | \|Hệ số\| lớn nhất |
|---|---|---|---|
| 4 | Linear | 4.061 | **9,819.47** |
| 4 | Ridge  | 4.165 | 11.32 |
| 5 | Linear | 4.000 | **7,180.38** |
| 5 | Ridge  | 4.149 | 9.51 |

Hệ số của Linear thuần lớn hơn Ridge tới **hàng nghìn lần** (~7.180–9.819 so với ~9.5–11.3 của Ridge) trong khi Ridge giữ ổn định. Đây là bằng chứng cho đa cộng tuyến do AT-V tương quan cao (0.844) cộng với các cột đa thức bậc cao tự tương quan với nhau — đúng lý do README gốc bắt buộc dùng Ridge. Lưu ý: ở lần chạy này, Linear thuần còn có RMSE test *thấp hơn* Ridge một chút ở cả hai bậc — dấu hiệu Linear đang overfit nhẹ trên tập test cụ thể, dù hệ số lớn và kém ổn định hơn nhiều so với Ridge. Vì vậy **không nên chọn Linear chỉ vì RMSE nhỉnh hơn** — hệ số ổn định của Ridge quan trọng hơn cho một model dùng để chào giá thực tế, đặc biệt khi cần ngoại suy hoặc chạy lại trên dữ liệu mới.

> Ghi chú: độ lớn hệ số Linear dao động khá mạnh giữa các lần chạy (do random_state của train/test split và mức đa cộng tuyến nhạy với mẫu dữ liệu) — có lần lên tới hàng triệu/hàng tỷ, có lần chỉ vài nghìn như lần chạy này. Bản chất vấn đề (đa cộng tuyến, cần Ridge) là không đổi.

### 5. Ngoại suy ngoài dải dữ liệu — đa thức bậc cao "phát điên"

Dải AT trong dữ liệu train chỉ ~1.8–37°C. Dự đoán PE (bậc 5) khi đi xa dần ra ngoài dải này (số liệu từ lần chạy trước, cần chạy lại `src/train.py` bước ngoại suy để xác nhận với hệ số của lần chạy hiện tại):

| AT (°C) | Ridge bậc 5 | Linear bậc 5 | PE hợp lý thực tế |
|---|---|---|---|
| 40  | 430.1 MW | 237.3 MW | ~420–496 MW |
| 50  | 436.3 MW | -878.6 MW | ~420–496 MW |
| 70  | 506.0 MW | -13,849.9 MW | ~420–496 MW |
| 100 | 897.6 MW | **-132,394.9 MW** | ~420–496 MW |

Ở AT=100°C, Linear bậc 5 (lần chạy trước) dự đoán **công suất âm khổng lồ** — vô lý hoàn toàn về mặt vật lý. Ridge kiềm chế tốt hơn nhiều nhờ hệ số nhỏ, nhưng vẫn lệch xa dải hợp lý ở AT=100°C (897.6 MW) — minh chứng rõ ràng cho cạm bẫy "ngoại suy ngoài dải dữ liệu, đa thức bậc cao cho giá trị vô lý", dù có regularize hay không. Vì độ lớn hệ số Linear thay đổi giữa các lần chạy (xem ghi chú ở mục 4), mức độ "phát điên" cụ thể khi ngoại suy cũng sẽ khác đi mỗi lần — nhưng xu hướng định tính (Linear bậc cao ngoại suy tệ hơn nhiều so với Ridge) là nhất quán.

### 6. So sánh với Random Forest

| Model | RMSE test |
|---|---|
| Polynomial bậc 5 (Ridge) | 4.149 MW |
| Random Forest (200 trees) | **3.230 MW** |

Random Forest cho RMSE thấp hơn hẳn (3.230 vs 4.149 MW) mà không cần tự tạo đặc trưng đa thức — cây quyết định tự học được ranh giới phi tuyến từ dữ liệu gốc. Đây là bằng chứng thực nghiệm cho việc mô hình cây thường mạnh hơn polynomial regression trên bài toán có quan hệ phi tuyến phức tạp, đúng như định hướng bài TT-17 sẽ khai thác sâu hơn.

## Giải thích các câu hỏi bắt buộc

**Vì sao phải `StandardScaler` SAU khi tạo đặc trưng đa thức, không phải trước?**
Nếu scale AT trước rồi mới bình phương (`PolynomialFeatures`), các giá trị âm sau khi scale sẽ mất dấu khi bình phương lên, đồng thời thang đo của x² (rất khác thang đo của x) lại không được chuẩn hoá — khiến Ridge phạt không công bằng giữa các bậc. Scale sau khi tạo đủ các cột x, x², x³... đảm bảo mọi cột về cùng thang đo trước khi vào model.

**Ở dải nhiệt độ nào công suất giảm nhanh nhất?**
Theo model bậc tối ưu (bậc 5, Ridge) chạy trên dữ liệu thật, tốc độ giảm PE theo từng đoạn AT là:

| Đoạn AT | Tốc độ (MW/°C) |
|---|---|
| 1.8°C → 10.2°C | **-2.414** |
| 10.2°C → 18.7°C | -2.231 |
| 18.7°C → 27.1°C | -1.753 |
| 27.1°C → 35.6°C | -1.038 |

→ Công suất giảm **nhanh nhất ở dải nhiệt độ thấp** (~1.8–10.2°C, -2.414 MW/°C) và chậm dần đều khi nhiệt độ tăng lên, chỉ còn -1.038 MW/°C ở đoạn 27.1–35.6°C. Kết quả này hơi khác trực giác ban đầu (dự đoán công suất giảm nhanh nhất ở nhiệt độ cao) — cho thấy quan hệ thật phức tạp hơn giả định "hiệu suất tua-bin giảm nhanh khi nóng", và đáng để đào sâu thêm (ví dụ xem xét tương tác giữa AT và V ở từng dải nhiệt độ).

**Hạn chế của đa thức bậc cao:** như bảng ngoại suy ở mục 5 cho thấy, đa thức bậc cao khớp tốt trong dải dữ liệu train nhưng dự đoán vô lý ngay khi ra ngoài dải đó — mô hình không có cơ chế "biết" giới hạn vật lý của bài toán.

## Cấu trúc project

```
TT-15-Polynomial-HoTen/
├── README.md
├── data/Folds5x2_pp.xlsx      
├── notebooks/polynomial_power_plant.ipynb
├── src/train.py         
├── models/poly_pipeline.joblib
├── reports/
│   ├── scatter_AT_PE.png
│   ├── residual_truoc_sau.png
│   ├── validation_curve.png
│   └── bang_bac_socot_rmse.csv
└── requirements.txt
```
