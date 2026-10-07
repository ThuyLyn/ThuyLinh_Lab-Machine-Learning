# TT-31 — Lag + Rolling Features: dự báo phụ tải điện ngày mai
## 1. Cách chạy

```bash
pip install -r requirements.txt
python src/train.py                  # chạy toàn bộ 14 bước, ghi vào reports/ và models/
# hoặc mở notebooks/lag_rolling_power.ipynb rồi Run All
```

Dữ liệu: [Individual Household Electric Power Consumption (UCI)](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption), đặt tại `data/household_power_consumption.txt`. Nhãn là `Global_active_power` (kW), đo mỗi phút, 2006–2010 (một hộ gia đình ở Pháp).

## 2. Tiền xử lý

| Bước | Cách làm | Kết quả |
|---|---|---|
| Đọc | `na_values="?"`, ghép `Date` + `Time` thành datetime, đặt làm index | |
| Gộp | `resample("D").mean()` (từ phút lên ngày) | 1.442 ngày |
| Lỗ hổng | Chỉ nội suy lỗ ≤ 3 ngày, lỗ dài giữ NaN | Thiếu gốc 9 ngày (**0,62%**), còn 4 ngày chưa vá |
| Đặc trưng | 6 lag, 12 thống kê rolling, 4 đặc trưng lịch/chu kỳ, 1 cờ ngày lễ Pháp | **23 đặc trưng** |
| NaN đầu chuỗi | Xoá dòng, **không** điền mean | |

**Chia theo thời gian** (không xáo trộn): train 2007–2008 (717 ngày) · val 2009 (365 ngày) · test 2010 (296 ngày). Siêu tham số chọn trên val; sau đó refit trên train+val và chạm tập test đúng một lần.

> Tập test có 296 ngày thay vì ~330 vì 4 ngày còn thiếu làm các dòng sau đó mất `lag_30` / `tb_30` (4 + 30 = 34 dòng bị loại). Đây là suy luận từ số liệu; có thể kiểm tra bằng `daily[daily.phu_tai.isna()]`.

### Quy tắc số 1: `shift(1)` trước `rolling()`

```python
base = y.shift(1)            # chỉ dùng thông tin đã biết đến hôm qua
r = base.rolling(w)          # tb_w, std_w, max_w, min_w
```

Khi dự báo ngày *t* chỉ biết dữ liệu đến *t−1*. Tháng được mã hoá bằng sin/cos để tháng 12 và tháng 1 nằm cạnh nhau.

## 3. Kết quả chính (tập test 2010)

| Mô hình | MAE (kW) | MAPE (%) | MAE giảm so với Naive |
|---|---|---|---|
| Naive (hôm qua) | 0,2239 | 21,19 | — |
| Seasonal naive (tuần trước) | 0,2583 | 27,18 | −15,4% (tệ hơn) |
| Ridge (alpha=100) | 0,1917 | 20,74 | 14,4% |
| Random Forest (leaf=5, max_features=0,5) | 0,1856 | 19,66 | 17,1% |
| **XGBoost (depth=2, 300 cây)** | **0,1793** | **18,40** | **19,9%** |

**Nhận xét**

- Cả ba mô hình thắng cả hai baseline, kể cả naive (baseline mạnh hơn ở đây). XGBoost giảm MAE khoảng 31% so với seasonal naive và 20% so với naive.
- Seasonal naive *tệ hơn* naive: với một hộ gia đình, phụ tải hôm nay giống hôm qua hơn là giống cùng thứ tuần trước. Chu kỳ tuần có tồn tại nhưng yếu so với nhiễu từng ngày.
- **MAPE 18–21% cao hơn mức tham chiếu 5–8%** của đề bài. Đề đã lưu ý dữ liệu một hộ gia đình nhiễu hơn nhiều so với phụ tải hệ thống, vì hành vi của một hộ (đi vắng, khách đến) không được san phẳng như khi cộng nhiều hộ.
- Khoảng cách giữa các mô hình nhỏ. Trên val, ba mô hình gần như bằng nhau (MAE 0,1860 / 0,1865 / 0,1866), và đây chỉ là một năm test. Có thể nói XGBoost nhỉnh hơn, chưa đủ để kết luận chắc chắn nó vượt Random Forest.

## 4. Feature importance

Top 5 (XGBoost): `tb_7` (0,183) · `min_7` (0,156) · `thang_cos` (0,103) · `lag_1` (0,069) · `max_14` (0,069).

- Thống kê của **7 ngày gần nhất** dẫn đầu, vì chúng gom nhiều ngày nên ít nhiễu hơn một lag đơn lẻ.
- `thang_cos` đứng thứ 3: mùa vụ ảnh hưởng rõ (sưởi ấm mùa đông, nghỉ hè tháng 8 ở Pháp).

## 5.  Thí nghiệm rò rỉ: quên `shift(1)`

Cùng XGBoost, cùng dữ liệu, chỉ khác việc có hay không `shift(1)` trước `rolling`.

| | MAE (kW) |
|---|---|
| Đúng (`shift(1)`) | 0,1793 |
| Rò rỉ (thiếu `shift`) | 0,1605 |
| **Giảm giả tạo** | **10,5%** |

**Nhận xét:** chỉ thiếu một lệnh `shift(1)` mà sai số "đẹp lên" 10,5%. Mức giảm không quá lớn vì chỉ các cột rolling bị rò, và trong mỗi cửa sổ chỉ có đúng một điểm là đáp án, nhưng đủ để model trông tốt hơn mà không ai nghi ngờ. Đây chính là điểm nguy hiểm: kết quả đẹp bất thường mà không đến từ năng lực dự báo. Khi triển khai thật, giá trị ngày mai chưa tồn tại nên model rò rỉ sẽ không thể đạt mức này. Cách phòng: kiểm tra mọi `rolling` có `shift(1)` đứng trước; gặp kết quả tốt bất thường thì nghi ngờ rò rỉ trước.

## 6. Thí nghiệm ngoại suy: dự báo tuyệt đối vs chênh lệch

Cây không dự đoán vượt khoảng giá trị đã thấy lúc train, nên lý thuyết là phụ tải có xu hướng tăng thì cây sẽ dự báo thấp. Cách chữa: dự đoán `y − lag` rồi cộng ngược lại.

| Mô hình | Cách dự đoán | MAE | MAPE (%) | Bias |
|---|---|---|---|---|
| XGBoost | **tuyệt đối** | **0,1793** | 18,40 | −0,0044 |
| XGBoost | chênh lệch lag_1 | 0,1806 | 18,62 | +0,0024 |
| XGBoost | chênh lệch lag_7 | 0,1869 | 19,56 | +0,0030 |
| Random Forest | tuyệt đối | 0,1856 | 19,66 | +0,0037 |
| Random Forest | **chênh lệch lag_1** | **0,1798** | 18,46 | +0,0077 |
| Random Forest | chênh lệch lag_7 | 0,1879 | 19,47 | −0,0002 |

**Nhận xét:**

- **Không có hiệu ứng ngoại suy rõ rệt** trên dữ liệu này. Bias của mọi cách đều rất nhỏ (|bias| < 0,008 kW), tức dự báo không bị thấp có hệ thống ở năm cuối. Dữ liệu của một hộ gia đình không có xu hướng tăng đủ mạnh để làm cây "đi ngang".
- Với XGBoost, dự đoán tuyệt đối tốt nhất (chênh lệch lag_1 kém 0,7%, trong ngưỡng nhiễu). Với Random Forest, chênh lệch lag_1 giảm MAE khoảng 3,1%, đây là cải thiện nhỏ nhưng nhất quán với lý thuyết.
- Chênh lệch `lag_7` kém hơn ở cả hai mô hình, khớp với phát hiện ở mục 4 rằng cùng thứ tuần trước là mốc tham chiếu kém hơn hôm qua.
- Kết luận: kỹ thuật chênh lệch hữu ích khi dữ liệu có xu hướng rõ; ở đây xu hướng yếu nên lợi ích không đáng kể. Nên kiểm tra thêm `xu_huong` (trung bình train / val / test) trong `reports/ketqua.json` để xác nhận.

## 7. Đánh giá ổn định: TimeSeriesSplit (5 fold)

Chạy trên giai đoạn train+val, mỗi fold train trên quá khứ và kiểm tra trên đoạn ngay sau (không dùng KFold thường vì sẽ để dữ liệu sau lọt vào tập train).

| Fold | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| MAE (kW) | 0,2303 | 0,2022 | 0,2437 | 0,2054 | 0,1655 |

**MAE trung bình 0,2094 ± 0,0269** (độ lệch chuẩn ≈ 13% giá trị trung bình).

- Sai số dao động khá giữa các giai đoạn: model không đồng đều theo thời gian, nên con số test 0,1793 là của một năm cụ thể chứ không phải hằng số.
- Con số này **không so sánh trực tiếp** với MAE test được: TimeSeriesSplit đánh giá trên 2007–2009 với lượng dữ liệu train tăng dần (fold đầu có ít dữ liệu nhất), còn test dùng toàn bộ train+val.
- Fold cuối (nhiều dữ liệu train nhất) cho MAE thấp nhất, 0,1655.

## 8. Quy đổi sai số ra tiền

Giả định giá điện **2.000 VND/kWh** (giả định, cần ghi nguồn khi nộp).

```
MAE 0,1793 kW × 24 giờ = 4,30 kWh sai số/ngày/hộ
4,30 kWh × 2.000 VND  ≈ 8.607 VND/ngày/hộ   (≈ 3,14 triệu VND/năm/hộ)
Kịch bản 1 triệu hộ giống nhau ≈ 3.141 tỷ VND/năm
```

**Lưu ý khi đọc con số:**

- Con số "1 triệu hộ" chỉ là phép nhân minh hoạ. Khi cộng nhiều hộ, sai số từng hộ **triệt tiêu một phần** (hộ thừa bù hộ thiếu), nên sai số tương đối của phụ tải hệ thống nhỏ hơn nhiều so với một hộ riêng lẻ.
- Chi phí thật không bằng giá bán lẻ × sai số: dự báo thiếu (mua giá cao trên thị trường tức thời, hoặc sa thải phụ tải) và dự báo thừa (nhà máy chạy không tải) có chi phí **bất đối xứng**.
- Giá trị của phép tính là cho thấy mức độ: cải thiện độ chính xác dự báo có thể quy ra tiền, chứ không phải ước tính ngân sách.

## 9. Hạn chế

- **Một hộ gia đình, một năm test:** MAPE cao, chênh lệch giữa các mô hình nhỏ, không đủ để kết luận chắc chắn mô hình nào tốt nhất.
- **Chưa phân tích ngày lễ:** mới có cờ `la_ngay_le` (lịch Pháp), chưa đo sai số riêng cho các ngày lễ.
- **Chưa làm phần mở rộng:** thời tiết (nhiệt độ), dự báo nhiều bước (1/3/7 ngày), dự báo phân vị.
- **Ngày thiếu:** 4 ngày còn thiếu bị loại cùng 30 ngày sau đó (chấp nhận mất dữ liệu thay vì bịa dữ liệu).

## 12. Cấu trúc thư mục

```
TT-31-LagRolling_Linh/
├── README.md
├── requirements.txt
├── data/household_power_consumption.txt
├── notebooks/lag_rolling_power.ipynb
├── src/{features.py, train.py}
├── models/xgb_power.json
└── reports/{eda_mua_vu, thi_nghiem_ro_ri, ngoai_suy, du_bao_60_ngay, feature_importance}.png
            · ketqua.json · bang_so_sanh.csv
```