# TT-30 — SARIMA dự báo lượng khách hàng tháng

Dự báo 24 tháng trên AirPassengers (144 tháng, 1949–1960) bằng SARIMA theo quy trình Box-Jenkins. Train 120 tháng, test 24 tháng, chia theo thời gian.

## Kết quả

| Mô hình | MAPE (24 tháng) | Tỉ lệ phủ 95% |
|---|---|---|
| Naive | 23,58% | — |
| Seasonal naive | 15,52% | — |
| **SARIMA(0,1,1)(0,1,1,12)** (chọn theo AIC) | **8,52%** | 100% |
| SARIMA(1,1,1)(1,1,1,12) | 7,80% | 100% |

SARIMA thắng seasonal naive. MAPE dự báo 1 bước là **2,55%**.

## Bảng ADF

| Phiên bản chuỗi | ADF stat | p-value | Kết luận |
|---|---|---|---|
| Gốc | 0,815 | 0,9919 | Không dừng |
| Log | −1,717 | 0,4224 | Không dừng |
| Log + diff(1) | −2,717 | 0,0711 | Không dừng |
| Log + diff(1) + diff(12) | −4,443 | 0,0002 | **Dừng** |

→ Chuỗi nhân tính nên lấy **log**; chọn **d = 1, D = 1, s = 12**.

## Chọn bậc

ACF nổi bật ở lag 1 và lag 12 → q = 1, Q = 1 → airline model (0,1,1)(0,1,1,12). Dò lưới AIC (p, q ≤ 2; P, Q ≤ 1) chọn đúng bậc này (AIC = −389,0). Model (1,1,1)(1,1,1,12) có MAPE test thấp hơn nhưng không được chọn vì chọn theo tập test là không công bằng; AIC của nó cao hơn và gồm nhiều hệ số thừa.

## Kiểm định phần dư (Ljung-Box)

| Model | p (lag 12) | p (lag 24) |
|---|---|---|
| (0,1,1)(0,1,1,12) | 0,912 | 0,804 |
| (1,1,1)(1,1,1,12) | 0,955 | 0,865 |

Cả hai p > 0,05 → phần dư là nhiễu trắng.

## Tỉ lệ phủ của khoảng dự báo 95%

| Model | Số điểm thật nằm trong khoảng | Tỉ lệ phủ |
|---|---|---|
| (0,1,1)(0,1,1,12) | 24 / 24 | 100% |
| (1,1,1)(1,1,1,12) | 24 / 24 | 100% |

Lý tưởng là khoảng 95%. Kết quả 100% nghĩa là khoảng đủ rộng, **không bị quá hẹp**, nên dùng được cho kế hoạch nhân sự. Tuy nhiên test chỉ có 24 điểm (mỗi điểm = 4,2%) nên chưa đủ để khẳng định khoảng chuẩn đúng 95%.

## Lưu ý

1. **Tắt `enforce_stationarity` làm hỏng dò lưới AIC:** lưới chọn nhầm model có MAPE test ≈ 14,7% và rớt Ljung-Box. Bài này giữ mặc định (`True`).
2. **"MAPE 3–5%" của đề là dự báo 1 bước** (2,55%). Dự báo 24 tháng liền sai số tích lũy nên là 8,52%.
3. **Dự báo bị lệch thấp có hệ thống:** thực tế cao hơn dự báo ở 24/24 tháng. Khi lập kế hoạch nhân sự nên bám cận trên của khoảng tin cậy.
4. **Tỉ lệ phủ 100% trên 24 điểm chưa đủ kết luận** khoảng tin cậy chuẩn 95%; cần đánh giá cuốn chiếu.

## Hạn chế

- ARIMA chỉ dùng quá khứ của chính chuỗi, không biết ngày lễ, khuyến mãi, thời tiết → dùng SARIMAX hoặc ML + lag (TT-31).
- Khoảng tin cậy giả định phần dư chuẩn, phương sai ổn định sau log.
- Chỉ một lần chia train/test, lưới dò hẹp.

## Chạy lại

```bash
pip install -r requirements.txt
python src/run.py
```

Hoặc mở `notebooks/arima_forecasting.ipynb` và Run All (cần internet để tải dữ liệu). Ảnh lưu ở `reports/`, model ở `models/`.