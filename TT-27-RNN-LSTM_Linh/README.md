# TT-27 — RNN / LSTM / GRU: Dự báo lưu lượng giao thông theo giờ

## 1. Tóm tắt kết quả

| | |
|---|---|
| Dự báo 1 giờ tới | **LSTM: MAE 156.5 xe/giờ**, tốt hơn seasonal naive (338.0) **~54%** và naive (564.2) **~72%** |
| Mô hình tốt nhất | **XGBoost + lag features: MAE 143.1**, thấp hơn LSTM 13.4 xe/giờ (~9%) và train nhanh hơn ~85 lần |
| Tiêu chí của đề | LSTM **thắng** seasonal naive → đạt. Nhưng XGBoost còn tốt hơn và rẻ hơn nhiều (xem mục 6) |

## 2. Cách chạy

```
pip install -r requirements.txt
python src/train.py
```
Biểu đồ và bảng CSV được lưu trong `reports/` của dự án, mô hình LSTM trong `models/lstm_traffic.keras`. Notebook `notebooks/lstm_traffic.ipynb` chạy cùng quy trình từng bước, có giải thích.

Tuỳ chọn thêm: `--seeds 3` (mỗi mô hình huấn luyện 3 seed, báo trung bình và độ lệch chuẩn), `--extra` (thêm thí nghiệm LSTM với đặc trưng mùa vụ của giờ đích).

Đường dẫn Windows trong Python nên viết dạng chuỗi thô `r"D:\TT_ML\...\file.csv"` hoặc dùng dấu `/`, nếu không Python báo `SyntaxWarning: invalid escape sequence`.

## 3. Xử lý dữ liệu

Sau xử lý còn **43.623 giờ** có dữ liệu và **22 đặc trưng** đầu vào.

| Vấn đề trong dữ liệu | Cách xử lý |
|---|---|
| Lỗ hổng thời gian | Reindex theo lưới giờ đầy đủ. Lỗ ≤ 6 giờ: nội suy tuyến tính + cột cờ `imputed`. Lỗ dài hơn: không bịa dữ liệu, cắt chuỗi thành các đoạn liên tục; cửa sổ trượt chỉ được tạo **bên trong một đoạn** |
| Ngoại lai (`temp` = 0 K, `rain_1h` = 9831 mm/giờ) | Đặt NaN khi `temp` < 200 K hoặc `rain_1h` > 100 mm/giờ, rồi nội suy |
| Trùng `date_time` | Gộp về 1 dòng/giờ |
| `holiday` chỉ đánh ở dòng đầu | Lan cờ ra cả ngày (pandas đọc chuỗi `"None"` thành NaN nên coi cả NaN lẫn `"None"` là không lễ) |

**Đặc trưng:** lưu lượng quá khứ, `temp`, `log1p(rain_1h)`, `snow_1h`, `clouds_all`, sin/cos của giờ và thứ, cờ ngày lễ, cờ `imputed`, one-hot `weather_main`.

**Chống rò rỉ:** chia theo thời gian 70% train / 15% val / 15% test (không shuffle khi chia); scaler chỉ `fit` trên train; cửa sổ không băng qua lỗ hổng hay ranh giới giữa các tập; chỉ shuffle các cửa sổ trong lúc train.

## 4. Mô hình và cách đánh giá

- **RNN / LSTM / GRU** cùng kiến trúc: `64 → Dropout(0.2) → 32 → Dropout(0.2) → Dense(1)` (không activation), Adam 1e-3, MSE, batch 128, tối đa 40 epoch, `EarlyStopping` (patience 5, khôi phục trọng số tốt nhất), seed 42.
- **Baseline:** naive (cùng giờ hôm qua), seasonal naive (cùng giờ tuần trước), XGBoost với lag features (lag 0–48 giờ, giá trị tại T−24h và T−168h, lịch của giờ đích, thời tiết hiện tại).
- **Chấm công bằng:** mọi mô hình và mọi horizon được đánh giá trên **cùng một tập giờ test** (những giờ mà cả hai baseline đều có giá trị và cửa sổ 48 giờ hợp lệ). Đơn vị MAE là xe/giờ.

## 5. Kết quả

### 5.1. So sánh mô hình — dự báo 1 giờ tới, cửa sổ 24 giờ

| Mô hình | Số tham số | Thời gian train (s) | Epoch | Val MAE | **Test MAE** |
|---|---|---|---|---|---|
| Naive (hôm qua) | – | – | – | – | 564.2 |
| Seasonal naive (tuần trước) | – | – | – | – | 338.0 |
| XGBoost + lag | – | 1.4 | – | – | **143.1** |
| SimpleRNN | 8.705 | 20.1 | 14 | 192.0 | 182.3 |
| LSTM | 34.721 | 118.5 | 37 | 157.6 | 156.5 |
| GRU | 26.337 | 165.2 | 40 | 160.4 | 159.5 |

Biểu đồ: `reports/rnn_lstm_gru.png`

Nhận xét:
- Cả ba mô hình hồi quy đều vượt xa hai baseline naive. Seasonal naive mạnh hơn naive rõ rệt (338 so với 564), đúng như dự đoán vì giao thông có chu kỳ tuần.
- LSTM (156.5) và GRU (159.5) cho kết quả gần nhau; chênh 3.0 xe/giờ là quá nhỏ để kết luận mô hình nào tốt hơn. Cả hai tốt hơn SimpleRNN (182.3) khoảng 12–14%, đúng với lý thuyết: cổng của LSTM/GRU giữ được ký ức dài hơn, còn SimpleRNN bị vanishing gradient. SimpleRNN dừng sớm ở epoch 14, còn GRU chạy hết 40 epoch (bằng mức tối đa) nên có thể chưa hội tụ hoàn toàn.
- Về chi phí, SimpleRNN nhẹ nhất (≈ 1.4 s mỗi epoch, 8.7 nghìn tham số). LSTM ≈ 3.2 s và GRU ≈ 4.1 s mỗi epoch; ở cấu hình này GRU không nhanh hơn LSTM như lý thuyết thường nêu.

### 5.2. Khảo sát độ dài cửa sổ (LSTM, dự báo 1 giờ)

| Cửa sổ (giờ) | Test MAE | Thời gian train (s) |
|---|---|---|
| 6 | 153.2 | 37.6 |
| 12 | 162.9 | 50.7 |
| 24 | 156.5 | 118.5 |
| 48 | 158.2 | 175.6 |

Biểu đồ: `reports/do_dai_cua_so.png`

Cửa sổ 6 giờ cho MAE thấp nhất (153.2) và cũng nhanh nhất. Kéo dài cửa sổ không cải thiện độ chính xác mà tốn thêm nhiều thời gian: cửa sổ 48 giờ mất gấp ~4.7 lần cửa sổ 6 giờ. Lý do hợp lý là thông tin về chu kỳ ngày và tuần đã được cung cấp qua đặc trưng lịch (sin/cos giờ, thứ, ngày lễ), nên mô hình chỉ cần vài giờ gần nhất để biết xu hướng hiện tại. Chênh lệch giữa các cỡ cửa sổ chỉ vài xe/giờ nên cần đọc như một xu hướng chung. Tất cả cỡ cửa sổ của LSTM đều còn cao hơn XGBoost (143.1) ít nhất ~10 xe/giờ. Chưa thử cửa sổ 168 giờ; đề cũng cảnh báo cỡ đó thường chậm mà không tốt hơn.

### 5.3. Dự báo nhiều bước

| Horizon | LSTM | XGBoost | Seasonal naive |
|---|---|---|---|
| 1 giờ | 156.5 | 143.1 | 338.0 |
| 3 giờ | 208.3 | 188.0 | 337.7 |
| 6 giờ | 242.3 | 202.7 | 337.8 |

- MAE tăng theo horizon đúng như kỳ vọng vì dự báo càng xa càng ít thông tin: LSTM từ 156.5 lên 242.3 (+85.8, ~55%), XGBoost từ 143.1 lên 202.7 (+59.6, ~42%).
- Ở cả 3 horizon, XGBoost đều thấp hơn LSTM, và khoảng cách rộng ra khi dự báo xa hơn: 13.4, 20.3 và 39.6 xe/giờ ở 1h, 3h, 6h.
- Ở horizon 6 giờ, LSTM vẫn tốt hơn seasonal naive khoảng 28%, XGBoost khoảng 40%. Nghĩa là dù dự báo xa, cả hai vẫn có giá trị hơn việc chỉ lấy số của tuần trước.
- Seasonal naive lệch rất nhẹ giữa các horizon (337.7–338.0) vì tập giờ so sánh của mỗi horizon hơi khác nhau.

### 5.4. Dự báo so với thực tế và phân tích lỗi

Biểu đồ: `reports/du_bao_vs_thuc_te.png` (7 ngày cuối của tập test), `reports/phan_tich_loi.png` (MAE theo 24 giờ và theo thứ).

- **Thứ sai nhiều nhất:** thứ Bảy.
- **Giờ sai nhiều nhất:** 22 giờ (MAE 247 xe/giờ). Xem đủ 24 giờ trong `phan_tich_loi.png` để thấy các giờ cao điểm khác cũng có sai số cao.
- **10 ngày sai nhiều nhất** (MAE trung bình trong ngày, xe/giờ):

| Ngày | Thứ | MAE |
|---|---|---|
| 2018-03-24 | Thứ Bảy | 370 |
| 2018-04-14 | Thứ Bảy | 335 |
| 2018-01-22 | Thứ Hai | 333 |
| 2018-05-21 | Thứ Hai | 312 |
| 2018-03-05 | Thứ Hai | 310 |
| 2018-01-17 | Thứ Tư | 294 |
| 2018-04-03 | Thứ Ba | 290 |
| 2018-09-20 | Thứ Năm | 266 |
| 2018-01-11 | Thứ Năm | 259 |
| 2018-04-15 | Chủ Nhật | 249 |

Chín trong mười ngày rơi vào tháng 1–5 (mùa đông và đầu xuân), và ngày sai nhiều nhất là 2018-03-24 (thứ Bảy) với MAE 370. Nguyên nhân chưa được kiểm chứng (có thể là thời tiết xấu, sự kiện đặc biệt hoặc lưu lượng bất thường); muốn khẳng định cần đối chiếu từng ngày với `weather_main`, `snow_1h`, `rain_1h`.

## 6. Kết luận

1. **Đạt tiêu chí của đề:** LSTM (MAE 156.5) và GRU (159.5) thắng seasonal naive (338.0) với biên rất lớn (~53–54%), nên deep learning có giá trị so với baseline "thuộc lòng chu kỳ tuần".
2. **Nhưng XGBoost + lag features tốt hơn và rẻ hơn:** MAE 143.1 so với 156.5 (thấp hơn ~9%), train 1.4 giây so với 118.5 giây (~85 lần nhanh hơn), và tốt hơn ở mọi horizon từ 1 đến 6 giờ. Theo đúng hướng dẫn của đề, kết luận trung thực là: **với bài này, XGBoost là lựa chọn triển khai hợp lý hơn; mạng hồi quy không mang lại lợi thế đủ để bù chi phí.**
3. **Lý do có thể (chưa kiểm chứng):** XGBoost được đưa trực tiếp giá trị lưu lượng tại đúng giờ đích của hôm qua và tuần trước (T−24h, T−168h) cùng lịch của giờ đích, trong khi LSTM chỉ nhìn cửa sổ quá khứ ngắn và không thấy giá trị của tuần trước. Tuỳ chọn `--extra` (LSTM có thêm các đặc trưng này) được thiết kế để kiểm tra giả thuyết.
4. **Nhóm RNN:** LSTM và GRU tương đương nhau, cùng tốt hơn SimpleRNN. Cửa sổ ngắn (6 giờ) là đủ và rẻ nhất; cửa sổ dài không giúp ích.
5. **Dự báo xa:** sai số tăng ~40–55% khi đi từ 1 giờ lên 6 giờ, nhưng vẫn tốt hơn seasonal naive ở cả hai mô hình.

**Mức tham chiếu của đề** là MAE ~250–400 xe/giờ cho dự báo 1 giờ; kết quả ở đây (143–160) thấp hơn khoảng đó. Mức tham chiếu chỉ mang tính ước chừng, và kết quả phụ thuộc vào cách làm sạch, cách nội suy và cách chọn tập so sánh, nên nên hiểu là "không tệ hơn mức kỳ vọng" thay vì so sánh trực tiếp con số.

## 7. Hạn chế

- Kết quả của mạng nơ-ron có thể dao động nhẹ (khoảng vài xe/giờ) do khởi tạo trọng số và thứ tự shuffle ngẫu nhiên, nên các chênh lệch nhỏ (như giữa LSTM và GRU, hay giữa các cỡ cửa sổ) không nên diễn giải là khác biệt thật. Khoảng cách với XGBoost và seasonal naive lớn hơn mức dao động này nhiều.
- Siêu tham số của LSTM và XGBoost chưa được tinh chỉnh kỹ; cả hai đều có thể còn dư địa cải thiện. GRU chạm mức tối đa 40 epoch nên có thể tăng số epoch tối đa.
- Lỗ hổng ≤ 6 giờ được nội suy nên một phần dữ liệu là ước lượng (đã có cờ `imputed`); các lỗ dài bị loại khỏi chuỗi.
- Nguyên nhân các ngày sai nhiều nhất và lý do XGBoost thắng mới ở mức giả thuyết.

## 8. Cấu trúc thư mục

```
TT-27-RNN-LSTM/
├── README.md
├── requirements.txt
├── notebooks/lstm_traffic.ipynb
├── src/{data.py, sequences.py, train.py}
├── models/lstm_traffic.keras
└── reports/
    ├── eda_theo_gio.png, rnn_lstm_gru.png, du_bao_vs_thuc_te.png, do_dai_cua_so.png, phan_tich_loi.png
```