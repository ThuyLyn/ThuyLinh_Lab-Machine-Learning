# TT-19 — XGBoost Regressor: Dự báo nhu cầu thuê xe đạp theo giờ

## 1. Mục tiêu

Dự báo `cnt` (tổng lượt thuê xe đạp trong 1 giờ) từ bộ dữ liệu UCI Bike Sharing
(`hour.csv`), phục vụ điều phối xe giữa các trạm.

## 2. Cách chạy

```bash
pip install -r requirements.txt
cd src
python train.py --data ../data/hour.csv

## 3. CHỨNG MINH RÒ RỈ (bắt buộc)

`cnt = casual + registered` là một quan hệ **đại số tuyệt đối**, không phải
tương quan thống kê. Hàm `prove_leakage()` trong `src/features.py` train một
model XGBoost nhanh CÓ giữ 2 cột `casual`/`registered` để minh hoạ điều này.

**Kết quả cần điền sau khi chạy (`reports/leakage_proof.json`):**

| | |
|---|---|
| R² khi giữ casual/registered | **0.9976** |
| RMSE khi giữ casual/registered | **10.75** lượt/giờ |

**Nhận xét:** R² = 0.9976 (gần tuyệt đối 1.0) và RMSE chỉ 10.75 lượt/giờ — thấp
hơn rất nhiều so với mọi kết quả ở các phần sau (RMSE ~118 lượt/giờ khi bỏ
2 cột này). Đây là bằng chứng rõ ràng cho rò rỉ: model gần như chỉ đang học
lại phép cộng `casual + registered`, không học được yếu tố ảnh hưởng thực sự
đến nhu cầu thuê xe. Trong tình huống dự báo thật, ta **không biết trước**
`casual` và `registered` của giờ tương lai — nên 2 cột này bị loại bỏ hoàn
toàn khỏi tập feature cho phần còn lại của bài (`drop_leaky_columns()`).

## 4. Chia dữ liệu theo thời gian

| Tập | Điều kiện | Lý do |
|---|---|---|
| Train | `yr == 0` (năm 1) | |
| Validation | `yr == 1` & `mnth <= 9` | dùng cho early stopping |
| Test | `yr == 1` & `mnth > 9` | 3 tháng cuối, không nhìn thấy khi train |

**Kích thước thực tế:** train = 8.645 dòng · validation = 6.566 dòng · test = 2.168 dòng.

Không dùng `train_test_split(shuffle=True)` vì đây là dữ liệu chuỗi thời gian
— shuffle sẽ khiến model học được thông tin từ "tương lai".

## 5. Baseline naive vs XGBoost (bắt buộc)

Baseline: `cnt(giờ này) = cnt(cùng giờ, tuần trước)` (lag 168 giờ).

| Model | RMSE (test) | R² (test) |
|---|---|---|
| Baseline naive (tuần trước) | 156.79 | 0.3951 |
| XGBoost (raw cnt) | **118.63** | **0.6537** |
| XGBoost (log1p cnt) | 126.38 | 0.6070 |

**Nhận xét:** XGBoost (raw cnt) THẮNG baseline naive, giảm RMSE từ 156.79
xuống 118.63 (**~24% thấp hơn**) và R² tăng từ 0.395 lên 0.654. Model được
chọn là bản **raw cnt**, không phải log1p — log1p cho RMSE cao hơn (126.38)
dù thường được kỳ vọng giúp ích với nhãn lệch phải. Lý do nhiều khả năng:
RMSE bị chi phối bởi các giờ cao điểm có `cnt` lớn (vài trăm lượt), trong khi
log1p "nén" các giá trị lớn lại nhiều hơn — khi đổi ngược bằng `expm1`, một
sai số nhỏ trong không gian log có thể phóng đại thành sai số lớn ở giờ cao
điểm, làm RMSE tổng thể xấu đi ngay cả khi phần lớn giờ trong ngày (đêm khuya,
`cnt` nhỏ) được dự báo tốt hơn.

 **RMSE ~118–126 cao hơn đáng kể so với mức tham chiếu của đề (RMSE ~40–55).**
Đây nhiều khả năng không phải lỗi mà là hệ quả của việc **chia dữ liệu đúng
theo thời gian**: nhiều benchmark phổ biến cho bộ dữ liệu này dùng chia ngẫu
nhiên (random split), vô tình để lộ các giờ liền kề của cùng một ngày giữa
train và test — mô hình "nhìn thấy" gần như cùng bối cảnh (ngày, thời tiết,
xu hướng tăng trưởng người dùng) và đạt R² cao giả tạo. Với time-based split
thật sự (test là 3 tháng chưa từng thấy), bài toán khó hơn nhiều vì phải
ngoại suy sang giai đoạn có thể có xu hướng người dùng khác. Hướng cải thiện
tiếp theo: áp dụng bộ tham số tối ưu từ RandomizedSearchCV (mục 6) để
retrain, và thử thêm đặc trưng lag/rolling (vd: `cnt` trung bình 24h/168h gần
nhất) — vốn không có trong tập feature hiện tại.

## 6. Kết quả model

- Số cây thực tế dùng (early stopping): **1591** (trên tổng 2000 cây cấu hình, dừng sớm khi validation RMSE hết cải thiện quanh mốc ~125.6)
- RMSE / R² trên test: **118.63 / 0.6537** (raw cnt) — mức tham chiếu đề bài: RMSE ~40–55, R² ~0.93–0.95 (xem giải thích chênh lệch ở mục 5)

**Top 10 feature quan trọng nhất (theo gain):**

| Hạng | Feature | Gain (tỷ lệ) |
|---|---|---|
| 1 | `workingday` | 0.2467 |
| 2 | `hr` | 0.2169 |
| 3 | `hr_sin` | 0.1258 |
| 4 | `hr_cos` | 0.0802 |
| 5 | `season` | 0.0617 |
| 6 | `temp` | 0.0577 |
| 7 | `atemp` | 0.0411 |
| 8 | `weathersit` | 0.0378 |
| 9 | `mnth` | 0.0304 |
| 10 | `weekday_cos` | 0.0229 |

**Nhận xét:** `workingday` và `hr` (cùng bản mã hoá chu kỳ `hr_sin`/`hr_cos`)
chiếm hơn 65% tổng gain — đúng như kỳ vọng, vì nhu cầu thuê xe gắn chặt với
giờ đi làm/tan làm trong ngày thường. Các yếu tố thời tiết (`temp`, `atemp`,
`weathersit`) và mùa (`season`) đóng góp ở mức vừa phải; `weekday_cos` lọt
top 10 cho thấy có khác biệt giữa các ngày trong tuần dù không mạnh bằng giờ
trong ngày.

**Kết quả RandomizedSearchCV (TimeSeriesSplit, bước 9):**

```
{'subsample': 1.0, 'min_child_weight': 1, 'max_depth': 4,
 'learning_rate': 0.1, 'colsample_bytree': 0.6}
```

Bộ tham số này nông hơn (`max_depth=4` so với 6) và learning rate cao hơn
(0.1 so với 0.03) so với cấu hình mặc định trong `train.py` — gợi ý model
mặc định có thể đang hơi phức tạp so với lượng dữ liệu train (8.645 dòng).
*Bước tiếp theo nên làm:* dùng đúng bộ tham số này để retrain model cuối và
so sánh RMSE/R² trên test với bản hiện tại (118.63 / 0.6537).

## 7. Phân tích lỗi

Output đã chạy chưa in ra số liệu lỗi theo giờ/thời tiết (phần này chỉ được
lưu trực tiếp vào `reports/phan_tich_loi.png`, không in ra console). Mở file
đó và điền nhận xét theo 2 câu hỏi:
- Model sai nhiều nhất ở giờ nào? (dự đoán: giờ cao điểm 8h và 17–18h, vì đây
  là các giờ có `cnt` lớn nên sai số tuyệt đối cũng dễ lớn theo)
- Model sai nhiều nhất ở điều kiện thời tiết nào? (dự đoán: `weathersit` xấu
  — mưa/tuyết — vì đây là các trường hợp hiếm gặp hơn trong dữ liệu train)

## 8. Cấu trúc thư mục

```
TT-19-XGBoostRegressor/
├── README.md
├── notebooks/xgboost_bike_demand.ipynb
├── src/{features.py, train.py}
├── models/xgb_bike.json
├── reports/{cnt_theo_gio.png, du_bao_vs_thuc_te.png, shap_summary.png, phan_tich_loi.png, leakage_proof.json}
├── dataset/hour.csv       
└── requirements.txt
```
