# TT-14-ElasticNet-Linh

Dự báo tải sưởi (Y1) và tải làm mát (Y2) toà nhà bằng ElasticNet, xử lý đa cộng tuyến
giữa các biến thiết kế (X1, X2, X4, X5) trên bộ dữ liệu Energy Efficiency (UCI, 768 dòng).

## Cấu trúc

```
TT-14-ElasticNet-Linh/
├── README.md
├── dataset/ENB2012_data.xlsx
├── notebooks/elasticnet_energy.ipynb
├── src/train.py
├── models/{elasticnet_Y1.joblib, elasticnet_Y2.joblib}
├── reports/{vif_table.csv, correlation_heatmap.png, so_sanh_model_full.csv,
│            heatmap_alpha_l1ratio_Y1.png, heatmap_alpha_l1ratio_Y2.png,
│            grouping_effect_Y1.csv, grouping_effect_Y2.csv,
│            bootstrap_std_Y1.csv, bootstrap_std_Y2.csv}
└── requirements.txt
```

## Cách chạy

```bash
pip install -r requirements.txt
python src/train.py
```

---

## 1. Bảng VIF — chứng minh đa cộng tuyến

| feature | VIF |
|---|---|
| X2 | 1.00×10¹⁵ |
| X3 | 1.00×10¹⁵ |
| X4 | 1.00×10¹⁵ |
| X1 | 105.52 |
| X5 | 31.21 |
| X7 | 1.00 |

**Nhận xét:** VIF của X2, X3, X4 lớn tới mức statsmodels phải cảnh báo *"design matrix
is rank-deficient"* — tức ma trận thiết kế gần như **suy biến hoàn toàn**, không chỉ là
tương quan cao mà gần như có **quan hệ tuyến tính chính xác** giữa các biến này (khớp với
mô tả trong đề bài: diện tích tường/mái/sàn/chiều cao bị ràng buộc bởi hình học toà nhà).
X1 (VIF ≈ 106) và X5 (VIF ≈ 31) cũng vượt xa ngưỡng cảnh báo thông thường (VIF > 10).
Chỉ X7 (diện tích kính) độc lập với nhóm này (VIF ≈ 1).

→ Đây là bằng chứng số học rõ ràng cho việc **không thể dùng Lasso một mình** — nó sẽ
loại bỏ ngẫu nhiên các biến trong nhóm suy biến này dù chúng đều có ý nghĩa vật lý.

## 2. Bảng so sánh Ridge / Lasso / ElasticNet

### Y1 — Tải sưởi (Heating Load)

| Model | RMSE | R² | Số biến giữ |
|---|---|---|---|
| Dummy (baseline) | 10.238 | -0.006 | — |
| Linear Regression | 2.872 | 0.921 | 14 |
| Ridge | 2.875 | 0.921 | 14 |
| Lasso | 2.875 | 0.921 | 14 |
| **ElasticNet** | **2.876** | **0.921** | **14** |

ElasticNet chọn **alpha = 0.00045, l1_ratio = 0.1**

### Y2 — Tải làm mát (Cooling Load)

| Model | RMSE | R² | Số biến giữ |
|---|---|---|---|
| Dummy (baseline) | 9.666 | -0.008 | — |
| Linear Regression | 3.120 | 0.895 | 14 |
| Ridge | 3.121 | 0.895 | 14 |
| Lasso | 3.120 | 0.895 | 14 |
| **ElasticNet** | **3.121** | **0.895** | **14** |

ElasticNet chọn **alpha = 0.00032, l1_ratio = 0.1**

**Nhận xét:** Cả 4 model (Linear/Ridge/Lasso/ElasticNet) đều cải thiện rất mạnh so với
baseline Dummy (RMSE giảm từ ~10 xuống ~2.9 cho Y1, từ ~9.7 xuống ~3.1 cho Y2), chứng tỏ
các biến thiết kế có sức giải thích cao. Tuy nhiên **alpha tối ưu rất nhỏ** (~0.0003–0.00045)
và **không model nào loại bỏ biến nào** (đều giữ đủ 14 biến) — điều này cho thấy trên bộ
dữ liệu mô phỏng "sạch" này, hầu như mọi biến đều mang thông tin hữu ích, quy mô phạt cần
rất nhẹ mới không làm giảm độ chính xác.

## 3. Hiệu ứng gom nhóm (grouping effect) 

So sánh hệ số hồi quy của Lasso và ElasticNet trên nhóm biến tương quan X1, X2, X4, X5:

### Y1

| Biến | Lasso | ElasticNet |
|---|---|---|
| X1 | -6.359 | -6.221 |
| X2 | -6.278 | -3.482 |
| X4 | -0.944 | -3.637 |
| X5 | 7.305 | 7.324 |

### Y2

| Biến | Lasso | ElasticNet |
|---|---|---|
| X1 | -7.357 | -7.180 |
| X2 | -7.294 | -3.995 |
| X4 | -0.683 | -3.825 |
| X5 | 7.170 | 7.214 |

**Nhận xét về gom nhóm:** Ở cả Y1 và Y2, hành vi giống nhau và rất rõ:
- **Lasso** đẩy gần hết "trọng lượng" của cặp (X2, X4) về phía X2 (hệ số ≈ -6.3 đến -7.3),
  gần như bỏ qua X4 (hệ số chỉ còn -0.68 đến -0.94) — mặc dù cả 2 biến này đều mô tả cùng
  một đặc tính hình học của toà nhà (diện tích bề mặt và diện tích mái, hai đại lượng gắn
  chặt với nhau qua công thức thiết kế).
- **ElasticNet** chia lại tải giữa X2 và X4 cân bằng hơn nhiều (X2 ≈ -3.5 đến -4.0,
  X4 ≈ -3.6 đến -3.8) — thay vì gán gần hết cho một biến rồi bỏ gần hết biến kia, nó phân
  bổ đóng góp đồng đều cho cả nhóm, đúng như lý thuyết "grouping effect" dự đoán.
- X1 và X5 ổn định ở cả 2 model vì mức độ tương quan của chúng với phần còn lại của nhóm
  thấp hơn X2-X4.

→ **Kết luận:** dù cả 2 model không loại biến nào về 0 tuyệt đối trên bộ dữ liệu này (do
alpha tối ưu rất nhỏ), Lasso vẫn thể hiện xu hướng "chọn đại diện, bỏ qua phần còn lại"
trong cặp tương quan mạnh nhất (X2, X4), trong khi ElasticNet phân bổ hệ số cân bằng và
ổn định hơn giữa các biến cùng nhóm — đây chính là ưu điểm cốt lõi của ElasticNet so với
Lasso khi dữ liệu có đa cộng tuyến nặng.

## 4. Ý nghĩa của l1_ratio máy chọn

Cả Y1 (l1_ratio = 0.1) và Y2 (l1_ratio = 0.1) đều nghiêng hẳn về phía **Ridge** (l1_ratio
gần 0), không phải Lasso. Điều này cho thấy: dữ liệu có **nhiều biến cùng đóng góp thông
tin** (không có vài biến "vô dụng" cần loại bỏ hẳn), nên mô hình ưu tiên phần phạt L2 để
co đều hệ số và giữ ổn định trước đa cộng tuyến, thay vì phần phạt L1 để triệt tiêu biến.
Kết quả này khớp với quan sát ở mục 2: không model nào loại bỏ biến nào trong 14 biến.

## 5. Heatmap alpha × l1_ratio

Xem `reports/heatmap_alpha_l1ratio_Y1.png` và `reports/heatmap_alpha_l1ratio_Y2.png`.
Vùng RMSE thấp nhất tập trung ở alpha rất nhỏ (< 0.01) trên toàn dải l1_ratio, phù hợp với
kết quả alpha tối ưu ở mục 2 — dữ liệu không cần regularization mạnh.

## 6. Kiểm tra ổn định hệ số (bootstrap, 100 lần)

Top 5 biến dao động mạnh nhất qua 100 lần bootstrap:

**Y1:** X1 (0.740) > X5 (0.589) > X2 (0.471) > X4 (0.457) > X8_2 (0.263)
**Y2:** X1 (0.970) > X5 (0.676) > X2 (0.629) > X4 (0.611) > X8_3 (0.269)

**Nhận xét:** Đúng như dự đoán, 4 biến trong nhóm tương quan (X1, X2, X4, X5) có độ lệch
chuẩn hệ số **cao hơn hẳn** so với các biến còn lại (X8_x chỉ 0.26-0.27, thấp hơn 2-3 lần).
Đây là hệ quả trực tiếp của đa cộng tuyến: khi mẫu bootstrap thay đổi nhẹ, mô hình có thể
"đánh đổi" trọng số qua lại giữa các biến cùng nhóm mà vẫn giữ dự đoán gần như không đổi —
nên các biến này kém ổn định hơn dù dự đoán tổng thể vẫn chính xác.

## 7. So sánh Y1 vs Y2 — biến nào quan trọng cho sưởi, biến nào cho làm mát

| Hạng | Y1 (tải sưởi) | Y2 (tải làm mát) |
|---|---|---|
| 1 | X5 (7.324) | X5 (7.214) |
| 2 | X1 (-6.221) | X1 (-7.180) |
| 3 | X4 (-3.637) | X2 (-3.995) |
| 4 | X2 (-3.482) | X4 (-3.825) |
| 5 | X7 (2.313) | X7 (1.814) |

**Nhận xét:** Thứ tự quan trọng gần như giống hệt nhau giữa 2 nhãn — X5 (chiều cao tổng)
và X1 (độ gọn tương đối) luôn là 2 yếu tố ảnh hưởng mạnh nhất tới cả tải sưởi lẫn tải làm
mát, do đây là 2 đặc trưng hình học tổng quát nhất chi phối diện tích trao đổi nhiệt của
toà nhà. Điểm khác biệt đáng chú ý: **X7 (diện tích kính) đóng góp mạnh hơn tương đối cho
Y1 (2.313) so với Y2 (1.814)** trên bộ dữ liệu này — dù về mặt vật lý dân dụng, kính thường
ảnh hưởng tải làm mát nhiều hơn qua bức xạ mặt trời; sự khác biệt này có thể do đặc thù
phân bố dữ liệu mô phỏng, nên khi có dữ liệu thật cần đối chiếu lại.

## 8. Đề xuất thiết kế

1. **Ưu tiên kiểm soát X5 (chiều cao tổng) và X1 (độ gọn tương đối)** — đây là 2 yếu tố
   ảnh hưởng mạnh nhất tới cả tải sưởi lẫn tải làm mát; giảm chiều cao trần hoặc tăng độ
   gọn (giảm tỷ lệ diện tích bề mặt / thể tích) sẽ giảm tải năng lượng ở cả 2 mùa.
2. **Không nên tối ưu X2 (diện tích bề mặt) hoặc X4 (diện tích mái) một cách riêng lẻ** —
   vì hai biến này dính chặt nhau về hình học (VIF ~ 10¹⁵), thay đổi một trong hai kéo theo
   thay đổi biến còn lại; cần đánh giá đồng thời cả cụm hình học (X1, X2, X4, X5) khi ra
   quyết định thiết kế, thay vì chỉnh từng biến độc lập.
3. **Cân nhắc diện tích kính (X7)** như một đòn bẩy thiết kế phụ — dù ảnh hưởng nhỏ hơn
   nhóm hình học chính, đây là biến độc lập duy nhất (VIF ≈ 1) và dễ điều chỉnh trong giai
   đoạn thiết kế mà không kéo theo thay đổi các thông số hình học khác.

---

**Lưu ý:** Model đạt R² ≈ 0.92 (Y1) và R² ≈ 0.89 (Y2), tốt hơn baseline rất nhiều —
đạt mức tham chiếu đề ra trong bài. Đây là bộ dữ liệu mô phỏng nên quan hệ rất sạch,
đừng kỳ vọng dữ liệu thật đẹp như vậy.