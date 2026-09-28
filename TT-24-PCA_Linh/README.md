# TT-24 — PCA trên dữ liệu cảm biến thiết bị đeo
# PCA giảm chiều cho nhận dạng hoạt động (UCI HAR)

Giảm 561 đặc trưng cảm biến điện thoại bằng PCA để mô hình nhận dạng 6 hoạt động chạy được trên thiết bị RAM nhỏ (~64 KB).

## Chạy
```bash
pip install -r requirements.txt
python src/pca_pipeline.py            # mặc định: dataset/UCI HAR Dataset, ghi reports/ và models/ ở gốc dự án
python src/pca_pipeline.py --data-dir "đường/dẫn/UCI HAR Dataset" --ram-kb 64
```
Kết quả: `reports/metrics.json`, `reports/bao_cao_ket_qua.md`, các hình `reports/*.png`, mô hình `models/pca_pipeline.joblib`.
Phân tích chi tiết nằm trong `notebooks/pca_har_sensors.ipynb`.

## Phương pháp
- `StandardScaler` và PCA chỉ fit trên train; giữ nguyên chia train/test theo người, có `assert` không trùng subject (21 người train, không giao với test).
- K chọn bằng **GroupKFold(5) theo subject** trên train, quy tắc **1-SE** (K nhỏ nhất có CV-accuracy >= tốt nhất - 1 SE). Test chỉ dùng để báo cáo cuối, không dùng chọn K.
- Có tính footprint mô hình theo K so với RAM đích và so sánh PCA với SelectKBest (cùng CV, cùng K).

## Kết quả
| Ngưỡng phương sai | 80% | 90% | 95% | 99% |
|---|---|---|---|---|
| Số PC cần | 26 | 63 | 102 | 179 |
| Giảm chiều | 95.4% | 88.8% | 81.8% | 68.1% |

| Mục | Giá trị |
|---|---|
| K chọn (1-SE) | **102** (giảm 81.8% số chiều) |
| CV accuracy (K=102 / K=561) | 0.9137 / 0.9305 |
| Test accuracy (K=102 / K=561) | 0.9301 / 0.9617 |
| Thời gian fit (K=102 / K=561) | 0.74 s / 14.55 s (~20 lần) |
| Footprint mô hình K=102 | 226 KB (fp32), 56.6 KB (int8) |
| K lớn nhất vừa 64 KB | 26 (fp32), 102 (int8) |
| PCA vs SelectKBest tại K=102 (CV) | 0.9137 vs 0.9172 |
| MSE tái tạo (test) tại K=102 | 0.056 |

**PC1** đại diện cho cường độ chuyển động (top loadings đều là `sma`/`mean`/`std` của AccJerk, Gyro), tách hoạt động động khỏi tĩnh.

## Hạn chế
- Test accuracy giảm ~3.2 điểm so với dùng đủ 561 đặc trưng (CV giảm ~1.7 điểm); độ lệch giữa các fold khoảng ±0.04.
- K=102 chỉ vừa 64 KB khi lượng tử hoá int8; ở fp32 chỉ K <= 26 vừa (CV 0.8757).
- PCA và LinearSVC đều tuyến tính nên khi triển khai có thể gộp thành một ma trận 6x561; khi đó lợi ích của PCA chủ yếu là tốc độ huấn luyện.

Chi tiết từng bảng: `reports/bao_cao_ket_qua.md`.