# TT-24 — PCA trên dữ liệu cảm biến thiết bị đeo

Nén 561 tín hiệu cảm biến (gia tốc kế + con quay hồi chuyển) xuống còn vài chục
chiều bằng PCA, phục vụ nhận dạng hoạt động (đi bộ / lên-xuống cầu thang / đứng /
ngồi / nằm) trên thiết bị phần cứng hạn chế (64 KB RAM).

## Bộ dữ liệu sử dụng

**Human Activity Recognition Using Smartphones (UCI HAR Dataset)**
— thu thập bởi Davide Anguita et al. (Smartlab, DIBRIS, Đại học Genoa).

- Nguồn: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
- 10.299 dòng × 561 đặc trưng, thu từ cảm biến accelerometer + gyroscope của
  smartphone Samsung Galaxy S II đeo ở thắt lưng
- 6 nhãn hoạt động: WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING,
  STANDING, LAYING
- Chia sẵn theo người: 21 người ở tập train, 9 người ở tập test (không được trộn
  lại và chia ngẫu nhiên — xem cảnh báo bên dưới)

## Cách chạy

### 1. Cài thư viện

```bash
pip install -r requirements.txt
```

### 2. Tải dữ liệu (làm 1 lần, thủ công)

Môi trường sinh code này không có quyền truy cập internet ra ngoài để tự tải bộ
dữ liệu, nên bạn cần tải thủ công:

1. Vào https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
2. Bấm **Download** để lấy `UCI HAR Dataset.zip` (khoảng 58 MB)
3. Giải nén, copy thư mục `UCI HAR Dataset/` vào `dataset/` của dự án này, sao cho có:

```
dataset/UCI HAR Dataset/
├── activity_labels.txt
├── features.txt
├── train/{X_train.txt, y_train.txt, subject_train.txt}
└── test/{X_test.txt, y_test.txt, subject_test.txt}
```

### 3. Chạy pipeline đầy đủ (script)

```bash
cd src
python pca_pipeline.py --data-dir "../dataset/UCI HAR Dataset" --out-dir ".."
```

Sinh ra:
- `reports/scree_plot.png`, `variance_tich_luy.png`, `danh_doi_chieu_accuracy.png`,
  `pc1_pc2_scatter.png`, `reconstruction_error.png`
- `reports/bao_cao_ket_qua.md` — bảng số liệu tổng hợp
- `models/pca_pipeline.joblib` — pipeline `StandardScaler → PCA(k tối ưu) → LinearSVC`

### 4. Hoặc chạy tương tác từng bước (notebook)

```bash
jupyter notebook notebooks/pca_har_sensors.ipynb
```

## Cấu trúc dự án

```
TT-24-PCA/
├── README.md
├── requirements.txt
├── notebooks/pca_har_sensors.ipynb   ← chạy tương tác, có giải thích từng bước
├── src/
│   └── pca_pipeline.py               ← TOÀN BỘ code trong 1 file: nạp dữ liệu (giữ
│                                        đúng cách chia theo người), chuẩn hoá, PCA,
│                                        các biểu đồ, thí nghiệm K vs accuracy, lưu model
├── dataset/UCI HAR Dataset/              ← bạn tự tải vào đây (xem hướng dẫn trên)
├── models/pca_pipeline.joblib         ← sinh ra sau khi chạy
└── reports/                           ← các biểu đồ + báo cáo, sinh ra sau khi chạy
```

## Những điểm quan trọng đã cài trong code

| Yêu cầu trong đề bài | Đã xử lý ở đâu (tất cả trong `src/pca_pipeline.py`) |
|---|---|
| Giữ đúng cách chia theo người, không trộn lại | `load_har_data()` — đọc trực tiếp `train/` và `test/` gốc, có `assert` kiểm tra không subject nào trùng giữa 2 tập |
| `StandardScaler` fit chỉ trên train | `fit_scaler_and_full_pca()` |
| PCA không được `fit` trên toàn bộ dữ liệu | PCA luôn `fit` trên `X_train_s`, `X_test_s` chỉ `transform` |
| Scree plot + phương sai tích luỹ | `ve_scree_plot()`, `ve_phuong_sai_tich_luy()` |
| Bảng ngưỡng phương sai → số chiều | `bang_nguong_phuong_sai()` |
| Đánh đổi K vs accuracy, tìm điểm ngọt | `thi_nghiem_danh_doi_K()` + `tim_diem_ngot()` (điểm ngọt = K nhỏ nhất mà accuracy giảm < 1% so với K=561) |
| Scatter PC1–PC2 theo 6 lớp | `ve_scatter_pc1_pc2()` |
| Phân tích ý nghĩa PC1 | `phan_tich_pc1()` — top 10 đặc trưng gốc theo `|trọng số|` |
| Đo dung lượng trước/sau PCA | `do_dung_luong()` |
| Tái tạo ngược + sai số theo K | `sai_so_tai_tao_theo_K()` |
| PCA nằm trong Pipeline | Pipeline cuối trong `main()`: `StandardScaler → PCA → LinearSVC` |

## Hạn chế của PCA (cần nêu trong báo cáo nộp)

- **Mất tính giải thích**: PC1, PC2... là tổ hợp tuyến tính của hàng chục/hàng
  trăm đặc trưng gốc, không có ý nghĩa vật lý trực tiếp như một cảm biến đơn lẻ.
- **Chỉ bắt được quan hệ tuyến tính**: nếu ranh giới giữa các hoạt động là phi
  tuyến (ví dụ phân biệt SITTING/STANDING chủ yếu qua tương tác phi tuyến giữa
  các trục cảm biến), PCA có thể bỏ lỡ — đây là lý do phần mở rộng đề xuất thử
  Kernel PCA.
- **Nhạy với thang đo**: bắt buộc chuẩn hoá trước, nếu không cột có phương sai
  lớn sẽ chi phối PC1 dù không quan trọng về mặt hoạt động.
