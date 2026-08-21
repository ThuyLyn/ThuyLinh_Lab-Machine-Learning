# TT-03 — Random Forest: Ai sẽ mở sổ tiết kiệm?

Dự án tối ưu danh sách gọi telesales cho chiến dịch mời mở sổ tiết kiệm có kỳ hạn,
sử dụng Random Forest trên bộ dữ liệu Bank Marketing (UCI).

## Cấu trúc thư mục

```
TT-03-RandomForest-<HoTen>/
├── README.md
├── data/bank-additional-full.csv      
├── notebooks/
│   ├── 01_leakage_demo.ipynb       
│   └── 02_random_forest.ipynb        
├── src/train.py                    
├── models/rf_pipeline.joblib          
├── outputs/danh_sach_goi_top5000.csv  
├── reports/
│   ├── auc_theo_so_cay.png
│   ├── permutation_importance.png
│   └── lift_curve.png
└── requirements.txt
```

## Cách chạy

```bash
python src/train.py
```

Script sẽ tự đọc `data/bank-additional-full.csv`, chạy toàn bộ pipeline, in các
chỉ số ra màn hình, lưu biểu đồ vào `reports/`, lưu model vào `models/`, và xuất
danh sách top 5000 khách hàng vào `outputs/`.


### Kết quả thực nghiệm (điền số liệu thật sau khi chạy `01_leakage_demo.ipynb`)

| Phiên bản | ROC-AUC | Nhận xét |
|---|---|---|
| CÓ cột `duration` | 0.9473 | Đẹp giả tạo — leakage |
| KHÔNG có cột `duration` | 0.7834 | Con số THẬT, dùng được |

### Kết luận

Từ bước xử lý dữ liệu chính thức trở đi (notebook `02_random_forest.ipynb` và
`src/train.py`), **cột `duration` bị loại bỏ hoàn toàn** khỏi tập đặc trưng. Đây
cũng là khuyến nghị chính thức của tài liệu UCI: *"should be discarded if the
intention is to have a realistic predictive model"*.

---

## Xử lý dữ liệu đáng chú ý

- **`pdays = 999`** ("chưa từng liên hệ trước đây") không phải là một con số ngày
  thật — được tách thành cột cờ nhị phân `pdays_contacted_before` + cột số
  `pdays_clean` (999 → -1) để tránh model hiểu nhầm là "999 ngày trước".
- **Giá trị `unknown`** trong `job`, `education`, `housing`, `loan`... được giữ
  nguyên như một mức (category) riêng, không impute, vì bản thân việc thiếu
  thông tin cũng có thể mang tín hiệu dự đoán.
- **Mất cân bằng lớp** (~11.3% yes): dùng `class_weight='balanced_subsample'`
  thay vì resample, và đánh giá bằng ROC-AUC / PR-AUC / Precision@K thay vì
  accuracy.

## Kết quả chính (điền sau khi chạy)

| Chỉ số | Giá trị |
|---|---|
| OOB score | | 0.8915
| ROC-AUC (test, không duration) | | 0.7863
| PR-AUC (test) | | 0.4308
| Precision@5000 | | 0.1594
| Lift@5000 | | 1.42x so với gọi ngẫu nhiên
| ROC-AUC cây đơn (baseline) so với Random Forest | | 0.7917

