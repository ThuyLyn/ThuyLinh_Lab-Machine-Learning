# TT-04 — Logistic Regression: Dự đoán nguy cơ bệnh tim

## Cấu trúc dự án

```
TT-04-LogisticRegression/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── logistic_heart.ipynb    
├── src/
│   └── train.py                 
├── data/
│   └── heart.csv              
├── models/
│   └── logreg_pipeline.joblib
└── reports/
    ├── metrics.json            
    ├── vif_table.csv
    ├── odds_ratio.png
    └── roc_pr_curve.png
```

## Cài đặt

```bash
python -m venv .venv
source .venv/bin/activate      
pip install -r requirements.txt
```

## Chạy

```bash
jupyter notebook notebooks/logistic_heart.ipynb
# hoặc
python src/train.py
```


---

## Danh sách lỗi đã sửa (đối chiếu nhận xét chấm bài)

### 1. `ColumnTransformer` bỏ sót cột — `sex`, `fbs`, `exang` bị vứt âm thầm

**Sai ở đâu:** `categorical_columns` + `numeric_columns` cộng lại chỉ có 10
cột, trong khi dữ liệu có 13 cột đặc trưng. `ColumnTransformer` mặc định dùng
`remainder="drop"`, nên 3 cột còn lại (`sex`, `fbs`, `exang`) **bị vứt bỏ mà
không có bất kỳ cảnh báo nào** — cả notebook lẫn `train.py` cũ đều không
nhắc tới việc này.

**Đã sửa:** thêm `sex`, `fbs`, `exang` vào `categorical_columns` (đều là biến
nhị phân 0/1, bản chất là phân loại chứ không phải liên tục). Thêm một dòng
`assert` kiểm tra tường minh: tổng số cột khai báo phải khớp đúng số cột
trong dữ liệu, để lỗi tương tự không thể lặp lại trong im lặng lần nữa.

```python
categorical_columns = [
    "cp", "restecg", "slope", "thal", "ca",
    "sex", "fbs", "exang",
]
numeric_columns = ["age", "trestbps", "chol", "thalach", "oldpeak"]

assert set(categorical_columns + numeric_columns + [target_column]) == set(df.columns)
```

### 2. VIF = inf — `OneHotEncoder` thiếu `drop="first"`

**Sai ở đâu:** không dùng `drop="first"`, nên với 1 biến phân loại có k mức,
tổng k cột one-hot của biến đó **luôn cộng lại đúng bằng 1** ở mọi dòng dữ
liệu — đây là đa cộng tuyến hoàn hảo với hệ số chặn (intercept) của mô hình
hồi quy dùng để tính VIF. Kết quả: toàn bộ cột phân loại đều ra `VIF = inf`
kèm `RuntimeWarning: divide by zero`, `vif_table.csv` lưu ra toàn `inf` —
vô dụng để đọc.

**Đã sửa:** thêm `drop="first"` vào `OneHotEncoder`, dùng chung 1
`preprocessor` cho cả việc train model, tính odds ratio, và tính VIF.

```python
preprocessor = ColumnTransformer([
    ("categorical", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_columns),
    ("numeric", StandardScaler(), numeric_columns),
])
```

**Kết quả (đo trên validation, áp dụng lên test):**

| | Bản cũ (sai) | Bản đã sửa |
|---|---|---|
| Ngưỡng | 0,00209 | **0,2255** |
| Recall | 1,0000 | 0,9394 (test) / 0,9091 (validation) |
| Precision | 0,5410 (= tỉ lệ nền, vô nghĩa) | **0,8378** (test) — cao hơn hẳn tỉ lệ nền 0,541 |
| Số bệnh nhân bị gắn cờ | gần như 100% | 37/61 (61%) |

## Kết quả cuối cùng (chạy thực tế, dataset 1025 dòng → 302 sau khi loại trùng)

### Làm sạch dữ liệu
- Loại **723 dòng trùng lặp** (1025 → 302), kiểm chứng lại `duplicated() == 0`.

### So sánh 3 model (trên tập test, 61 dòng)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Dummy baseline | 0,5410 | 0,5410 | 1,0000 | 0,7021 | 0,5000 |
| Logistic Regression | 0,8361 | 0,8710 | 0,8182 | 0,8438 | **0,9286** |
| L1 | 0,8361 | 0,8710 | 0,8182 | 0,8438 | 0,9232 |
| L2 | 0,8361 | 0,8710 | 0,8182 | 0,8438 | 0,9286 |

L1 đưa **5/22 hệ số về 0** — loại bớt được các biến ít đóng góp mà vẫn giữ
nguyên accuracy/precision/recall so với L2.

### Chọn ngưỡng đạt Recall ≥ 0,90 (đã sửa rò rỉ + suy biến)

- Ngưỡng chọn trên **validation**: `0.2255`
- Áp dụng lên **test** (chưa từng dùng để chọn gì trước đó):
  Precision = **0,8378**, Recall = **0,9394**, gắn cờ 37/61 bệnh nhân.

### VIF (đa cộng tuyến)
- `vif_inf_count = 0` (trước đây toàn bộ cột đều `inf`).
- VIF cao nhất: `thal_2 ≈ 15,7` — xem đầy đủ trong `reports/vif_table.csv`.

### Top hệ số Odds Ratio (ảnh hưởng mạnh nhất tới nguy cơ bệnh tim)
`cp_2` (OR ≈ 4,93), `cp_3` (OR ≈ 3,02), `cp_1` (OR ≈ 1,98),
`restecg_1` (OR ≈ 1,89), `thal_2` (OR ≈ 1,82) — xem đầy đủ trong
`reports/odds_ratio.png` và `reports/vif_table.csv`.

---