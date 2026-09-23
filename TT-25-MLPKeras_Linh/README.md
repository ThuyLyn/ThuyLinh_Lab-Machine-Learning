# TT-25 — MLP với Keras
## Chấm điểm khách hàng tiềm năng mua bảo hiểm ô tô
---

## 1. Bài toán

Công ty bảo hiểm có 380.000 khách đang mua bảo hiểm sức khoẻ, muốn bán chéo bảo hiểm ô tô.
Chỉ khoảng 12,3% khách thực sự quan tâm. Đội telesales chỉ gọi được 3.000 cuộc/ngày, nên
bài toán không phải là phân loại 0/1 mà là **xếp hạng** — chọn ra 3.000 khách có khả năng
quan tâm cao nhất để gọi trước.

**Metric chính:** PR-AUC (Precision-Recall AUC) và Precision@3000 — *không* dùng accuracy,
vì đoán bừa "không quan tâm" cho tất cả đã đạt 87,7% accuracy, không nói lên điều gì.

---

## 2. Dữ liệu

| | |
|---|---|
| Nguồn | [Health Insurance Cross Sell Prediction](https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction) (Kaggle) |
| Kích thước | 381.109 dòng × 12 cột |
| Nhãn | `Response` (0/1), ~12,3% dương |
| Đặc trưng | `Gender`, `Age`, `Driving_License`, `Region_Code`, `Previously_Insured`, `Vehicle_Age`, `Vehicle_Damage`, `Annual_Premium`, `Policy_Sales_Channel`, `Vintage` |

### Các lưu ý quan trọng về dữ liệu

- **`Previously_Insured` không phải rò rỉ dữ liệu** — thông tin này đã biết trước khi gọi
  điện, nên được phép dùng làm đặc trưng. Đây cũng là tín hiệu cực mạnh: khách đã có bảo
  hiểm xe gần như chắc chắn không mua nữa.
- **`Region_Code` (53 mức)** và **`Policy_Sales_Channel` (155 mức)** là biến phân loại mã
  hoá bằng số. One-hot sẽ tạo hơn 200 cột thưa → cân nhắc dùng Embedding layer (mục 5.7).
- **`Annual_Premium` lệch phải mạnh** → áp dụng `log1p` trước khi chuẩn hoá.

---

## 3. EDA (tóm tắt)

*(Phần này cần chạy riêng `notebooks/mlp_keras_insurance.ipynb`, mục EDA, để điền số liệu
thật — log huấn luyện ở mục 6 không chứa các thống kê này)*

| Biến | Nhóm | Tỉ lệ Response=1 |
|---|---|---|
| `Previously_Insured` | 0 | ... |
| `Previously_Insured` | 1 | ... |
| `Vehicle_Damage` | Yes | ... |
| `Vehicle_Damage` | No | ... |
| Nhóm tuổi | 25–45 | ... |

Biểu đồ chi tiết (phân phối `Annual_Premium` trước/sau log1p, tỉ lệ theo nhóm tuổi) nằm
trong notebook, phần EDA.

---

## 4. Tiền xử lý

1. `Annual_Premium_log = log1p(Annual_Premium)` — xử lý lệch phải.
2. One-hot encode các biến phân loại (`Gender`, `Vehicle_Age`, `Vehicle_Damage`, và luồng
   riêng cho `Region_Code`/`Policy_Sales_Channel` — xem mục 5.7).
3. `StandardScaler` fit trên tập train, áp dụng lại cho val/test (tránh rò rỉ).
4. Chia tập **70/15/15** train/val/test, có `stratify=y` vì dữ liệu lệch 12,3%.

Code: `src/data.py` — `load_and_split_onehot()` và `load_and_split_embedding()`.

---

## 5. Kiến trúc & phương pháp

### 5.1. Baseline bắt buộc — LightGBM

Với dữ liệu dạng bảng, gradient boosting thường thắng hoặc ngang mạng nơ-ron. Baseline này
được train trước, dùng làm mốc so sánh trung thực cho mọi thử nghiệm MLP.

```python
lgb.LGBMClassifier(n_estimators=500, class_weight="balanced")
```

### 5.2. MLP cơ bản (không Dropout/BatchNorm)

Sequential 2 tầng ẩn (128 → 64), sigmoid đầu ra, dùng làm điểm xuất phát để đo hiệu quả
của regularization.

### 5.3. MLP + Dropout + BatchNorm

```
Dense(128, relu) → BatchNorm → Dropout(0.3)
→ Dense(64, relu) → BatchNorm → Dropout(0.2)
→ Dense(1, sigmoid)
```

- **BatchNorm**: ổn định quá trình huấn luyện, cho phép learning rate cao hơn.
- **Dropout**: chống overfit — dữ liệu bảng với MLP rất dễ overfit vì không có cấu trúc
  không gian/thời gian để mạng khai thác như ảnh hoặc chuỗi.

### 5.4. Ba callback bắt buộc

| Callback | Vai trò |
|---|---|
| `EarlyStopping(monitor='val_pr_auc', patience=10, restore_best_weights=True)` | Dừng khi val PR-AUC không cải thiện, khôi phục trọng số tốt nhất |
| `ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)` | Giảm learning rate khi loss chững lại |
| `ModelCheckpoint(monitor='val_pr_auc', save_best_only=True)` | Lưu model tốt nhất theo PR-AUC, không phải theo epoch cuối |

### 5.5. Xử lý dữ liệu lệch — class_weight

```python
compute_class_weight('balanced', classes=[0, 1], y=y_train)
```
Tăng trọng số cho lớp thiểu số (Response=1) trong hàm loss, thử nghiệm so sánh có/không
class_weight ở mục 5.8.

### 5.6. Ba kiến trúc được thử nghiệm

| Kiến trúc | Số tầng ẩn | Mục đích so sánh |
|---|---|---|
| `(64,)` | 1 | Model nhỏ, ít tham số |
| `(128, 64)` | 2 | Kiến trúc mặc định |
| `(256, 128, 64)` | 3 | Model sâu hơn, nhiều tham số hơn |

### 5.7. Embedding cho biến phân loại nhiều mức

Thay vì one-hot 208 cột cho `Region_Code` (53 mức) + `Policy_Sales_Channel` (155 mức),
dùng `layers.Embedding` để học biểu diễn liên tục, kích thước vector chọn theo kinh
nghiệm `min(50, round(n_categories**0.25 * 4))` (ở đây chọn 8 và 12). Dùng Functional
API vì cần nhiều input riêng biệt (region, channel, numeric).

### 5.8. class_weight có/không

So sánh trực tiếp trên kiến trúc thắng ở mục 5.6, để tách bạch tác động của class_weight
khỏi tác động của kiến trúc.

Code: `src/model.py` — `build_mlp_basic()`, `build_mlp_regularized()`, `build_mlp_embedding()`.

---

## 6. Kết quả

Chạy trên `Train (266.776, 215)` / `Val (57.166, 215)`, tỉ lệ `Response=1` = 12,3%
(khớp với thống kê ở mục 3). Số cột sau one-hot là **215** — xác nhận đúng lưu ý ở mục 2:
`Region_Code` + `Policy_Sales_Channel` khiến one-hot phình to đáng kể.

| Model | PR-AUC | Precision@3000 | Thời gian train | Ghi chú |
|---|---|---|---|---|
| LightGBM (baseline) | 0,3675 | 0,4210 | 2,9s | class_weight=balanced |
| MLP cơ bản | 0,3672 | 0,4190 | 41,0s | không Dropout/BN |
| MLP + Dropout/BN (128,64) | 0,3724 | 0,4190 | 80,5s | class_weight=balanced — **kiến trúc tốt nhất** |
| MLP + Dropout/BN (64,) | 0,3620 | 0,4087 | 68,0s | class_weight=balanced |
| MLP + Dropout/BN (256,128,64) | 0,3705 | 0,4220 | 88,3s | class_weight=balanced |
| MLP (128,64), không class_weight | 0,3738 | 0,4227 | 65,2s | class_weight=None |
| **MLP + Embedding** | **0,3744** | 0,4227 | 70,7s | 15.213 tham số — **PR-AUC cao nhất** |

**Mức tham chiếu của README gốc:** PR-AUC ~0,30–0,35 — kết quả thực tế (0,362–0,374) nằm
đúng trong và hơi vượt khoảng này, khớp với dự đoán rằng LightGBM và MLP sẽ ở mức ngang nhau.

Biểu đồ liên quan: `reports/learning_curves.png`, `reports/kien_truc_comparison.png`,
`reports/embedding_vs_onehot.png` (PR-AUC và số tham số: one-hot tốt nhất vs Embedding),
`reports/pr_curve.png`.

---

## 7. Kết luận 

- **Dropout/BatchNorm có giúp ích không?** Có: PR-AUC tăng từ 0,3672 (MLP cơ bản) lên
  0,3724 (MLP + Dropout/BN, cùng kiến trúc `(128,64)`) — Precision@3000 giữ nguyên 0,4190.
  Khác với một lần chạy trước đó (nơi Dropout/BN không tạo khác biệt), lần này regularization
  cho thấy lợi ích rõ, dù không lớn.

- **Kiến trúc nào tốt nhất?** `(128,64)` (PR-AUC 0,3724) > `(256,128,64)` (0,3705) >
  `(64,)` (0,3620). Đáng chú ý là kiến trúc sâu nhất **không** phải kiến trúc tốt nhất lần
  này — khác biệt so với chạy trước, cho thấy quan hệ giữa độ sâu và hiệu năng không đơn
  điệu ở quy mô dữ liệu này, dễ bị ảnh hưởng bởi khởi tạo ngẫu nhiên hơn là do bản chất kiến
  trúc quá nông hay quá sâu.

- **class_weight có cải thiện Precision@3000 không?** Lần chạy này thì **không** —
  trên kiến trúc `(128,64)`, bỏ class_weight cho cả PR-AUC cao hơn (0,3738 vs 0,3724) *và*
  Precision@3000 cao hơn (0,4227 vs 0,4190). Đây là kết quả ngược với suy luận lý thuyết
  thường gặp ("class_weight giúp lớp thiểu số"), và ngược cả với một lần chạy khác của
  chính dự án này — cho thấy tác dụng của class_weight trên bộ dữ liệu này không ổn định,
  nên cần chạy nhiều seed rồi lấy trung bình trước khi kết luận chắc chắn nên dùng hay không.

- **Embedding có thắng one-hot không?** Thắng, nhưng rất sít sao: PR-AUC 0,3744 so với
  0,3738 của MLP one-hot tốt nhất (không class_weight) — chênh lệch 0,0006, gần như trong
  sai số ngẫu nhiên. Điểm mạnh thực sự của Embedding ở đây là **hiệu quả tham số**: chỉ
  15.213 tham số nhưng đạt PR-AUC cao nhất toàn bộ thí nghiệm, cho thấy đáng cân nhắc khi
  cần một model nhẹ hơn, không hẳn vì nó vượt trội hẳn về độ chính xác.

- **MLP có thắng được LightGBM không?** Model tốt nhất (MLP + Embedding, PR-AUC 0,3744)
  chỉ cao hơn LightGBM (0,3675) đúng **0,0069** — dưới ngưỡng 0,01 mà nhóm coi là "không
  đáng kể". LightGBM train trong 2,9 giây và cho kết quả ổn định, có thể tái lập; trong khi
  để tìm ra MLP tốt nhất phải chạy 7 model Keras (tổng ~350 giây CPU) và kết quả từng model
  còn dao động giữa các lần chạy (xem lưu ý ở mục 6).

  **→ Kết luận trung thực: MLP không thắng LightGBM một cách có ý nghĩa thực tế, và ngay cả
  nội bộ các thử nghiệm MLP với nhau cũng chưa cho kết luận ổn định về kiến trúc hay
  class_weight tốt nhất.** Với dữ liệu dạng bảng ở quy mô này, LightGBM vẫn là lựa chọn
  thực dụng hơn nhờ vừa nhanh vừa ổn định giữa các lần chạy. Nếu tiếp tục theo hướng MLP,
  bước hợp lý tiếp theo là cố định `random seed` và chạy lại mỗi cấu hình 3-5 lần để lấy
  trung bình ± độ lệch chuẩn, thay vì kết luận dựa trên 1 lần chạy duy nhất.

---

## 8. Cạm bẫy đã tránh

| Cạm bẫy | Cách xử lý trong dự án |
|---|---|
| Không có baseline cây | LightGBM luôn chạy trước tiên (bước 4) |
| Dùng `accuracy` làm metric | Dùng PR-AUC + Precision@3000 |
| Quên chuẩn hoá đầu vào | `StandardScaler` fit trên train, áp lại cho val/test |
| Không `EarlyStopping` | Có sẵn trong `get_callbacks()`, áp dụng cho mọi model Keras |
| One-hot 155 mức `Policy_Sales_Channel` | Có luồng Embedding riêng để so sánh |
| `softmax` + 2 neuron cho bài toán nhị phân | Dùng `sigmoid` 1 neuron |

---

## 9. Cấu trúc dự án

```
TT-25-MLPKeras-<HoTen>/
├── README.md                             
├── requirements.txt
├── data/
│   └── train.csv                         
├── notebooks/
│   └── mlp_keras_insurance.ipynb         
├── src/
│   ├── data.py                            ← tiền xử lý, chia tập, class_weight
│   ├── model.py                           ← kiến trúc MLP (cơ bản/regularized/embedding), callbacks
│   └── train.py                           ← chạy toàn bộ pipeline bước 4→12 tự động
├── models/
│   └── best.keras                         ← model tốt nhất, tự lưu bởi ModelCheckpoint
└── reports/
    ├── learning_curves.png
    ├── kien_truc_comparison.png
    ├── embedding_vs_onehot.png
    ├── pr_curve.png
    └── ket_qua_so_sanh.csv                ← bảng kết quả đầy đủ, xuất tự động
```

---

## 10. Cách chạy

```bash
pip install -r requirements.txt
cd src
python train.py --data ../data/train.csv
jupyter notebook ../notebooks/mlp_keras_insurance.ipynb
```

`train.py` in kết quả ra console theo từng bước, đồng thời lưu:
- `models/*.keras` — checkpoint tốt nhất của mỗi kiến trúc (theo `val_pr_auc`)
- `reports/*.png` — learning curve, so sánh kiến trúc, PR-curve
- `reports/ket_qua_so_sanh.csv` và `reports/ket_luan.json` — bảng kết quả đầy đủ

---