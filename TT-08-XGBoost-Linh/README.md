# TT-08 — XGBoost: Phát hiện gian lận thẻ tín dụng theo thời gian thực

## 1. Bài toán & dữ liệu

- Dataset: [Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 284.807 giao dịch, 492 gian lận (**0,172%** — lệch cực đoan)
- V1–V28: đã PCA hoá, `Amount`: chưa scale, `Time`: giây kể từ giao dịch đầu tiên

## 2. Pipeline

1. `Time` → `Hour` (giờ trong ngày); `Amount` → `log1p` + `StandardScaler`
2. EDA: tỉ lệ gian lận theo giờ, phân phối Amount theo lớp (`reports/eda_overview.png`)
3. **Chia dữ liệu theo thời gian** (70/15/15, không shuffle) — *lý do*: gian lận xảy ra
   theo đợt; chia ngẫu nhiên khiến model "nhìn thấy tương lai" → điểm ảo, không phản ánh
   production (model luôn chỉ có dữ liệu quá khứ khi dự đoán).
4. Baseline → Logistic Regression (`class_weight=balanced`) → XGBoost
   (`scale_pos_weight`, `eval_metric=aucpr`, early stopping) → so sánh Random Forest, LightGBM

## 3. VÌ SAO ROC-AUC ĐÁNH LỪA

Ở mức lệch 0,172%, số True Negative (giao dịch thật, đoán đúng) áp đảo tuyệt đối.
ROC-AUC dùng False Positive Rate = FP / (FP + TN) — mẫu số quá lớn nên FPR luôn nhỏ,
kể cả khi model bắt được rất ít gian lận. Kết quả: ROC-AUC ~0,97 **kể cả với model kém**.

PR-AUC (Average Precision) chỉ đánh giá trên lớp dương (gian lận): Precision = TP/(TP+FP).
Khi lớp dương hiếm, FP ảnh hưởng trực tiếp và rõ rệt tới Precision → PR-AUC phản ánh
đúng năng lực bắt gian lận của model.

| Metric | Giá trị (điền sau khi chạy) | Ý nghĩa |
|---|---|---|
| ROC-AUC | 0.9831| 	Đẹp giả tạo do dữ liệu lệch |
| PR-AUC |  0.7571 | Con số thật, dùng để so sánh model|

Xem `reports/pr_vs_roc.png`.

## 4. Ngưỡng phân loại

- Ngưỡng đạt Precision ≥ 0,90: 0.9902864098548889 (Recall tương ứng: 0.712)
- Ngưỡng tối ưu lợi nhuận (chặn nhầm = 200.000đ, bỏ lọt = số tiền giao dịch): 0.99
  → xem `reports/chi_phi_theo_nguong.png`

## 5. Hiệu năng thời gian thực

- Độ trễ dự đoán 1 giao dịch: p50 = 1.43 ms | p99 = 2.30 ms (yêu cầu < 100ms -> ĐẠT)

## 6. Hạn chế

- V1–V28 đã qua PCA → **không thể diễn giải theo nghiệp vụ** (không biết cột nào là
  "số tiền bất thường", "vị trí lạ"...). `feature_importance.png` chỉ cho biết cột nào
  quan trọng về mặt thống kê, không giải thích được **vì sao**.
- Dataset chỉ gồm 2 ngày giao dịch → chưa chắc đại diện cho mùa vụ / hành vi dài hạn.

## 7. Theo dõi Concept Drift

Kẻ gian liên tục đổi chiêu thức, nên phân phối dữ liệu gian lận sẽ trôi (drift) theo
thời gian. Đề xuất:
1. Log lại `proba` dự đoán và nhãn thật (khi có phản hồi từ ngân hàng) mỗi ngày.
2. Theo dõi PR-AUC trên dữ liệu mới theo tuần; cảnh báo khi giảm > X% so với baseline.
3. Theo dõi phân phối input (PSI/KS-test trên V1–V28, Amount, Hour) để phát hiện
   dịch chuyển phân phối trước khi hiệu năng model giảm rõ rệt.
4. Lên lịch **re-train định kỳ** (vd. hàng tháng) hoặc khi drift alert kích hoạt.
5. (Mở rộng) So sánh với Isolation Forest — vì nó không cần nhãn, có thể dùng làm
   tín hiệu cảnh báo sớm khi pattern gian lận thay đổi mà XGBoost (học từ nhãn cũ)
   chưa kịp thích nghi.

## 8. Cấu trúc thư mục

```
TT-08-XGBoost-<HoTen>/
├── README.md
├── notebooks/xgboost_fraud.ipynb
├── src/train.py
├── models/xgb_fraud.json
├── reports/{eda_overview.png, pr_vs_roc.png, chi_phi_theo_nguong.png, feature_importance.png}
└── requirements.txt
```

## 9. Cách chạy

```bash
pip install -r requirements.txt
python src/train.py --data dataset/creditcard.csv
```