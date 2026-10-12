# KT-01 — MLP PyTorch dự đoán khách hàng rời bỏ 
## 1. Mục tiêu

Dự đoán khách hàng viễn thông có rời bỏ hay không bằng MLP viết bằng PyTorch, trọng tâm là **tự viết vòng lặp huấn luyện**. Bài có hai phần chính: làm **2 thí nghiệm cố tình sai** để thấy hậu quả, và **so sánh MLP với XGBoost** trên cùng dữ liệu.

## 2. Dữ liệu và mô hình

- **Dữ liệu:** Telco Customer Churn (Kaggle), 7.043 dòng, khoảng 26,5% khách rời bỏ. Chia 60/20/20 theo tỉ lệ nhãn; bộ tiền xử lý chỉ học trên tập train để tránh rò rỉ.
- **Mô hình:** MLP 46 → 128 → 64 → 1 (BatchNorm, ReLU, Dropout), đầu ra là logit, không có sigmoid.
- **Huấn luyện:** `BCEWithLogitsLoss` với `pos_weight` để bù lệch lớp, AdamW, batch 32, early stopping và lưu checkpoint tốt nhất.
- **Chỉ số:** PR-AUC, vì nhãn lệch nên phù hợp hơn Accuracy.
- Phải dùng `drop_last=True` vì 4.225 mẫu train chia 32 dư 1, batch cuối chỉ có 1 mẫu sẽ làm `BatchNorm` báo lỗi.

Epoch tốt nhất là epoch 1: sau đó train loss vẫn giảm nhưng val không cải thiện, tức mô hình overfit sớm. Early stopping giữ lại bản tốt nhất.

## 3. Thí nghiệm sai thứ nhất: bỏ `optimizer.zero_grad()`

PyTorch cộng dồn gradient vào `.grad` sau mỗi lần `backward()`. Nếu không xoá, mỗi bước cập nhật dùng tổng gradient của các batch trước nên đi sai hướng.

Khi chạy, loss của run chuẩn giảm đều từ khoảng 0,78 xuống 0,66, còn run bỏ `zero_grad` không giảm mà dao động quanh 0,75–0,79. Điều đáng chú ý là PR-AUC của hai run lại gần như bằng nhau (khoảng 0,63). Lý do là PR-AUC chỉ đo thứ hạng xác suất nên không phản ánh việc tối ưu bị hỏng, và Adam tự chuẩn hoá độ lớn bước cập nhật nên hậu quả được làm dịu. Vì vậy cần theo dõi **loss**, đừng chỉ nhìn một metric.

## 4. Thí nghiệm sai thứ hai: bỏ `model.eval()` khi đánh giá

Ở chế độ train, Dropout vẫn tắt nơ-ron ngẫu nhiên và BatchNorm dùng thống kê của từng batch, nên cùng một model có thể cho điểm khác nhau.

Mình đánh giá cùng một model 10 lần liên tiếp. Có `eval()` thì cả 10 lần đều ra đúng **0,6331**. Bỏ `eval()` thì điểm dao động từ **0,6008 đến 0,6355** và trung bình thấp hơn đúng khoảng 0,014. Mức dao động này lớn hơn mọi chênh lệch cần so sánh trong bài (giữa các optimizer hay giữa MLP và XGBoost), nên quên `eval()` sẽ làm mọi kết luận mất giá trị.

## 5. So sánh với XGBoost

| Mô hình | Test PR-AUC | Thời gian train |
|---|---|---|
| MLP (PyTorch) | 0,637 | 1,83 s |
| XGBoost | **0,647** | **0,17 s** |

MLP **không thắng** XGBoost nhưng chỉ kém khoảng 0,011, nằm trong ngưỡng 0,03 của đề bài. XGBoost còn nhanh hơn khoảng 11 lần và ít công sức hơn nhiều (một lệnh `fit()`, trong khi MLP phải tự viết Dataset, model, vòng lặp và early stopping). Lý do là dữ liệu bảng nhỏ phù hợp với cây quyết định, còn lợi thế của deep learning nằm ở ảnh, văn bản và âm thanh. Với dữ liệu bảng, nên thử XGBoost trước.

Ba optimizer SGD, Adam, AdamW cho kết quả ngang nhau (khoảng 0,641 trên tập val, chênh lệch nhỏ hơn độ lệch chuẩn giữa các seed).

## 7. Cấu trúc và cách chạy

```
├── notebooks/mlp_pytorch_churn.ipynb
├── src/{dataset.py, model.py, train.py}
├── models/best.pt
├── reports/{learning_curve.png, optimizer_comparison.png, vs_xgboost.csv}
├── data/         
└── requirements.txt
```

```bash
pip install -r requirements.txt
python src/train.py        # hoặc mở notebook và chạy Run All
```