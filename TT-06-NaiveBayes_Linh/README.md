# TT-06 — Naive Bayes: Lọc tin nhắn rác cho tổng đài viễn thông

Bộ lọc SMS spam dùng Naive Bayes, tối ưu cho **precision cao** (chặn nhầm tin thật,
ví dụ OTP ngân hàng, tốn kém hơn nhiều so với lọt 1 tin rác).

## Cấu trúc dự án

```
TT-06-NaiveBayes/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── naive_bayes_sms.ipynb   
├── src/
│   └── train.py                
├── dataset/
│   └── spam.csv                
├── models/
│   ├── nb_pipeline.joblib       
│   └── threshold.json          
└── reports/
    ├── do_dai_tin.png
    ├── top_tu_spam.png
    └── confusion_matrix.png
```

## Cài đặt

```bash
python -m venv .venv
source .venv/bin/activate      
pip install -r requirements.txt
```

## Chạy

```bash
# Chạy notebook (khám phá đầy đủ)
jupyter notebook notebooks/naive_bayes_sms.ipynb

# Hoặc chạy script production
python src/train.py
python src/train.py --data data/spam.csv --target-precision 0.98
```

## Dữ liệu

| | |
|---|---|
| Tổng số tin | 5.572 (5.169 sau khi loại trùng lặp — **403 dòng trùng**, 7,2%) |
| Nhãn | `ham` (4.825 ≈ 87%) / `spam` (747 ≈ 13%) |
| Encoding | `latin-1` (không phải utf-8) |

## Kết quả

### So sánh 3 tổ hợp vectorizer × Naive Bayes

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Count + MultinomialNB | 0.9756 | 0.9160 | 0.9449 |
| TFIDF + BernoulliNB | 0.9907 | 0.8168 | 0.8954 |
| TFIDF + MultinomialNB (mặc định, chưa tune) | 1.0000 | 0.6336 | 0.7757 |


### Model tốt nhất sau khi tune

- **Cấu hình:** `TfidfVectorizer(ngram_range=(1,1))` + `MultinomialNB(alpha=0.01)`
- **CV f1_weighted:** 0.9847

| Ngưỡng | Precision (spam) | Recall (spam) |
|---|---|---|
| 0.5 (mặc định) | 0.9746 | 0.8779 |
| **0.571 (đã chỉnh để đạt mục tiêu)** | **0.9829** | **0.8779** |

→ Đạt mục tiêu **precision ≥ 0,98** với recall giữ nguyên ở 0,878.

### Top từ/cụm từ đặc trưng nhất cho SPAM

(xếp theo `log P(từ|spam) − log P(từ|ham)`, xem `reports/top_tu_spam.png`)

`claim`, `prize`, `150p`, `nokia`, `co`, `18`, `guaranteed`, `1000`, `500`, `16`,
`tone`, `000`, `cs`, `ringtone`, `awarded`, `150ppm`, `attempt`, `http`, `10p`, `tones`

→ Toàn từ liên quan đến quảng cáo/trúng thưởng/dịch vụ tính phí cao (150p, 150ppm là
cước SMS premium ở Anh) — hợp lý về mặt ngữ nghĩa, không phải nhiễu.

### So sánh với Logistic Regression

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Naive Bayes (threshold-tuned) | 0.9829 | 0.8779 | 0.9274 |
| Logistic Regression + TFIDF | 0.9894 | 0.7099 | 0.8267 |

→ Logistic Regression có precision nhỉnh hơn nhưng **recall thấp hơn đáng kể**
(lọt nhiều spam hơn). Naive Bayes cho trade-off tốt hơn ở bài toán này, đồng thời train
nhanh hơn nhiều (≈49ms vs ≈124ms).

### Hiệu năng

| Chỉ số | Giá trị | Yêu cầu |
|---|---|---|
| Thời gian train (4.135 tin) | ≈ 49 ms | — |
| Thời gian dự đoán 1 tin | ≈ 0,36–0,41 ms | < 5 ms Đạt |

### Phân tích lỗi

18/1.034 tin bị phân loại sai (≈1,7%). Đa số lỗi là **false negative** (spam bị lọt),
thường là spam viết dạng "nói chuyện tự nhiên" (mời hẹn hò, không dùng từ khoá quảng cáo
điển hình) hoặc chứa ký tự đặc biệt/encoding lỗi (`å£` thay cho `£`) khiến tokenizer không
bắt được pattern quen thuộc. Chỉ có 1 false positive: tin ham ngắn, nhiều emoticon
(`K:)eng rocking in ashes:)`) bị hiểu nhầm do quá ít ngữ cảnh.

## Những điều cần nhớ khi làm lại / mở rộng

- **`alpha = 0` tuyệt đối không dùng** — 1 từ lạ chưa từng thấy trong 1 lớp sẽ làm
  xác suất lớp đó = 0 do phép nhân, phá hỏng toàn bộ dự đoán.
- Luôn `drop_duplicates()` **trước** khi `train_test_split`.
- Đọc file bằng `encoding='latin-1'`, không phải utf-8.
- Fit vectorizer chỉ trên tập train (dùng `Pipeline` để tránh lỗi vô tình fit trên test).
- Accuracy không đáng tin với dữ liệu lệch lớp — baseline đoán "ham" hết đã đạt 87,3%.

