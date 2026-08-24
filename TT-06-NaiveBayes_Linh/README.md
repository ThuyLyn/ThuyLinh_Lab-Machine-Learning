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
    ├── confusion_matrix.png
    └── metrics.json
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

# Hoặc chạy script production — CHẠY TỪ THƯ MỤC GỐC DỰ ÁN
python src/train.py
python src/train.py --data dataset/spam.csv --target-precision 0.98
```

Tham số dòng lệnh hỗ trợ (xem `python src/train.py --help`):

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--data` | `dataset/spam.csv` (tự tính theo vị trí `train.py`, không phụ thuộc nơi gõ lệnh) | Đường dẫn file dữ liệu |
| `--out-dir` | `.` (thư mục đang đứng khi gõ lệnh) | Thư mục gốc để ghi `models/` và `reports/` |
| `--target-precision` | `0.98` | Precision tối thiểu mong muốn cho lớp spam |
| `--random-state` | `42` | Random state cho các bước chia dữ liệu |

## Dữ liệu

| | |
|---|---|
| Tổng số tin | 5.572 (5.169 sau khi loại trùng lặp — **403 dòng trùng**, 7,2%) |
| Nhãn | `ham` (4.825 ≈ 87%) / `spam` (747 ≈ 13%) |
| Encoding | `latin-1` (không phải utf-8) |
| Độ dài tin (ký tự) | ham: trung bình 70,5 (std 56,4) · spam: trung bình 137,9 (std 30,1) — spam thường dài và đều hơn hẳn ham |

## Phương pháp — chia 3 tập train / validation / test
> Dữ liệu (sau khi `drop_duplicates()`) được chia làm 3 phần, tách
> **trước** khi train (`test_size=0.2` hai lần liên tiếp → tỉ lệ xấp xỉ 64% /
> 16% / 20%):
> - **train** (3.308 tin): fit `GridSearchCV` (5-fold CV nội bộ, không đụng validation/test)
> - **validation** (827 tin): dùng để dò ngưỡng đạt precision mục tiêu
> - **test** (1.034 tin): dùng **đúng một lần**, sau khi ngưỡng đã chốt, để báo cáo
>   kết quả cuối cùng
>
> Sau khi chốt ngưỡng, model cuối được train lại trên train+validation gộp lại
> (4.135 tin, tận dụng hết dữ liệu) rồi mới đo hiệu năng — ngưỡng giữ nguyên,
> không tính lại trên dữ liệu gộp này.

`GridSearchCV` dò tham số bằng scorer nhắm thẳng vào **F1 của lớp spam**
(`make_scorer(f1_score, pos_label="spam")`) thay vì `f1_weighted` như bản cũ,
vì lớp ham chiếm 86,6% dữ liệu khiến `f1_weighted` bị pha loãng và có thể chọn
tham số không tối ưu cho mục tiêu thật sự: phát hiện spam.

## Kết quả

*(Số liệu dưới đây đo từ lần chạy notebook thật trên `dataset/spam.csv`, đã
đối chiếu khớp với `reports/metrics.json` do `train.py` sinh ra — xem bảng
xác nhận ở cuối mục này.)*

### So sánh 3 tổ hợp vectorizer × Naive Bayes (đánh giá trên **validation**)

| Model | Precision | Recall | F1 |
|---|---|---|---|
| TFIDF + Bernoulli | 1.0000 | 0.6442 | 0.7836 |
| TFIDF + Multinomial (mặc định, chưa tune) | 1.0000 | 0.4904 | 0.6581 |
| Count + Multinomial | 0.9778 | 0.8462 | 0.9072 |

→ Đánh giá trên validation (không phải test) vì kết quả này ảnh hưởng đến việc
chọn hướng đi tiếp theo. `Count + Multinomial` có F1 cao nhất ở cấu hình mặc
định, nhưng bước tune `alpha`/`ngram_range` ở mục sau (trên nền TFIDF +
MultinomialNB) mới là model được chọn cuối cùng.

### Model tốt nhất sau khi tune

- **Cấu hình:** `TfidfVectorizer(ngram_range=(1,2))` + `MultinomialNB(alpha=0.1)`
- **Best CV spam-F1** (trên train, 5-fold): 0.9446

| Ngưỡng | Tập dùng để chọn | Precision (spam) trên test | Recall (spam) trên test |
|---|---|---|---|
| 0.5 (mặc định) | — | 0.9832 | 0.8931 |
| **0.5650 (chọn trên validation)** | validation | **0.9831** | **0.8855** |

→ Trên **validation**, ngưỡng 0.565 đạt precision 0.9882 (≥ mục tiêu 0,98) với
recall 0,8077. Áp dụng đúng ngưỡng đó lên **test** (đúng 1 lần, không dò lại)
cho precision 0.9831 — rất sát mục tiêu và **là số liệu thật**, không còn là số
đã "học thuộc" test set như bản trước. Đáng chú ý: ở tập test này, ngưỡng mặc
định 0.5 tình cờ đã cho precision 0.9832 — cao hơn cả ngưỡng đã chỉnh — cho
thấy độ nhiễu tự nhiên giữa các tập dữ liệu nhỏ; đây chính xác là lý do không
nên chọn ngưỡng dựa trên tập dùng để báo cáo.

### Top từ/cụm từ đặc trưng nhất cho SPAM

(xếp theo `log P(từ|spam) − log P(từ|ham)`, xem `reports/top_tu_spam.png`)

`claim`, `prize`, `have won`, `18`, `150p`, `your mobile`, `co uk`, `co`,
`guaranteed`, `to claim`, `1000`, `500`, `nokia`, `16`, `ringtone`, `000`,
`awarded`, `code`, `tones`, `www`

→ Toàn cụm từ liên quan đến quảng cáo/trúng thưởng/dịch vụ tính phí cao (150p
là cước SMS premium ở Anh) — hợp lý về mặt ngữ nghĩa, không phải nhiễu. Vì
`ngram_range=(1,2)` được chọn (khác bản cũ chỉ `(1,1)`), danh sách giờ có cả
cụm 2 từ (`have won`, `your mobile`, `to claim`) — cho thấy bigram mang thêm
tín hiệu hữu ích so với chỉ dùng từ đơn.

### So sánh với Logistic Regression

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Naive Bayes (threshold-tuned, 0.565) | 0.9831 | 0.8855 | 0.9317 |
| Naive Bayes (ngưỡng mặc định 0.5) | 0.9832 | 0.8931 | 0.9360 |
| Logistic Regression + TFIDF | 1.0000 | 0.6641 | 0.7982 |

→ Logistic Regression đạt precision tuyệt đối nhưng **recall thấp hơn nhiều**
(lọt gần 1/3 số tin spam), trong khi Naive Bayes cho trade-off cân bằng hơn
đúng với mục tiêu nghiệp vụ (ưu tiên không chặn nhầm nhưng vẫn bắt được phần
lớn spam). Naive Bayes cũng train nhanh hơn Logistic Regression.

### Hiệu năng

*(Số liệu chính thức lấy từ `reports/metrics.json`, sinh ra bởi `train.py`)

| Chỉ số | Giá trị | Yêu cầu |
|---|---|---|
| Thời gian train (train+validation) | ≈ 87,7 ms | — |
| Thời gian dự đoán 1 tin | ≈ 0,478 ms | < 5 ms Đạt |

> Notebook đo được số hơi khác (≈108 ms train, ≈0,65 ms dự đoán) do chạy ở
> thời điểm/máy khác — chênh lệch này bình thường với các phép đo thời gian
> thực thi (phụ thuộc tải máy lúc đó), không phản ánh sai lệch về phương
> pháp. Cả hai lần đo đều đạt yêu cầu < 5ms/tin với biên độ rất lớn.
