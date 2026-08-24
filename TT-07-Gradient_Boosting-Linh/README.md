# TT-07 — Gradient Boosting: Chấm điểm khả năng tài chính cho hồ sơ vay tiêu dùng

Dự đoán thu nhập (`>50K` / `<=50K`) từ bộ dữ liệu **Adult Census Income**, dùng làm
một đầu vào cho mô hình chấm điểm tín dụng khi khách không khai hoặc khai sai thu nhập.

## Cấu trúc thư mục

```
├── README.md
├── requirements.txt
├── train.py                                  # script chính, chạy toàn bộ pipeline
├── dataset/                                   # dữ liệu gốc tải từ UCI (không sửa tên file)
│   ├── adult.data
│   ├── adult.test
│   ├── adult.names          (chỉ để tham khảo mô tả cột, code không dùng tới)
│   ├── Index                 (chỉ để tham khảo, code không dùng tới)
│   └── old.adult.names       (chỉ để tham khảo, code không dùng tới)
├── notebooks/
│   └── gradient_boosting_income.ipynb        # notebook giải thích từng bước
├── models/
│   └── gb_pipeline.joblib                     # sinh ra sau khi chạy train.py
└── reports/                                    # sinh ra sau khi chạy train.py
    ├── loss_theo_so_cay.png
    ├── lr_vs_nestimators.png
    ├── bias_by_group.png
    ├── bias_report.json
    └── model_comparison.csv
```

## Cách chạy

```bash
python train.py
```

Mặc định `train.py` đọc trực tiếp `dataset/adult.data` (train) và `dataset/adult.test`
(test) — **không cần gộp thành 1 file csv**, dùng đúng cách chia train/test mà UCI đã
định sẵn. 

```bash
python train.py --data-dir ten_thu_muc_cua_ban
```

Tải dữ liệu tại: https://archive.ics.uci.edu/dataset/2/adult

Notebook đầy đủ (EDA + giải thích từng bước): `notebooks/gradient_boosting_income.ipynb`
— mở bằng `jupyter notebook`, sửa biến `DATA_DIR` ở cell đầu nếu cần.

### Kiểm tra đọc đủ dữ liệu

Ngay khi chạy, `train.py` in ra tổng số dòng đọc được và so với con số kỳ vọng
(**48.842 dòng, 14 cột đặc trưng**). Nếu báo "KHÔNG KHỚP", kiểm tra lại file
`adult.data`/`adult.test` đã tải đủ và đúng định dạng chưa.

## BAGGING vs BOOSTING

| | Bagging (Random Forest) | Boosting (Gradient Boosting — bài này) |
|---|---|---|
| Cách học | Nhiều cây học **song song, độc lập**, rồi bỏ phiếu | Cây sau học để sửa lỗi (residual) của cây trước, **tuần tự** |
| Mục tiêu giảm | Variance | Bias |
| Độ sâu cây | Sâu, mỗi cây là "học viên mạnh" | **Nông** (`max_depth=3`), mỗi cây là "học viên yếu" |
| Rủi ro chính | Ít overfit hơn nếu đủ cây | Dễ overfit nếu `max_depth` lớn hoặc quá nhiều cây với `learning_rate` lớn |
| Siêu tham số quan trọng nhất | `n_estimators`, `max_depth` | `learning_rate` × `n_estimators` (quan hệ nghịch) |

Vì cơ chế ngược nhau, dùng cây sâu cho boosting sẽ khiến mô hình tổng overfit gần như
ngay lập tức (cây đầu đã "học thuộc" residual quá tốt); ngược lại Random Forest cần
cây đủ mạnh (sâu) để mỗi lá phiếu có ý nghĩa khi bỏ phiếu đa số.

## THIÊN LỆCH ⚖️

Bộ dữ liệu Adult Census Income lấy từ điều tra dân số Mỹ năm 1994, phản ánh **định
kiến lịch sử thật** về giới tính và chủng tộc trong thu nhập/nghề nghiệp thời điểm đó.
`train.py` (hàm `check_bias`) đo:

- Tỉ lệ dự đoán `>50K` theo `sex` và `race`
- So sánh với tỉ lệ thực tế trong tập test
- Xuất biểu đồ `reports/bias_by_group.png` và số liệu chi tiết `reports/bias_report.json`

Việc bỏ cột `sex`/`race` khỏi đặc trưng đầu vào **không** loại bỏ hoàn toàn thiên lệch,
vì thông tin vẫn "rò rỉ" qua các biến tương quan (`occupation`, `relationship`,
`hours-per-week`...) — đáng thử nghiệm thêm ở phần mở rộng.

**Mô hình này chỉ dùng cho mục đích học tập, tuyệt đối không dùng để ra quyết định
thật về con người.**

## Cạm bẫy đã xử lý

- `' ?'` (có dấu cách) được nhận diện qua `na_values="?"` kết hợp `sep=r",\s*"` khi đọc file
- Mọi cột chuỗi được `.str.strip()` lại lần nữa trong `clean_raw` để chắc chắn không sót dấu cách
- Bỏ `education` (giữ `education-num` — cùng thông tin, dạng số dễ dùng hơn)
- Bỏ `fnlwgt` (trọng số điều tra dân số, không liên quan cá nhân)
- Không scale biến số (cây quyết định/boosting không cần)
- Nhãn `income` trong `adult.test` có thêm dấu `.` cuối chuỗi (`>50K.`) — đã chuẩn hoá
  cho khớp với `adult.data`
- `adult.test` có 1 dòng chú thích ở đầu file, không phải dữ liệu — đã `skiprows=1`