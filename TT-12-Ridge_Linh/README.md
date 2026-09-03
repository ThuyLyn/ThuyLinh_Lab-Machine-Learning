# TT-12 — Ridge Regression (L2)
## Phân bổ ngân sách quảng cáo đa kênh khi các kênh chạy cùng lúc

## Cấu trúc thư mục

```
TT-12-Ridge/
├── README.md
├── notebooks/ridge_marketing.ipynb
├── src/train.py
├── models/ridge_pipeline.joblib
├── reports/
│   ├── vif_table.csv
│   ├── so_sanh_he_so.csv
│   ├── de_xuat_phan_bo_ngan_sach.csv
│   ├── bootstrap_he_so.png
│   ├── coefficient_path.png
│   └── rmse_theo_alpha.png
└── requirements.txt
```

## Cách chạy

```bash
pip install -r requirements.txt
python src/train.py
```

## Dữ liệu

Dữ liệu được tự sinh (500 dòng) với 3 kênh: TV, Facebook, Google. Facebook được sinh
có chủ đích tương quan cao với TV (r ≈ 0.98) để mô phỏng việc ngân sách các kênh
thường tăng/giảm cùng nhau trong thực tế — gây đa cộng tuyến.

## Kết quả chính

### Bước 2 — Ma trận tương quan & VIF

Ma trận tương quan:

|          | TV       | Facebook | Google   |
|----------|----------|----------|----------|
| TV       | 1.000000 | 0.980895 | 0.016020 |
| Facebook | 0.980895 | 1.000000 | 0.011397 |
| Google   | 0.016020 | 0.011397 | 1.000000 |

Bảng VIF:

| feature  | VIF       |
|----------|-----------|
| TV       | 26.439999 |
| Facebook | 26.436647 |
| Google   | 1.000750  |

→ TV và Facebook tương quan gần như tuyệt đối (r ≈ 0.98) và VIF >> 10 ở cả hai →
xác nhận đa cộng tuyến nghiêm trọng. Google gần như độc lập (VIF ≈ 1).

### Bước 3 — Linear Regression baseline

| feature  | Hệ số      |
|----------|------------|
| TV       | 406.919195 |
| Facebook | 140.989237 |
| Google   | 206.739992 |

### Bước 4 — ⭐ Bootstrap ổn định hệ số (100 lần)

| feature  | std_linear | std_ridge |
|----------|------------|-----------|
| TV       | 7.558479   | 4.100672  |
| Facebook | 5.657777   | 3.538353  |
| Google   | 2.324139   | 2.261789  |

→ Độ lệch chuẩn của Linear cao hơn Ridge rõ rệt ở TV và Facebook (hai biến tương
quan cao) — bằng chứng trực tiếp cho việc Ridge cho hệ số ổn định hơn.

### Bước 5 — RidgeCV dò alpha

**alpha tối ưu = 0.001**

### Bước 8 — So sánh RMSE trên tập test

| Model  | RMSE (test) |
|--------|-------------|
| Linear | 47.946      |
| Ridge  | 47.946      |

(Hai giá trị gần như bằng nhau vì alpha tối ưu do CV chọn ra rất nhỏ — nhiễu trong
dữ liệu mô phỏng thấp nên phạt L2 gần như không cần thiết để đạt RMSE tốt nhất;
lợi ích thực sự của Ridge ở đây nằm ở **độ ổn định hệ số**, không phải RMSE.)

### Bước 9 — So sánh hệ số Ridge vs Lasso vs ElasticNet

| feature  | Ridge      | Lasso      | ElasticNet |
|----------|------------|------------|------------|
| TV       | 406.902458 | 407.752136 | 220.387401 |
| Facebook | 141.005297 | 139.560412 | 209.578198 |
| Google   | 206.739713 | 206.115715 | 137.255987 |

### Bước 10 — ✍️ Đề xuất phân bổ ngân sách (dựa trên |hệ số Ridge|)

Trên tổng ngân sách 2 tỷ/tháng:

| Kênh     | Hệ số Ridge (chuẩn hoá) | Tỷ trọng | Ngân sách đề xuất |
|----------|-------------------------|----------|--------------------|
| TV       | 406.902458              | 53.92%   | 1,078,391,000 đ    |
| Facebook | 141.005297              | 18.68%   | 373,698,500 đ      |
| Google   | 206.739713              | 27.40%   | 547,910,700 đ      |

## VÌ SAO RIDGE ỔN ĐỊNH HƠN

Khi hai biến (TV và Facebook) tương quan rất cao, có vô số cách chia "công lao" giữa
chúng mà vẫn cho ra tổng dự đoán gần như giống nhau — ví dụ (TV=+500, Facebook=−480)
hay (TV=+10, Facebook=+10) đều có thể khớp dữ liệu train gần như tốt như nhau. Linear
Regression (OLS) không có cơ chế nào để chọn giữa các lời giải này, nên chỉ cần dữ
liệu train thay đổi nhẹ (như trong bootstrap), hệ số có thể nhảy rất mạnh — thể hiện
qua độ lệch chuẩn cao ở bảng trên.

Ridge thêm số hạng phạt `λ·Σwᵢ²` vào hàm mất mát. Vì phạt này tăng theo bình phương,
nó "phạt nặng" các lời giải có hệ số lớn trái dấu triệt tiêu nhau (ví dụ +500/−480 có
tổng bình phương rất lớn) hơn là lời giải chia đều ảnh hưởng cho cả hai biến tương
quan. Do đó Ridge có xu hướng **chia đều đóng góp** giữa các biến tương quan cao thay
vì để một biến "cướp" hết hoặc "bù trừ" cho biến kia một cách tuỳ tiện — kết quả là hệ
số ổn định hơn nhiều qua các lần lấy mẫu khác nhau, như thấy rõ trong thí nghiệm
bootstrap ở Bước 4.

Ridge **không bao giờ đưa hệ số về đúng 0** vì đạo hàm của phạt L2 (`2λw`) tiến về 0
khi w tiến về 0, nên "lực co" cũng yếu dần và không đủ để đẩy hệ số qua điểm 0 tuyệt
đối — khác với phạt L1 (Lasso) có đạo hàm không đổi (`±λ`), đủ mạnh để đưa hệ số về
chính xác 0.



