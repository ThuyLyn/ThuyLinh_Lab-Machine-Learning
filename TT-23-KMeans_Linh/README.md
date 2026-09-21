# TT-23 — K-Means Clustering: Gom nhóm sản phẩm để sắp xếp lại kệ hàng siêu thị

## Tóm tắt

Gom **4.397 sản phẩm** (sau làm sạch, từ 4.732 mã hàng gốc) của bộ dữ liệu Online Retail II thành **5 cụm** theo hành vi bán hàng, dùng K-Means trên pipeline `log1p → StandardScaler → KMeans`. Kết quả: file CSV gán mỗi mã hàng vào 1 cụm kèm tên cụm tiếng Việt, sẵn sàng bàn giao cho bộ phận trưng bày.

## Quy trình

1. **Làm sạch giao dịch** (1.067.371 dòng gốc → 1.037.082 dòng): loại hoá đơn huỷ, Quantity/Price ≤ 0, các mã không phải sản phẩm (POST, M, BANK CHARGES, DOT, ADJUST, C2, D, S, AMAZONFEE, TEST001/2, PADS, CRUK), dòng thiếu mô tả.
2. **Tổng hợp cấp sản phẩm**: 8 đặc trưng hành vi (tổng số lượng, tổng doanh thu, giá TB, số đơn, số khách, độ lệch số lượng, số lượng/đơn, tỷ lệ mua lại). Loại sản phẩm có < 5 đơn hàng.
3. **Xử lý phân phối lệch**: `np.log1p()` trước `StandardScaler` — bắt buộc vì doanh thu/số lượng lệch phải cực nặng do vài "sản phẩm cá voi".
4. **Chọn K = 5**: căn cứ kỹ thuật (silhouette cao nhất trong khoảng kinh doanh chấp nhận được) + căn cứ kinh doanh (K=4–6 để bộ phận trưng bày quản lý nổi).
5. **Thí nghiệm n_init**: xác nhận `n_init=1` cho kết quả không ổn định giữa các seed; `n_init=10` ổn định — dùng `n_init=10` cho mô hình cuối.
6. **So sánh với DBSCAN**: DBSCAN đánh dấu ~11% sản phẩm là nhiễu (ngoại lệ), minh hoạ giả định "cụm hình cầu" của K-Means không hoàn toàn đúng với dữ liệu này.

Chi tiết đầy đủ, biểu đồ, và code nằm trong `notebooks/kmeans_san_pham.ipynb`.

## Bảng mô tả cụm & đề xuất trưng bày

| Cụm | Tên | Số mã hàng | % doanh thu | Đặc điểm | Đề xuất trưng bày |
|---|---|---|---|---|---|
| 3 | **Hàng chủ lực – bán chạy đều** | 1.237 | 76,6% | Giá thấp-vừa, số đơn/khách mua rất cao, đóng góp phần lớn doanh thu | Vị trí trung tâm, đầu lối đi chính, ngang tầm mắt |
| 0 | **Hàng giá trị cao – ổn định** | 1.144 | 17,0% | Giá trung bình cao nhất, doanh thu/mã hàng cao, bán đều đặn | Khu trưng bày nổi bật, đầu kệ, ánh sáng tốt |
| 1 | **Hàng giá rẻ – số lượng lớn** | 960 | 4,9% | Giá thấp nhất, số lượng bán/đơn cao, doanh thu/mã hàng thấp | Kệ thấp dễ lấy số lượng lớn, gần lối ra |
| 4 | **Hàng bán chậm – ít khách** | 798 | 1,0% | Số đơn và số khách thấp nhất, đóng góp doanh thu không đáng kể | Góc khuất kệ, cân nhắc thanh lý / khuyến mãi đẩy hàng |
| 2 | **Hàng ngách – khách quen mua lại** | 258 | 0,6% | Ít mã hàng nhất nhưng tỷ lệ mua lại cao vượt trội | Khu ưu đãi thành viên / gần quầy thu ngân |

## Ba giả định của K-Means — đối chiếu với bài toán

| Giả định | Có thoả mãn? |
|---|---|
| Cụm hình cầu, kích thước tương đương | Không hoàn toàn — cụm lớn nhất (1.237 mã) gấp ~5 lần cụm nhỏ nhất (258 mã) |
| Đặc trưng quan trọng ngang nhau | Có — đã xử lý bằng log1p + StandardScaler |
| Biết trước K | Chấp nhận được — K=5 chọn có căn cứ kỹ thuật + kinh doanh |

## Cấu trúc thư mục

```
TT-23-KMeans/
├── README.md
├── notebooks/kmeans_san_pham.ipynb
├── src/{features.py, cluster.py}    
├── outputs/san_pham_theo_cum.csv    
├── reports/{truoc_sau_log.png, elbow_silhouette.png, pca_scatter.png, mo_ta_cum.png}
└── requirements.txt
```
