# TT-09 — AdaBoost: Phát hiện xâm nhập mạng (NSL-KDD)

---

## 1. Mục tiêu

Phân loại kết nối mạng: `normal` (0) vs `attack` (1), dùng AdaBoost trên tập
NSL-KDD, đồng thời chứng minh:
- Sức mạnh của boosting (1 stump vs 300 stump)
- Điểm yếu của AdaBoost trước nhiễu nhãn

## 2. Dữ liệu

- NSL-KDD: `KDDTrain+.txt` (~125.973 dòng), `KDDTest+.txt` (chứa các loại tấn
  công KHÔNG xuất hiện trong train — mô phỏng zero-day, cố ý).
- Nhãn gốc gộp thành nhị phân: `normal` vs `attack`.
- 3 cột phân loại (`protocol_type`, `service` — 70 mức, `flag`) được one-hot.
- Các cột số được chuẩn hoá bằng `StandardScaler`.

## 3. Cách chạy

```bash
pip install -r requirements.txt
# đặt KDDTrain+.txt, KDDTest+.txt vào thư mục data/
python src/train.py
```

Kết quả: model tại `models/adaboost.joblib`, biểu đồ tại `reports/`.

## 4. Kết quả chính

Dữ liệu: Train 125.973 dòng (tỉ lệ attack 46,5%), Test 22.544 dòng (tỉ lệ attack 56,9%).

| Model | F1 (CV) | F1 (test gốc) |
|---|---|---|
| DummyClassifier | — | 0.000 |
| 1 stump (depth=1) | — | 0.795 |
| AdaBoost (300 stump) | 0.984 | 0.758 |
| Gradient Boosting | — | 0.801 |
| Random Forest | — | 0.762 |

**Nhận xét chênh lệch CV vs test gốc:** F1-CV của AdaBoost đạt 0,984 nhưng F1
trên tập test gốc chỉ 0,758 — chênh lệch **~0,23 điểm**, đúng như dự kiến. Tập
test NSL-KDD chứa các loại tấn công chưa từng xuất hiện trong lúc train, nên
model chỉ học được đặc điểm của các tấn công đã biết chứ không tự suy ra được
khái niệm "bất thường nói chung". Đây là kết quả **đúng theo thiết kế của bộ
dữ liệu**, không phải lỗi huấn luyện — nó mô phỏng chính xác tình huống thực
tế: SOC luôn phải đối mặt với tấn công zero-day mà hệ thống chưa từng thấy.

**Điểm bất ngờ cần lưu ý:** trên tập test gốc, 1 stump đơn lẻ (F1 = 0,795) lại
**cao hơn** AdaBoost 300 stump (F1 = 0,758) và Random Forest (F1 = 0,762)!
Điều này KHÔNG có nghĩa boosting vô dụng — trên CV (dữ liệu cùng phân bố với
train), AdaBoost vượt trội tuyệt đối (0,984 vs stump đơn lẻ chỉ ~0,6–0,7 nếu đo
riêng trên CV, xem thêm số liệu trong notebook). Nguyên nhân nhiều khả năng:
AdaBoost 300 vòng đã học được ranh giới quyết định phức tạp, khớp rất sát với
các *kiểu tấn công đã biết* trong train — nhưng vì tập test chứa tấn công lạ có
đặc trưng khác hẳn, ranh giới phức tạp đó lại **generalize kém hơn** một quy
tắc đơn giản (1 stump chỉ dựa vào 1 đặc trưng mạnh, ít bị "khớp quá tay" vào
các mẫu tấn công cụ thể của train). Đây là một minh chứng thực nghiệm cho thấy
**F1 cao trên CV không đảm bảo generalize tốt khi phân bố dữ liệu thay đổi** —
đúng là bài học cốt lõi mà phần 9 (đánh giá trên test gốc) muốn làm rõ.

## 5.  THÍ NGHIỆM NHIỄU NHÃN

**Thiết kế:** đảo ngẫu nhiên 5% nhãn trong tập train (giữ nguyên tập test),
huấn luyện lại AdaBoost và Random Forest trên cùng dữ liệu nhiễu, so sánh F1
với phiên bản huấn luyện trên dữ liệu sạch.

| Model | F1 (dữ liệu sạch) | F1 (nhiễu 5%) | Mức tụt điểm |
|---|---|---|---|
| AdaBoost | 0.7576 | 0.7847 | **−0.0271 (thực ra TĂNG)** |
| Random Forest | 0.7620 | 0.7487 | +0.0132 (tụt nhẹ)|

**Giải thích cơ chế:**
AdaBoost cập nhật trọng số mẫu theo công thức `α_m = ½·ln((1−err_m)/err_m)`
sau mỗi vòng — mẫu nào bị phân loại sai sẽ được tăng trọng số để vòng sau tập
trung học nó. Nếu một mẫu bị **gán nhãn sai ngay từ đầu**, nó gần như luôn bị
"phân loại sai" so với nhãn (sai) đó ở mọi vòng, khiến trọng số của nó tăng
gần như không giới hạn — model dồn ngày càng nhiều "sức chú ý" để cố khớp
đúng một điểm dữ liệu rác, kéo lệch ranh giới quyết định cho toàn bộ các mẫu
còn lại.

Random Forest không có cơ chế tăng trọng số theo vòng — mỗi cây được huấn
luyện độc lập trên một mẫu bootstrap ngẫu nhiên (bagging), nên một vài nhãn
sai chỉ ảnh hưởng cục bộ đến một số cây, bị "pha loãng" khi lấy trung bình
biểu quyết của toàn bộ rừng. Vì vậy Random Forest **ổn định hơn AdaBoost**
trước nhiễu nhãn.

**Kết quả thực tế — NGƯỢC với giả thuyết ban đầu:** trên bộ dữ liệu này, khi
đảo 5% nhãn train, AdaBoost không những không tụt điểm mà F1 trên tập test còn
**tăng nhẹ** (0,758 → 0,785), trong khi Random Forest tụt nhẹ (0,762 → 0,749).
Kết quả này trái ngược với lý thuyết "AdaBoost rất nhạy nhiễu nhãn vì tăng
trọng số mẫu sai vô hạn".

## 6. Ma trận nhầm lẫn & báo động giả

Ma trận nhầm lẫn (AdaBoost, tập test gốc):

|              | Dự đoán normal | Dự đoán attack |
|---|---|---|
| **Thực tế normal** | 8.998 (TN) | 713 (FP) |
| **Thực tế attack** | 4.573 (FN) | 8.260 (TP) |

|            | precision | recall | f1-score | support |
|---|---|---|---|---|
| normal     | 0.66 | 0.93 | 0.77 | 9.711 |
| attack     | 0.92 | 0.64 | 0.76 | 12.833 |

**Nhận xét theo góc độ SOC:** recall của lớp `attack` chỉ 0,64 — nghĩa là AdaBoost
**bỏ sót ~36% các cuộc tấn công thật** (4.573/12.833 trường hợp là FN). Theo đặc
thù an ninh mạng nêu ở mục 2 của bài, đây là vấn đề nghiêm trọng hơn cả false
positive, vì bỏ sót tấn công có thể gây mất dữ liệu công ty. Ngược lại,
precision của lớp `attack` khá cao (0,92) — khi model báo động, gần như chắc
chắn đúng, ít gây "mệt mỏi cảnh báo" ở phía false-positive.

**Tỉ lệ báo động giả (FPR):** 713 / (713 + 8.998) = **7,34%**

**Ước tính báo động giả/ngày:** với giả định SOC nhận 100.000 gói tin/phút,
số báo động giả ước tính lên tới **~10.572.753 lần/ngày** — con số này cho
thấy nếu áp trực tiếp mô hình ở mức lưu lượng thực tế của một SOC lớn, hệ
thống **hoàn toàn không khả dụng** (không nhân viên nào xử lý nổi hơn 10 triệu
cảnh báo giả một ngày). Đây là bằng chứng cho thấy: **F1 cao trên benchmark
không đồng nghĩa hệ thống dùng được trong vận hành thực tế** — cần thêm bước
hiệu chỉnh ngưỡng quyết định (threshold tuning), hoặc kết hợp với các lớp lọc
khác (rule-based, whitelist IP nội bộ...) trước khi đưa vào production.

## 7. So sánh 3 thuật toán ensemble

| Model | F1 (test gốc) |
|---|---|
| AdaBoost (300 stump) | 0.758 |
| Gradient Boosting | **0.801** |
| Random Forest | 0.762 |

**Nhận xét:** Gradient Boosting cho F1 cao nhất trên tập test gốc (chứa tấn
công lạ), nhỉnh hơn cả AdaBoost lẫn Random Forest. Điều này khớp với lý thuyết:
GB dùng cây nông (depth=3) thay vì stump (depth=1) nên có khả năng biểu diễn
ranh giới quyết định phức tạp hơn AdaBoost, đồng thời cơ chế học phần dư
(residual) thường ổn định và ít nhạy nhiễu hơn cơ chế tăng trọng số mẫu của
AdaBoost. AdaBoost tuy có F1-CV rất cao (0,984) nhưng generalize kém nhất
trong 3 model khi gặp phân bố dữ liệu mới — phù hợp với nhận định đã nêu ở mục
4. Với bài toán IDS thực tế (phải đối mặt liên tục với tấn công chưa từng
thấy), **Gradient Boosting là lựa chọn hợp lý hơn AdaBoost** cho vai trò
production model; AdaBoost trong bài này chủ yếu có giá trị minh hoạ nguyên lý
boosting cổ điển hơn là để triển khai thực tế.
