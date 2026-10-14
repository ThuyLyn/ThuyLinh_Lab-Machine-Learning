# KT-02 — BÊN TRONG MẠNG NƠ-RON
## Tự code backpropagation và so sánh 7 hàm kích hoạt

## 0. Đọc trước — những giới hạn của bản nộp này

| Điều | Chi tiết |
|---|---|
| **Không dùng MNIST thật** | Môi trường làm bài không có mạng nên không tải được MNIST. Mọi thí nghiệm "MNIST" chạy trên **sklearn digits** (1.797 ảnh 8×8, 10 lớp; 1.347 train / 450 val). Kết luận định tính (ai tiêu biến, ai chết, khởi tạo nào học được) vẫn đúng, nhưng **con số tuyệt đối không so được với MNIST**. Đặt `USE_MNIST=1` để dùng MNIST thật qua `fetch_openml`, đường này **chưa được chạy thử**. |
| **Không dùng PyTorch** | Môi trường không cài PyTorch. Thay vì đo gradient bằng `named_parameters()`, mình tự cài mạng sâu tổng quát bằng NumPy (`src/nn_numpy.py`: 7 activation, 5 cách khởi tạo, BatchNorm) và đo gradient từ chính `backward()` đã được kiểm chứng. |
| **Mẫu nhỏ** | Phần lớn thí nghiệm dùng 3 seed. Chênh lệch dưới ~1 độ lệch chuẩn không đủ để kết luận ai hơn ai. |

## 1. Cấu trúc (đúng mục 8 của đề)

```
KT-02-NNInternals-HoTen/
├── README.md          
├── notebooks/{01_backprop_numpy.ipynb, 02_activations.ipynb, 03_initialization.ipynb}   
├── src/{nn_numpy.py, gradient_check.py}
├── reports/{7_activations.png, vanishing_gradient.png, dying_relu.csv, init_comparison.png}
└── requirements.txt
```
`src/nn_numpy.py` gồm: 7 activation + đạo hàm, SGD/Adam, mạng XOR 2→4→1 và 2→1, `DeepNet` (5 cách khởi tạo, BatchNorm cả forward lẫn backward), vòng train, đếm neuron chết, nạp dữ liệu.
Các hình/bảng khác (ranh giới XOR, đường loss, 7 activation trên digits, phân phối activation, BatchNorm, Adam, bề mặt loss...) nằm **ngay trong output của notebook**.

## 2. Kết quả chính

**Phần 1 (NumPy thuần).**
- Kiểm chứng gradient: sai lệch lớn nhất **5.4e-12** cho mạng XOR; **≤ 2.8e-10** cho mạng sâu với cả 7 activation, có và không có BatchNorm (ngưỡng 1e-7).
- Bộ kiểm chứng *bắt được lỗi thật*: cố tình quên chia N → sai lệch **1.0e-1** → báo SAI. Và `eps` quá lớn (1e-1) hay quá nhỏ (1e-10 → 2.4e-7, 1e-12 → 4.9e-5) cũng làm kiểm chứng báo SAI dù backprop đúng; `eps=1e-5` cho 5.4e-12.
- XOR: `lr=0.5`, khởi tạo nhỏ. Loss đứng ở **cao nguyên ≈0.25 trong ~1.590 epoch**, rơi xuống <0.1 ở epoch **2.399**, cuối đạt 7.1e-4. Dự đoán `[0.03, 0.975, 0.974, 0.026]` ≈ `[0,1,1,0]` → **100%**.
- Bỏ tầng ẩn: loss kẹt **0.2500**, mọi dự đoán = 0.5; vét cạn 360 hướng × 401 ngưỡng, đường thẳng tốt nhất chỉ đúng **75%**.

**Phần 2 — 7 activation** (64→128→64→10, Xavier cho cả 7, SGD lr 0.1, batch 64, 60 epoch, 3 seed, dữ liệu digits):

| Activation | val acc (%) | epoch tới ≥95% | loss cuối | thời gian (s) |
|---|---|---|---|---|
| Sigmoid | 88.52 ± 0.46 | **không đạt (0/3)** | 0.514 | 0.9 |
| Tanh | 97.33 ± 0.31 | 8.0 | 0.039 | 0.8 |
| ReLU | 97.41 ± 0.21 | 8.7 | 0.029 | 0.6 |
| LeakyReLU | 97.41 ± 0.21 | 8.7 | 0.030 | 1.0 |
| ELU | 97.11 ± 0.18 | 7.3 | 0.033 | 1.1 |
| GELU | 96.96 ± 0.28 | 9.7 | 0.032 | 1.5 |
| SiLU/Swish | 96.67 ± 0.63 | 10.7 | 0.041 | 1.0 |

Chỉ Sigmoid tụt rõ rệt. Sáu hàm còn lại chênh nhau trong khoảng ~1 độ lệch chuẩn, nên **bài này không đủ bằng chứng để xếp hạng chúng**, đặc biệt không kết luận được "GELU/SiLU tốt hơn" hay "ReLU luôn tốt nhất" (cạm bẫy số 5): lợi thế của GELU/SiLU được nói tới ở Transformer và CNN hiện đại, còn ở đây là mạng MLP nhỏ trên ảnh 8×8.

**Vanishing gradient** (mạng 15 tầng ẩn × 64 neuron, `reports/vanishing_gradient.png`):

| Cấu hình | ‖grad‖ tầng 1 | ‖grad‖ tầng cuối | tầng cuối / tầng 1 |
|---|---|---|---|
| Sigmoid (He) | 2.4e-8 | 0.95 | 4.0e+7 |
| Sigmoid (Xavier) | 3.5e-10 | 0.82 | 2.3e+9 |
| Tanh (Xavier) | 0.31 | 0.21 | 0.7 (**không** tiêu biến) |
| ReLU (He) | 1.7 | 2.1 | 1.2 |

Đường sigmoid bám sát lý thuyết `0.25^k` (`0.25¹⁰ = 9.5e-7`, khớp README). Lưu ý: bảng trong README gốc ghi Tanh "vẫn tiêu biến", nhưng **ở lúc khởi tạo với Xavier thì gradient tanh gần như không tiêu biến** (đạo hàm tối đa = 1 và Xavier giữ phương sai). Tanh chỉ tiêu biến khi bão hoà hoặc khởi tạo quá nhỏ (xem Phần 3).

**Dying ReLU** (`reports/dying_relu.csv`; mạng 64→128→128→64→10, He, 5 epoch ở lr lớn, 3 seed; rồi hạ lr về 0.05 train 30 epoch):

| lr | ReLU: % chết → sau hạ lr | ReLU val acc sau | LeakyReLU: % chết → sau hạ lr | LeakyReLU val acc sau |
|---|---|---|---|---|
| 0.05 | 6.0 → 5.6 | 96.7 | 5.7 → 5.6 | 96.6 |
| 0.5 | 2.1 → 2.8 | 97.1 | 2.8 → 3.4 | 97.2 |
| 1.0 | 42.1 → 39.6 | 91.9 | 42.7 → 44.2 (1/3 phân kỳ) | 84.0 |
| 2.0 | **87.6 → 87.6** | **10.2** | **phân kỳ NaN 3/3** | – |
| 5.0 | 87.3 → 87.3 | 10.1 | phân kỳ NaN 3/3 | – |

*LeakyReLU có cứu được không?* **Không, trong thí nghiệm này.** Neuron ReLU chết ở lr lớn **không hồi phục** khi hạ lr (87.6% → 87.6%, accuracy 10% = đoán bừa). Nhưng LeakyReLU không "thắng": ở lr=1.0 tỉ lệ chết tương đương và kết quả sau đó còn thấp hơn ReLU (84.0% so với 91.9%, chỉ 2 lần chạy hợp lệ), còn từ lr=2.0 nó **phân kỳ hoàn toàn** (không có gradient clipping). Độ dốc 0.01 ở nhánh âm giúp gradient khác 0 nhưng không bù được lr quá lớn. Với lr hợp lý (≤0.5) cả hai đều ổn.

**Phần 3 — khởi tạo** (6 tầng ẩn × 64 neuron, SGD lr 0.05, 40 epoch, 3 seed):

| Activation | Khởi tạo | loss epoch 1 | loss epoch 40 | val acc (%) |
|---|---|---|---|---|
| Tanh | zeros | 2.303 | 2.302 | 10.2 |
| Tanh | random nhỏ N(0,0.01²) | 2.303 | 2.302 | 10.2 |
| Tanh | Xavier | 1.489 | 0.043 | 96.7 |
| Tanh | He | 0.872 | 0.018 | 97.3 |
| ReLU | zeros | 2.303 | 2.302 | 10.2 |
| ReLU | random nhỏ | 2.303 | 2.302 | 10.2 |
| ReLU | Xavier | 2.277 | 0.029 | 96.2 |
| ReLU | He | 1.725 | 0.006 | 96.2 |

(`ln 10 = 2.303` là loss của mạng đoán đều 10 lớp, tức chưa học gì.) Random nhỏ **kẹt hẳn**, không chỉ chậm: train ReLU + random nhỏ tới 300 epoch loss vẫn 2.3025. Lý do (notebook 03, bước 12): độ lệch chuẩn activation ở tầng 10 chỉ còn **4.9e-9** (tanh) và **1.6e-10** (ReLU), tín hiệu tắt dần theo chiều sâu. Với ReLU, Xavier co dần (0.36 → 0.018 qua 10 tầng) còn He giữ ổn định (0.81 → 0.92), đúng như công thức `Var(W)=2/n_in`. Với tanh, He làm nhiều activation tầng đầu **bão hoà ở ±1** (notebook 03, bước 12), nên He không "luôn tốt hơn": nó được thiết kế cho ReLU. (Ở đây He+tanh vẫn học tốt; chỉ nêu cảnh báo về bão hoà.)

**Bằng chứng "đối xứng không phá vỡ được"** (bước 11, tanh 64→32→32→10, 20 epoch):
- `zeros`: toàn bộ trọng số tầng ẩn **vẫn đúng bằng 0**, loss 2.302 (gradient bằng 0 vì activation tầng trước bằng 0).
- `const` (mọi trọng số = 0.1, **khác 0**): sau train trọng số đã đổi (0.09994…) nhưng **5 neuron giống nhau đến từng chữ số**, `spread = 0.0e+00` ở cả 2 tầng ẩn; val acc 10%. 32 neuron chỉ hoạt động như 1.
- `he`: các neuron khác nhau (spread 1.3), val acc 96.0%.

**BatchNorm** (ReLU, 3 seed, val acc %):

| Khởi tạo | không BN | có BN |
|---|---|---|
| zeros | 10.2 | **10.2** |
| random nhỏ | 10.2 | **96.7** |
| Xavier | 96.2 | 97.7 |
| He | 96.2 | 97.0 |

Chênh lệch val acc giữa 3 cách khởi tạo ngẫu nhiên: **86.0 điểm** (không BN) → **1.0 điểm** (có BN). Vậy BatchNorm làm mạng **bớt nhạy hẳn** với *độ lớn* trọng số khởi tạo (cứu được random nhỏ), nhưng **không cứu được zeros** vì không phá được đối xứng.