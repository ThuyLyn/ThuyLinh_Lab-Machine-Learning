# Báo cáo kết quả (sinh tự động từ pca_pipeline.py)

## 1. Bảng ngưỡng phương sai
| nguong_phuong_sai | so_chieu_can | phan_tram_giam_chieu |
|---|---|---|
| 80% | 26 | 95.4% |
| 90% | 63 | 88.8% |
| 95% | 102 | 81.8% |
| 99% | 179 | 68.1% |

## 2. Chọn K (GroupKFold theo subject, trên TRAIN)
- K tốt nhất theo CV: **561** (CV acc = 0.9305)
- Quy tắc 1-SE -> ngưỡng 0.9120 -> **K chọn = 102**
- Giảm chiều thực sự: 81.8%
- Accuracy TEST tại K chọn (chỉ để báo cáo): **0.9301**;
  baseline K=561: 0.9617

## 3. Footprint mô hình vs RAM 64.0 KB
| K | cv_acc_mean | pca_fp32_KB | pca_int8_KB | vua_fp32 | vua_int8 |
|---|---|---|---|---|---|
| 2 | 0.498 | 4.461 | 1.115 | True | True |
| 5 | 0.790 | 11.117 | 2.779 | True | True |
| 10 | 0.830 | 22.211 | 5.553 | True | True |
| 15 | 0.858 | 33.305 | 8.326 | True | True |
| 20 | 0.872 | 44.398 | 11.100 | True | True |
| 26 | 0.876 | 57.711 | 14.428 | True | True |
| 30 | 0.879 | 66.586 | 16.646 | False | True |
| 40 | 0.886 | 88.773 | 22.193 | False | True |
| 50 | 0.896 | 110.961 | 27.740 | False | True |
| 63 | 0.904 | 139.805 | 34.951 | False | True |
| 80 | 0.906 | 177.523 | 44.381 | False | True |
| 102 | 0.914 | 226.336 | 56.584 | False | True |
| 125 | 0.915 | 277.367 | 69.342 | False | False |
| 150 | 0.911 | 332.836 | 83.209 | False | False |
| 179 | 0.917 | 397.180 | 99.295 | False | False |
| 200 | 0.923 | 443.773 | 110.943 | False | False |
| 561 | 0.930 | 1244.742 | 311.186 | False | False |

- K lớn nhất vừa 64.0 KB ở fp32: 26, ở int8: 102

## 4. PCA vs SelectKBest (CV accuracy)
| K | pca_cv_acc | selectkbest_cv_acc |
|---|---|---|
| 10 | 0.8298 | 0.7518 |
| 26 | 0.8757 | 0.8665 |
| 63 | 0.9040 | 0.8928 |
| 102 | 0.9137 | 0.9172 |
| 179 | 0.9172 | 0.9237 |

## 5. 10 đặc trưng đóng góp nhiều nhất vào PC1
| dac_trung_goc | trong_so_PC1 |
|---|---|
| fBodyAcc-sma() | 0.0586 |
| fBodyAccJerk-sma() | 0.0586 |
| tBodyAccJerk-sma() | 0.0585 |
| fBodyGyro-sma() | 0.0585 |
| tBodyAccJerkMag-mean() | 0.0585 |
| tBodyAccJerkMag-sma() | 0.0585 |
| fBodyBodyAccJerkMag-sma() | 0.0581 |
| fBodyBodyAccJerkMag-mean() | 0.0581 |
| tBodyAccJerkMag-mad() | 0.0580 |
| tBodyAccJerkMag-std() | 0.0580 |

## 6. Sai số tái tạo (test)
| K | mse_tai_tao |
|---|---|
| 2 | 0.39169 |
| 10 | 0.28804 |
| 25 | 0.20489 |
| 50 | 0.13245 |
| 100 | 0.05784 |
| 102 | 0.05583 |
| 200 | 0.00805 |
