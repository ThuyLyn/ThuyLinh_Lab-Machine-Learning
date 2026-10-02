# TT-29 — TF-IDF: Tự động phân loại ticket hỗ trợ khách hàng về đúng bộ phận

## Tóm tắt kết quả

| Hạng mục | Kết quả |
|---|---|
| Accuracy TF-IDF + LinearSVC (test) | **0,6818** (calibrated: **0,6887**) |
| Baseline: Dummy / CountVectorizer + NB | 0,0530 / 0,6305 |
| Rò rỉ nhãn: không remove → remove đủ 3 | 0,8573 → 0,6903 (giảm **16,7 điểm %**) |
| Ngưỡng tin cậy 0,6 | tự động **49,9%** ticket, accuracy phần tự động **91,6%**, **50,1%** chuyển người |
| Thời gian dự đoán 1 văn bản | **3,29 ms** (p95 3,95 ms) < 5 ms  |
| Thời gian train toàn pipeline | 7,3 s |

Accuracy ~0,69 thấp hơn nhiều so với mức ảo khi để lọt header, và đó chính là bài học của bài này.

## Cách chạy

```bash
pip install -r requirements.txt
python src/train.py                  # chạy đủ 12 bước, lưu ảnh vào reports/ và model vào models/
python src/train.py --skip-leakage   # bỏ Bước 1 cho nhanh
```
Notebook `notebooks/tfidf_ticket_classification.ipynb` chứa cùng các bước kèm nhận xét.

---

## 1.  Bằng chứng rò rỉ do headers / footers / quotes

Cùng một pipeline (TF-IDF unigram + LinearSVC, vectorizer chỉ `fit` trên train), thay đổi mức `remove`:

| Cấu hình | Accuracy (test) | Giảm so với bước trước |
|---|---|---|
| Không remove gì | 0,8573 | — |
| Chỉ bỏ headers | 0,8218 | −3,55 điểm |
| Bỏ headers + footers | 0,8096 | −1,22 điểm |
| Bỏ cả 3 (headers + footers + quotes) | **0,6903** | −11,93 điểm |

**Nhận xét.**
- Khi để nguyên dữ liệu, accuracy cao giả tạo (0,857). Bỏ cả ba thành phần thì chỉ còn 0,690, tức mất 16,7 điểm %. Đề bài nêu mức ~0,95 khi chưa remove; số đo của mình thấp hơn (0,857) vì dùng pipeline đơn giản (unigram, LinearSVC mặc định), nhưng xu hướng giống nhau.
- **Headers** chứa dòng `Newsgroups: ...` (đáp án nằm sẵn trong đầu vào) cùng các dòng như `Organization`, `Subject`, tên máy chủ; **footers** là chữ ký/email đặc trưng của từng người, mà mỗi người thường chỉ đăng ở một nhóm.
- Phần tụt mạnh nhất lại là bước bỏ **quotes** (−11,93 điểm). Giải thích hợp lý (mình chưa kiểm chứng riêng): phần trích dẫn lặp lại nội dung bài trước trong cùng chuỗi thảo luận, nên bài train và bài test cùng thread chia sẻ nhiều từ giống nhau, model "nhận ra" thread thay vì hiểu chủ đề.
- Từ Bước 2 trở đi chỉ dùng bản đã `remove`.

## 2. EDA (dữ liệu đã remove)

- 11.314 văn bản train, 20 lớp, tương đối cân bằng (Dummy chọn lớp đông nhất chỉ đạt 0,053 ≈ 1/20).
- Độ dài: trung bình 185,8 từ nhưng trung vị chỉ 83, lệch phải rất mạnh (tối đa 11.765 từ).
- **300 văn bản rỗng (2,7%)** và **1.186 văn bản dưới 20 từ (10,5%)**, phần lớn do bỏ header/quotes. Đây là nguồn lỗi chính ở mục 10.
- 20 từ phổ biến nhất (bỏ stopword): `ax` (62.387), `max`, `people`, `like`, `don`, `just`, `know`, `use`, `think`, `time`, … Từ `ax` xuất hiện bất thường, nhiều khả năng đến từ khối dữ liệu mã hoá/dán rác lặp lại trong vài bài. Từ `edu` (2.436 lần) vẫn còn trong thân bài (địa chỉ email), nhưng không lọt vào top từ nào ở mục 6.

## 3. Baseline và mô hình chính

| Mô hình | Accuracy (test) |
|---|---|
| Dummy (lớp phổ biến nhất) | 0,0530 |
| CountVectorizer + Naive Bayes | 0,6305 |
| **TF-IDF (1–2 gram, min_df=3, max_df=0,8, sublinear_tf, ≤50.000 đặc trưng) + LinearSVC** | **0,6818** |
| Như trên, bọc calibrate (cv=3) | 0,6887 |

TF-IDF + LinearSVC hơn baseline Naive Bayes 5,1 điểm. Mức 0,6818 nằm sát dưới khoảng tham chiếu 0,69–0,73 của đề; bản calibrate (0,6887) và pipeline unigram ở mục 1 (0,6903) nằm sát ngưỡng dưới của khoảng này.

## 4. Khảo sát tham số vectorizer (ba tham số, đo trên tập validation 20% tách từ train)

| ngram_range | min_df | sublinear_tf | Số đặc trưng | Accuracy (val) |
|---|---|---|---|---|
| (1, 1) | 1 | True | 90.291 | **0,7684** |
| (1, 2) | 1 | False | 798.320 | 0,7645 |
| (1, 2) | 1 | True | 798.320 | 0,7636 |
| (1, 1) | 1 | False | 90.291 | 0,7627 |
| (1, 2) | 3 | True | 99.145 | 0,7521 |
| (1, 1) | 3 | True | 23.420 | 0,7517 |
| (1, 2) | 3 | False | 99.145 | 0,7464 |
| (1, 1) | 3 | False | 23.420 | 0,7455 |
| (1, 1) | 5 | True | 15.695 | 0,7441 |
| (1, 2) | 5 | True | 51.653 | 0,7437 |
| (1, 2) | 5 | False | 51.653 | 0,7349 |
| (1, 1) | 5 | False | 15.695 | 0,7340 |

**Nhận xét.**
- **`ngram_range`:** thêm bigram **không giúp** (ở `min_df=3`: 0,7521 so với 0,7517; ở `min_df=5`: 0,7437 so với 0,7441) nhưng số đặc trưng tăng gấp 4–9 lần (ở `min_df=1`: 90 nghìn → gần 800 nghìn).
- **`min_df`:** `min_df=1` cho accuracy cao nhất (+1,7 điểm so với `min_df=3` ở unigram) nhưng số chiều gấp ~4 lần. Điều này ngược với cảnh báo "min_df=1 gây overfit" trong đề: trên bộ này từ hiếm (tên riêng, mã sản phẩm) vẫn mang thông tin. Tuy vậy mục 6 cho thấy một số từ hiếm đó là tên tác giả hoặc từ của một thread cụ thể, tức là phần nào đang "nhớ" thay vì tổng quát hoá, nên mình **không chọn `min_df=1`** cho bản nộp.
- **`sublinear_tf`:** bật lên tăng khoảng 0,6–1,0 điểm ở 5/6 cặp so sánh (ví dụ min_df=5, unigram: 0,7340 → 0,7441); chỉ một cặp (bigram, min_df=1) gần như hoà (−0,09 điểm).
- Accuracy trên validation (0,75–0,77) cao hơn test (0,68) với cùng cấu hình. Lý do hợp lý: validation tách ngẫu nhiên từ train nên chung thread/tác giả với phần train, còn test của 20 Newsgroups tách theo thời gian. Vì vậy con số validation lạc quan hơn thực tế; chỉ nên dùng để so sánh tương đối giữa các cấu hình.

Cấu hình giữ cho bản chính là cấu hình của đề (1–2 gram, `min_df=3`, `sublinear_tf=True`, ≤50.000 đặc trưng).

## 5. So sánh 3 bộ phân loại trên cùng TF-IDF

| Mô hình | Accuracy (test) | Thời gian train |
|---|---|---|
| **LinearSVC** | **0,6818** | 1,42 s |
| LogReg (C=10) | 0,6758 | 12,79 s |
| Naive Bayes (alpha=0,1) | 0,6705 | 0,04 s |

Ba mô hình chênh nhau không nhiều (1,1 điểm giữa tốt nhất và kém nhất). LinearSVC thắng về accuracy và nhanh gấp ~9 lần LogReg; Naive Bayes nhanh nhất nhưng kém nhất. Văn bản TF-IDF nhiều chiều và thưa nên gần như tách tuyến tính được, hợp với SVM tuyến tính. Naive Bayes trên TF-IDF (0,6705) hơn Naive Bayes trên CountVectorizer (0,6305) 4 điểm; phần chênh này đến từ TF-IDF và cả việc chỉnh `alpha` (0,1 so với mặc định 1), nên không quy hết cho TF-IDF.

## 6. Top 15 từ có trọng số cao nhất mỗi lớp (LinearSVC)

| Lớp | Top 15 từ |
|---|---|
| alt.atheism | atheism, atheists, religion, bobby, islamic, islam, deletion, motto, atheist, cruel, punishment, the motto, timmons, species, bake |
| comp.graphics | graphics, 3d, image, images, pov, animation, 68070, tiff, viewer, polygon, format, 3do, sphere, cview, vesa |
| comp.os.ms-windows.misc | windows, cica, file, win3, risc, ini, nt, w4wg, win, characters, mfc, microsoft, ms, drivers, norton |
| comp.sys.ibm.pc.hardware | vlb, ide, controller, bios, gateway, bus, pc, 486, os, irq, port, isa, scsi, cmos, adaptec |
| comp.sys.mac.hardware | mac, apple, powerbook, quadra, centris, duo, se, lc, adb, nubus, simms, c650, iisi, lciii, vram |
| comp.windows.x | motif, server, widget, xterm, x11r5, window, sun, widgets, mit, clients, xlib, x11, openwindows, hi, comp windows |
| misc.forsale | shipping, sell, offer, for sale, sale, forsale, wanted, condition, asking, new, postage, if interested, includes, su, summer |
| rec.autos | car, cars, dealer, vw, ford, toyota, gt, oil, engine, autos, wagon, saturn, the car, sho, mileage |
| rec.motorcycles | bike, dod, bikes, motorcycle, ride, riding, helmet, harley, bmw, the bike, motorcycles, kawasaki, dog, rider, moto |
| rec.sport.baseball | baseball, stadium, phillies, alomar, braves, cubs, runs, ball, era, pitching, jewish, mets, uniforms, pitchers, bat |
| rec.sport.hockey | hockey, nhl, playoff, team, mask, season, puck, roger, devils, game, coach, the nhl, playoffs, goals, detroit |
| sci.crypt | encryption, clipper, nsa, key, security, crypto, keys, tempest, pgp, privacy, des, the nsa, encrypted, secret, vesselin |
| sci.electronics | circuit, electronics, 8051, voltage, power, motorola, dial, dsp, scope, catalog, cci, circuits, current, the number, copy protection |
| sci.med | msg, doctor, medical, disease, treatment, patients, diet, cancer, health, pain, needles, syndrome, effects, symptoms, photography |
| sci.space | space, orbit, launch, nasa, spacecraft, moon, shuttle, solar, lunar, flight, the moon, rockets, earth, mining, funding |
| soc.religion.christian | god, church, christ, christianity, christians, easter, scripture, christian, marriage, jesus, sin, abstinence, resurrection, catholic, heaven |
| talk.politics.guns | gun, guns, weapons, firearms, firearm, weapon, nra, fire, hunting, fbi, the fire, atf, rkba, criminals, batf |
| talk.politics.mideast | israel, israeli, arab, turkish, jews, arabs, loser, armenians, turkey, serdar, mr, turks, armenian, palestinian, israelis |
| talk.politics.misc | tax, clinton, libertarians, gay, homosexuals, drugs, jobs, deane, taxes, libertarian, trial, bush, blacks, narrative, congress |
| talk.religion.misc | koresh, kent, rosicrucian, objective, amorc, christian, 666, christians, paradise, such, hudson, jesus, how about, values, thou |


**Kiểm tra bằng mắt.**
- **Phần lớn hợp lý:** `rec.autos` (car, engine, dealer, toyota), `sci.space` (orbit, nasa, shuttle), `comp.sys.mac.hardware` (powerbook, quadra, nubus), `talk.politics.guns` (nra, atf, firearms), `rec.sport.hockey` (nhl, puck, playoff)…
- **Không có dấu vết header/email** (tự quét `edu, com, writes, nntp, organization, …` đều không lọt vào top 15 lớp nào), nên việc bỏ headers/footers đã hiệu quả.
- **Nhưng còn "rò rỉ nhẹ" kiểu tên người và thread:** `bobby`, `timmons` (alt.atheism), `serdar` (talk.politics.mideast), `deane` (talk.politics.misc), `kent`, `hudson` (talk.religion.misc), `vesselin` (sci.crypt), `roger` (hockey). Đó là tên người viết hoặc người được nhắc nhiều trong vài chuỗi thảo luận. `deletion`, `motto`, `cruel`, `punishment`, `bake` (alt.atheism) và `jewish` (baseball) cũng giống từ của một thread riêng hơn là của cả chủ đề. Model có phần học thuộc tác giả/thread nên sẽ yếu hơn với bài viết mới.
- Một số từ lạ nhưng có lý do: `dod` (câu lạc bộ "Denizens of Doom" của rec.motorcycles), `msg` (bột ngọt, sci.med), `cica` (kho phần mềm Windows), `w4wg` (Windows for Workgroups). Bigram `comp windows` (comp.windows.x) đến từ việc bài viết nhắc tên chính nhóm.

## 7. Ma trận nhầm lẫn

10 cặp lớp bị nhầm nhiều nhất (số văn bản test):

| Thật | Bị đoán thành | Số văn bản |
|---|---|---|
| talk.politics.misc | talk.politics.guns | 85 |
| talk.religion.misc | soc.religion.christian | 64 |
| comp.windows.x | comp.graphics | 54 |
| alt.atheism | soc.religion.christian | 50 |
| talk.religion.misc | alt.atheism | 38 |
| comp.os.ms-windows.misc | comp.sys.ibm.pc.hardware | 38 |
| comp.sys.ibm.pc.hardware | comp.os.ms-windows.misc | 37 |
| comp.sys.mac.hardware | comp.sys.ibm.pc.hardware | 33 |
| sci.electronics | comp.sys.ibm.pc.hardware | 29 |
| comp.sys.ibm.pc.hardware | sci.electronics | 27 |

**Nhận xét.** Nhầm lẫn tập trung ở các nhóm có từ vựng chồng lấn: **tôn giáo** (religion.misc ↔ christian ↔ atheism), **chính trị** (politics.misc → guns), và **phần cứng/hệ điều hành máy tính** (ms-windows ↔ pc.hardware ↔ mac.hardware, sci.electronics). Hai lớp `*.misc` ("khác") vừa là nguồn nhầm lớn nhất vừa mơ hồ về bản chất, vì chúng gom mọi thứ còn lại. Bài học cho bài toán ticket thật: không nên đặt một nhãn "Khác/Khiếu nại chung" quá rộng cạnh các nhãn cụ thể, vì model và cả nhân viên đều dễ nhầm.

## 8. Cơ chế ngưỡng tin cậy

Quy tắc: nếu `max(proba) ≥ ngưỡng` thì ticket được xử lý tự động, ngược lại chuyển người. (Xác suất lấy từ `CalibratedClassifierCV` bọc quanh LinearSVC.)

**Trên tập test** :

| Ngưỡng | % tự động | Accuracy phần tự động | % cần người |
|---|---|---|---|
| 0,3 | 82,2% | 78,8% | 17,8% |
| 0,4 | 72,7% | 83,6% | 27,3% |
| 0,5 | 61,6% | 88,2% | 38,4% |
| **0,6** | **49,9%** | **91,6%** | **50,1%** |
| 0,7 | 37,3% | 94,2% | 62,7% |
| 0,8 | 19,2% | 96,8% | 80,8% |
| 0,9 | 1,5% | 99,1% | 98,5% |

Accuracy nếu tự động hoá toàn bộ (không ngưỡng): **0,6887**.

**Trên tập validation** (dùng để chọn ngưỡng):

| Ngưỡng | % tự động | Accuracy phần tự động | % cần người |
|---|---|---|---|
| 0,3 | 83,6% | 84,4% | 16,4% |
| 0,5 | 65,4% | 91,6% | 34,6% |
| 0,6 | 52,9% | 94,8% | 47,1% |
| 0,7 | 41,1% | 96,6% | 58,9% |
| 0,8 | 21,9% | 99,0% | 78,1% |

**Nhận xét.**
- Ngưỡng cho phép **đổi khối lượng tự động lấy độ chính xác**: ở 0,6, một nửa ticket được xử lý tự động với accuracy 91,6% (so với 68,9% nếu tự động hết), phần còn lại giao người. Tăng ngưỡng lên 0,8 thì accuracy 96,8% nhưng chỉ còn 19,2% ticket tự động, giảm hẳn giá trị của hệ thống. Ngưỡng 0,9 gần như vô dụng (1,5% tự động).
- Nếu mục tiêu là sai ≤ 10% ở phần tự động thì chọn **0,6** (tỉ lệ sai 8,4%); ngưỡng 0,5 có tỉ lệ sai 11,8% nhưng tự động được nhiều hơn 11,7 điểm %. Mình chọn 0,6 theo đề. Chỉ là so sánh tham khảo: đề nêu ~18% ticket thật bị chuyển sai, nhưng dữ liệu ở đây là 20 Newsgroups chứ không phải ticket, nên không so sánh trực tiếp được.
- Validation lạc quan hơn test khoảng 2–6 điểm tuỳ ngưỡng (ngưỡng 0,6: 94,8% so với 91,6%; chênh nhiều nhất ở ngưỡng thấp, ví dụ 0,3: 84,4% so với 78,8%), cùng nguyên nhân đã nêu ở mục 4. Khi triển khai thật, nên chọn ngưỡng trên dữ liệu gần với dữ liệu thực tế (tách theo thời gian) và theo dõi lại định kỳ.
- Độ tin cậy phân biệt tốt ca đúng/sai: confidence trung bình 0,656 (ca đúng) so với 0,363 (ca sai), nên phần lớn ca sai bị chặn lại ở ngưỡng.

## 9. Thời gian

| Hạng mục | Kết quả |
|---|---|
| Train toàn pipeline (TF-IDF + LinearSVC calibrate cv=3) | 7,3 s |
| Dự đoán 1 văn bản (trung bình 300 lần) | **3,29 ms** |
| Dự đoán 1 văn bản (p95) | 3,95 ms |

Đạt yêu cầu < 5 ms, nhưng biên an toàn không lớn (p95 gần 4 ms); dùng `max_features` nhỏ hơn hoặc bỏ bigram sẽ nhanh hơn. Train 7,3 s nằm trong mức "vài chục giây" của đề nên có thể train lại thường xuyên khi có nhãn mới.

## 10. Phân tích 10 ca sai

Ca sai có confidence thấp hơn (trung bình 0,363 so với 0,656 của ca đúng) và **ngắn hơn** (trung vị 50 từ so với 99 từ).

| # | Thật → Đoán | Conf | Số từ | Nguyên nhân |
|---|---|---|---|---|
| 624 | misc.forsale → rec.autos | 0,09 | 0 | **Văn bản rỗng** (sau khi bỏ header/quotes) |
| 5747 | misc.forsale → rec.autos | 0,09 | 0 | **Văn bản rỗng** |
| 3261 | comp.windows.x → rec.autos | 0,09 | 0 | **Văn bản rỗng** |
| 6421 | sci.crypt → rec.autos | 0,09 | 0 | **Văn bản rỗng** |
| 647 | misc.forsale → talk.politics.guns | 0,23 | 9 | **Quá ngắn**, không có từ khoá chủ đề ("tell us about it Ken!") |
| 1510 | sci.space → rec.motorcycles | 0,27 | 22 | **Quá ngắn / câu đùa**, không có từ khoá chuyên ngành, cần hiểu ngữ cảnh |
| 4871 | alt.atheism → talk.religion.misc | 0,43 | 220 | **Chủ đề giao nhau**: bàn về lập luận/tranh luận, từ vựng không phân biệt hai nhóm tôn giáo |
| 3228 | talk.religion.misc → talk.politics.misc | 0,57 | 111 | **Nhãn mơ hồ**: nội dung bàn về phá thai, vốn nằm giữa tôn giáo và chính trị |
| 699 | talk.politics.guns → talk.politics.mideast | 0,20 | 68 | **Chủ đề giao nhau**: bài về vụ thảm sát (Việt Nam/Mỹ Lai) mượn từ vựng chiến tranh |
| 5205 | talk.politics.misc → talk.politics.guns | **0,78** | 92 | **Chủ đề giao nhau**: bài về vụ BATF/Waco chứa nhiều từ về súng và cơ quan thi hành luật |

**Nhận xét.**
- **4/10 ca là văn bản rỗng**: vector toàn số 0, model chỉ còn dựa vào hệ số chặn nên luôn đoán `rec.autos` với độ tin cậy ~0,09. Cách xử lý: đặt quy tắc **văn bản rỗng hoặc dưới ~5 từ thì chuyển thẳng cho người**, không cần gọi model.
- 2/10 là văn bản quá ngắn, thiếu từ khoá; 4/10 là chủ đề giao nhau hoặc nhãn mơ hồ, đúng với các cặp nhầm lớn ở mục 7.
- **9/10 ca có confidence dưới 0,6** nên sẽ bị chuyển người ở ngưỡng đề xuất. Ngoại lệ là ca #5205 (0,78): model tự tin nhưng sai vì bài thực sự bàn về súng/BATF, nhãn gốc `politics.misc` không còn chỗ dựa trong từ vựng. Loại lỗi "tự tin mà sai" này không ngưỡng nào chặn hết được và là rủi ro còn lại của hệ thống.

## 11. Cấu trúc thư mục

```
TT-29-TFIDF/
├── README.md
├── notebooks/tfidf_ticket_classification.ipynb
├── src/{preprocess.py, train.py}
├── models/tfidf_pipeline.joblib
├── reports/{leakage_comparison.png, top_tu_moi_lop.png, confusion_matrix.png, nguong_tin_cay.png, eda.png}
└── requirements.txt
```