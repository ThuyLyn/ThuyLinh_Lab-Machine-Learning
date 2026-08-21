import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# Tạo sẵn thư mục output nếu chưa có
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)
# BƯỚC 1: Nạp dữ liệu

DATA_PATH = r"D:\TT_ML\TT-06-NaiveBayes_Linh\dataset\spam.csv"

# Cạm bẫy: file gốc lưu bằng encoding latin-1, không phải utf-8
df = pd.read_csv(DATA_PATH, encoding="latin-1")

# Chỉ giữ 2 cột cần thiết (2 cột còn lại là rác do dấu phẩy dư trong file gốc)
df = df.iloc[:, :2]
df.columns = ["label", "text"]

print(f"Số dòng dữ liệu: {len(df)}")
print(df["label"].value_counts())

# BƯỚC 2: Loại tin nhắn trùng lặp
# Nếu không loại trùng, cùng 1 tin có thể vừa ở train vừa ở test -> đánh giá sai
so_dong_truoc = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Đã loại {so_dong_truoc - len(df)} dòng trùng lặp, còn lại {len(df)} dòng")

# BƯỚC 3: Chia tập train/test
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"],  # giữ nguyên tỉ lệ ham/spam ở cả 2 tập
)
print(f"Train: {len(X_train)} tin | Test: {len(X_test)} tin")

# BƯỚC 4: Dò tham số tốt nhất (alpha, ngram_range)

pipe = make_pipeline(
    TfidfVectorizer(lowercase=True, min_df=2, max_df=0.9, sublinear_tf=True),
    MultinomialNB(),
)

param_grid = {
    "tfidfvectorizer__ngram_range": [(1, 1), (1, 2)],
    "multinomialnb__alpha": [0.01, 0.1, 0.5, 1.0],
}

grid = GridSearchCV(pipe, param_grid, scoring="f1_weighted", cv=5, n_jobs=-1)
grid.fit(X_train, y_train)

print("Tham số tốt nhất:", grid.best_params_)

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred, digits=4))

# BƯỚC 5: Chỉnh ngưỡng để đạt precision cao (ưu tiên không chặn nhầm)
TARGET_PRECISION = 0.98

# Lấy xác suất dự đoán là "spam" cho từng tin trong tập test
classes = list(best_model.named_steps["multinomialnb"].classes_)
spam_index = classes.index("spam")
y_proba = best_model.predict_proba(X_test)[:, spam_index]

precisions, recalls, thresholds = precision_recall_curve(
    (y_test == "spam").astype(int), y_proba
)

# Tìm ngưỡng thấp nhất mà vẫn đạt được precision mục tiêu
valid = np.where(precisions[:-1] >= TARGET_PRECISION)[0]
if len(valid) > 0:
    threshold = thresholds[valid[0]]
else:
    threshold = 0.5
    print(f"Không đạt được precision >= {TARGET_PRECISION}, dùng ngưỡng mặc định 0.5")

print(f"Ngưỡng đã chọn: {threshold:.4f}")

y_pred_final = np.where(y_proba >= threshold, "spam", "ham")
print("Kết quả sau khi chỉnh ngưỡng:")
print(classification_report(y_test, y_pred_final, digits=4))

# BƯỚC 6: Vẽ confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])
fig, ax = plt.subplots(figsize=(5, 5))
ConfusionMatrixDisplay(cm, display_labels=["ham", "spam"]).plot(ax=ax, cmap="Blues")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("reports/confusion_matrix.png", dpi=120)
plt.close()
print("Đã lưu reports/confusion_matrix.png")

# BƯỚC 7: Top 20 từ đặc trưng của spam
vectorizer = best_model.named_steps["tfidfvectorizer"]
nb_model = best_model.named_steps["multinomialnb"]

ham_index = classes.index("ham")
feature_names = vectorizer.get_feature_names_out()

# log-ratio: từ nào "nghiêng" về spam nhiều hơn hẳn ham
log_ratio = nb_model.feature_log_prob_[spam_index] - nb_model.feature_log_prob_[ham_index]
top20 = pd.Series(log_ratio, index=feature_names).sort_values(ascending=False).head(20)

print("Top 20 từ đặc trưng cho spam:")
print(top20)

fig, ax = plt.subplots(figsize=(7, 6))
top20.sort_values().plot(kind="barh", ax=ax, color="crimson")
plt.title("Top 20 từ đặc trưng cho SPAM")
plt.tight_layout()
plt.savefig("reports/top_tu_spam.png", dpi=120)
plt.close()
print("Đã lưu reports/top_tu_spam.png")

# BƯỚC 8: Đo thời gian train + dự đoán (yêu cầu < 5ms/tin)
t0 = time.perf_counter()
best_model.fit(X_train, y_train)  # train lại lần cuối trên model đã chọn
train_time = time.perf_counter() - t0
print(f"Thời gian train: {train_time*1000:.1f} ms")

sample = X_test.iloc[[0]]
best_model.predict(sample)  # warm-up lần đầu

t0 = time.perf_counter()
for _ in range(100):
    best_model.predict(sample)
avg_ms = (time.perf_counter() - t0) / 100 * 1000
print(f"Thời gian dự đoán 1 tin: {avg_ms:.3f} ms", "(ĐẠT < 5ms)" if avg_ms < 5 else "(CHƯA ĐẠT)")

# BƯỚC 9: Lưu model để dùng lại sau này
joblib.dump(best_model, "models/nb_pipeline.joblib")
print("Đã lưu model: models/nb_pipeline.joblib")
