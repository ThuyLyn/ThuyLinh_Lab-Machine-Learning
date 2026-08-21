import json
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix,
)
from statsmodels.stats.outliers_influence import variance_inflation_factor

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# BƯỚC 1: Nạp dữ liệu + loại trùng lặp
DATA_PATH = "data/heart.csv"

df = pd.read_csv(DATA_PATH)
print(f"Kích thước ban đầu: {df.shape}")

n_dup = df.duplicated().sum()
df = df.drop_duplicates().reset_index(drop=True)
print(f"Số dòng trùng đã loại: {n_dup} -> kích thước còn lại: {df.shape}")
print(f"Kiểm chứng lại số dòng trùng: {df.duplicated().sum()}")

# BƯỚC 2: Xác định biến 
categorical_columns = [
    "cp", "restecg", "slope", "thal", "ca",   # nhiều mức
    "sex", "fbs", "exang",                     # nhị phân, cũng là phân loại
]
numeric_columns = ["age", "trestbps", "chol", "thalach", "oldpeak"]
target_column = "target"

assert set(categorical_columns + numeric_columns + [target_column]) == set(df.columns), \
    "Có cột trong dữ liệu chưa được khai báo ở categorical_columns/numeric_columns!"

print(f"Biến phân loại ({len(categorical_columns)}): {categorical_columns}")
print(f"Biến số ({len(numeric_columns)}): {numeric_columns}")

# BƯỚC 3: Chia Train (60%) / Validation (20%) / Test (20%)
X = df.drop(columns=[target_column])
y = df[target_column]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)
print(f"Train: {len(X_train)} | Validation: {len(X_val)} | Test: {len(X_test)}")

# BƯỚC 4: Preprocessing 
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_columns),
        ("numeric", StandardScaler(), numeric_columns),
    ]
)

logistic_pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("model", LogisticRegression(max_iter=2000, random_state=42)),
])
logistic_pipeline.fit(X_train, y_train)
print("Đã train Logistic Regression.")

# BƯỚC 5: Đánh giá đầy đủ trên TẬP TEST 
def evaluate(model, X, y, name):
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, proba),
    }
    print(f"[{name}] " + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    return metrics

# Baseline Dummy — mốc so sánh tối thiểu
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
dummy_metrics = evaluate(dummy, X_test, y_test, "Dummy baseline")

# Logistic Regression (ngưỡng mặc định 0.5)
logreg_metrics = evaluate(logistic_pipeline, X_test, y_test, "Logistic Regression")

# BƯỚC 6: Bảng Odds Ratio
trained_model = logistic_pipeline.named_steps["model"]
coefficients = trained_model.coef_[0]
feature_names = logistic_pipeline.named_steps["preprocessing"].get_feature_names_out()
odds_ratio = np.exp(coefficients)

odds_table = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients,
    "Odds_Ratio": odds_ratio,
}).sort_values("Odds_Ratio", ascending=False)

print("\nBẢNG ODDS RATIO (top 10):")
print(odds_table.head(10).to_string(index=False))

# BƯỚC 7: So sánh L1 và L2
model_l1 = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("model", LogisticRegression(penalty="l1", solver="liblinear", max_iter=2000, random_state=42)),
])
model_l2 = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("model", LogisticRegression(penalty="l2", max_iter=2000, random_state=42)),
])
model_l1.fit(X_train, y_train)
model_l2.fit(X_train, y_train)

l1_metrics = evaluate(model_l1, X_test, y_test, "L1")
l2_metrics = evaluate(model_l2, X_test, y_test, "L2")

coef_l1 = model_l1.named_steps["model"].coef_[0]
zero_count = int(np.sum(np.isclose(coef_l1, 0)))
print(f"Số hệ số bị L1 đưa về 0: {zero_count}/{len(coef_l1)}")

# BƯỚC 8: Chọn ngưỡng đạt Recall >= 0.90 TRÊN TẬP VALIDATION
target_recall = 0.90

y_val_proba = logistic_pipeline.predict_proba(X_val)[:, 1]
precision_values, recall_values, pr_thresholds = precision_recall_curve(y_val, y_val_proba)
valid_idx = np.where(recall_values[:-1] >= target_recall)[0]

if len(valid_idx) > 0:
    idx = valid_idx[-1]  
    best_threshold = float(pr_thresholds[idx])
    best_precision_val = float(precision_values[idx])
    best_recall_val = float(recall_values[idx])
else:
    best_threshold = 0.5
    best_precision_val = None
    best_recall_val = None
    print(f"CẢNH BÁO: không tìm được ngưỡng nào đạt recall >= {target_recall} trên validation.")

print(f"Ngưỡng chọn (trên validation): {best_threshold:.4f}")
print(f"  Precision (validation): {best_precision_val}")
print(f"  Recall (validation):    {best_recall_val}")

y_test_proba = logistic_pipeline.predict_proba(X_test)[:, 1]
y_pred_threshold = (y_test_proba >= best_threshold).astype(int)

precision_thresh = precision_score(y_test, y_pred_threshold, zero_division=0)
recall_thresh = recall_score(y_test, y_pred_threshold, zero_division=0)
n_flagged = int(y_pred_threshold.sum())

print(f"\nKẾT QUẢ TRÊN TEST với ngưỡng {best_threshold:.4f}:")
print(f"  Precision: {precision_thresh:.4f}")
print(f"  Recall:    {recall_thresh:.4f}")
print(f"  Số bệnh nhân bị gắn cờ dương tính: {n_flagged}/{len(y_test)}")
base_rate = y_test.mean()
if n_flagged >= 0.95 * len(y_test) or abs(precision_thresh - base_rate) < 0.01:
    print("  CẢNH BÁO: ngưỡng gần như flag TOÀN BỘ bệnh nhân là dương tính — "
          "precision xấp xỉ tỉ lệ nền, model không cho thêm thông tin gì so với "
          "DummyClassifier ở ngưỡng này. Cân nhắc hạ target_recall hoặc xem lại model.")

# BƯỚC 9: VIF — kiểm tra đa cộng tuyến 
X_vif = preprocessor.fit_transform(X_train)
if hasattr(X_vif, "toarray"):
    X_vif = X_vif.toarray()
vif_features = preprocessor.get_feature_names_out()

vif_table = pd.DataFrame({
    "Feature": vif_features,
    "VIF": [variance_inflation_factor(X_vif, i) for i in range(X_vif.shape[1])],
}).sort_values("VIF", ascending=False)

n_inf = np.isinf(vif_table["VIF"]).sum()
print(f"\nBẢNG VIF — số dòng còn bị inf: {n_inf} (kỳ vọng: 0 sau khi thêm drop='first')")
print(vif_table.head(10).to_string(index=False))

vif_table.to_csv("reports/vif_table.csv", index=False)
print("Đã lưu reports/vif_table.csv")

# BƯỚC 10: Vẽ và lưu Odds Ratio
plt.figure(figsize=(10, 8))
plt.barh(odds_table["Feature"], odds_table["Odds_Ratio"])
plt.axvline(1, linestyle="--", color="gray")
plt.xlabel("Odds Ratio")
plt.ylabel("Feature")
plt.title("Logistic Regression - Odds Ratio")
plt.tight_layout()
plt.savefig("reports/odds_ratio.png", dpi=150)
plt.close()
print("Đã lưu reports/odds_ratio.png")

# BƯỚC 11: Vẽ và lưu ROC + Precision-Recall (trên TEST)
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_test_proba)
auc_test = roc_auc_score(y_test, y_test_proba)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(fpr, tpr, label=f"AUC = {auc_test:.3f}")
axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve (test)")
axes[0].legend()

axes[1].plot(recall_curve, precision_curve)
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve (test)")
axes[1].grid(True)

plt.tight_layout()
plt.savefig("reports/roc_pr_curve.png", dpi=150)
plt.close()
print("Đã lưu reports/roc_pr_curve.png")

# BƯỚC 12: Lưu toàn bộ chỉ số vào metrics.json
metrics_summary = {
    "data": {
        "rows_before_dedup": int(n_dup + len(df)),
        "duplicates_removed": int(n_dup),
        "rows_after_dedup": int(len(df)),
    },
    "dummy_baseline": dummy_metrics,
    "logistic_regression_default_threshold": logreg_metrics,
    "l1": l1_metrics,
    "l2": l2_metrics,
    "l1_coefficients_zeroed": zero_count,
    "threshold_tuning": {
        "target_recall": target_recall,
        "chosen_threshold": best_threshold,
        "precision_on_validation": best_precision_val,
        "recall_on_validation": best_recall_val,
        "precision_on_test": precision_thresh,
        "recall_on_test": recall_thresh,
        "n_flagged_positive_on_test": n_flagged,
        "test_set_size": len(y_test),
    },
    "vif_max_value": None if vif_table["VIF"].replace([np.inf], np.nan).isna().all()
        else float(vif_table["VIF"].replace([np.inf], np.nan).max()),
    "vif_inf_count": int(n_inf),
}

with open("reports/metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics_summary, f, ensure_ascii=False, indent=2)
print("Đã lưu reports/metrics.json")

# BƯỚC 13: Lưu model
joblib.dump(logistic_pipeline, "models/logreg_pipeline.joblib")
print("Đã lưu model: models/logreg_pipeline.joblib")

print("\n=== HOÀN TẤT ===")