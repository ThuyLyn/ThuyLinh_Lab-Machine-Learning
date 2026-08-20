import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, recall_score, precision_score,
    ConfusionMatrixDisplay, make_scorer
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))      
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)                    
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def main():
# BƯỚC 1 — Nạp dữ liệu, XÁC NHẬN quy ước nhãn (0 = ác tính!)
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target

    print("=" * 70)
    print("BƯỚC 1: Nạp dữ liệu")
    print("=" * 70)
    print("Kích thước X:", X.shape)
    print("Tên nhãn:", data.target_names)  # ['malignant' 'benign'] => 0=ác tính, 1=lành tính
    print("Phân bố nhãn:\n", y.value_counts())
    print("Số giá trị thiếu (NaN):", X.isnull().sum().sum())
    print()

# BƯỚC 2 — EDA: heatmap tương quan 30 biến
    print("BƯỚC 2: Vẽ heatmap tương quan...")
    plt.figure(figsize=(14, 12))
    sns.heatmap(X.corr(), cmap="coolwarm", center=0)
    plt.title("Tương quan giữa 30 đặc trưng")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "correlation_heatmap.png"))
    plt.close()
    print(f"-> Đã lưu {REPORTS_DIR}/correlation_heatmap.png\n")

# BƯỚC 3 — Chia train/test, stratify theo y
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print("BƯỚC 3: Chia train/test")
    print("Train:", X_train.shape, "| Test:", X_test.shape, "\n")

# BƯỚC 4 — SVM KHÔNG scale 
    print("BƯỚC 4: SVM KHÔNG scale")
    svm_noscale = SVC(random_state=42)
    svm_noscale.fit(X_train, y_train)
    pred_noscale = svm_noscale.predict(X_test)

    print(classification_report(y_test, pred_noscale, target_names=data.target_names))
    recall_noscale = recall_score(y_test, pred_noscale, pos_label=0)
    print(f"Recall lớp ÁC TÍNH (không scale): {recall_noscale:.3f}\n")

# BƯỚC 5 — SVM CÓ scale (Pipeline)
    print("BƯỚC 5: SVM CÓ scale (Pipeline)")
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("svm", SVC(class_weight="balanced", random_state=42)),
    ])
    pipe.fit(X_train, y_train)
    pred_scaled = pipe.predict(X_test)

    print(classification_report(y_test, pred_scaled, target_names=data.target_names))
    recall_scaled = recall_score(y_test, pred_scaled, pos_label=0)
    print(f"Recall lớp ÁC TÍNH (có scale): {recall_scaled:.3f}\n")

    compare_scale = pd.DataFrame({
        "Không scale": [recall_noscale],
        "Có scale": [recall_scaled],
    }, index=["Recall (ác tính)"])
    print("Bảng so sánh scale vs không scale:\n", compare_scale, "\n")

    plt.figure(figsize=(5, 4))
    compare_scale.T.plot(kind="bar", legend=False)
    plt.ylabel("Recall lớp ác tính")
    plt.title("SVM: Có scale vs Không scale")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "scale_vs_noscale.png"))
    plt.close()
    print(f"-> Đã lưu {REPORTS_DIR}/scale_vs_noscale.png\n")

# BƯỚC 6 — So sánh 3 kernel (cùng C=1)
    print("BƯỚC 6: So sánh 3 kernel (C=1)")
    kernel_results = {}
    for kernel in ["linear", "rbf", "poly"]:
        p = Pipeline([
            ("scale", StandardScaler()),
            ("svm", SVC(kernel=kernel, C=1, class_weight="balanced", random_state=42)),
        ])
        p.fit(X_train, y_train)
        pred = p.predict(X_test)
        kernel_results[kernel] = recall_score(y_test, pred, pos_label=0)

    kernel_df = pd.Series(kernel_results, name="Recall (ác tính)")
    print(kernel_df, "\n")

    plt.figure(figsize=(5, 4))
    kernel_df.plot(kind="bar", color=["#4C72B0", "#DD8452", "#55A868"])
    plt.ylabel("Recall lớp ác tính")
    plt.title("So sánh kernel (C=1)")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "kernel_comparison.png"))
    plt.close()
    print(f"-> Đã lưu {REPORTS_DIR}/kernel_comparison.png\n")

 # BƯỚC 7 — GridSearchCV dò C, gamma, kernel (scoring = recall lớp ác tính)
    print("BƯỚC 7: GridSearchCV (có thể mất 1-2 phút)...")
    recall_malignant_scorer = make_scorer(recall_score, pos_label=0)

    grid = {
        "svm__C": [0.1, 1, 10, 100],
        "svm__gamma": ["scale", 0.001, 0.01, 0.1],
        "svm__kernel": ["linear", "rbf", "poly"],
    }

    gs = GridSearchCV(pipe, grid, cv=5, scoring=recall_malignant_scorer, n_jobs=-1)
    gs.fit(X_train, y_train)

    print("Best params:", gs.best_params_)
    print("Best CV recall (ác tính):", round(gs.best_score_, 4), "\n")


# BƯỚC 8 — Heatmap điểm CV theo (C, gamma) — kernel rbf
    print("BƯỚC 8: Heatmap C x gamma (kernel=rbf)")
    cv_results = pd.DataFrame(gs.cv_results_)
    rbf_results = cv_results[cv_results["param_svm__kernel"] == "rbf"]
    pivot = rbf_results.pivot_table(
        index="param_svm__gamma", columns="param_svm__C", values="mean_test_score"
    )

    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis")
    plt.title("Recall CV (ác tính) theo C và gamma — kernel RBF")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "C_gamma_heatmap.png"))
    plt.close()
    print(f"-> Đã lưu {REPORTS_DIR}/C_gamma_heatmap.png\n")

# BƯỚC 9 — Đếm support vectors
    print("BƯỚC 9: Support vectors")
    best_svm = gs.best_estimator_.named_steps["svm"]
    n_sv = best_svm.n_support_
    pct_sv = n_sv.sum() / len(X_train) * 100
    print(f"Support vectors mỗi lớp: {n_sv}")
    print(f"Tổng support vectors: {n_sv.sum()} / {len(X_train)} ({pct_sv:.1f}%)\n")

# BƯỚC 10 — Chọn ngưỡng đạt recall ác tính >= 0.98

    print("BƯỚC 10: Dò ngưỡng để recall ác tính >= 0.98")
    best_params_clean = {k.replace("svm__", ""): v for k, v in gs.best_params_.items()}

    pipe_proba = Pipeline([
        ("scale", StandardScaler()),
        ("svm", SVC(**best_params_clean, probability=True,
                    class_weight="balanced", random_state=42)),
    ])
    pipe_proba.fit(X_train, y_train)

    proba_malignant = pipe_proba.predict_proba(X_test)[:, 0]  # xác suất lớp 0 = ác tính

    threshold_results = []
    chosen_threshold = None
    for threshold in np.arange(0.05, 0.55, 0.05):
        pred_thresh = np.where(proba_malignant >= threshold, 0, 1)
        r = recall_score(y_test, pred_thresh, pos_label=0)
        p = precision_score(y_test, pred_thresh, pos_label=0)
        threshold_results.append((round(threshold, 2), r, p))
        if r >= 0.98 and chosen_threshold is None:
            chosen_threshold = threshold

    threshold_df = pd.DataFrame(threshold_results, columns=["threshold", "recall", "precision"])
    print(threshold_df, "\n")

    if chosen_threshold is None:
        chosen_threshold = threshold_df["threshold"].min()
        print("CẢNH BÁO: không tìm được threshold đạt recall >= 0.98 trong khoảng quét, "
              "dùng threshold nhỏ nhất đã thử.\n")
    else:
        r_chosen = threshold_df.loc[threshold_df.threshold == round(chosen_threshold, 2), "recall"].values[0]
        print(f"-> Chọn threshold = {chosen_threshold:.2f} (recall ác tính = {r_chosen:.3f})\n")

# BƯỚC 11 — Ma trận nhầm lẫn với ngưỡng đã chọn
    print("BƯỚC 11: Ma trận nhầm lẫn (sau khi chỉnh ngưỡng)")
    pred_final = np.where(proba_malignant >= chosen_threshold, 0, 1)

    cm_disp = ConfusionMatrixDisplay.from_predictions(
        y_test, pred_final, display_labels=data.target_names, cmap="Blues"
    )
    plt.title(f"Ma trận nhầm lẫn (threshold={chosen_threshold:.2f})")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "confusion_matrix.png"))
    plt.close()

    cm = cm_disp.confusion_matrix
    missed_malignant = cm[0, 1]  # thực tế ác tính (0), dự đoán lành tính (1)
    print(f"-> Số ca ÁC TÍNH bị bỏ sót: {missed_malignant}")
    print(f"-> Đã lưu {REPORTS_DIR}/confusion_matrix.png\n")

# BƯỚC 12 — So sánh với Logistic Regression (TT-04)
    print("BƯỚC 12: So sánh với Logistic Regression")
    logreg = Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
    ])
    logreg.fit(X_train, y_train)
    pred_lr = logreg.predict(X_test)
    recall_lr = recall_score(y_test, pred_lr, pos_label=0)

    print(f"Logistic Regression - recall (ác tính): {recall_lr:.3f}")
    print(f"SVM (đã chỉnh ngưỡng)  - recall (ác tính): "
          f"{recall_score(y_test, pred_final, pos_label=0):.3f}\n")

# Lưu model cuối cùng
    model_path = os.path.join(MODELS_DIR, "svm_pipeline.joblib")
    joblib.dump(gs.best_estimator_, model_path)
    print(f"-> Đã lưu {model_path}")
    print("\nHOÀN THÀNH toàn bộ 12 bước.")


if __name__ == "__main__":
    main()