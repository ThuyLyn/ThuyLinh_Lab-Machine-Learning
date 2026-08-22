import os
import json
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
    classification_report, recall_score, precision_score, fbeta_score,
    ConfusionMatrixDisplay, make_scorer
)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def main():
    metrics_report = {} 
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target

    print("=" * 70)
    print("BƯỚC 1: Nạp dữ liệu")
    print("=" * 70)
    print("Kích thước X:", X.shape)
    print("Tên nhãn:", data.target_names) 
    print("Phân bố nhãn:\n", y.value_counts())
    print("Số giá trị thiếu (NaN):", X.isnull().sum().sum())
    print()

    print("BƯỚC 2: Vẽ heatmap tương quan...")
    plt.figure(figsize=(14, 12))
    sns.heatmap(X.corr(), cmap="coolwarm", center=0)
    plt.title("Tương quan giữa 30 đặc trưng")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "correlation_heatmap.png"))
    plt.close()
    print(f"-> Đã lưu {REPORTS_DIR}/correlation_heatmap.png\n")

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=42
    )  

    print("BƯỚC 3: Chia train/validation/test")
    print("Train:", X_train.shape, "| Validation:", X_val.shape,
          "| Test:", X_test.shape, "\n")

    print("BƯỚC 4: SVM KHÔNG scale")
    svm_noscale = SVC(random_state=42)
    svm_noscale.fit(X_train, y_train)
    pred_noscale = svm_noscale.predict(X_val)

    print(classification_report(y_val, pred_noscale, target_names=data.target_names))
    recall_noscale = recall_score(y_val, pred_noscale, pos_label=0)
    print(f"Recall lớp ÁC TÍNH (không scale, trên validation): {recall_noscale:.3f}\n")

    print("BƯỚC 5: SVM CÓ scale (Pipeline)")
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("svm", SVC(class_weight="balanced", random_state=42)),
    ])
    pipe.fit(X_train, y_train)
    pred_scaled = pipe.predict(X_val)

    print(classification_report(y_val, pred_scaled, target_names=data.target_names))
    recall_scaled = recall_score(y_val, pred_scaled, pos_label=0)
    print(f"Recall lớp ÁC TÍNH (có scale, trên validation): {recall_scaled:.3f}\n")

    compare_scale = pd.DataFrame({
        "Không scale": [recall_noscale],
        "Có scale": [recall_scaled],
    }, index=["Recall (ác tính)"])
    print("Bảng so sánh scale vs không scale:\n", compare_scale, "\n")

    plt.figure(figsize=(5, 4))
    compare_scale.T.plot(kind="bar", legend=False)
    plt.ylabel("Recall lớp ác tính (validation)")
    plt.title("SVM: Có scale vs Không scale")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "scale_vs_noscale.png"))
    plt.close()
    print(f"-> Đã lưu {REPORTS_DIR}/scale_vs_noscale.png\n")

    metrics_report["scale_comparison_on_validation"] = {
        "recall_no_scale": round(float(recall_noscale), 4),
        "recall_scaled": round(float(recall_scaled), 4),
    }
    print("BƯỚC 6: So sánh 3 kernel (C=1)")
    kernel_results = {}
    for kernel in ["linear", "rbf", "poly"]:
        p = Pipeline([
            ("scale", StandardScaler()),
            ("svm", SVC(kernel=kernel, C=1, class_weight="balanced", random_state=42)),
        ])
        p.fit(X_train, y_train)
        pred = p.predict(X_val)
        kernel_results[kernel] = recall_score(y_val, pred, pos_label=0)

    kernel_df = pd.Series(kernel_results, name="Recall (ác tính)")
    print(kernel_df, "\n")

    plt.figure(figsize=(5, 4))
    kernel_df.plot(kind="bar", color=["#4C72B0", "#DD8452", "#55A868"])
    plt.ylabel("Recall lớp ác tính (validation)")
    plt.title("So sánh kernel (C=1)")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "kernel_comparison.png"))
    plt.close()
    print(f"-> Đã lưu {REPORTS_DIR}/kernel_comparison.png\n")

    metrics_report["kernel_comparison_on_validation"] = {
        k: round(float(v), 4) for k, v in kernel_results.items()
    }
    print("BƯỚC 7: GridSearchCV (có thể mất 1-2 phút)...")
    fbeta2_malignant_scorer = make_scorer(fbeta_score, pos_label=0, beta=2)

    grid = {
        "svm__C": [0.1, 1, 10, 100],
        "svm__gamma": ["scale", 0.001, 0.01, 0.1],
        "svm__kernel": ["linear", "rbf", "poly"],
    }
    gs = GridSearchCV(pipe, grid, cv=5, scoring=fbeta2_malignant_scorer, n_jobs=-1)
    gs.fit(X_train, y_train)
    print("Best params:", gs.best_params_)
    print("Best CV F2-score (ác tính):", round(gs.best_score_, 4), "\n")

    metrics_report["grid_search"] = {
        "scoring": "fbeta(beta=2, pos_label=malignant) trên CV=5 của TRAIN",
        "best_params": gs.best_params_,
        "best_cv_score": round(float(gs.best_score_), 4),
    }

    print("BƯỚC 8: Heatmap C x gamma (kernel=rbf)")
    cv_results = pd.DataFrame(gs.cv_results_)
    rbf_results = cv_results[cv_results["param_svm__kernel"] == "rbf"]
    pivot = rbf_results.pivot_table(
        index="param_svm__gamma", columns="param_svm__C", values="mean_test_score"
    )
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis")
    plt.title("F2-score CV (ác tính) theo C và gamma — kernel RBF")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "C_gamma_heatmap.png"))
    plt.close()
    print(f"-> Đã lưu {REPORTS_DIR}/C_gamma_heatmap.png\n")

    print("BƯỚC 9: Support vectors")
    best_svm = gs.best_estimator_.named_steps["svm"]
    n_sv = best_svm.n_support_
    pct_sv = n_sv.sum() / len(X_train) * 100
    print(f"Support vectors mỗi lớp: {n_sv}")
    print(f"Tổng support vectors: {n_sv.sum()} / {len(X_train)} ({pct_sv:.1f}%)\n")

    metrics_report["support_vectors"] = {
        "per_class": n_sv.tolist(),
        "total": int(n_sv.sum()),
        "train_size": int(len(X_train)),
        "pct_of_train": round(float(pct_sv), 2),
    }
    print("BƯỚC 10: Dò ngưỡng trên VALIDATION để recall ác tính >= 0.98")
    best_params_clean = {k.replace("svm__", ""): v for k, v in gs.best_params_.items()}
    pipe_proba = Pipeline([
        ("scale", StandardScaler()),
        ("svm", SVC(**best_params_clean, probability=True,
                    class_weight="balanced", random_state=42)),
    ])
    pipe_proba.fit(X_train, y_train)
    proba_val_malignant = pipe_proba.predict_proba(X_val)[:, 0]
    threshold_results = []
    candidates = [] 
    for threshold in np.arange(0.05, 0.55, 0.05):
        pred_thresh = np.where(proba_val_malignant >= threshold, 0, 1)
        r = recall_score(y_val, pred_thresh, pos_label=0)
        p = precision_score(y_val, pred_thresh, pos_label=0, zero_division=0)
        threshold_results.append((round(threshold, 2), r, p))
        if r >= 0.98:
            candidates.append((threshold, r, p))

    threshold_df = pd.DataFrame(threshold_results, columns=["threshold", "recall", "precision"])
    print(threshold_df, "\n")

    if candidates:
        chosen_threshold, r_chosen, p_chosen = max(candidates, key=lambda t: t[0])
        print(f"-> Chọn threshold = {chosen_threshold:.2f} "
              f"(validation recall = {r_chosen:.3f}, precision = {p_chosen:.3f})\n")
    else:
        chosen_threshold = threshold_df["threshold"].min()
        r_chosen = threshold_df["recall"].min()
        p_chosen = threshold_df.loc[threshold_df.threshold == chosen_threshold, "precision"].values[0]
        print("CẢNH BÁO: không tìm được threshold đạt recall >= 0.98 trong khoảng quét, "
              "dùng threshold nhỏ nhất đã thử.\n")

    metrics_report["threshold_selection"] = {
        "evaluated_on": "VALIDATION (không phải test)",
        "target_recall": 0.98,
        "scan": threshold_df.to_dict(orient="records"),
        "chosen_threshold": round(float(chosen_threshold), 3),
        "recall_at_chosen_on_validation": round(float(r_chosen), 4),
        "precision_at_chosen_on_validation": round(float(p_chosen), 4),
        "tradeoff_note": (
            "Ngưỡng thấp hơn làm recall cao hơn nhưng precision thấp hơn -> "
            "nhiều ca lành tính bị gán nhầm 'nghi ác tính', dẫn đến sinh "
            "thiết/xét nghiệm thêm không cần thiết."
        ),
    }
    print("BƯỚC 11: Refit trên TRAIN+VALIDATION, đánh giá 1 lần trên TEST")
    final_pipe = Pipeline([
        ("scale", StandardScaler()),
        ("svm", SVC(**best_params_clean, probability=True,
                    class_weight="balanced", random_state=42)),
    ])
    final_pipe.fit(X_trainval, y_trainval)

    proba_test_malignant = final_pipe.predict_proba(X_test)[:, 0]
    pred_final = np.where(proba_test_malignant >= chosen_threshold, 0, 1)

    cm_disp = ConfusionMatrixDisplay.from_predictions(
        y_test, pred_final, display_labels=data.target_names, cmap="Blues"
    )
    plt.title(f"Ma trận nhầm lẫn trên TEST (threshold={chosen_threshold:.2f})")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "confusion_matrix.png"))
    plt.close()

    cm = cm_disp.confusion_matrix
    missed_malignant = cm[0, 1]  # thực tế ác tính (0), dự đoán lành tính (1)
    recall_test = recall_score(y_test, pred_final, pos_label=0)
    precision_test = precision_score(y_test, pred_final, pos_label=0, zero_division=0)
    print(f"-> Recall ác tính (TEST, đánh giá 1 lần): {recall_test:.3f}")
    print(f"-> Precision ác tính (TEST, đánh giá 1 lần): {precision_test:.3f}")
    print(f"-> Số ca ÁC TÍNH bị bỏ sót: {missed_malignant}")
    print(f"-> Đã lưu {REPORTS_DIR}/confusion_matrix.png\n")

    metrics_report["final_test_evaluation"] = {
        "note": "TEST chỉ được dùng 1 lần duy nhất cho đánh giá cuối cùng",
        "threshold_used": round(float(chosen_threshold), 3),
        "recall_malignant": round(float(recall_test), 4),
        "precision_malignant": round(float(precision_test), 4),
        "missed_malignant_cases": int(missed_malignant),
        "test_size": int(len(y_test)),
    }

    print("BƯỚC 12: So sánh với Logistic Regression")
    logreg = Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
    ])
    logreg.fit(X_trainval, y_trainval)
    pred_lr = logreg.predict(X_test)
    recall_lr = recall_score(y_test, pred_lr, pos_label=0)

    print(f"Logistic Regression - recall (ác tính): {recall_lr:.3f}")
    print(f"SVM (đã chỉnh ngưỡng)  - recall (ác tính): {recall_test:.3f}\n")

    metrics_report["logreg_baseline_comparison"] = {
        "logreg_recall_malignant_test": round(float(recall_lr), 4),
        "svm_recall_malignant_test": round(float(recall_test), 4),
    }
    model_path = os.path.join(MODELS_DIR, "svm_pipeline.joblib")
    bundle = {
        "pipeline": final_pipe,
        "threshold": float(chosen_threshold),
        "positive_class_for_threshold": "malignant (label 0)",
        "predict_instructions": (
            "Không dùng pipeline.predict() trực tiếp (nó dùng ngưỡng 0.5). "
            "Thay vào đó: proba = pipeline.predict_proba(X)[:, 0]; "
            "pred = np.where(proba >= threshold, 0, 1)"
        ),
        "best_params": best_params_clean,
    }
    joblib.dump(bundle, model_path)
    print(f"-> Đã lưu {model_path} (kèm threshold = {chosen_threshold:.2f})")
    metrics_path = os.path.join(REPORTS_DIR, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, ensure_ascii=False, indent=2)
    print(f"-> Đã lưu {metrics_path}")
    print("\nHOÀN THÀNH")

if __name__ == "__main__":
    main()