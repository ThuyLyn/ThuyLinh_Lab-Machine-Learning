import os
import json
import time
import argparse

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
    make_scorer,
)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_PATH = os.path.join(PROJECT_ROOT, "dataset", "spam.csv")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train mô hình Naive Bayes lọc SMS rác."
    )
    parser.add_argument(
        "--data", type=str, default=DEFAULT_DATA_PATH,
        help="Đường dẫn tới file spam.csv ",
    )
    parser.add_argument(
        "--out-dir", type=str, default=PROJECT_ROOT,
        help="Thư mục gốc để ghi models/ và reports/ ",
    )
    parser.add_argument(
        "--target-precision", type=float, default=0.98,
        help="Precision tối thiểu mong muốn cho lớp spam (mặc định: 0.98)",
    )
    parser.add_argument(
        "--random-state", type=int, default=42,
        help="Random state dùng cho các bước chia dữ liệu (mặc định: 42)",
    )
    return parser.parse_args()

def load_data(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path, encoding="latin-1")
    df = df.iloc[:, :2]
    df.columns = ["label", "text"]

    print(f"Số dòng dữ liệu: {len(df)}")
    print(df["label"].value_counts())

    so_dong_truoc = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Đã loại {so_dong_truoc - len(df)} dòng trùng lặp, còn lại {len(df)} dòng")
    return df

def split_data(df: pd.DataFrame, random_state: int):
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        df["text"], df["label"],
        test_size=0.2,
        random_state=random_state,
        stratify=df["label"],
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.2,
        random_state=random_state,
        stratify=y_train_full,
    )
    print(
        f"Train: {len(X_train)} tin | Validation: {len(X_val)} tin | "
        f"Test: {len(X_test)} tin"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def compare_vectorizer_nb_combos(X_train, y_train, X_val, y_val):
    configs = {
        "Count + Multinomial": make_pipeline(
            CountVectorizer(lowercase=True), MultinomialNB()
        ),
        "TFIDF + Multinomial": make_pipeline(
            TfidfVectorizer(lowercase=True, sublinear_tf=True), MultinomialNB()
        ),
        "TFIDF + Bernoulli": make_pipeline(
            TfidfVectorizer(lowercase=True, sublinear_tf=True), BernoulliNB()
        ),
    }

    rows = []
    for name, pipe in configs.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_val)
        rows.append({
            "model": name,
            "precision": precision_score(y_val, y_pred, pos_label="spam"),
            "recall": recall_score(y_val, y_pred, pos_label="spam"),
            "f1": f1_score(y_val, y_pred, pos_label="spam"),
        })

    comparison_df = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)

    print("\n=== So sánh 3 tổ hợp vectorizer x Naive Bayes (đánh giá trên VALIDATION) ===")
    print(comparison_df.to_string(index=False))
    return comparison_df

def train_model(X_train, y_train):
    pipe = make_pipeline(
        TfidfVectorizer(lowercase=True, min_df=2, max_df=0.9, sublinear_tf=True),
        MultinomialNB(),
    )

    param_grid = {
        "tfidfvectorizer__ngram_range": [(1, 1), (1, 2)],
        "multinomialnb__alpha": [0.01, 0.1, 0.5, 1.0],
    }

    spam_f1_scorer = make_scorer(f1_score, pos_label="spam")

    grid = GridSearchCV(
        pipe, param_grid, scoring=spam_f1_scorer, cv=5, n_jobs=-1
    )
    grid.fit(X_train, y_train)

    print("Tham số tốt nhất:", grid.best_params_)
    print(f"Best CV spam-F1: {grid.best_score_:.4f}")
    return grid.best_estimator_, grid.best_params_


def pick_threshold(model, X_val, y_val, target_precision: float):
    classes = list(model.named_steps["multinomialnb"].classes_)
    spam_index = classes.index("spam")

    y_val_proba = model.predict_proba(X_val)[:, spam_index]
    precisions, recalls, thresholds = precision_recall_curve(
        (y_val == "spam").astype(int), y_val_proba
    )

    valid = np.where(precisions[:-1] >= target_precision)[0]
    if len(valid) > 0:
        idx = valid[0]  
        threshold = thresholds[idx]
        print(
            f"[Validation] Ngưỡng chọn: {threshold:.4f} | "
            f"precision={precisions[idx]:.4f} | recall={recalls[idx]:.4f}"
        )
    else:
        threshold = 0.5
        print(
            f"[Validation] Không đạt được precision >= {target_precision} "
            f"trên validation, dùng ngưỡng mặc định 0.5"
        )

    return threshold, spam_index

def evaluate_on_test(model, X_test, y_test, threshold, spam_index):
    y_test_proba = model.predict_proba(X_test)[:, spam_index]
    y_pred_default = model.predict(X_test)
    y_pred_final = np.where(y_test_proba >= threshold, "spam", "ham")

    print("\n=== Kết quả trên TEST — ngưỡng mặc định 0.5 ===")
    print(classification_report(y_test, y_pred_default, digits=4))

    print(f"=== Kết quả trên TEST — ngưỡng đã chỉnh ({threshold:.4f}) ===")
    print(classification_report(y_test, y_pred_final, digits=4))

    metrics = {
        "threshold": float(threshold),
        "test_precision_spam": float(
            precision_score(y_test, y_pred_final, pos_label="spam")
        ),
        "test_recall_spam": float(
            recall_score(y_test, y_pred_final, pos_label="spam")
        ),
        "test_f1_spam": float(f1_score(y_test, y_pred_final, pos_label="spam")),
    }
    return y_pred_final, metrics

def plot_confusion_matrix(y_test, y_pred_final, out_path):
    cm = confusion_matrix(y_test, y_pred_final, labels=["ham", "spam"])
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(cm, display_labels=["ham", "spam"]).plot(
        ax=ax, cmap="Blues"
    )
    plt.title("Confusion Matrix (ngưỡng đã chỉnh)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Đã lưu {out_path}")

def plot_top_spam_words(model, out_path, top_n=20):
    vectorizer = model.named_steps["tfidfvectorizer"]
    nb_model = model.named_steps["multinomialnb"]

    classes = list(nb_model.classes_)
    spam_index = classes.index("spam")
    ham_index = classes.index("ham")
    feature_names = vectorizer.get_feature_names_out()

    log_ratio = (
        nb_model.feature_log_prob_[spam_index] - nb_model.feature_log_prob_[ham_index]
    )
    top_words = (
        pd.Series(log_ratio, index=feature_names)
        .sort_values(ascending=False)
        .head(top_n)
    )

    print(f"Top {top_n} từ đặc trưng cho spam:")
    print(top_words)

    fig, ax = plt.subplots(figsize=(7, 6))
    top_words.sort_values().plot(kind="barh", ax=ax, color="crimson")
    plt.title(f"Top {top_n} từ đặc trưng cho SPAM")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Đã lưu {out_path}")


def measure_latency(model, X_train, y_train, X_test, n_runs=100):
    """Đo thời gian train + dự đoán 1 tin (yêu cầu < 5ms/tin)."""
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    print(f"Thời gian train: {train_time * 1000:.1f} ms")

    sample = X_test.iloc[[0]]
    model.predict(sample)  # warm-up lần đầu

    t0 = time.perf_counter()
    for _ in range(n_runs):
        model.predict(sample)
    avg_ms = (time.perf_counter() - t0) / n_runs * 1000

    print(
        f"Thời gian dự đoán 1 tin: {avg_ms:.3f} ms",
        "(ĐẠT < 5ms)" if avg_ms < 5 else "(CHƯA ĐẠT)",
    )
    return train_time, avg_ms

def save_artifacts(model, threshold, metrics, comparison_df, out_dir):
    models_dir = os.path.join(out_dir, "models")
    reports_dir = os.path.join(out_dir, "reports")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "nb_pipeline.joblib")
    joblib.dump(model, model_path)
    print(f"Đã lưu model: {model_path}")

    threshold_path = os.path.join(models_dir, "threshold.json")
    with open(threshold_path, "w", encoding="utf-8") as f:
        json.dump(
            {"threshold": float(threshold), "positive_class": "spam"},
            f, ensure_ascii=False, indent=2,
        )
    print(f"Đã lưu ngưỡng: {threshold_path}")

    metrics_path = os.path.join(reports_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu metrics: {metrics_path}")

    comparison_path = os.path.join(reports_dir, "model_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Đã lưu bảng so sánh 3 tổ hợp: {comparison_path}")
    return models_dir, reports_dir

def main():
    args = parse_args()

    df = load_data(args.data)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df, random_state=args.random_state
    )
    comparison_df = compare_vectorizer_nb_combos(X_train, y_train, X_val, y_val)
    best_model, best_params = train_model(X_train, y_train)
    threshold, spam_index = pick_threshold(
        best_model, X_val, y_val, args.target_precision
    )
    y_pred_final, metrics = evaluate_on_test(
        best_model, X_test, y_test, threshold, spam_index
    )
    metrics["best_params"] = {
        k: (list(v) if isinstance(v, tuple) else v) for k, v in best_params.items()
    }
    X_train_final = pd.concat([X_train, X_val])
    y_train_final = pd.concat([y_train, y_val])
    train_time, avg_ms = measure_latency(
        best_model, X_train_final, y_train_final, X_test
    )
    metrics["train_time_sec"] = train_time
    metrics["avg_predict_ms"] = avg_ms

    models_dir, reports_dir = save_artifacts(
        best_model, threshold, metrics, comparison_df, args.out_dir
    )

    plot_confusion_matrix(
        y_test, y_pred_final, os.path.join(reports_dir, "confusion_matrix.png")
    )
    plot_top_spam_words(
        best_model, os.path.join(reports_dir, "top_tu_spam.png")
    )
    print("\nHoàn tất.")

if __name__ == "__main__":
    main()