import os
import json
import time
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # không cần GUI backend -> chạy được cả khi không có màn hình
import matplotlib.pyplot as plt
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    GradientBoostingClassifier, HistGradientBoostingClassifier,
    RandomForestClassifier, AdaBoostClassifier,
)
from sklearn.metrics import average_precision_score, roc_auc_score, log_loss
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
EXPECTED_TOTAL_ROWS = 48_842

COLS = ["age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country",
        "income"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train income classifier (UCI Adult).")
    parser.add_argument(
        "--data-dir", default="dataset"
    )
    parser.add_argument(
        "--output-dir", default="."
    )
    return parser.parse_args()

def clean_raw(df):
    df = df.copy()
    str_cols = df.select_dtypes(include="object").columns
    for c in str_cols:
        df[c] = df[c].str.strip()
    df = df.replace("?", np.nan)
    df = df.drop(columns=["education", "fnlwgt"])
    df["income"] = df["income"].str.replace(".", "", regex=False)
    return df

def load_data(data_dir):
    train_path = os.path.join(data_dir, "adult.data")
    test_path = os.path.join(data_dir, "adult.test")

    train_df = pd.read_csv(train_path, names=COLS, sep=r",\s*",
                            engine="python", na_values="?")
    test_df = pd.read_csv(test_path, names=COLS, sep=r",\s*",
                           engine="python", na_values="?", skiprows=1)

    train_df, test_df = clean_raw(train_df), clean_raw(test_df)

    total_rows = len(train_df) + len(test_df)
    if total_rows != EXPECTED_TOTAL_ROWS:
        warnings.warn(
            f"Số dòng sau khi đọc ({total_rows}) khác với UCI Adult chuẩn "
            f"({EXPECTED_TOTAL_ROWS}). Kiểm tra lại file dữ liệu ở {data_dir}."
        )
    else:
        print(f"OK: tổng số dòng = {total_rows} (khớp UCI Adult chuẩn).")

    print("Train:", train_df.shape, "| Test:", test_df.shape)
    return train_df, test_df

def build_pipeline(X, model):
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = X.select_dtypes(exclude="object").columns.tolist()
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", "passthrough", num_cols),
    ])
    return Pipeline([("pre", pre), ("model", model)])

def plot_eda(train_df, reports_dir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    train_df.groupby("education-num")["income"].apply(lambda s: (s == ">50K").mean()).plot(
        kind="bar", ax=axes[0], title="Tỉ lệ >50K theo education-num", color="#4C72B0")

    train_df.assign(hours_bin=pd.cut(train_df["hours-per-week"], [0, 20, 40, 60, 100])).groupby(
        "hours_bin", observed=True)["income"].apply(lambda s: (s == ">50K").mean()).plot(
        kind="bar", ax=axes[1], title="Tỉ lệ >50K theo giờ làm/tuần", color="#DD8452")

    train_df.groupby("marital-status")["income"].apply(lambda s: (s == ">50K").mean()).plot(
        kind="bar", ax=axes[2], title="Tỉ lệ >50K theo hôn nhân", color="#55A868")

def evaluate(pipe, X, y):
    proba = pipe.predict_proba(X)[:, 1]
    return {
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
    }

def check_bias(pipe, X, y, col):
    proba = pipe.predict_proba(X)[:, 1]
    pred = pipe.predict(X)
    out = []
    for g in X[col].dropna().unique():
        mask = X[col] == g
        if mask.sum() < 20:
            continue
        out.append({
            "group": g, "n": int(mask.sum()),
            "predicted_positive_rate": float(pred[mask].mean()),
            "actual_positive_rate": float(y[mask].mean()),
            "avg_predicted_probability": float(proba[mask].mean()),
        })
    return pd.DataFrame(out)

def main():
    args = parse_args()
    models_dir = os.path.join(args.output_dir, "models")
    reports_dir = os.path.join(args.output_dir, "reports")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # ---------------- Load & EDA ----------------
    train_df, test_df = load_data(args.data_dir)
    plot_eda(train_df, reports_dir)

    y_train_full = (train_df["income"] == ">50K").astype(int)
    X_train_full = train_df.drop(columns=["income"])
    y_test = (test_df["income"] == ">50K").astype(int)
    X_test = test_df.drop(columns=["income"])

    # ---------------- Validation split (KHÔNG đụng test set) ----------------
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.15,
        stratify=y_train_full, random_state=RANDOM_STATE,
    )

    # ---------------- Baselines (đánh giá trên validation) ----------------
    baselines = {
        "Dummy (most_frequent)": DummyClassifier(strategy="most_frequent"),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
    }
    baseline_rows = []
    for name, model in baselines.items():
        pipe = build_pipeline(X_train, model)
        pipe.fit(X_train, y_train)
        metrics = evaluate(pipe, X_val, y_val)
        baseline_rows.append({"model": name, **metrics})
    print(pd.DataFrame(baseline_rows))

    # ---------------- Gradient Boosting chính + early stopping curve ----------------
    gb = GradientBoostingClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=3,
        subsample=0.8, validation_fraction=0.1, n_iter_no_change=20,
        random_state=RANDOM_STATE,
    )
    gb_pipe = build_pipeline(X_train, gb)
    gb_pipe.fit(X_train, y_train)

    pre = gb_pipe.named_steps["pre"]
    model = gb_pipe.named_steps["model"]
    Xtr_enc, Xval_enc = pre.transform(X_train), pre.transform(X_val)

    train_loss = [log_loss(y_train, p) for p in model.staged_predict_proba(Xtr_enc)]
    val_loss = [log_loss(y_val, p) for p in model.staged_predict_proba(Xval_enc)]
    best_iter = int(np.argmin(val_loss))

    fig = plt.figure(figsize=(7, 4.5))
    plt.plot(train_loss, label="Train loss")
    plt.plot(val_loss, label="Validation loss")
    plt.axvline(best_iter, color="red", ls="--", label=f"Tốt nhất tại cây #{best_iter}")
    plt.xlabel("Số cây"); plt.ylabel("Log loss"); plt.legend()
    plt.title("Overfit xuất hiện khi validation loss bắt đầu tăng trở lại")
    plt.savefig(os.path.join(reports_dir, "loss_theo_so_cay.png"), dpi=120)
    plt.close(fig)
    print(f"Sau cây #{best_iter}, train loss tiếp tục giảm nhưng validation loss tăng -> overfit.")

    # ---------------- Grid tuning lr x n_estimators (đánh giá trên validation) ----------------
    lrs, n_ests = [0.3, 0.1, 0.05], [50, 200, 500]
    grid = np.zeros((len(lrs), len(n_ests)))
    for i, lr in enumerate(lrs):
        for j, n in enumerate(n_ests):
            m = GradientBoostingClassifier(n_estimators=n, learning_rate=lr, max_depth=3,
                                            random_state=RANDOM_STATE)
            p = build_pipeline(X_train, m)
            p.fit(X_train, y_train)
            grid[i, j] = average_precision_score(y_val, p.predict_proba(X_val)[:, 1])

    fig = plt.figure(figsize=(6, 5))
    plt.imshow(grid, cmap="YlGnBu")
    plt.xticks(range(len(n_ests)), n_ests); plt.yticks(range(len(lrs)), lrs)
    plt.xlabel("n_estimators"); plt.ylabel("learning_rate")
    for i in range(len(lrs)):
        for j in range(len(n_ests)):
            plt.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center")
    plt.title("PR-AUC (validation): lr nhỏ + nhiều cây thường tốt/ổn định hơn")
    plt.colorbar()
    plt.savefig(os.path.join(reports_dir, "lr_vs_nestimators.png"), dpi=120)
    plt.close(fig)

    # ---------------- So sánh model family (đánh giá trên validation) ----------------
    compare = {
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": gb,
        "AdaBoost": AdaBoostClassifier(n_estimators=300, learning_rate=0.5, random_state=RANDOM_STATE),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.05, early_stopping=True, random_state=RANDOM_STATE),
    }
    compare_rows = []
    for name, m in compare.items():
        pipe = build_pipeline(X_train, m)
        t0 = time.time()
        pipe.fit(X_train, y_train)
        dt = time.time() - t0
        metrics = evaluate(pipe, X_val, y_val)
        compare_rows.append({"model": name, **metrics, "train_time_s": dt})
    comparison_df = pd.DataFrame(compare_rows).sort_values("pr_auc", ascending=False)
    comparison_df.to_csv(os.path.join(reports_dir, "model_comparison.csv"), index=False)
    print(comparison_df)

    # ---------------- Refit model cuối cùng trên toàn bộ train, đánh giá 1 LẦN trên test ----------------
    final_gb_pipe = build_pipeline(X_train_full, gb)
    final_gb_pipe.fit(X_train_full, y_train_full)
    final_metrics = evaluate(final_gb_pipe, X_test, y_test)
    print("Final (test set, chỉ đánh giá 1 lần) ->", final_metrics)
    with open(os.path.join(reports_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, ensure_ascii=False, indent=2)

    # ---------------- Bias / fairness report (trên test set) ----------------
    bias_sex = check_bias(final_gb_pipe, X_test, y_test, "sex")
    bias_race = check_bias(final_gb_pipe, X_test, y_test, "race")
    print(bias_sex)
    print(bias_race)

    with open(os.path.join(reports_dir, "bias_report.json"), "w", encoding="utf-8") as f:
        json.dump({
            "sex": bias_sex.to_dict(orient="records"),
            "race": bias_race.to_dict(orient="records"),
        }, f, ensure_ascii=False, indent=2)

    fig = plt.figure(figsize=(5, 4))
    plt.bar(bias_sex["group"], bias_sex["predicted_positive_rate"], color=["#4C72B0", "#DD8452"])
    plt.ylabel("Tỉ lệ dự đoán income > 50K")
    plt.title("Thiên lệch dự đoán theo giới tính")
    plt.savefig(os.path.join(reports_dir, "bias_by_group.png"), dpi=120)
    plt.close(fig)

    # ---------------- Lưu model ----------------
    model_path = os.path.join(models_dir, "gb_pipeline.joblib")
    joblib.dump(final_gb_pipe, model_path)
    print(f"\nĐã lưu model tại: {model_path}")


if __name__ == "__main__":
    main()