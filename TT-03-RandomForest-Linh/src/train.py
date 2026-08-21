import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.inspection import permutation_importance

RANDOM_STATE = 42

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
DATA_PATH = os.path.join(BASE_DIR, "dataset", "bank-additional-full.csv")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
MODELS_DIR = os.path.join(BASE_DIR, "models")

for d in [REPORTS_DIR, OUTPUTS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

CAT_COLS = ["job", "marital", "education", "default", "housing", "loan",
            "contact", "month", "day_of_week", "poutcome"]
NUM_COLS = ["age", "campaign", "previous", "pdays_clean", "pdays_contacted_before",
            "emp.var.rate", "cons.price.idx", "cons.conf.idx",
            "euribor3m", "nr.employed"]

def build_pipeline(model, cat_cols=CAT_COLS):
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
                             remainder="passthrough")
    return Pipeline([("pre", pre), ("model", model)])


def make_rf(n_estimators=400):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_features="sqrt",          # nguồn ngẫu nhiên chính của Random Forest
        class_weight="balanced_subsample",
        oob_score=True,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

# Bước 1: Đọc dữ liệu
def load_data():
    df = pd.read_csv(DATA_PATH, sep=";")
    print("Số dòng x cột:", df.shape)
    print(df["y"].value_counts(normalize=True))
    return df

# Bước 2: Demo rò rỉ dữ liệu (duration)
def demo_leakage(df, metrics):
    print("\n=== Demo rò rỉ dữ liệu: duration ===")
    y = (df["y"] == "yes").astype(int)
    num_cols_raw = ["age", "campaign", "pdays", "previous",
                     "emp.var.rate", "cons.price.idx", "cons.conf.idx",
                     "euribor3m", "nr.employed"]

    def quick_auc(feature_cols):
        X = df[feature_cols]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
        pipe = build_pipeline(make_rf(n_estimators=200))
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        return roc_auc_score(y_test, proba)

    auc_with = quick_auc(CAT_COLS + num_cols_raw + ["duration"])
    auc_without = quick_auc(CAT_COLS + num_cols_raw)

    print(f"AUC CÓ duration     : {auc_with:.4f}")
    print(f"AUC KHÔNG có duration: {auc_without:.4f}")
    print("=> duration gây rò rỉ dữ liệu, sẽ bị loại bỏ từ đây trở đi.")

    metrics["auc_with_duration"] = auc_with
    metrics["auc_without_duration"] = auc_without

# Bước 3: Xử lý pdays = 999 ("chưa từng liên hệ")
def clean_pdays(df):
    df = df.copy()
    df["pdays_contacted_before"] = (df["pdays"] != 999).astype(int)
    df["pdays_clean"] = df["pdays"].replace(999, -1)
    return df

# Bước 4: Baseline (Dummy + Decision Tree đơn)
def train_baselines(X_train, X_test, y_train, y_test, metrics):
    print("\n=== Baseline ===")

    dummy = DummyClassifier(strategy="stratified", random_state=RANDOM_STATE)
    dummy.fit(X_train, y_train)
    dummy_auc = roc_auc_score(y_test, dummy.predict_proba(X_test)[:, 1])
    print(f"Dummy AUC: {dummy_auc:.4f}")

    tree = DecisionTreeClassifier(max_depth=6, class_weight="balanced",
                                   random_state=RANDOM_STATE)
    tree_pipe = build_pipeline(tree)
    tree_pipe.fit(X_train, y_train)
    tree_auc = roc_auc_score(y_test, tree_pipe.predict_proba(X_test)[:, 1])
    print(f"Decision Tree đơn AUC: {tree_auc:.4f}")

    metrics["dummy_auc"] = dummy_auc
    metrics["decision_tree_auc"] = tree_auc
    return tree_auc

# Bước 5: Random Forest chính thức + OOB score
def train_random_forest(X_train, X_test, y_train, y_test, tree_auc, metrics):
    print("\n=== Random Forest ===")

    rf_pipe = build_pipeline(make_rf(n_estimators=400))
    rf_pipe.fit(X_train, y_train)

    rf_proba = rf_pipe.predict_proba(X_test)[:, 1]
    rf_auc = roc_auc_score(y_test, rf_proba)
    rf_pr_auc = average_precision_score(y_test, rf_proba)
    oob = rf_pipe.named_steps["model"].oob_score_

    print(f"Random Forest ROC-AUC: {rf_auc:.4f}")
    print(f"Random Forest PR-AUC : {rf_pr_auc:.4f}")
    print(f"OOB score             : {oob:.4f}")
    print(f"Cải thiện so với cây đơn: {rf_auc - tree_auc:+.4f}")

    metrics["random_forest_auc"] = rf_auc
    metrics["random_forest_pr_auc"] = rf_pr_auc
    metrics["oob_score"] = oob
    metrics["improvement_over_single_tree"] = rf_auc - tree_auc

    return rf_pipe, rf_proba

# Bước 6: AUC theo số cây (n_estimators) — tìm điểm bão hòa
def plot_auc_vs_n_estimators(X_train, X_test, y_train, y_test, metrics):
    print("\n=== AUC theo số cây ===")

    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS)],
                             remainder="passthrough")
    X_train_pre = pre.fit_transform(X_train)
    X_test_pre = pre.transform(X_test)

    n_range = [10, 20, 50, 100, 150, 200, 300, 400, 500]
    aucs = []
    for n in n_range:
        rf = make_rf(n_estimators=n)
        rf.fit(X_train_pre, y_train)
        auc = roc_auc_score(y_test, rf.predict_proba(X_test_pre)[:, 1])
        aucs.append(auc)
        print(f"  n_estimators={n:<4} AUC={auc:.4f}")

    plt.figure(figsize=(7, 5))
    plt.plot(n_range, aucs, marker="o")
    plt.xlabel("Số cây (n_estimators)")
    plt.ylabel("ROC-AUC")
    plt.title("AUC theo số cây trong rừng")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(REPORTS_DIR, "auc_theo_so_cay.png"), dpi=150)
    plt.close()

    metrics["best_n_estimators"] = n_range[int(np.argmax(aucs))]
    metrics["best_auc_from_curve"] = max(aucs)

# Bước 7: Feature importance vs permutation
def compare_feature_importance(rf_pipe, X_test, y_test, metrics):
    print("\n=== Feature importance ===")

    ohe = rf_pipe.named_steps["pre"].named_transformers_["cat"]
    feature_names = list(ohe.get_feature_names_out(CAT_COLS)) + NUM_COLS

    df_default = pd.DataFrame({
        "feature": feature_names,
        "importance": rf_pipe.named_steps["model"].feature_importances_,
    }).sort_values("importance", ascending=False).head(15)

    perm = permutation_importance(rf_pipe, X_test, y_test, n_repeats=10,
                                   scoring="average_precision",
                                   random_state=RANDOM_STATE, n_jobs=-1)
    df_perm = pd.DataFrame({
        "feature": X_test.columns,
        "importance": perm.importances_mean,
    }).sort_values("importance", ascending=False).head(15)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].barh(df_default["feature"][::-1], df_default["importance"][::-1])
    axes[0].set_title("feature_importances_ mặc định (thiên lệch)")
    axes[1].barh(df_perm["feature"][::-1], df_perm["importance"][::-1])
    axes[1].set_title("Permutation importance (đáng tin hơn)")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "permutation_importance.png"), dpi=150)
    plt.close()

    print("Top 1 (mặc định)   :", df_default.iloc[0]["feature"])
    print("Top 1 (permutation):", df_perm.iloc[0]["feature"])

    metrics["top_feature_default"] = df_default.iloc[0]["feature"]
    metrics["top_feature_permutation"] = df_perm.iloc[0]["feature"]

# Bước 8: Precision@5000, Lift, Cumulative gain
def evaluate_ranking(y_test, rf_proba, metrics, k=5000):
    print("\n=== Precision@K và Lift ===")

    k = min(k, len(y_test))
    y_test_arr = np.asarray(y_test)
    idx_top_k = np.argsort(rf_proba)[::-1][:k]

    precision_k = y_test_arr[idx_top_k].mean()
    lift_k = precision_k / y_test_arr.mean()

    print(f"Precision@{k}: {precision_k:.4f}")
    print(f"Lift@{k}     : {lift_k:.2f}x so với gọi ngẫu nhiên")

    metrics["precision_at_5000"] = precision_k
    metrics["lift_at_5000"] = lift_k

    # Cumulative gain curve
    order = np.argsort(rf_proba)[::-1]
    y_sorted = y_test_arr[order]
    pct_contacted = np.arange(1, len(y_sorted) + 1) / len(y_sorted)
    pct_captured = np.cumsum(y_sorted) / y_sorted.sum()

    plt.figure(figsize=(7, 5))
    plt.plot(pct_contacted, pct_captured, label="Random Forest")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Gọi ngẫu nhiên")
    plt.xlabel("% khách hàng được gọi")
    plt.ylabel('% khách hàng "yes" thu được')
    plt.title("Cumulative Gain Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(REPORTS_DIR, "lift_curve.png"), dpi=150)
    plt.close()

# Bước 9: Fit trên toàn bộ dữ liệu + xuất danh sách gọi top 5000
def export_call_list(df, X, y):
    print("\n=== Xuất danh sách gọi + lưu model ===")

    final_pipe = build_pipeline(make_rf(n_estimators=400))
    final_pipe.fit(X, y)

    df_out = df.copy()
    df_out["score"] = final_pipe.predict_proba(X)[:, 1]
    df_out = df_out.drop(columns=["duration"], errors="ignore")

    top5000 = df_out.sort_values("score", ascending=False).head(5000)
    top5000.to_csv(os.path.join(OUTPUTS_DIR, "danh_sach_goi_top5000.csv"), index=False)
    joblib.dump(final_pipe, os.path.join(MODELS_DIR, "rf_pipeline.joblib"))

    print("Đã lưu outputs/danh_sach_goi_top5000.csv và models/rf_pipeline.joblib")

# Bước 10: Lưu toàn bộ metric
def save_metrics(metrics):
    with open(os.path.join(REPORTS_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    pd.DataFrame(metrics.items(), columns=["metric", "value"]).to_csv(
        os.path.join(REPORTS_DIR, "metrics.csv"), index=False)

    print("\n=== TÓM TẮT METRIC (đã lưu reports/metrics.json và .csv) ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
        
def main():
    metrics = {}

    df = load_data()
    demo_leakage(df, metrics)
    df = clean_pdays(df)

    X = df[CAT_COLS + NUM_COLS]
    y = (df["y"] == "yes").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    tree_auc = train_baselines(X_train, X_test, y_train, y_test, metrics)
    rf_pipe, rf_proba = train_random_forest(
        X_train, X_test, y_train, y_test, tree_auc, metrics)

    plot_auc_vs_n_estimators(X_train, X_test, y_train, y_test, metrics)
    compare_feature_importance(rf_pipe, X_test, y_test, metrics)
    evaluate_ranking(y_test, rf_proba, metrics)

    export_call_list(df, X, y)
    save_metrics(metrics)


if __name__ == "__main__":
    main()