import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_recall_curve, roc_curve
)
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

RANDOM_STATE = 42
COST_CHAN_NHAM = 200_000         
PRECISION_TARGET = 0.90
DATA_PATH = r"dataset/creditcard.csv"

# 1-2. NẠP DỮ LIỆU + FEATURE ENGINEERING TỐI THIỂU
def load_and_prepare(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    fraud_rate = df["Class"].mean()
    print(f"[1] Tổng giao dịch: {len(df):,} | Tỉ lệ gian lận: {fraud_rate:.4%} "
          f"({df['Class'].sum()} ca)")
    assert 0.001 < fraud_rate < 0.003, "Tỉ lệ lệch khác kỳ vọng — kiểm tra lại dữ liệu."

    # Time (giây từ giao dịch đầu tiên) -> giờ trong ngày [0-23]
    df["Hour"] = (df["Time"] // 3600) % 24
    df["Amount_log"] = np.log1p(df["Amount"])
    scaler = StandardScaler()
    df["Amount_scaled"] = scaler.fit_transform(df[["Amount_log"]])

    print("[2] Đã tạo Hour (giờ trong ngày) và Amount_scaled (log1p + StandardScaler)")
    return df

# 3. EDA 
def eda(df: pd.DataFrame, out_dir: str = "reports"):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    fraud_by_hour = df.groupby("Hour")["Class"].mean()
    axes[0].bar(fraud_by_hour.index, fraud_by_hour.values, color="crimson")
    axes[0].set_title("Tỉ lệ gian lận theo giờ trong ngày")
    axes[0].set_xlabel("Giờ"); axes[0].set_ylabel("Tỉ lệ gian lận")

    axes[1].hist(df.loc[df.Class == 0, "Amount_log"], bins=50, alpha=0.6,
                 label="Thật (0)", density=True)
    axes[1].hist(df.loc[df.Class == 1, "Amount_log"], bins=50, alpha=0.6,
                 label="Gian lận (1)", density=True)
    axes[1].set_title("Phân phối log(Amount) theo lớp")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{out_dir}/eda_overview.png", dpi=120)
    plt.close()
    print("[3] Đã lưu reports/eda_overview.png")

# 4. CHIA DỮ LIỆU THEO THỜI GIAN (KHÔNG SHUFFLE)
def time_based_split(df: pd.DataFrame):
    df = df.sort_values("Time").reset_index(drop=True)
    n = len(df)
    i_train = int(n * 0.70)
    i_val = int(n * 0.85)

    feature_cols = [c for c in df.columns if c.startswith("V")] + ["Hour", "Amount_scaled"]

    train, val, test = df.iloc[:i_train], df.iloc[i_train:i_val], df.iloc[i_val:]
    print(f"[4] Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,} "
          f"(chia theo Time, không shuffle)")

    X_train, y_train = train[feature_cols], train["Class"]
    X_val, y_val = val[feature_cols], val["Class"]
    X_test, y_test = test[feature_cols], test["Class"]
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols

# 5-7. BASELINE -> LOGISTIC REGRESSION -> XGBOOST
def evaluate(name, model, X_test, y_test, results: dict, predict_fn=None):
    proba = predict_fn(X_test) if predict_fn else model.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, proba)
    roc_auc = roc_auc_score(y_test, proba)
    results[name] = {"pr_auc": pr_auc, "roc_auc": roc_auc, "proba": proba}
    print(f"    {name:22s} | PR-AUC = {pr_auc:.4f} | ROC-AUC = {roc_auc:.4f}")
    return proba

def train_all_models(X_train, y_train, X_val, y_val, X_test, y_test):
    results = {}
    print("\n[5-7] Huấn luyện và so sánh các model:")

    # 5. Baseline ngẫu nhiên
    dummy = DummyClassifier(strategy="stratified", random_state=RANDOM_STATE)
    dummy.fit(X_train, y_train)
    evaluate("Dummy (baseline)", dummy, X_test, y_test, results)

    # 6. Logistic Regression — baseline mạnh, có xử lý lệch bằng class_weight
    logreg = LogisticRegression(class_weight="balanced", max_iter=1000,
                                 random_state=RANDOM_STATE)
    logreg.fit(X_train, y_train)
    evaluate("Logistic Regression", logreg, X_test, y_test, results)

    # 7. XGBoost — model chính
    ty_le = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"    scale_pos_weight = {ty_le:.1f}")

    xgb_model = xgb.XGBClassifier(
        n_estimators=1000, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=ty_le,
        reg_lambda=1.0, reg_alpha=0.1,
        eval_metric="aucpr",           # PR-AUC, KHÔNG dùng 'auc'
        early_stopping_rounds=50,
        tree_method="hist", n_jobs=-1, random_state=RANDOM_STATE,
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    evaluate("XGBoost", xgb_model, X_test, y_test, results)

    # 12. So sánh với Random Forest và LightGBM
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, class_weight="balanced",
        n_jobs=-1, random_state=RANDOM_STATE
    )
    rf.fit(X_train, y_train)
    evaluate("Random Forest", rf, X_test, y_test, results)

    lgb_model = lgb.LGBMClassifier(
            n_estimators=1000, learning_rate=0.05, max_depth=4,
            scale_pos_weight=ty_le, random_state=RANDOM_STATE, verbose=-1
        )
    lgb_model.fit(X_train, y_train)
    evaluate("LightGBM", lgb_model, X_test, y_test, results)
    return xgb_model, results
# 8. VÌ SAO ROC-AUC ĐÁNH LỪA — so sánh 2 đường cong
def plot_pr_vs_roc(results: dict, y_test, out_dir: str = "reports"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, r in results.items():
        prec, rec, _ = precision_recall_curve(y_test, r["proba"])
        axes[0].plot(rec, prec, label=f"{name} (AP={r['pr_auc']:.3f})")
        fpr, tpr, _ = roc_curve(y_test, r["proba"])
        axes[1].plot(fpr, tpr, label=f"{name} (AUC={r['roc_auc']:.3f})")

    axes[0].set_xlabel("Recall"); axes[0].set_ylabel("Precision")
    axes[0].set_title("Precision-Recall Curve (metric THẬT)")
    axes[0].legend(fontsize=8)

    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve (metric ẢO ở dữ liệu lệch)")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/pr_vs_roc.png", dpi=120)
    plt.close()
    print("    Đã lưu reports/pr_vs_roc.png")

# 9-10. NGƯỠNG THEO PRECISION VÀ NGƯỠNG THEO CHI PHÍ
def choose_thresholds(y_test, proba, amounts, out_dir: str = "reports"):
    prec, rec, thr = precision_recall_curve(y_test, proba)

    # 9. Ngưỡng đạt Precision >= 0.90
    idx = np.argmax(prec[:-1] >= PRECISION_TARGET)  # điểm đầu tiên đạt precision mong muốn
    thr_precision = thr[idx] if prec[idx] >= PRECISION_TARGET else None
    print(f"\n[9] Ngưỡng để Precision >= {PRECISION_TARGET}: "
          f"{thr_precision if thr_precision is not None else 'không đạt được ở bất kỳ ngưỡng nào'} "
          f"(Recall tương ứng = {rec[idx]:.3f})")

    # 10. Ngưỡng tối ưu lợi nhuận
    # Chi phí = FP * 200.000đ + FN * (số tiền giao dịch bị bỏ lọt)
    thresholds_grid = np.linspace(0.01, 0.99, 99)
    costs = []
    for t in thresholds_grid:
        pred = (proba >= t).astype(int)
        fp_mask = (pred == 1) & (y_test.values == 0)
        fn_mask = (pred == 0) & (y_test.values == 1)
        cost = fp_mask.sum() * COST_CHAN_NHAM + amounts[fn_mask].sum()
        costs.append(cost)
    costs = np.array(costs)
    best_idx = costs.argmin()
    best_threshold = thresholds_grid[best_idx]

    print(f"[10] Ngưỡng tối ưu lợi nhuận = {best_threshold:.2f} "
          f"| Tổng chi phí ước tính = {costs[best_idx]:,.0f}đ")

    plt.figure(figsize=(6, 4))
    plt.plot(thresholds_grid, costs)
    plt.axvline(best_threshold, color="red", linestyle="--",
                label=f"Ngưỡng tối ưu = {best_threshold:.2f}")
    plt.xlabel("Ngưỡng phân loại"); plt.ylabel("Tổng chi phí (đ)")
    plt.title("Chi phí theo ngưỡng (chặn nhầm 200.000đ vs bỏ lọt = số tiền giao dịch)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/chi_phi_theo_nguong.png", dpi=120)
    plt.close()
    print("     Đã lưu reports/chi_phi_theo_nguong.png")

    return thr_precision, best_threshold

# 11. ĐO THỜI GIAN DỰ ĐOÁN 1 GIAO DỊCH
def measure_latency(model, X_test, n_runs: int = 200):
    sample = X_test.iloc[[0]]
    # warm-up
    for _ in range(5):
        model.predict_proba(sample)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.predict_proba(sample)
        times.append((time.perf_counter() - t0) * 1000)  # ms

    p50, p99 = np.percentile(times, [50, 99])
    print(f"\n[11] Độ trễ dự đoán 1 giao dịch: p50 = {p50:.2f} ms | p99 = {p99:.2f} ms "
          f"(yêu cầu < 100ms -> {'ĐẠT' if p99 < 100 else 'KHÔNG ĐẠT'})")
    return p50, p99

# FEATURE IMPORTANCE
def plot_feature_importance(model, feature_cols, out_dir: str = "reports"):
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:15]
    plt.figure(figsize=(7, 5))
    plt.barh([feature_cols[i] for i in order][::-1], importances[order][::-1])
    plt.title("Feature Importance (XGBoost, top 15)")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/feature_importance.png", dpi=120)
    plt.close()
    print("     Đã lưu reports/feature_importance.png")

# MAIN
def main():
    df = load_and_prepare(DATA_PATH)
    eda(df)
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = time_based_split(df)

    xgb_model, results = train_all_models(X_train, y_train, X_val, y_val, X_test, y_test)
    plot_pr_vs_roc(results, y_test)
    print(f"\n[8] Kết luận: ROC-AUC ~{results['XGBoost']['roc_auc']:.2f} 'đẹp giả tạo', "
          f"PR-AUC ~{results['XGBoost']['pr_auc']:.2f} mới là năng lực THẬT của model "
          f"ở mức lệch 0,172%.")

    test_df = df.sort_values("Time").iloc[len(df) - len(X_test):]
    amounts = test_df["Amount"].values

    choose_thresholds(y_test, results["XGBoost"]["proba"], amounts)
    measure_latency(xgb_model, X_test)
    plot_feature_importance(xgb_model, feature_cols)

    xgb_model.save_model("models/xgb_fraud.json")
    print("\nĐã lưu model tại models/xgb_fraud.json")
    print("\n=== HOÀN TẤT — xem toàn bộ biểu đồ trong thư mục reports/ ===")


if __name__ == "__main__":
    main()