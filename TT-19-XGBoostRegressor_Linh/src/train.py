"""
train.py
--------
Pipeline huấn luyện đầy đủ cho TT-19 - XGBoost Regressor (Bike Sharing).

Chạy:
    python src/train.py --data data/hour.csv

Kết quả:
    - models/xgb_bike.json
    - reports/cnt_theo_gio.png
    - reports/du_bao_vs_thuc_te.png
    - reports/shap_summary.png
    - reports/phan_tich_loi.png
    - In ra console: kết quả chứng minh rò rỉ, so sánh baseline vs XGBoost,
      so sánh log1p vs không log1p, feature importance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # để chạy được cả khi không có màn hình (server/CI)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

from features import (
    add_cyclical_features,
    drop_leaky_columns,
    evaluate,
    load_data,
    naive_weekly_baseline,
    prove_leakage,
    time_based_split,
)

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    drop_cols = {"cnt", "dteday", "instant"}
    return [c for c in df.columns if c not in drop_cols]


def train_xgb(X_train, y_train, X_val, y_val, **overrides) -> XGBRegressor:
    params = dict(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.1,
        min_child_weight=3,
        objective="reg:squarederror",
        early_stopping_rounds=100,
        eval_metric="rmse",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )
    params.update(overrides)
    model = XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=200)
    return model


def main(data_path: str):
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    # ---------------------------------------------------------------- #
    # Bước 1 & 2: nạp dữ liệu + CHỨNG MINH RÒ RỈ
    # ---------------------------------------------------------------- #
    df_raw = load_data(data_path)

    print("\n=== BƯỚC 2: CHỨNG MINH RÒ RỈ (casual + registered) ===")
    leak_result = prove_leakage(df_raw)
    print(json.dumps(leak_result, indent=2, ensure_ascii=False))
    (REPORTS_DIR / "leakage_proof.json").write_text(
        json.dumps(leak_result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    df = drop_leaky_columns(df_raw)

    # ---------------------------------------------------------------- #
    # Bước 4: EDA nhanh — cnt trung bình theo giờ
    # ---------------------------------------------------------------- #
    hourly_avg = df.groupby("hr")["cnt"].mean()
    plt.figure(figsize=(8, 4))
    hourly_avg.plot(kind="bar", color="#4C72B0")
    plt.title("Lượt thuê trung bình theo giờ trong ngày")
    plt.xlabel("Giờ")
    plt.ylabel("cnt trung bình")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "cnt_theo_gio.png", dpi=150)
    plt.close()

    # ---------------------------------------------------------------- #
    # Bước 3: chia theo thời gian
    # ---------------------------------------------------------------- #
    train_df, val_df, test_df = time_based_split(df)
    print(f"\nKích thước: train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    # ---------------------------------------------------------------- #
    # Bước 5: mã hoá chu kỳ
    # ---------------------------------------------------------------- #
    train_df = add_cyclical_features(train_df)
    val_df = add_cyclical_features(val_df)
    test_df = add_cyclical_features(test_df)

    feature_cols = get_feature_columns(train_df)
    X_train, y_train = train_df[feature_cols], train_df["cnt"]
    X_val, y_val = val_df[feature_cols], val_df["cnt"]
    X_test, y_test = test_df[feature_cols], test_df["cnt"]

    # ---------------------------------------------------------------- #
    # Bước 6: baseline naive "cùng giờ tuần trước"
    # (tính trên toàn bộ df đã sort thời gian rồi mới cắt ra phần test)
    # ---------------------------------------------------------------- #
    df_with_lag = df.copy()
    df_with_lag["baseline_pred"] = naive_weekly_baseline(df_with_lag)
    baseline_test = df_with_lag.loc[test_df.index, "baseline_pred"]
    valid_mask = baseline_test.notna()

    print("\n=== BƯỚC 6: BASELINE NAIVE vs XGBOOST ===")
    baseline_metrics = evaluate(
        y_test[valid_mask], baseline_test[valid_mask], label="Baseline naive (tuần trước)"
    )

    # ---------------------------------------------------------------- #
    # Bước 7 & 8: train XGBoost, so sánh có/không log1p
    # ---------------------------------------------------------------- #
    print("\n=== BƯỚC 7: XGBoost (không log1p) ===")
    model_plain = train_xgb(X_train, y_train, X_val, y_val)
    pred_plain = np.clip(model_plain.predict(X_test), 0, None)
    metrics_plain = evaluate(y_test, pred_plain, label="XGBoost (raw cnt)")

    print("\n=== BƯỚC 8: XGBoost (log1p) ===")
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)
    model_log = train_xgb(X_train, y_train_log, X_val, y_val_log)
    pred_log = np.clip(np.expm1(model_log.predict(X_test)), 0, None)
    metrics_log = evaluate(y_test, pred_log, label="XGBoost (log1p cnt)")

    # chọn model tốt hơn theo RMSE trên test
    best_model, best_pred, best_label = (
        (model_log, pred_log, "log1p")
        if metrics_log["rmse"] < metrics_plain["rmse"]
        else (model_plain, pred_plain, "raw")
    )
    print(f"\n>>> Model được chọn: {best_label} (RMSE thấp hơn trên test)")

    if metrics_plain["rmse"] < baseline_metrics["rmse"] or metrics_log["rmse"] < baseline_metrics["rmse"]:
        print(">>> XGBoost THẮNG baseline naive.")
    else:
        print(">>> CẢNH BÁO: XGBoost CHƯA thắng được baseline naive — cần xem lại feature/tuning.")

    best_model.save_model(MODELS_DIR / "xgb_bike.json")
    print(f"\nSố cây thực tế dùng (early stopping): {best_model.best_iteration}")

    # ---------------------------------------------------------------- #
    # Bước 9: RandomizedSearchCV với TimeSeriesSplit (tuỳ chọn, tốn thời gian)
    # ---------------------------------------------------------------- #
   
    param_dist = {
       "max_depth": [3, 4, 5, 6, 8],
       "learning_rate": [0.01, 0.03, 0.05, 0.1],
       "subsample": [0.6, 0.8, 1.0],
       "colsample_bytree": [0.6, 0.8, 1.0],
       "min_child_weight": [1, 3, 5, 7],
    }
    search = RandomizedSearchCV(
        XGBRegressor(n_estimators=500, tree_method="hist", random_state=42),
        param_distributions=param_dist,
        n_iter=25,
        cv=TimeSeriesSplit(n_splits=5),
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print(search.best_params_)

    # ---------------------------------------------------------------- #
    # Bước 10: feature importance + SHAP
    # ---------------------------------------------------------------- #
    importance = pd.Series(
        best_model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)
    print("\n=== Feature importance (gain, top 10) ===")
    print(importance.head(10))

    try:
        import shap

        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_test)
        plt.figure()
        shap.summary_plot(shap_values, X_test, show=False)
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "shap_summary.png", dpi=150)
        plt.close()
    except ImportError:
        print("(shap chưa được cài — bỏ qua bước vẽ SHAP summary)")

    # ---------------------------------------------------------------- #
    # Bước 11: dự báo vs thực tế trên 2 tuần cuối
    # ---------------------------------------------------------------- #
    last_2w = test_df.tail(24 * 14)
    idx_2w = last_2w.index
    pred_2w = pd.Series(best_pred, index=test_df.index).loc[idx_2w]

    plt.figure(figsize=(12, 4))
    plt.plot(last_2w["dteday"].astype(str) + " " + last_2w["hr"].astype(str) + "h",
              last_2w["cnt"].values, label="Thực tế", linewidth=1)
    plt.plot(last_2w["dteday"].astype(str) + " " + last_2w["hr"].astype(str) + "h",
              pred_2w.values, label="Dự báo", linewidth=1)
    plt.xticks([]) 
    plt.legend()
    plt.title("Dự báo vs thực tế — 2 tuần cuối tập test")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "du_bao_vs_thuc_te.png", dpi=150)
    plt.close()

    # ---------------------------------------------------------------- #
    # Bước 12: phân tích lỗi theo giờ / thời tiết
    # ---------------------------------------------------------------- #
    err_df = test_df.copy()
    err_df["pred"] = best_pred
    err_df["abs_err"] = (err_df["cnt"] - err_df["pred"]).abs()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    err_df.groupby("hr")["abs_err"].mean().plot(kind="bar", ax=axes[0], color="#DD8452")
    axes[0].set_title("Lỗi tuyệt đối trung bình theo giờ")
    err_df.groupby("weathersit")["abs_err"].mean().plot(kind="bar", ax=axes[1], color="#55A868")
    axes[1].set_title("Lỗi tuyệt đối trung bình theo thời tiết")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "phan_tich_loi.png", dpi=150)
    plt.close()

    print("\nHoàn tất. Xem kết quả trong thư mục reports/ và models/.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=str(ROOT / "dataset" / "hour.csv"))
    args = parser.parse_args()
    main(args.dataset)