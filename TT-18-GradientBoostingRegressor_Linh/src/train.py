import os
import time

import joblib
import matplotlib
matplotlib.use("Agg")  # để chạy được cả khi không có màn hình
import matplotlib.pyplot as plt
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from features import add_engineered_features, build_feature_pipeline, load_and_clean

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "train.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
RANDOM_STATE = 42

def rmse(y_true, y_pred):
    # Không dùng squared=False vì tham số này đã bị loại khỏi sklearn >= 1.6
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    # ---- Bước 1-2: load + xử lý NaN --------------------------------------
    print("=" * 60)
    print("BƯỚC 1-2: Load & xử lý giá trị thiếu")
    print("=" * 60)
    df = load_and_clean(DATA_PATH)

    # ---- Bước 3-5: encode + feature engineering (bên trong build_feature_pipeline)
    print("\n" + "=" * 60)
    print("BƯỚC 3-5: Encoding + kiểm tra skew nhãn")
    print("=" * 60)
    X, y_log, encoder = build_feature_pipeline(df, fit=True)
    print(f"Số đặc trưng sau encoding: {X.shape[1]}")
    y_raw = np.expm1(y_log)
    print(f"Skew SalePrice gốc: {y_raw.skew():.3f}")
    print(f"Skew SalePrice sau log1p: {y_log.skew():.3f}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_log, test_size=0.2, random_state=RANDOM_STATE
    )

    # ---- Bước 6: Baseline ---------------------------------------------------
    print("\n" + "=" * 60)
    print("BƯỚC 6: Baseline models")
    print("=" * 60)
    baselines = {
        "Dummy (mean)": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Ridge (alpha=10)": Ridge(alpha=10),
    }
    for name, model in baselines.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        print(f"  {name:20s} RMSE(log) = {rmse(y_val, pred):.4f}")

    # ---- Bước 7: Gradient Boosting + early stopping -------------------------
    print("\n" + "=" * 60)
    print("BƯỚC 7: Gradient Boosting Regressor")
    print("=" * 60)
    gbr = GradientBoostingRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        max_features="sqrt",
        validation_fraction=0.1,
        n_iter_no_change=50,
        random_state=RANDOM_STATE,
    )
    gbr.fit(X_train, y_train)
    pred_val_log = gbr.predict(X_val)
    print(f"  Số cây thực tế dùng (early stopping): {gbr.n_estimators_}")
    print(f"  RMSE(log) trên validation: {rmse(y_val, pred_val_log):.4f}")

    # ---- Bước 8: vẽ train/validation loss theo số cây -----------------------
    print("\n" + "=" * 60)
    print("BƯỚC 8: Biểu đồ train/validation loss")
    print("=" * 60)
    val_score = np.zeros(gbr.n_estimators_, dtype=np.float64)
    for i, y_pred_stage in enumerate(gbr.staged_predict(X_val)):
        val_score[i] = mean_squared_error(y_val, y_pred_stage)

    plt.figure(figsize=(8, 5))
    plt.plot(gbr.train_score_, label="Train loss (MSE, log scale)")
    plt.plot(val_score, label="Validation loss (MSE, log scale)")
    plt.xlabel("Số cây (boosting stage)")
    plt.ylabel("MSE trên thang log")
    plt.title("Train vs Validation loss theo số cây")
    plt.legend()
    plt.tight_layout()
    loss_path = os.path.join(REPORT_DIR, "loss_theo_so_cay.png")
    plt.savefig(loss_path, dpi=120)
    plt.close()
    print(f"  Đã lưu: {loss_path}")

    # ---- Bước 9: Feature engineering — so sánh trước/sau ---------------------
    print("\n" + "=" * 60)
    print("BƯỚC 9: Đánh giá hiệu quả feature engineering")
    print("=" * 60)
    df_raw_features = df.drop(columns=[c for c in ["TotalSF", "TuoiNha", "DaSuaChua"] if c in df.columns])
    X_before, y_log_before, _ = build_feature_pipeline(
        df_raw_features.assign(SalePrice=df["SalePrice"]) if "SalePrice" not in df_raw_features.columns else df_raw_features,
        fit=True,
    )
    # Vô hiệu hoá add_engineered_features tạm thời bằng cách bỏ 3 cột mới nếu lỡ được thêm lại
    for c in ["TotalSF", "TuoiNha", "DaSuaChua", "TotalBath"]:
        if c in X_before.columns:
            X_before = X_before.drop(columns=[c])

    Xb_train, Xb_val, yb_train, yb_val = train_test_split(
        X_before, y_log_before, test_size=0.2, random_state=RANDOM_STATE
    )
    gbr_before = GradientBoostingRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE
    ).fit(Xb_train, yb_train)
    rmse_before = rmse(yb_val, gbr_before.predict(Xb_val))

    gbr_after = GradientBoostingRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE
    ).fit(X_train, y_train)
    rmse_after = rmse(y_val, gbr_after.predict(X_val))

    print(f"  RMSE(log) TRƯỚC feature engineering: {rmse_before:.4f}")
    print(f"  RMSE(log) SAU feature engineering:   {rmse_after:.4f}")
    print(f"  Cải thiện: {(rmse_before - rmse_after) / rmse_before * 100:.2f}%")

    # ---- Bước 10-11: hồi quy phân vị + Median APE -----------------------------
    print("\n" + "=" * 60)
    print("BƯỚC 10-11: Hồi quy phân vị (10/50/90) & Median APE")
    print("=" * 60)
    quantile_models = {}
    for q in [0.1, 0.5, 0.9]:
        m = GradientBoostingRegressor(
            loss="quantile", alpha=q,
            n_estimators=500, learning_rate=0.05, max_depth=3,
            random_state=RANDOM_STATE,
        )
        m.fit(X_train, y_train)
        quantile_models[q] = m

    pred_low = np.expm1(quantile_models[0.1].predict(X_val))
    pred_mid = np.expm1(quantile_models[0.5].predict(X_val))
    pred_high = np.expm1(quantile_models[0.9].predict(X_val))
    y_true = np.expm1(y_val)

    ape = np.abs(pred_mid - y_true) / y_true * 100
    median_ape = np.median(ape)
    coverage = ((y_true >= pred_low) & (y_true <= pred_high)).mean()

    print(f"  Median APE: {median_ape:.2f}%  (mục tiêu < 12%)")
    print(f"  Tỉ lệ phủ thực tế của khoảng 10-90%: {coverage * 100:.1f}%  (kỳ vọng ~80%)")

    plt.figure(figsize=(8, 5))
    idx = np.argsort(y_true.values)
    plt.fill_between(range(len(idx)), pred_low.values[idx] if hasattr(pred_low, "values") else pred_low[idx],
                      pred_high.values[idx] if hasattr(pred_high, "values") else pred_high[idx],
                      alpha=0.3, label="Khoảng dự báo 10-90%")
    plt.plot(range(len(idx)), np.array(y_true)[idx], "k.", markersize=3, label="Giá thật")
    plt.xlabel("Căn nhà (sắp xếp theo giá thật)")
    plt.ylabel("Giá (USD)")
    plt.title("Khoảng dự báo giá 10-90% vs giá thật")
    plt.legend()
    plt.tight_layout()
    khoang_gia_path = os.path.join(REPORT_DIR, "khoang_gia.png")
    plt.savefig(khoang_gia_path, dpi=120)
    plt.close()
    print(f"  Đã lưu: {khoang_gia_path}")

    plt.figure(figsize=(8, 5))
    plt.hist(ape, bins=40)
    plt.axvline(median_ape, color="red", linestyle="--", label=f"Median APE = {median_ape:.1f}%")
    plt.xlabel("APE (%)")
    plt.ylabel("Số lượng căn nhà")
    plt.title("Phân phối sai số phần trăm tuyệt đối (APE)")
    plt.legend()
    plt.tight_layout()
    ape_path = os.path.join(REPORT_DIR, "ape_distribution.png")
    plt.savefig(ape_path, dpi=120)
    plt.close()
    print(f"  Đã lưu: {ape_path}")

    # ---- Bước 12: human-in-the-loop --------------------------------------
    print("\n" + "=" * 60)
    print("BƯỚC 12: Cơ chế human-in-the-loop")
    print("=" * 60)
    width_pct = (pred_high - pred_low) / pred_mid * 100
    auto_approved = width_pct <= 25
    print(f"  Tự động duyệt (khoảng hẹp <=25%): {auto_approved.mean() * 100:.1f}%")
    print(f"  Chuyển thẩm định viên (khoảng rộng >25%): {(~auto_approved).mean() * 100:.1f}%")

    # ---- Bước 13: so sánh tốc độ train ------------------------------------
    print("\n" + "=" * 60)
    print("BƯỚC 13: So sánh thời gian train GBR vs HistGBR")
    print("=" * 60)
    speed_models = {
        "GradientBoostingRegressor": GradientBoostingRegressor(
            n_estimators=500, max_depth=3, random_state=RANDOM_STATE
        ),
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(
            max_iter=500, max_depth=3, random_state=RANDOM_STATE
        ),
    }
    for name, model in speed_models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0
        print(f"  {name:32s}: {elapsed:.2f}s")

    # ---- Lưu model chính (GBR mean, dùng cho dự đoán điểm) ------------------
    model_path = os.path.join(MODEL_DIR, "gbr_pipeline.joblib")
    joblib.dump(
        {
            "encoder": encoder,
            "model_mean": gbr,
            "quantile_models": quantile_models,
            "feature_columns": X.columns.tolist(),
        },
        model_path,
    )
    print(f"\nĐã lưu pipeline đầy đủ (encoder + models) tại: {model_path}")

if __name__ == "__main__":
    main()