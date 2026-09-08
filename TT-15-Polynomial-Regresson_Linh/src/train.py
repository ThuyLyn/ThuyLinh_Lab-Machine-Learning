import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, validation_curve
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "Folds5x2_pp.xlsx")
REPORTS = os.path.join(os.path.dirname(__file__), "..", "reports")
MODELS = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(REPORTS, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)

def load_data():
    return pd.read_excel(DATA_PATH)

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def header(step_no, title):
    print(f"\n{'='*70}\nBƯỚC {step_no}. {title}\n{'='*70}")

def main():
    df = load_data()
    print(f"Đã đọc {len(df)} dòng từ {DATA_PATH}")
    X = df[["AT", "V", "AP", "RH"]]
    y = df["PE"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Tương quan AT-V: {X_train['AT'].corr(X_train['V']):.3f}")

    # ================================================================
    # BƯỚC 1. EDA: scatter AT vs PE → NHÌN THẤY đường cong bằng mắt
    # ================================================================
    header(1, "EDA — Scatter AT vs PE")
    plt.figure(figsize=(6, 4))
    plt.scatter(df["AT"], df["PE"], alpha=0.25, s=6, color="#2E5EAA")
    plt.xlabel("AT — Nhiệt độ (°C)")
    plt.ylabel("PE — Công suất phát (MW)")
    plt.title("Quan hệ AT ↔ PE: rõ ràng là ĐƯỜNG CONG, không phải đường thẳng")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS, "scatter_AT_PE.png"), dpi=120)
    plt.close()
    print("Đã lưu reports/scatter_AT_PE.png")

    # ================================================================
    # BƯỚC 2. Baseline: Linear Regression bậc 1 → ghi RMSE
    # ================================================================
    header(2, "Baseline — Linear Regression bậc 1")
    lr1 = LinearRegression().fit(X_train, y_train)
    pred1_test = lr1.predict(X_test)
    rmse_baseline = rmse(y_test, pred1_test)
    print(f"RMSE test (bậc 1) = {rmse_baseline:.3f} MW")

    # ================================================================
    # BƯỚC 3. Vẽ RESIDUAL PLOT của model bậc 1
    #          → phần dư có hình chữ U không? → bằng chứng thiếu bậc phi tuyến
    # ================================================================
    header(3, "Residual plot của model bậc 1 (TRƯỚC)")
    residuals_before = y_test - pred1_test
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(X_test["AT"], residuals_before, alpha=0.25, s=6, color="#C0392B")
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_title("TRƯỚC — Bậc 1 (Linear)")
    axes[0].set_xlabel("AT — Nhiệt độ (°C)")
    axes[0].set_ylabel("Residual (y_thực - y_dự đoán)")
    print("Residual plot TRƯỚC đã vẽ (sẽ lưu chung ảnh với bước 8 ở cuối)")

    # ================================================================
    # BƯỚC 4. Chạy bậc 1 → 5, ghi RMSE train và validation
    # ================================================================
    header(4, "Chạy bậc 1→5, ghi RMSE train và validation")
    pipe = Pipeline(
        [
            ("poly", PolynomialFeatures(include_bias=False)),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )
    degrees = [1, 2, 3, 4, 5]
    train_scores, val_scores = validation_curve(
        pipe,
        X_train,
        y_train,
        param_name="poly__degree",
        param_range=degrees,
        cv=5,
        scoring="neg_root_mean_squared_error",
    )
    train_rmse = -train_scores.mean(axis=1)
    val_rmse = -val_scores.mean(axis=1)
    best_degree = degrees[int(np.argmin(val_rmse))]
    for d, tr, va in zip(degrees, train_rmse, val_rmse):
        print(f"  Bậc {d}: RMSE train={tr:.3f}  RMSE val={va:.3f}")
    print(f"  => Bậc tối ưu theo validation: {best_degree}")

    # ================================================================
    # BƯỚC 5. Vẽ đường cong xác thực → chỉ ra bậc tối ưu và điểm bắt đầu overfit
    # ================================================================
    header(5, "Vẽ đường cong xác thực (validation curve)")
    plt.figure(figsize=(6, 4))
    plt.plot(degrees, train_rmse, "o-", label="RMSE train")
    plt.plot(degrees, val_rmse, "o-", label="RMSE validation")
    plt.axvline(best_degree, color="gray", linestyle="--", alpha=0.6,
                label=f"Bậc tối ưu = {best_degree}")
    plt.xlabel("Bậc đa thức (degree)")
    plt.ylabel("RMSE (MW)")
    plt.title("Đường cong xác thực theo bậc đa thức")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS, "validation_curve.png"), dpi=120)
    plt.close()
    print("Đã lưu reports/validation_curve.png")

    # ================================================================
    # BƯỚC 6. Đếm số cột sinh ra ở mỗi bậc → lập bảng (bậc | số cột | RMSE)
    # ================================================================
    header(6, "Đếm số cột sinh ra ở mỗi bậc + bảng RMSE")
    rows = []
    for d in degrees:
        p = Pipeline(
            [
                ("poly", PolynomialFeatures(degree=d, include_bias=False)),
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=1.0)),
            ]
        )
        p.fit(X_train, y_train)
        n_cols = p.named_steps["poly"].n_output_features_
        tr_rmse = rmse(y_train, p.predict(X_train))
        te_rmse = rmse(y_test, p.predict(X_test))
        rows.append((d, n_cols, tr_rmse, te_rmse))
        print(f"  bậc={d:>2} | số cột={n_cols:>4} | RMSE train={tr_rmse:.3f} | RMSE test={te_rmse:.3f}")

    table_df = pd.DataFrame(rows, columns=["degree", "n_columns", "rmse_train", "rmse_test"])
    table_df.to_csv(os.path.join(REPORTS, "bang_bac_socot_rmse.csv"), index=False)
    print("Đã lưu reports/bang_bac_socot_rmse.csv")

    # ================================================================
    # BƯỚC 7. So sánh Linear vs Ridge ở bậc cao (bậc 4–5) → Ridge cứu được bao nhiêu?
    # ================================================================
    header(7, "So sánh Linear vs Ridge ở bậc cao (4-5)")
    compare_rows = []
    for d in [4, 5]:
        for name, Model, kwargs in [
            ("Linear", LinearRegression, {}),
            ("Ridge", Ridge, {"alpha": 1.0}),
        ]:
            p = Pipeline(
                [
                    ("poly", PolynomialFeatures(degree=d, include_bias=False)),
                    ("scale", StandardScaler()),
                    ("model", Model(**kwargs)),
                ]
            )
            p.fit(X_train, y_train)
            te_rmse = rmse(y_test, p.predict(X_test))
            max_coef = np.max(np.abs(p.named_steps["model"].coef_))
            compare_rows.append((d, name, te_rmse, max_coef))
            print(f"  bậc={d} | {name:6s} | RMSE test={te_rmse:.3f} | |hệ số| lớn nhất={max_coef:.2f}")

    # ================================================================
    # BƯỚC 8. Vẽ lại residual plot của model bậc tối ưu → chữ U đã biến mất chưa?
    # ================================================================
    header(8, f"Residual plot của model bậc tối ưu (bậc {best_degree}) (SAU)")
    final_pipe = Pipeline(
        [
            ("poly", PolynomialFeatures(degree=best_degree, include_bias=False)),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )
    final_pipe.fit(X_train, y_train)
    pred_final_test = final_pipe.predict(X_test)
    rmse_final = rmse(y_test, pred_final_test)
    residuals_after = y_test - pred_final_test

    axes[1].scatter(X_test["AT"], residuals_after, alpha=0.25, s=6, color="#27AE60")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_title(f"SAU — Bậc {best_degree} (Ridge)")
    axes[1].set_xlabel("AT — Nhiệt độ (°C)")
    axes[1].set_ylabel("Residual")
    plt.tight_layout()
    fig.savefig(os.path.join(REPORTS, "residual_truoc_sau.png"), dpi=120)
    plt.close(fig)
    print(f"RMSE test (bậc {best_degree}, Ridge) = {rmse_final:.3f} MW"
          f" (so với baseline bậc 1: {rmse_baseline:.3f} MW)")
    print("Đã lưu reports/residual_truoc_sau.png (gộp cả TRƯỚC bậc 1 và SAU bậc tối ưu)")

    # Lưu model tối ưu ngay sau khi có (dùng lại ở bước diễn giải)
    joblib.dump(final_pipe, os.path.join(MODELS, "poly_pipeline.joblib"))
    print(f"Đã lưu models/poly_pipeline.joblib")

    # ================================================================
    # BƯỚC 9. So sánh với Random Forest Regressor (TT-17)
    #          — cây bắt phi tuyến tự nhiên, không cần feature engineering
    # ================================================================
    header(9, "So sánh với Random Forest Regressor (TT-17)")
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rmse_rf = rmse(y_test, rf.predict(X_test))
    print(f"  Polynomial bậc {best_degree} (Ridge) : RMSE test = {rmse_final:.3f} MW")
    print(f"  Random Forest (200 trees)          : RMSE test = {rmse_rf:.3f} MW")

    # ================================================================
    # BƯỚC 10. Diễn giải: ở dải nhiệt độ nào công suất giảm nhanh nhất?
    # ================================================================
    header(10, "Diễn giải kết quả")
    at_min, at_max = X_train["AT"].min(), X_train["AT"].max()
    at_grid = np.linspace(at_min, at_max, 5)
    base_row = X_train.drop(columns="AT").mean()
    preds_grid = []
    for at_val in at_grid:
        row = pd.DataFrame([{**base_row.to_dict(), "AT": at_val}])[X_train.columns]
        preds_grid.append(final_pipe.predict(row)[0])
    slopes = np.diff(preds_grid) / np.diff(at_grid)
    print("Tốc độ giảm PE theo từng đoạn AT (MW / °C), dùng model bậc tối ưu:")
    for i in range(len(slopes)):
        print(f"  AT {at_grid[i]:.1f}°C -> {at_grid[i+1]:.1f}°C : {slopes[i]:+.3f} MW/°C")
    steepest_idx = int(np.argmin(slopes))
    print(f"=> Công suất giảm NHANH NHẤT trong đoạn AT "
          f"{at_grid[steepest_idx]:.1f}–{at_grid[steepest_idx+1]:.1f}°C "
          f"({slopes[steepest_idx]:+.3f} MW/°C).")
          
if __name__ == "__main__":
    main()