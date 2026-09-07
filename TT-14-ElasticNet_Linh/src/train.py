import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (
    LinearRegression, RidgeCV, LassoCV, ElasticNet, ElasticNetCV,
)
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import resample
from statsmodels.stats.outliers_influence import variance_inflation_factor

RANDOM_STATE = 42
DEFAULT_DATA_PATH = os.path.join("dataset", "ENB2012_data.xlsx")
DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_PATH

REPORTS_DIR = "reports"
MODELS_DIR = "models"

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ----------------------------------------------------------------------
# BƯỚC 1a — NẠP DỮ LIỆU
# ----------------------------------------------------------------------
def load_data(path=DATA_PATH):
    df = pd.read_excel(path)
    # Chuẩn hoá tên cột về đúng chuẩn X1..X8, Y1, Y2 nếu file gốc có tên khác
    df.columns = [str(c).strip() for c in df.columns]
    expected = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "Y1", "Y2"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(
            f"File dữ liệu thiếu cột: {missing}. Các cột hiện có: {list(df.columns)}. "
            "Kiểm tra lại tên cột trong file xlsx của bạn (có thể cần đổi tên cho khớp)."
        )
    df = df[expected].dropna()
    return df

# ----------------------------------------------------------------------
# BƯỚC 1b — MA TRẬN TƯƠNG QUAN + VIF (chứng minh đa cộng tuyến)
# ----------------------------------------------------------------------
def compute_correlation(df_numeric: pd.DataFrame):
    corr = df_numeric.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Ma trận tương quan giữa các biến thiết kế")
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "correlation_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Đã lưu ma trận tương quan: {path}")
    return corr


def compute_vif(X_numeric: pd.DataFrame) -> pd.DataFrame:
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_numeric.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X_numeric.values, i)
        for i in range(X_numeric.shape[1])
    ]
    return vif_data.sort_values("VIF", ascending=False).reset_index(drop=True)

# ----------------------------------------------------------------------
# BƯỚC 2 — TIỀN XỬ LÝ: one-hot X6, X8 (biến phân loại)
# ----------------------------------------------------------------------
def preprocess_features(df: pd.DataFrame):
    X = df.drop(columns=["Y1", "Y2"])
    y1 = df["Y1"]
    y2 = df["Y2"]

    categorical_cols = [c for c in ["X6", "X8"] if c in X.columns]
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    return X_encoded, y1, y2

# ----------------------------------------------------------------------
# BƯỚC 3-9 — HUẤN LUYỆN, SO SÁNH, GOM NHÓM, HEATMAP, BOOTSTRAP CHO 1 NHÃN
# ----------------------------------------------------------------------
def run_for_target(X: pd.DataFrame, y: pd.Series, target_name: str):
    print(f"\n{'=' * 60}\nNHÃN: {target_name}\n{'=' * 60}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    results = {}

    # --- BƯỚC 3: Baseline ---
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_train, y_train)
    pred = dummy.predict(X_test)
    results["Dummy"] = {
        "rmse": np.sqrt(mean_squared_error(y_test, pred)),
        "r2": r2_score(y_test, pred),
        "n_features": None,
    }

    lr_pipe = Pipeline([("scale", StandardScaler()), ("lr", LinearRegression())])
    lr_pipe.fit(X_train, y_train)
    pred = lr_pipe.predict(X_test)
    results["LinearRegression"] = {
        "rmse": np.sqrt(mean_squared_error(y_test, pred)),
        "r2": r2_score(y_test, pred),
        "n_features": X.shape[1],
    }

    # --- BƯỚC 4a: Ridge (CV chọn alpha) ---
    ridge_pipe = Pipeline([
        ("scale", StandardScaler()),
        ("ridge", RidgeCV(alphas=np.logspace(-4, 4, 100), cv=5)),
    ])
    ridge_pipe.fit(X_train, y_train)
    pred = ridge_pipe.predict(X_test)
    ridge_coef = ridge_pipe["ridge"].coef_
    results["Ridge"] = {
        "rmse": np.sqrt(mean_squared_error(y_test, pred)),
        "r2": r2_score(y_test, pred),
        "n_features": int((np.abs(ridge_coef) > 1e-6).sum()),
        "alpha": ridge_pipe["ridge"].alpha_,
    }

    # --- BƯỚC 4b: Lasso (CV chọn alpha) ---
    lasso_pipe = Pipeline([
        ("scale", StandardScaler()),
        ("lasso", LassoCV(alphas=np.logspace(-4, 1, 100), cv=5,
                           max_iter=50000, random_state=RANDOM_STATE)),
    ])
    lasso_pipe.fit(X_train, y_train)
    pred = lasso_pipe.predict(X_test)
    lasso_coef = lasso_pipe["lasso"].coef_
    results["Lasso"] = {
        "rmse": np.sqrt(mean_squared_error(y_test, pred)),
        "r2": r2_score(y_test, pred),
        "n_features": int((np.abs(lasso_coef) > 1e-6).sum()),
        "alpha": lasso_pipe["lasso"].alpha_,
    }

    # --- BƯỚC 4c: ElasticNet (CV chọn cả alpha + l1_ratio) ---
    en_pipe = Pipeline([
        ("scale", StandardScaler()),
        ("en", ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0],
            alphas=np.logspace(-4, 1, 100),
            cv=5, max_iter=50000, random_state=RANDOM_STATE,
        )),
    ])
    en_pipe.fit(X_train, y_train)
    pred = en_pipe.predict(X_test)
    en_coef = en_pipe["en"].coef_
    best_alpha = en_pipe["en"].alpha_
    best_l1_ratio = en_pipe["en"].l1_ratio_
    results["ElasticNet"] = {
        "rmse": np.sqrt(mean_squared_error(y_test, pred)),
        "r2": r2_score(y_test, pred),
        "n_features": int((np.abs(en_coef) > 1e-6).sum()),
        "alpha": best_alpha,
        "l1_ratio": best_l1_ratio,
    }

    # --- BƯỚC 5: BẢNG SO SÁNH 3 MODEL ---
    print(pd.DataFrame(results).T[["rmse", "r2", "n_features"]])
    print(f"\nElasticNet chọn alpha={best_alpha:.5f}, l1_ratio={best_l1_ratio}")

    # --- BƯỚC 6: KIỂM CHỨNG HIỆU ỨNG GOM NHÓM ---
    group_cols = [c for c in ["X1", "X2", "X4", "X5"] if c in X.columns]
    coef_compare = pd.DataFrame({
        "Lasso": pd.Series(lasso_coef, index=X.columns)[group_cols],
        "ElasticNet": pd.Series(en_coef, index=X.columns)[group_cols],
    })
    print(f"\nSo sánh hệ số nhóm tương quan (X1, X2, X4, X5) cho {target_name}:")
    print(coef_compare)
    coef_compare.to_csv(
        os.path.join(REPORTS_DIR, f"grouping_effect_{target_name}.csv")
    )

    # --- BƯỚC 7: HEATMAP RMSE THEO LƯỚI (alpha x l1_ratio) ---
    alphas_grid = np.logspace(-3, 1, 15)
    l1_ratios_grid = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]
    param_grid = {"en__alpha": alphas_grid, "en__l1_ratio": l1_ratios_grid}
    grid_pipe = Pipeline([
        ("scale", StandardScaler()),
        ("en", ElasticNet(max_iter=50000, random_state=RANDOM_STATE)),
    ])
    gs = GridSearchCV(
        grid_pipe, param_grid, cv=5,
        scoring="neg_root_mean_squared_error", n_jobs=-1,
    )
    gs.fit(X_train, y_train)

    cv_results = pd.DataFrame(gs.cv_results_)
    pivot = cv_results.pivot_table(
        index="param_en__l1_ratio", columns="param_en__alpha",
        values="mean_test_score",
    ) * -1

    plt.figure(figsize=(12, 5))
    sns.heatmap(pivot, cmap="viridis_r", annot=False)
    plt.title(f"RMSE theo alpha x l1_ratio ({target_name})")
    plt.xlabel("alpha")
    plt.ylabel("l1_ratio")
    plt.tight_layout()
    heatmap_path = os.path.join(REPORTS_DIR, f"heatmap_alpha_l1ratio_{target_name}.png")
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    print(f"Đã lưu heatmap: {heatmap_path}")

    # --- BƯỚC 9: BOOTSTRAP KIỂM TRA ỔN ĐỊNH HỆ SỐ ---
    n_boot = 100
    coefs_boot = []
    for i in range(n_boot):
        X_bs, y_bs = resample(X_train, y_train, random_state=i)
        model_bs = Pipeline([
            ("scale", StandardScaler()),
            ("en", ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio,
                               max_iter=50000, random_state=RANDOM_STATE)),
        ])
        model_bs.fit(X_bs, y_bs)
        coefs_boot.append(model_bs["en"].coef_)
    coefs_boot = np.array(coefs_boot)
    boot_std = pd.Series(coefs_boot.std(axis=0), index=X.columns).sort_values(ascending=False)
    print(f"\nĐộ lệch chuẩn hệ số qua {n_boot} lần bootstrap (top 5):")
    print(boot_std.head())
    boot_std.to_csv(os.path.join(REPORTS_DIR, f"bootstrap_std_{target_name}.csv"))

    # --- Lưu model ElasticNet cuối cùng ---
    model_path = os.path.join(MODELS_DIR, f"elasticnet_{target_name}.joblib")
    joblib.dump(en_pipe, model_path)
    print(f"Đã lưu model: {model_path}")

    # --- Bảng hệ số đầy đủ để xem biến nào quan trọng ---
    full_coef = pd.Series(en_coef, index=X.columns).sort_values(key=abs, ascending=False)
    print(f"\nHệ số ElasticNet đầy đủ cho {target_name} (sắp theo |hệ số|):")
    print(full_coef)

    return {
        "results_table": pd.DataFrame(results).T,
        "coef_compare_group": coef_compare,
        "elasticnet_full_coef": full_coef,
        "bootstrap_std": boot_std,
        "best_alpha": best_alpha,
        "best_l1_ratio": best_l1_ratio,
    }


def main():
    # BƯỚC 1a
    df = load_data()
    print(f"Đã nạp {df.shape[0]} dòng, {df.shape[1]} cột.")
    # BƯỚC 1b — ma trận tương quan + VIF
    numeric_cols = [c for c in ["X1", "X2", "X3", "X4", "X5", "X7"] if c in df.columns]
    compute_correlation(df[numeric_cols])
    vif_table = compute_vif(df[numeric_cols])
    vif_table.to_csv(os.path.join(REPORTS_DIR, "vif_table.csv"), index=False)
    print("\nBảng VIF (đa cộng tuyến):")
    print(vif_table)
    # BƯỚC 2 — tiền xử lý
    X, y1, y2 = preprocess_features(df)
    # BƯỚC 3-9 — chạy cho từng nhãn Y1, Y2 (yêu cầu ở bước 8 trong README)
    out_y1 = run_for_target(X, y1, "Y1")
    out_y2 = run_for_target(X, y2, "Y2")
    # BƯỚC 5 (gộp) — bảng so sánh 3 model, cả 2 nhãn, lưu CSV + hình
    combo = pd.concat(
        [out_y1["results_table"].assign(target="Y1"),
         out_y2["results_table"].assign(target="Y2")]
    )
    combo.to_csv(os.path.join(REPORTS_DIR, "so_sanh_model_full.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (target, sub) in zip(axes, combo.groupby("target")):
        sub = sub.drop(columns="target")
        sub["rmse"].plot(kind="bar", ax=ax, color="steelblue")
        ax.set_title(f"RMSE các model — {target}")
        ax.set_ylabel("RMSE")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "so_sanh_3_model.png"), dpi=150)
    plt.close()
    # BƯỚC 8 — so sánh biến quan trọng giữa Y1 và Y2
    print("\n\n== TỔNG KẾT ==")
    print(combo[["rmse", "r2", "n_features"]])
    print(f"\nY1 -> alpha={out_y1['best_alpha']:.5f}, l1_ratio={out_y1['best_l1_ratio']}")
    print(f"Y2 -> alpha={out_y2['best_alpha']:.5f}, l1_ratio={out_y2['best_l1_ratio']}")
    print("\nTop 5 biến quan trọng nhất cho Y1 (tải sưởi):")
    print(out_y1["elasticnet_full_coef"].head())
    print("\nTop 5 biến quan trọng nhất cho Y2 (tải làm mát):")
    print(out_y2["elasticnet_full_coef"].head())

if __name__ == "__main__":
    main()