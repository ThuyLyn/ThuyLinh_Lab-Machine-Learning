import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

import os

TARGET = "Concrete compressive strength"
DATA_PATH = "dataset/Concrete_Data.xls"  
REPORTS_DIR = "reports"
MODELS_DIR = "models"

COLUMN_RENAME_MAP = {
    "Cement (component 1)(kg in a m^3 mixture)": "Cement",
    "Blast Furnace Slag (component 2)(kg in a m^3 mixture)": "Blast Furnace Slag",
    "Fly Ash (component 3)(kg in a m^3 mixture)": "Fly Ash",
    "Water  (component 4)(kg in a m^3 mixture)": "Water",
    "Superplasticizer (component 5)(kg in a m^3 mixture)": "Superplasticizer",
    "Coarse Aggregate  (component 6)(kg in a m^3 mixture)": "Coarse Aggregate",
    "Fine Aggregate (component 7)(kg in a m^3 mixture)": "Fine Aggregate",
    "Age (day)": "Age",
    "Concrete compressive strength(MPa, megapascals) ": TARGET,
}


def load_data(path):
    if path.lower().endswith((".xls", ".xlsx")):
        df = pd.read_excel(path)
        df = df.rename(columns=COLUMN_RENAME_MAP)
    else:
        df = pd.read_csv(path)
    assert TARGET in df.columns, (
        f"Không tìm thấy cột target '{TARGET}'. Kiểm tra lại tên cột trong file dữ liệu."
    )
    return df

def line(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

# =====================================================================
# BƯỚC 1: EDA
# =====================================================================
def step1_eda(df, features):
    line("BƯỚC 1 — EDA: scatter & heatmap tương quan")

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, col in zip(axes.flat, features):
        ax.scatter(df[col], df[TARGET], s=8, alpha=0.4)
        ax.set_xlabel(col)
        ax.set_ylabel(TARGET)
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/scatter_features.png", dpi=110)
    plt.close()

    plt.figure(figsize=(9, 7))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/correlation_heatmap.png", dpi=110)
    plt.close()

    print("Tương quan với target (giảm dần theo |r|):")
    print(corr[TARGET].drop(TARGET).sort_values(key=lambda s: -s.abs()))
    print("\n-> Đã lưu reports/scatter_features.png, reports/correlation_heatmap.png")

# =====================================================================
# BƯỚC 2: Đặc trưng miền
# =====================================================================
def step2_add_domain_features(df):
    line("BƯỚC 2 — Tạo đặc trưng miền: water_cement_ratio, tong_chat_ket_dinh, log_age")

    df = df.copy()
    df["water_cement_ratio"] = df["Water"] / df["Cement"]
    df["tong_chat_ket_dinh"] = df["Cement"] + df["Blast Furnace Slag"] + df["Fly Ash"]
    df["log_age"] = np.log1p(df["Age"])

    print("Tương quan đặc trưng mới với target:")
    print(df[["water_cement_ratio", "tong_chat_ket_dinh", "log_age", "Age", TARGET]].corr()[TARGET])
    return df

# =====================================================================
# BƯỚC 3: Baseline
# =====================================================================
def step3_baseline(X_train, X_test, y_train, y_test):
    line("BƯỚC 3 — Baseline: DummyRegressor + Linear Regression")

    for name, model in [("DummyRegressor (mean)", DummyRegressor(strategy="mean")),
                         ("Linear Regression", LinearRegression())]:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        r2 = r2_score(y_test, pred)
        rmse = mean_squared_error(y_test, pred) ** 0.5
        print(f"{name:25s}  R2={r2:.4f}   RMSE={rmse:.3f} MPa")

# =====================================================================
# BƯỚC 4-5: SVR không scale vs scale X&y
# =====================================================================
def step4_5_scaling(X_train, X_test, y_train, y_test):
    line("BƯỚC 4-5 — SVR: KHÔNG scale vs scale cả X và y")

    results = {}

    svr_raw = SVR(kernel="rbf")  # epsilon mặc định = 0.1, không scale gì
    svr_raw.fit(X_train, y_train)
    pred = svr_raw.predict(X_test)
    results["SVR - KHÔNG scale"] = (r2_score(y_test, pred), mean_squared_error(y_test, pred) ** 0.5)

    svr_x_only = Pipeline([("scale", StandardScaler()),
                            ("svr", SVR(kernel="rbf", C=100, gamma="scale", epsilon=0.1))])
    svr_x_only.fit(X_train, y_train)
    pred = svr_x_only.predict(X_test)
    results["SVR - chỉ scale X"] = (r2_score(y_test, pred), mean_squared_error(y_test, pred) ** 0.5)

    svr_pipeline = Pipeline([("scale", StandardScaler()),
                              ("svr", SVR(kernel="rbf", C=100, gamma="scale", epsilon=0.1))])
    model_scaled = TransformedTargetRegressor(regressor=svr_pipeline, transformer=StandardScaler())
    model_scaled.fit(X_train, y_train)
    pred = model_scaled.predict(X_test)
    results["SVR - scale cả X và y"] = (r2_score(y_test, pred), mean_squared_error(y_test, pred) ** 0.5)

    print(f"{'Cấu hình':30s} {'R2':>8s} {'RMSE':>10s}")
    for name, (r2, rmse) in results.items():
        print(f"{name:30s} {r2:8.4f} {rmse:10.3f}")

    names = list(results.keys())
    r2s = [results[n][0] for n in names]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, r2s, color=["#d62728", "#ff7f0e", "#2ca02c"])
    plt.ylabel("R² (test)")
    plt.title("So sánh SVR: không scale vs scale X/y")
    plt.xticks(rotation=15, ha="right")
    for b, v in zip(bars, r2s):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/scale_vs_noscale.png", dpi=110)
    plt.close()
    print("\n-> Đã lưu reports/scale_vs_noscale.png")

# =====================================================================
# BƯỚC 6: So sánh kernel
# =====================================================================
def step6_kernels(X_train, X_test, y_train, y_test):
    line("BƯỚC 6 — So sánh 3 kernel: linear / rbf / poly (degree 2, 3)")

    configs = {
        "linear": dict(kernel="linear", C=100),
        "rbf": dict(kernel="rbf", C=100, gamma="scale"),
        "poly (degree=2)": dict(kernel="poly", degree=2, C=100, gamma="scale"),
        "poly (degree=3)": dict(kernel="poly", degree=3, C=100, gamma="scale"),
    }

    rows = []
    for name, params in configs.items():
        pipe = Pipeline([("scale", StandardScaler()), ("svr", SVR(epsilon=0.1, **params))])
        m = TransformedTargetRegressor(regressor=pipe, transformer=StandardScaler())
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        r2 = r2_score(y_test, pred)
        rmse = mean_squared_error(y_test, pred) ** 0.5
        n_sv = m.regressor_.named_steps["svr"].n_support_.sum()
        rows.append((name, r2, rmse, n_sv))

    df_res = pd.DataFrame(rows, columns=["kernel", "R2", "RMSE", "n_support_vectors"])
    print(df_res.to_string(index=False))

    plt.figure(figsize=(7, 5))
    plt.bar(df_res["kernel"], df_res["R2"], color="#1f77b4")
    plt.ylabel("R² (test)")
    plt.title("So sánh kernel SVR")
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/kernel_comparison.png", dpi=110)
    plt.close()
    print("\n-> Đã lưu reports/kernel_comparison.png")

# =====================================================================
# BƯỚC 7-8: GridSearchCV + heatmap C x gamma
# =====================================================================
def step7_8_gridsearch(X_train, X_test, y_train, y_test):
    line("BƯỚC 7-8 — GridSearchCV (C x gamma x epsilon) + heatmap RMSE")

    svr_pipeline = Pipeline([("scale", StandardScaler()), ("svr", SVR(kernel="rbf"))])
    base_model = TransformedTargetRegressor(regressor=svr_pipeline, transformer=StandardScaler())

    param_grid = {
        "regressor__svr__C": [1, 10, 100, 1000],
        "regressor__svr__gamma": ["scale", 0.01, 0.1, 1],
        "regressor__svr__epsilon": [0.01, 0.1, 0.5],
    }

    grid = GridSearchCV(base_model, param_grid, scoring="neg_root_mean_squared_error", cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)

    print("Bộ tham số tốt nhất:", grid.best_params_)
    print("RMSE CV tốt nhất:", -grid.best_score_)

    best_model = grid.best_estimator_
    pred = best_model.predict(X_test)
    r2 = r2_score(y_test, pred)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    print(f"Test set: R2={r2:.4f}  RMSE={rmse:.3f} MPa")

    results_df = pd.DataFrame(grid.cv_results_)
    best_eps = grid.best_params_["regressor__svr__epsilon"]
    sub = results_df[results_df["param_regressor__svr__epsilon"] == best_eps]
    pivot = sub.pivot_table(index="param_regressor__svr__gamma", columns="param_regressor__svr__C",
                             values="mean_test_score", aggfunc="mean")
    pivot_rmse = -pivot

    plt.figure(figsize=(7, 5))
    sns.heatmap(pivot_rmse, annot=True, fmt=".2f", cmap="viridis_r")
    plt.title(f"RMSE (CV) theo C & gamma (epsilon={best_eps})")
    plt.xlabel("C")
    plt.ylabel("gamma")
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/C_gamma_heatmap.png", dpi=110)
    plt.close()
    print("\n-> Đã lưu reports/C_gamma_heatmap.png")

    return best_model, grid.best_params_

# =====================================================================
# BƯỚC 9: Support vectors
# =====================================================================
def step9_support_vectors(best_model, X_train):
    line("BƯỚC 9 — Đếm support vectors")

    svr = best_model.regressor_.named_steps["svr"]
    n_sv = svr.n_support_.sum()
    n_train = len(X_train)
    print(f"Số support vectors: {n_sv} / {n_train} mẫu train  ({n_sv/n_train*100:.1f}%)")
    print("(tỉ lệ % thấp = model gọn, tổng quát hoá tốt hơn)")

# =====================================================================
# BƯỚC 10: Hiệu quả đặc trưng miền
# =====================================================================
def step10_domain_effect(df, best_params):
    line("BƯỚC 10 — Đo hiệu quả đặc trưng miền: RMSE trước vs sau")

    raw_features = ["Cement", "Blast Furnace Slag", "Fly Ash", "Water",
                     "Superplasticizer", "Coarse Aggregate", "Fine Aggregate", "Age"]
    domain_features = raw_features + ["water_cement_ratio", "tong_chat_ket_dinh", "log_age"]

    rows = []
    for label, cols in [("KHÔNG có đặc trưng miền (8 cột gốc)", raw_features),
                         ("CÓ đặc trưng miền (11 cột)", domain_features)]:
        X = df[cols]
        y = df[TARGET]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        pipe = Pipeline([("scale", StandardScaler()),
                          ("svr", SVR(kernel="rbf",
                                      C=best_params["regressor__svr__C"],
                                      gamma=best_params["regressor__svr__gamma"],
                                      epsilon=best_params["regressor__svr__epsilon"]))])
        m = TransformedTargetRegressor(regressor=pipe, transformer=StandardScaler())
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        r2 = r2_score(y_test, pred)
        rmse = mean_squared_error(y_test, pred) ** 0.5
        rows.append((label, r2, rmse))

    print(f"{'Cấu hình':40s} {'R2':>8s} {'RMSE':>8s}")
    for label, r2, rmse in rows:
        print(f"{label:40s} {r2:8.4f} {rmse:8.3f}")

    improvement = rows[0][2] - rows[1][2]
    print(f"\n=> Cải thiện RMSE nhờ đặc trưng miền: {improvement:.3f} MPa "
          f"({improvement/rows[0][2]*100:.1f}% tốt hơn)")

# =====================================================================
# BƯỚC 11: So sánh SVR vs XGBoost vs Random Forest
# =====================================================================
def step11_compare_models(X_train, X_test, y_train, y_test, best_params):
    line("BƯỚC 11 — So sánh SVR vs XGBoost vs Random Forest")

    svr_pipeline = Pipeline([("scale", StandardScaler()),
                              ("svr", SVR(kernel="rbf",
                                          C=best_params["regressor__svr__C"],
                                          gamma=best_params["regressor__svr__gamma"],
                                          epsilon=best_params["regressor__svr__epsilon"]))])
    models = {
        "SVR (RBF, tuned)": TransformedTargetRegressor(regressor=svr_pipeline, transformer=StandardScaler()),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.08, random_state=42),
    }

    rows = []
    for name, model in models.items():
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0
        pred = model.predict(X_test)
        r2 = r2_score(y_test, pred)
        rmse = mean_squared_error(y_test, pred) ** 0.5
        rows.append((name, r2, rmse, train_time))

    df_res = pd.DataFrame(rows, columns=["model", "R2", "RMSE", "train_time_s"])
    print(df_res.to_string(index=False))

# =====================================================================
# BƯỚC 12: Thời gian train theo kích thước dữ liệu
# =====================================================================
def step12_scaling_time(X_train, y_train, best_params, multipliers=(1, 2, 4, 8)):
    line("BƯỚC 12 — Đo thời gian train khi nhân dữ liệu lên nhiều lần")

    rows = []
    for m in multipliers:
        X_big = pd.concat([X_train] * m, ignore_index=True)
        y_big = pd.concat([y_train] * m, ignore_index=True)
        n = len(X_big)

        svr_pipeline = Pipeline([("scale", StandardScaler()),
                                  ("svr", SVR(kernel="rbf",
                                              C=best_params["regressor__svr__C"],
                                              gamma=best_params["regressor__svr__gamma"],
                                              epsilon=best_params["regressor__svr__epsilon"]))])
        svr_model = TransformedTargetRegressor(regressor=svr_pipeline, transformer=StandardScaler())
        t0 = time.perf_counter()
        svr_model.fit(X_big, y_big)
        svr_time = time.perf_counter() - t0

        xgb_model = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.08, random_state=42)
        t0 = time.perf_counter()
        xgb_model.fit(X_big, y_big)
        xgb_time = time.perf_counter() - t0

        rows.append((n, svr_time, xgb_time))
        print(f"n={n:6d}  SVR={svr_time:7.3f}s   XGBoost={xgb_time:7.3f}s")

    df_res = pd.DataFrame(rows, columns=["n_samples", "svr_time_s", "xgb_time_s"])

    plt.figure(figsize=(7, 5))
    plt.plot(df_res["n_samples"], df_res["svr_time_s"], "o-", label="SVR (RBF)")
    plt.plot(df_res["n_samples"], df_res["xgb_time_s"], "o-", label="XGBoost")
    plt.xlabel("Số mẫu train")
    plt.ylabel("Thời gian train (giây)")
    plt.title("Thời gian train theo kích thước dữ liệu")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/thoi_gian_train.png", dpi=110)
    plt.close()
    print("\n-> Đã lưu reports/thoi_gian_train.png")
    print("=> SVR tăng thời gian ~O(n^2), không phù hợp dữ liệu lớn (>50.000 dòng).")

# =====================================================================
# MAIN — chạy tuần tự cả 12 bước
# =====================================================================
def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = load_data(DATA_PATH)
    features = [c for c in df.columns if c != TARGET]

    # Bước 1
    step1_eda(df, features)

    # Bước 2
    df = step2_add_domain_features(df)

    # Chia train/test (dùng log_age thay Age gốc)
    X = df.drop(columns=[TARGET, "Age"])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Bước 3
    step3_baseline(X_train, X_test, y_train, y_test)

    # Bước 4-5
    step4_5_scaling(X_train, X_test, y_train, y_test)

    # Bước 6
    step6_kernels(X_train, X_test, y_train, y_test)

    # Bước 7-8
    best_model, best_params = step7_8_gridsearch(X_train, X_test, y_train, y_test)

    # Bước 9
    step9_support_vectors(best_model, X_train)

    # Bước 10
    step10_domain_effect(df, best_params)

    # Bước 11
    step11_compare_models(X_train, X_test, y_train, y_test, best_params)

    # Bước 12
    step12_scaling_time(X_train, y_train, best_params)

    # Lưu model cuối cùng
    joblib.dump(best_model, f"{MODELS_DIR}/svr_pipeline.joblib")

    line("HOÀN THÀNH")
    print(f"Model đã lưu -> {MODELS_DIR}/svr_pipeline.joblib")
    print(f"6 hình báo cáo đã lưu trong thư mục -> {REPORTS_DIR}/")


if __name__ == "__main__":
    main()