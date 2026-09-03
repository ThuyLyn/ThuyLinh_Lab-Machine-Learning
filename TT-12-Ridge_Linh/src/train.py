import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, LassoCV, ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# Bước 1 — Tạo dữ liệu có đa cộng tuyến rõ rệt
def generate_data(n=500, seed=42):
    rng = np.random.default_rng(seed)
    tv = rng.uniform(50, 500, n)
    fb = tv * 0.6 + rng.normal(0, 15, n)  # tương quan cao với TV (r ~ 0.95)
    gg = rng.uniform(20, 300, n)
    doanh_thu = 3.2 * tv + 1.8 * fb + 2.5 * gg + rng.normal(0, 50, n)
    return pd.DataFrame({"TV": tv, "Facebook": fb, "Google": gg, "DoanhThu": doanh_thu})

# Bước 2 — Ma trận tương quan + VIF (tính thủ công, không cần statsmodels)
#   VIF_i = 1 / (1 - R_i^2), với R_i^2 là hồi quy biến i theo các biến còn lại
def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    vif_data = []
    for col in X.columns:
        y_col = X[col]
        X_others = X.drop(columns=[col])
        r2 = LinearRegression().fit(X_others, y_col).score(X_others, y_col)
        vif = np.inf if r2 >= 1.0 else 1.0 / (1.0 - r2)
        vif_data.append({"feature": col, "VIF": vif})
    return pd.DataFrame(vif_data)


def main():
    df = generate_data()
    X = df[["TV", "Facebook", "Google"]]
    y = df["DoanhThu"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("=== Bước 2: Tương quan & VIF ===")
    corr = X.corr()
    print(corr, "\n")

    vif_df = compute_vif(X)
    print(vif_df, "\n")
    vif_df.to_csv(os.path.join(REPORTS_DIR, "vif_table.csv"), index=False)

    # Bước 3 — Linear Regression baseline
    print("=== Bước 3: Linear Regression baseline ===")
    lr_pipe = Pipeline([("scale", StandardScaler()), ("lr", LinearRegression())])
    lr_pipe.fit(X_train, y_train)
    lr_coefs = dict(zip(X.columns, lr_pipe["lr"].coef_))
    print(lr_coefs, "\n")

    # Bước 4 — Thí nghiệm ổn định (bootstrap): Linear vs Ridge
    print("=== Bước 4: Bootstrap ổn định hệ số (100 lần) ===")
    n_boot = 100
    coefs_lr, coefs_ridge = [], []

    for i in range(n_boot):
        sample = df.sample(frac=0.8, random_state=i)
        Xs, ys = sample[["TV", "Facebook", "Google"]], sample["DoanhThu"]
        scaler = StandardScaler().fit(Xs)
        Xs_scaled = scaler.transform(Xs)

        coefs_lr.append(LinearRegression().fit(Xs_scaled, ys).coef_)
        coefs_ridge.append(Ridge(alpha=10).fit(Xs_scaled, ys).coef_)

    coefs_lr = np.array(coefs_lr)
    coefs_ridge = np.array(coefs_ridge)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, name in enumerate(X.columns):
        axes[i].hist(coefs_lr[:, i], alpha=0.5, label="Linear", bins=20, color="tab:red")
        axes[i].hist(coefs_ridge[:, i], alpha=0.5, label="Ridge", bins=20, color="tab:blue")
        axes[i].set_title(f"Hệ số: {name}")
        axes[i].legend()
    plt.suptitle("Độ ổn định hệ số qua Bootstrap: Linear vs Ridge")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "bootstrap_he_so.png"), dpi=120)
    plt.close()

    std_compare = pd.DataFrame(
        {
            "feature": X.columns,
            "std_linear": coefs_lr.std(axis=0),
            "std_ridge": coefs_ridge.std(axis=0),
        }
    )
    print(std_compare, "\n")

    # Bước 5 — RidgeCV dò alpha
    print("=== Bước 5: RidgeCV dò alpha tối ưu ===")
    ridge_pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5)),
        ]
    )
    ridge_pipe.fit(X_train, y_train)
    best_alpha = ridge_pipe["ridge"].alpha_
    print("alpha tối ưu:", best_alpha, "\n")

    # Bước 6 — Coefficient path
    print("=== Bước 6: Coefficient path ===")
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    alphas = np.logspace(-3, 4, 100)
    paths = np.array(
        [Ridge(alpha=a).fit(X_train_scaled, y_train).coef_ for a in alphas]
    )

    plt.figure(figsize=(8, 5))
    for i, name in enumerate(X.columns):
        plt.plot(alphas, paths[:, i], label=name)
    plt.xscale("log")
    plt.xlabel("alpha (log scale)")
    plt.ylabel("Hệ số")
    plt.axvline(best_alpha, color="gray", linestyle="--", label="alpha tối ưu (CV)")
    plt.legend()
    plt.title("Coefficient Path (Ridge)")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "coefficient_path.png"), dpi=120)
    plt.close()

    # Bước 7 — RMSE train/test theo alpha
    print("=== Bước 7: RMSE train/test theo alpha ===")
    X_test_scaled = scaler.transform(X_test)
    train_rmse, test_rmse = [], []

    for a in alphas:
        m = Ridge(alpha=a).fit(X_train_scaled, y_train)
        train_rmse.append(
            np.sqrt(mean_squared_error(y_train, m.predict(X_train_scaled)))
        )
        test_rmse.append(
            np.sqrt(mean_squared_error(y_test, m.predict(X_test_scaled)))
        )

    plt.figure(figsize=(8, 5))
    plt.plot(alphas, train_rmse, label="Train RMSE")
    plt.plot(alphas, test_rmse, label="Test RMSE")
    plt.xscale("log")
    plt.axvline(best_alpha, color="gray", linestyle="--", label="alpha tối ưu (CV)")
    plt.legend()
    plt.xlabel("alpha")
    plt.ylabel("RMSE")
    plt.title("RMSE theo alpha (Bias-Variance tradeoff)")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "rmse_theo_alpha.png"), dpi=120)
    plt.close()

    # Bước 8 — So sánh RMSE Linear vs Ridge trên test
    print("=== Bước 8: So sánh RMSE trên test ===")
    rmse_lr_test = np.sqrt(mean_squared_error(y_test, lr_pipe.predict(X_test)))
    rmse_ridge_test = np.sqrt(mean_squared_error(y_test, ridge_pipe.predict(X_test)))
    print(f"Linear RMSE (test): {rmse_lr_test:.3f}")
    print(f"Ridge  RMSE (test): {rmse_ridge_test:.3f}\n")

    # Bước 9 — So sánh với Lasso / ElasticNet
    print("=== Bước 9: So sánh hệ số Ridge vs Lasso vs ElasticNet ===")
    lasso_pipe = Pipeline(
        [("scale", StandardScaler()), ("lasso", LassoCV(cv=5, random_state=42))]
    ).fit(X_train, y_train)
    elastic_pipe = Pipeline(
        [("scale", StandardScaler()), ("en", ElasticNetCV(cv=5, random_state=42))]
    ).fit(X_train, y_train)

    compare_coefs = pd.DataFrame(
        {
            "feature": X.columns,
            "Ridge": ridge_pipe["ridge"].coef_,
            "Lasso": lasso_pipe["lasso"].coef_,
            "ElasticNet": elastic_pipe["en"].coef_,
        }
    )
    print(compare_coefs, "\n")
    compare_coefs.to_csv(os.path.join(REPORTS_DIR, "so_sanh_he_so.csv"), index=False)

    # Bước 10 — Đề xuất phân bổ ngân sách dựa trên hệ số Ridge
    print("=== Bước 10: Đề xuất phân bổ ngân sách (dựa trên |hệ số Ridge|) ===")
    ridge_coefs = dict(zip(X.columns, ridge_pipe["ridge"].coef_))
    total = sum(abs(v) for v in ridge_coefs.values())
    budget_total_vnd = 2_000_000_000  # 2 tỷ / tháng

    allocation = []
    for k, v in ridge_coefs.items():
        pct = abs(v) / total * 100
        allocation.append(
            {
                "kenh": k,
                "he_so_ridge_chuan_hoa": v,
                "ty_trong_%": pct,
                "de_xuat_ngan_sach_vnd": budget_total_vnd * pct / 100,
            }
        )
    alloc_df = pd.DataFrame(allocation)
    print(alloc_df, "\n")
    alloc_df.to_csv(os.path.join(REPORTS_DIR, "de_xuat_phan_bo_ngan_sach.csv"), index=False)

    # Lưu model
    joblib.dump(ridge_pipe, os.path.join(MODELS_DIR, "ridge_pipeline.joblib"))
    print(f"Đã lưu model tại: {os.path.join(MODELS_DIR, 'ridge_pipeline.joblib')}")
    print("Hoàn tất. Xem toàn bộ biểu đồ & bảng trong thư mục reports/.")


if __name__ == "__main__":
    main()