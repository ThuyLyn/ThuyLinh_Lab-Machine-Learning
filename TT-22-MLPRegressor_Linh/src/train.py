import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import warnings

RANDOM_STATE = 42
DATA_PATH = "dataset/auto-mpg.data"
REPORTS_DIR = "reports"
MODELS_DIR = "models"

COLUMN_NAMES = [
    "mpg", "cylinders", "displacement", "horsepower", "weight",
    "acceleration", "model_year", "origin", "car_name",
]


# --------------------------------------------------------------------------
# 1. LOAD & CLEAN DATA
# --------------------------------------------------------------------------
def load_data():
    df = pd.read_csv(
        DATA_PATH, sep=r"\s+", names=COLUMN_NAMES, na_values="?",
        quotechar='"',
    )
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # (1) horsepower co '?' -> ep ve numeric, dien median
    df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")
    n_missing = df["horsepower"].isna().sum()
    print(f"[clean] So gia tri thieu o horsepower: {n_missing} -> dien median")
    df["horsepower"] = df["horsepower"].fillna(df["horsepower"].median())

    # (2) bo car_name (dinh danh, khong du bao duoc)
    if "car_name" in df.columns:
        df = df.drop(columns=["car_name"])

    # (3) origin la ma vung phan loai -> one-hot
    df["origin"] = df["origin"].astype(int)
    df = pd.get_dummies(df, columns=["origin"], prefix="origin")

    return df

# --------------------------------------------------------------------------
# 2. EDA (luu hinh, khong chan chuong trinh)
# --------------------------------------------------------------------------
def run_eda(df: pd.DataFrame):
    os.makedirs(REPORTS_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(df["weight"], df["mpg"], alpha=0.5)
    axes[0].set_xlabel("weight")
    axes[0].set_ylabel("mpg")
    axes[0].set_title("weight vs mpg (quan he nghich, hoi cong)")

    axes[1].scatter(df["model_year"], df["mpg"], alpha=0.5)
    axes[1].set_xlabel("model_year")
    axes[1].set_ylabel("mpg")
    axes[1].set_title("mpg theo model_year (xu huong tang)")

    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/eda.png", dpi=120)
    plt.close()
    print(f"[eda] Da luu {REPORTS_DIR}/eda.png")

# --------------------------------------------------------------------------
# 3. BASELINES
# --------------------------------------------------------------------------
def eval_model(name, model, X_train, X_test, y_train, y_test, results):
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    rmse_train = mean_squared_error(y_train, pred_train) ** 0.5
    rmse_test = mean_squared_error(y_test, pred_test) ** 0.5
    r2_test = r2_score(y_test, pred_test)

    results.append({
        "model": name,
        "rmse_train": round(rmse_train, 3),
        "rmse_test": round(rmse_test, 3),
        "r2_test": round(r2_test, 4),
        "train_time_s": round(train_time, 3),
    })
    print(f"[{name}] RMSE train={rmse_train:.3f}  RMSE test={rmse_test:.3f}  "
          f"R2 test={r2_test:.4f}  time={train_time:.3f}s")
    return model

# -------------------------------------------------------------------------
# 4. MLP: ANH HUONG CUA SCALING (khong scale / chi X / X va y)
# --------------------------------------------------------------------------
def compare_scaling(X_train, X_test, y_train, y_test):
    results = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # (a) Khong scale gi ca
        mlp_raw = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000,
                                early_stopping=True, n_iter_no_change=30,
                                random_state=RANDOM_STATE)
        eval_model("MLP (khong scale)", mlp_raw, X_train, X_test, y_train, y_test, results)

        # (b) Chi scale X
        mlp_x = Pipeline([
            ("scale", StandardScaler()),
            ("mlp", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000,
                                  early_stopping=True, n_iter_no_change=30,
                                  random_state=RANDOM_STATE)),
        ])
        eval_model("MLP (scale X)", mlp_x, X_train, X_test, y_train, y_test, results)

        # (c) Scale ca X va y
        net = Pipeline([
            ("scale", StandardScaler()),
            ("mlp", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000,
                                  early_stopping=True, n_iter_no_change=30,
                                  random_state=RANDOM_STATE)),
        ])
        mlp_xy = TransformedTargetRegressor(regressor=net, transformer=StandardScaler())
        eval_model("MLP (scale X va y)", mlp_xy, X_train, X_test, y_train, y_test, results)

    df_res = pd.DataFrame(results)
    df_res.to_csv(f"{REPORTS_DIR}/scale_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(df_res["model"], df_res["rmse_test"])
    ax.set_ylabel("RMSE (test)")
    ax.set_title("Anh huong cua scaling toi MLP")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/scale_comparison.png", dpi=120)
    plt.close()

    return df_res

# --------------------------------------------------------------------------
# 5. SO SANH KIEN TRUC (chung minh mang to overfit voi du lieu nho)
# --------------------------------------------------------------------------
def compare_architectures(X_train, X_test, y_train, y_test):
    architectures = [(16,), (64,), (64, 32), (256, 128, 64)]
    results = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for arch in architectures:
            net = Pipeline([
                ("scale", StandardScaler()),
                ("mlp", MLPRegressor(hidden_layer_sizes=arch, max_iter=2000,
                                      early_stopping=True, n_iter_no_change=30,
                                      random_state=RANDOM_STATE)),
            ])
            model = TransformedTargetRegressor(regressor=net, transformer=StandardScaler())
            name = f"MLP{arch}"
            eval_model(name, model, X_train, X_test, y_train, y_test, results)

            n_features = X_train.shape[1]
            n_params = 0
            prev = n_features
            for h in arch:
                n_params += prev * h + h
                prev = h
            n_params += prev * 1 + 1
            results[-1]["n_params"] = n_params
            results[-1]["overfit_gap"] = round(
                results[-1]["rmse_test"] - results[-1]["rmse_train"], 3
            )

    df_res = pd.DataFrame(results)
    df_res.to_csv(f"{REPORTS_DIR}/architecture_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(df_res))
    width = 0.35
    ax.bar(x - width / 2, df_res["rmse_train"], width, label="RMSE train")
    ax.bar(x + width / 2, df_res["rmse_test"], width, label="RMSE test")
    ax.set_xticks(x)
    ax.set_xticklabels(df_res["model"], rotation=20, ha="right")
    ax.set_ylabel("RMSE")
    ax.set_title("Kien truc vs Overfit (398 mau)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/kien_truc_overfit.png", dpi=120)
    plt.close()

    return df_res


# --------------------------------------------------------------------------
# 6. LOSS CURVE cua mo hinh MLP tot nhat (scale X va y, (64,32))
# --------------------------------------------------------------------------
def plot_loss_curve(best_model):
    mlp = best_model.regressor_.named_steps["mlp"]
    plt.figure(figsize=(7, 5))
    plt.plot(mlp.loss_curve_, label="loss (train)")
    if hasattr(mlp, "validation_scores_") and mlp.validation_scores_ is not None:
        plt.plot(mlp.validation_scores_, label="validation score")
    plt.xlabel("Epoch (iteration)")
    plt.ylabel("Loss / Score")
    plt.title("Loss curve cua MLP")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/loss_curve.png", dpi=120)
    plt.close()
    print(f"[loss] Da luu {REPORTS_DIR}/loss_curve.png")


# --------------------------------------------------------------------------
# 7. DO ALPHA
# --------------------------------------------------------------------------
def sweep_alpha(X_train, X_test, y_train, y_test):
    alphas = [1e-4, 1e-3, 1e-2, 1e-1]
    rows = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for a in alphas:
            net = Pipeline([
                ("scale", StandardScaler()),
                ("mlp", MLPRegressor(hidden_layer_sizes=(64, 32), alpha=a,
                                      max_iter=2000, early_stopping=True,
                                      n_iter_no_change=30, random_state=RANDOM_STATE)),
            ])
            model = TransformedTargetRegressor(regressor=net, transformer=StandardScaler())
            model.fit(X_train, y_train)
            rmse_train = mean_squared_error(y_train, model.predict(X_train)) ** 0.5
            rmse_test = mean_squared_error(y_test, model.predict(X_test)) ** 0.5
            rows.append({"alpha": a, "rmse_train": rmse_train, "rmse_test": rmse_test})
            print(f"[alpha={a}] RMSE train={rmse_train:.3f}  RMSE test={rmse_test:.3f}")

    df_res = pd.DataFrame(rows)
    df_res.to_csv(f"{REPORTS_DIR}/alpha_sweep.csv", index=False)

    plt.figure(figsize=(7, 5))
    plt.plot(df_res["alpha"], df_res["rmse_train"], marker="o", label="RMSE train")
    plt.plot(df_res["alpha"], df_res["rmse_test"], marker="o", label="RMSE test")
    plt.xscale("log")
    plt.xlabel("alpha (log scale)")
    plt.ylabel("RMSE")
    plt.title("Anh huong cua alpha (regularization)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/alpha_sweep.png", dpi=120)
    plt.close()

    return df_res


# --------------------------------------------------------------------------
# 8. SO SANH ACTIVATION
# --------------------------------------------------------------------------
def compare_activations(X_train, X_test, y_train, y_test):
    activations = ["relu", "tanh", "logistic"]
    rows = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for act in activations:
            net = Pipeline([
                ("scale", StandardScaler()),
                ("mlp", MLPRegressor(hidden_layer_sizes=(64, 32), activation=act,
                                      alpha=1e-2, max_iter=2000, early_stopping=True,
                                      n_iter_no_change=30, random_state=RANDOM_STATE)),
            ])
            model = TransformedTargetRegressor(regressor=net, transformer=StandardScaler())
            model.fit(X_train, y_train)
            rmse_test = mean_squared_error(y_test, model.predict(X_test)) ** 0.5
            r2_test = r2_score(y_test, model.predict(X_test))
            rows.append({"activation": act, "rmse_test": rmse_test, "r2_test": r2_test})
            print(f"[activation={act}] RMSE test={rmse_test:.3f}  R2 test={r2_test:.4f}")

    df_res = pd.DataFrame(rows)
    df_res.to_csv(f"{REPORTS_DIR}/activation_comparison.csv", index=False)
    return df_res

# --------------------------------------------------------------------------
# 9. QUY DOI SANG L/100km
# --------------------------------------------------------------------------
def mpg_to_l_per_100km(mpg):
    return 235.215 / mpg


def estimate_yearly_fuel_cost(mpg, km_per_year=15000, price_per_liter_vnd=23000):
    l_per_100km = mpg_to_l_per_100km(mpg)
    liters_per_year = l_per_100km * (km_per_year / 100)
    cost_vnd = liters_per_year * price_per_liter_vnd
    return l_per_100km, cost_vnd


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("== 1. Doc va lam sach du lieu ==")
    df = load_data()
    df = clean_data(df)
    print(df.head())
    print(f"So dong: {len(df)}, so cot: {df.shape[1]}")

    run_eda(df)

    X = df.drop(columns=["mpg"])
    y = df["mpg"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    print("\n== 2. Baseline: Dummy / Linear / RandomForest ==")
    baseline_results = []
    eval_model("DummyRegressor", DummyRegressor(strategy="mean"),
               X_train, X_test, y_train, y_test, baseline_results)
    eval_model("LinearRegression", LinearRegression(),
               X_train, X_test, y_train, y_test, baseline_results)
    eval_model("RandomForest", RandomForestRegressor(random_state=RANDOM_STATE),
               X_train, X_test, y_train, y_test, baseline_results)
    eval_model("SVR", Pipeline([("scale", StandardScaler()), ("svr", SVR())]),
               X_train, X_test, y_train, y_test, baseline_results)

    print("\n== 3. MLP: anh huong cua scaling ==")
    scale_df = compare_scaling(X_train, X_test, y_train, y_test)

    print("\n== 4. MLP: so sanh kien truc ==")
    arch_df = compare_architectures(X_train, X_test, y_train, y_test)

    print("\n== 5. Huan luyen mo hinh tot nhat (scale X&y, (64,32)) de ve loss curve ==")
    net = Pipeline([
        ("scale", StandardScaler()),
        ("mlp", MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-2,
                              max_iter=2000, early_stopping=True,
                              n_iter_no_change=30, random_state=RANDOM_STATE)),
    ])
    best_model = TransformedTargetRegressor(regressor=net, transformer=StandardScaler())
    best_model.fit(X_train, y_train)
    plot_loss_curve(best_model)

    print("\n== 6. Do alpha ==")
    alpha_df = sweep_alpha(X_train, X_test, y_train, y_test)

    print("\n== 7. So sanh activation ==")
    act_df = compare_activations(X_train, X_test, y_train, y_test)

    print("\n== 8. Luu model ==")
    joblib.dump(best_model, f"{MODELS_DIR}/mlp_reg.joblib")
    print(f"Da luu {MODELS_DIR}/mlp_reg.joblib")

    print("\n== 9. Vi du quy doi ra L/100km va tien xang/nam ==")
    sample_pred = best_model.predict(X_test.iloc[:3])
    for mpg_val in sample_pred:
        l100, cost = estimate_yearly_fuel_cost(mpg_val)
        print(f"MPG={mpg_val:.1f} -> {l100:.2f} L/100km -> ~{cost:,.0f} VND/nam "
              f"(15.000km/nam, 23.000 VND/lit)")

    print("\n== HOAN THANH ==")
    print("Xem cac file ket qua trong thu muc reports/")


if __name__ == "__main__":
    main()