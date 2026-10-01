import argparse
import json
import os
import time
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, cross_validate, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

RANDOM_STATE = 42
REPORTS_DIR = "reports"
MODELS_DIR = "models"
DATA_CANDIDATES = ["dataset/auto-mpg.data", "data/auto-mpg.data"]

COLUMN_NAMES = [
    "mpg", "cylinders", "displacement", "horsepower", "weight",
    "acceleration", "model_year", "origin", "car_name",
]

CV = RepeatedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)
# --------------------------------------------------------------------------
# 1. LOAD & CLEAN
# --------------------------------------------------------------------------
def find_data_path(cli_path=None):
    candidates = [cli_path] if cli_path else DATA_CANDIDATES
    for p in candidates:
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError(
        "Khong tim thay auto-mpg.data. Da thu: " + ", ".join(c for c in candidates if c)
        + ". Dat file vao dataset/ (hoac data/) hoac dung --data <duong_dan>."
    )


def load_data(path):
    return pd.read_csv(path, sep=r"\s+", names=COLUMN_NAMES, na_values="?", quotechar='"')


def clean_data(df):
    df = df.copy()
    df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")
    n_missing = int(df["horsepower"].isna().sum())
    print(f"[clean] So gia tri thieu o horsepower: {n_missing} -> dien median")
    # Luu y: median tinh tren toan bo du lieu la mot ro ri nho (6 dong). Chap nhan duoc
    # o day, nhung cach chuan la dua SimpleImputer vao Pipeline.
    df["horsepower"] = df["horsepower"].fillna(df["horsepower"].median())
    df = df.drop(columns=["car_name"])
    df["origin"] = df["origin"].astype(int)
    return pd.get_dummies(df, columns=["origin"], prefix="origin")


# --------------------------------------------------------------------------
# 2. EDA
# --------------------------------------------------------------------------
def run_eda(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(df["weight"], df["mpg"], alpha=0.5)
    axes[0].set(xlabel="weight", ylabel="mpg", title="weight vs mpg (nghich, hoi cong)")
    axes[1].scatter(df["model_year"], df["mpg"], alpha=0.5)
    axes[1].set(xlabel="model_year", ylabel="mpg", title="mpg theo model_year")
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/eda.png", dpi=120)
    plt.close()


# --------------------------------------------------------------------------
# 3. HAM DANH GIA
# --------------------------------------------------------------------------
def make_mlp(arch=(64, 32), alpha=1e-4, activation="relu", scale_x=True, scale_y=True):
    """Tao MLP voi tuy chon scale X / scale y."""
    mlp = MLPRegressor(
        hidden_layer_sizes=arch, alpha=alpha, activation=activation,
        max_iter=2000, early_stopping=True, n_iter_no_change=30,
        random_state=RANDOM_STATE,
    )
    model = Pipeline([("scale", StandardScaler()), ("mlp", mlp)]) if scale_x else mlp
    if scale_y:
        model = TransformedTargetRegressor(regressor=model, transformer=StandardScaler())
    return model


def cv_eval(name, model, X_train, y_train, extra=None):
    """Danh gia bang CV tren TAP TRAIN. Tra ve dict mean +- std, gap train/val."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv = cross_validate(
            model, X_train, y_train, cv=CV, n_jobs=-1, return_train_score=True,
            scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2"},
        )
    rmse_val = -cv["test_rmse"]
    rmse_tr = -cv["train_rmse"]
    row = {
        "model": name,
        "cv_rmse_val_mean": round(rmse_val.mean(), 3),
        "cv_rmse_val_std": round(rmse_val.std(), 3),
        "cv_rmse_train_mean": round(rmse_tr.mean(), 3),
        "cv_r2_val_mean": round(cv["test_r2"].mean(), 4),
        # gap = val - train, tinh trung binh 15 fold; kem std de biet co khac 0 khong
        "overfit_gap": round((rmse_val - rmse_tr).mean(), 3),
        "overfit_gap_std": round((rmse_val - rmse_tr).std(), 3),
    }
    if extra:
        row.update(extra)
    print(f"[CV] {name:<28} RMSE val={row['cv_rmse_val_mean']:.3f}+-{row['cv_rmse_val_std']:.3f}  "
          f"train={row['cv_rmse_train_mean']:.3f}  gap={row['overfit_gap']:+.3f}")
    return row


def count_params(n_features, arch):
    n, prev = 0, n_features
    for h in arch:
        n += prev * h + h
        prev = h
    return n + prev + 1


# --------------------------------------------------------------------------
# 4. CAC THI NGHIEM (tat ca chon bang CV tren train)
# --------------------------------------------------------------------------
def compare_scaling(X_train, y_train):
    rows = [
        cv_eval("MLP (khong scale)", make_mlp(scale_x=False, scale_y=False), X_train, y_train),
        cv_eval("MLP (scale X)", make_mlp(scale_x=True, scale_y=False), X_train, y_train),
        cv_eval("MLP (scale X va y)", make_mlp(scale_x=True, scale_y=True), X_train, y_train),
    ]
    df = pd.DataFrame(rows)
    df.to_csv(f"{REPORTS_DIR}/scale_comparison.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(df["model"], df["cv_rmse_val_mean"], yerr=df["cv_rmse_val_std"], capsize=4)
    ax.set(ylabel="RMSE (CV val)", title="Anh huong cua scaling toi MLP")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/scale_comparison.png", dpi=120)
    plt.close()
    return df


def compare_architectures(X_train, y_train):
    rows = []
    for arch in [(16,), (64,), (64, 32), (256, 128, 64)]:
        rows.append(cv_eval(f"MLP{arch}", make_mlp(arch=arch), X_train, y_train,
                            extra={"arch": arch, "n_params": count_params(X_train.shape[1], arch)}))
    df = pd.DataFrame(rows)
    df.to_csv(f"{REPORTS_DIR}/architecture_comparison.csv", index=False)
    x, w = np.arange(len(df)), 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w / 2, df["cv_rmse_train_mean"], w, label="RMSE train (CV)")
    ax.bar(x + w / 2, df["cv_rmse_val_mean"], w, yerr=df["cv_rmse_val_std"], capsize=3,
           label="RMSE val (CV)")
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=20, ha="right")
    ax.set(ylabel="RMSE", title="Kien truc vs overfit (CV tren tap train)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/kien_truc_overfit.png", dpi=120)
    plt.close()
    return df


def sweep_alpha(X_train, y_train, arch):
    rows = [cv_eval(f"alpha={a}", make_mlp(arch=arch, alpha=a), X_train, y_train, extra={"alpha": a})
            for a in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]]
    df = pd.DataFrame(rows)
    df.to_csv(f"{REPORTS_DIR}/alpha_sweep.csv", index=False)
    plt.figure(figsize=(7, 5))
    plt.errorbar(df["alpha"], df["cv_rmse_val_mean"], yerr=df["cv_rmse_val_std"],
                 marker="o", capsize=3, label="RMSE val (CV)")
    plt.plot(df["alpha"], df["cv_rmse_train_mean"], marker="o", label="RMSE train (CV)")
    plt.xscale("log")
    plt.xlabel("alpha (log)")
    plt.ylabel("RMSE")
    plt.title("Anh huong cua alpha")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/alpha_sweep.png", dpi=120)
    plt.close()
    return df


def compare_activations(X_train, y_train, arch, alpha):
    rows = [cv_eval(f"activation={a}", make_mlp(arch=arch, alpha=alpha, activation=a),
                    X_train, y_train, extra={"activation": a})
            for a in ["relu", "tanh", "logistic"]]
    df = pd.DataFrame(rows)
    df.to_csv(f"{REPORTS_DIR}/activation_comparison.csv", index=False)
    return df


def pick_best(df, col):
    """Chon cau hinh co CV RMSE nho nhat (khong dung test)."""
    return df.loc[df["cv_rmse_val_mean"].idxmin(), col]


# --------------------------------------------------------------------------
# 5. LOSS CURVE
# --------------------------------------------------------------------------
def plot_loss_curve(model):
    mlp = model.regressor_.named_steps["mlp"]
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(mlp.loss_curve_, color="tab:blue", label="loss (train)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (train, tren y da scale)", color="tab:blue")
    # validation_scores_ la R2 (cang cao cang tot), khac don vi voi loss -> truc phu
    if getattr(mlp, "validation_scores_", None) is not None:
        ax2 = ax1.twinx()
        ax2.plot(mlp.validation_scores_, color="tab:orange", label="validation R2")
        ax2.set_ylabel("Validation R2", color="tab:orange")
    best_it = int(np.argmax(mlp.validation_scores_)) if getattr(mlp, "validation_scores_", None) else None
    if best_it is not None:
        ax1.axvline(best_it, ls="--", color="gray", alpha=0.6)
    plt.title(f"Loss curve MLP (dung o epoch {mlp.n_iter_})")
    fig.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/loss_curve.png", dpi=120)
    plt.close()


# --------------------------------------------------------------------------
# 6. DANH GIA CUOI TREN TEST (DUNG 1 LAN)
# --------------------------------------------------------------------------
def bootstrap_rmse_diff(y_true, pred_a, pred_b, n_boot=2000):
    """CI 95% cua RMSE(a) - RMSE(b) tren test. Neu CI chua 0 -> khong phan biet duoc."""
    rng = np.random.default_rng(RANDOM_STATE)
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    n, diffs = len(y_true), []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ra = mean_squared_error(y_true[idx], pred_a[idx]) ** 0.5
        rb = mean_squared_error(y_true[idx], pred_b[idx]) ** 0.5
        diffs.append(ra - rb)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def final_evaluation(models, X_train, X_test, y_train, y_test):
    rows, preds = [], {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, model in models.items():
            t0 = time.time()
            model.fit(X_train, y_train)
            fit_time = time.time() - t0
            p = model.predict(X_test)
            preds[name] = p
            rows.append({
                "model": name,
                "rmse_test": round(mean_squared_error(y_test, p) ** 0.5, 3),
                "r2_test": round(r2_score(y_test, p), 4),
                "train_time_s": round(fit_time, 3),
            })
    df = pd.DataFrame(rows)
    return df, preds


# --------------------------------------------------------------------------
# 7. QUY DOI L/100km
# --------------------------------------------------------------------------
def mpg_to_l_per_100km(mpg):
    return 235.215 / mpg


def estimate_yearly_fuel_cost(mpg, km_per_year=15000, price_per_liter_vnd=23000):
    l100 = mpg_to_l_per_100km(mpg)
    return l100, l100 * (km_per_year / 100) * price_per_liter_vnd


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="duong dan auto-mpg.data")
    args = ap.parse_args()

    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("== 1. Doc va lam sach du lieu ==")
    df = clean_data(load_data(find_data_path(args.data)))
    print(f"So dong: {len(df)}, so cot: {df.shape[1]}")
    run_eda(df)

    X, y = df.drop(columns=["mpg"]), df["mpg"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"Train={len(X_train)}  Test={len(X_test)} (test CHUA dung de chon gi ca)")

    print("\n== 2. Baseline (CV tren train) ==")
    baselines = {
        "DummyRegressor": DummyRegressor(strategy="mean"),
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(random_state=RANDOM_STATE),
        "SVR": Pipeline([("scale", StandardScaler()), ("svr", SVR())]),
    }
    cv_rows = [cv_eval(n, m, X_train, y_train) for n, m in baselines.items()]

    print("\n== 3. Scaling (CV) ==")
    scale_df = compare_scaling(X_train, y_train)

    print("\n== 4. Kien truc (CV) ==")
    arch_df = compare_architectures(X_train, y_train)
    best_arch = pick_best(arch_df, "arch")

    print(f"\n== 5. Alpha (CV), kien truc={best_arch} ==")
    alpha_df = sweep_alpha(X_train, y_train, best_arch)
    best_alpha = float(pick_best(alpha_df, "alpha"))

    print(f"\n== 6. Activation (CV), alpha={best_alpha} ==")
    act_df = compare_activations(X_train, y_train, best_arch, best_alpha)
    best_act = pick_best(act_df, "activation")

    print(f"\n== 7. MLP cuoi: arch={best_arch}, alpha={best_alpha}, act={best_act} ==")
    best_mlp = make_mlp(arch=best_arch, alpha=best_alpha, activation=best_act)
    cv_rows.append(cv_eval("MLP (best by CV)", best_mlp, X_train, y_train))
    cv_df = pd.DataFrame(cv_rows)

    print("\n== 8. Danh gia TEST (1 lan duy nhat) ==")
    all_models = {**baselines, "MLP (best by CV)": best_mlp}
    test_df, preds = final_evaluation(all_models, X_train, X_test, y_train, y_test)
    final_df = cv_df.merge(test_df, on="model")
    final_df.to_csv(f"{REPORTS_DIR}/final_comparison.csv", index=False)
    print(final_df.to_string(index=False))

    lo, hi = bootstrap_rmse_diff(y_test, preds["MLP (best by CV)"], preds["RandomForest"])
    print(f"\nRMSE(MLP) - RMSE(RF) tren test, CI95% = [{lo:+.3f}, {hi:+.3f}]  "
          f"-> {'KHONG phan biet duoc' if lo <= 0 <= hi else 'co khac biet'}")

    plot_loss_curve(best_mlp)
    joblib.dump(best_mlp, f"{MODELS_DIR}/mlp_reg.joblib")

    metrics = {
        "n_train": len(X_train), "n_test": len(X_test),
        "selected": {"arch": list(best_arch), "alpha": best_alpha, "activation": best_act},
        "selection_method": "RepeatedKFold(5x3) tren tap train; test chi dung 1 lan",
        "final": final_df.to_dict(orient="records"),
        "mlp_minus_rf_rmse_test_ci95": [lo, hi],
    }
    with open(f"{REPORTS_DIR}/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)

    print("\n== 9. Vi du quy doi L/100km va tien xang/nam ==")
    for mpg_val in best_mlp.predict(X_test.iloc[:3]):
        l100, cost = estimate_yearly_fuel_cost(mpg_val)
        print(f"MPG={mpg_val:.1f} -> {l100:.2f} L/100km -> ~{cost:,.0f} VND/nam "
              f"(15.000km/nam, 23.000 VND/lit)")

    print("\nHOAN THANH. Xem reports/ (final_comparison.csv, metrics.json, *.png)")


if __name__ == "__main__":
    main()