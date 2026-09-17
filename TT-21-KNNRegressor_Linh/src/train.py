import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
REPORTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

RANDOM_STATE = 42


def rmse(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred)

# ----------------------------------------------------------------------
# 1. Nạp dữ liệu + xử lý outlier
# ----------------------------------------------------------------------
from sklearn.datasets import fetch_california_housing


def load_data():
    data = fetch_california_housing(as_frame=True)
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # Xử lý outlier trên tập train (AveOccup, AveRooms có vài dòng cực đoan)
    mask = (X_train["AveOccup"] < 10) & (X_train["AveRooms"] < 20)
    n_removed = (~mask).sum()
    X_train_clean = X_train[mask]
    y_train_clean = y_train.loc[X_train_clean.index]

    print(f"[1] Đã loại {n_removed} dòng outlier khỏi tập train "
          f"({len(X_train)} -> {len(X_train_clean)})")

    return X_train_clean, X_test, y_train_clean, y_test


# ----------------------------------------------------------------------
# 2. Baseline: Dummy + Linear Regression
# ----------------------------------------------------------------------
def baseline(X_train, X_test, y_train, y_test, results):
    dummy = DummyRegressor(strategy="mean").fit(X_train, y_train)
    lin = LinearRegression().fit(X_train, y_train)

    results["Dummy"] = {
        "rmse": rmse(y_test, dummy.predict(X_test)),
        "r2": r2_score(y_test, dummy.predict(X_test)),
    }
    results["Linear"] = {
        "rmse": rmse(y_test, lin.predict(X_test)),
        "r2": r2_score(y_test, lin.predict(X_test)),
    }
    print(f"[2] Dummy RMSE={results['Dummy']['rmse']:.4f} | "
          f"Linear RMSE={results['Linear']['rmse']:.4f} "
          f"(R2={results['Linear']['r2']:.4f})")
    return lin


# ----------------------------------------------------------------------
# 3. KNN KHÔNG scale -> chứng minh vì sao phải scale
# ----------------------------------------------------------------------
def knn_no_scale(X_train, X_test, y_train, y_test, results):
    knn = KNeighborsRegressor(n_neighbors=5, n_jobs=-1).fit(X_train, y_train)
    r = rmse(y_test, knn.predict(X_test))
    results["KNN_khong_scale"] = {"rmse": r}
    print(f"[3] KNN KHÔNG scale, K=5 -> RMSE={r:.4f}  "
          f"(so với Linear={results['Linear']['rmse']:.4f}: "
          f"{'TỆ HƠN' if r > results['Linear']['rmse'] else 'tốt hơn'})")
    return r


# ----------------------------------------------------------------------
# 4. KNN có scale, K=5 mặc định
# ----------------------------------------------------------------------
def knn_scaled_default(X_train, X_test, y_train, y_test, results):
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("knn", KNeighborsRegressor(n_neighbors=5, weights="distance", n_jobs=-1)),
    ]).fit(X_train, y_train)
    r = rmse(y_test, pipe.predict(X_test))
    results["KNN_scaled_K5"] = {"rmse": r}
    print(f"[4] KNN có scale, K=5, weights=distance -> RMSE={r:.4f}")
    return pipe

# ----------------------------------------------------------------------
# 5. RMSE train/test theo K = 1..50
# ----------------------------------------------------------------------
def sweep_k(X_train, X_test, y_train, y_test, results):
    K_range = range(1, 51)
    train_rmse, test_rmse = [], []
    for k in K_range:
        p = Pipeline([
            ("scale", StandardScaler()),
            ("knn", KNeighborsRegressor(n_neighbors=k, weights="distance", n_jobs=-1)),
        ]).fit(X_train, y_train)
        train_rmse.append(rmse(y_train, p.predict(X_train)))
        test_rmse.append(rmse(y_test, p.predict(X_test)))

    best_k = list(K_range)[int(np.argmin(test_rmse))]
    results["best_K"] = best_k
    results["K1_train_rmse"] = train_rmse[0]
    results["K1_test_rmse"] = test_rmse[0]

    plt.figure(figsize=(8, 5))
    plt.plot(list(K_range), train_rmse, label="Train RMSE", marker=".")
    plt.plot(list(K_range), test_rmse, label="Test RMSE", marker=".")
    plt.axvline(best_k, color="gray", linestyle="--", alpha=0.6,
                label=f"K tối ưu = {best_k}")
    plt.xlabel("K (số hàng xóm)")
    plt.ylabel("RMSE")
    plt.title("RMSE Train/Test theo K — KNN Regressor")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORTS / "rmse_theo_K.png", dpi=150)
    plt.close()

    print(f"[5] K=1: train RMSE={train_rmse[0]:.4f} (≈0, overfit hoàn toàn), "
          f"test RMSE={test_rmse[0]:.4f}")
    print(f"    K tối ưu theo test RMSE = {best_k} (test RMSE={min(test_rmse):.4f})")
    return best_k

# ----------------------------------------------------------------------
# 6. weights: uniform vs distance | 7. metric: euclidean vs manhattan
# ----------------------------------------------------------------------
def compare_weights_metric(X_train, X_test, y_train, y_test, best_k, results):
    for w in ["uniform", "distance"]:
        p = Pipeline([
            ("scale", StandardScaler()),
            ("knn", KNeighborsRegressor(n_neighbors=best_k, weights=w, n_jobs=-1)),
        ]).fit(X_train, y_train)
        r = rmse(y_test, p.predict(X_test))
        results.setdefault("weights", {})[w] = r
        print(f"[6] weights={w:9s} -> RMSE={r:.4f}")

    for m in ["euclidean", "manhattan"]:
        p = Pipeline([
            ("scale", StandardScaler()),
            ("knn", KNeighborsRegressor(n_neighbors=best_k, weights="distance",
                                         metric=m, n_jobs=-1)),
        ]).fit(X_train, y_train)
        r = rmse(y_test, p.predict(X_test))
        results.setdefault("metric", {})[m] = r
        print(f"[7] metric={m:10s} -> RMSE={r:.4f}")

# ----------------------------------------------------------------------
# 8. Thí nghiệm trọng số vị trí: nhân Lat/Lon x {1,2,3,5}
# ----------------------------------------------------------------------
def location_weight_experiment(X_train, X_test, y_train, y_test, best_k, results):
    he_so_list = [1, 2, 3, 5]
    rmses = []

    scaler = StandardScaler().fit(X_train)
    X_train_s = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns,
                              index=X_train.index)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns,
                             index=X_test.index)

    for he_so in he_so_list:
        Xtr = X_train_s.copy()
        Xte = X_test_s.copy()
        Xtr[["Latitude", "Longitude"]] *= he_so
        Xte[["Latitude", "Longitude"]] *= he_so

        knn = KNeighborsRegressor(n_neighbors=best_k, weights="distance",
                                   n_jobs=-1).fit(Xtr, y_train)
        r = rmse(y_test, knn.predict(Xte))
        rmses.append(r)
        print(f"[8] Hệ số Lat/Lon x{he_so} -> RMSE={r:.4f}")

    results["location_weight"] = dict(zip(he_so_list, rmses))
    best_he_so = he_so_list[int(np.argmin(rmses))]
    results["best_location_weight"] = best_he_so

    plt.figure(figsize=(7, 5))
    plt.plot(he_so_list, rmses, marker="o")
    plt.xlabel("Hệ số nhân Latitude/Longitude")
    plt.ylabel("RMSE")
    plt.title("Ảnh hưởng của trọng số vị trí lên RMSE")
    plt.axvline(best_he_so, color="gray", linestyle="--", alpha=0.6,
                label=f"Tốt nhất: x{best_he_so}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORTS / "trong_so_vi_tri.png", dpi=150)
    plt.close()

    print(f"    -> Hệ số tốt nhất: x{best_he_so}")


# ----------------------------------------------------------------------
# 9. In ra 5 "căn tương tự" cho 3 căn bất kỳ
# ----------------------------------------------------------------------
def similar_houses_examples(pipe, X_train, X_test, y_train, y_test, results):
    scaler = pipe["scale"]
    knn = pipe["knn"]
    X_test_scaled = scaler.transform(X_test)

    sample_idx = [0, 50, 100]  # 3 căn bất kỳ trong test
    lines = []
    for i in sample_idx:
        dist, idx = knn.kneighbors(X_test_scaled[[i]], n_neighbors=5)
        can_can_dinh_gia = X_test.iloc[i]
        gia_thuc = y_test.iloc[i]
        can_tuong_tu = X_train.iloc[idx[0]].copy()
        can_tuong_tu["Gia_ban"] = y_train.iloc[idx[0]].values
        can_tuong_tu["Khoang_cach"] = dist[0]
        du_doan = np.average(y_train.iloc[idx[0]].values, weights=1 / (dist[0] + 1e-9))

        lines.append(f"\n=== Căn cần định giá (test idx={i}) ===")
        lines.append(can_can_dinh_gia.to_string())
        lines.append(f"Giá thực tế: {gia_thuc:.3f} (x100k USD)")
        lines.append(f"Giá dự đoán (KNN, K=5): {du_doan:.3f}")
        lines.append("\n5 căn tương tự dùng để dự đoán:")
        lines.append(can_tuong_tu.to_string())
        lines.append("-" * 60)

    text = "\n".join(lines)
    (REPORTS / "can_tuong_tu_vi_du.txt").write_text(text, encoding="utf-8")
    print(f"[9] Đã ghi ví dụ 3 căn + 5 hàng xóm mỗi căn vào "
          f"reports/can_tuong_tu_vi_du.txt")

    # Hình minh hoạ: khoảng cách của 5 hàng xóm cho từng căn mẫu
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, i in zip(axes, sample_idx):
        dist, idx = knn.kneighbors(X_test_scaled[[i]], n_neighbors=5)
        gia = y_train.iloc[idx[0]].values
        ax.bar(range(5), gia, color="steelblue")
        ax.axhline(y_test.iloc[i], color="red", linestyle="--", label="Giá thực")
        ax.set_title(f"Test idx={i}")
        ax.set_xlabel("Hàng xóm thứ")
        ax.set_ylabel("Giá (x100k USD)")
        ax.legend(fontsize=8)
    plt.suptitle("Giá của 5 căn tương tự vs giá thực tế")
    plt.tight_layout()
    plt.savefig(REPORTS / "can_tuong_tu_vi_du.png", dpi=150)
    plt.close()

# ----------------------------------------------------------------------
# 10. Bảng so sánh Linear vs KNN vs RandomForest + thời gian
# ----------------------------------------------------------------------
def compare_models(X_train, X_test, y_train, y_test, lin, best_k, results):
    knn_pipe = Pipeline([
        ("scale", StandardScaler()),
        ("knn", KNeighborsRegressor(n_neighbors=best_k, weights="distance", n_jobs=-1)),
    ])
    rf = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE)

    models = {"Linear": lin, "KNN": knn_pipe, "RandomForest": rf}
    table = {}

    for name, m in models.items():
        t0 = time.time()
        m.fit(X_train, y_train)
        train_time = time.time() - t0

        t0 = time.time()
        preds = m.predict(X_test)
        predict_time_per_sample = (time.time() - t0) / len(X_test) * 1000  # ms

        table[name] = {
            "RMSE": round(rmse(y_test, preds), 4),
            "R2": round(r2_score(y_test, preds), 4),
            "train_time_s": round(train_time, 3),
            "predict_time_ms_per_sample": round(predict_time_per_sample, 5),
            "giai_thich_duoc": "Có" if name != "RandomForest" else "Một phần",
        }
        print(f"[10] {name:12s} RMSE={table[name]['RMSE']:.4f}  "
              f"R2={table[name]['R2']:.4f}  "
              f"train={train_time:.2f}s  "
              f"predict/sample={predict_time_per_sample:.4f}ms")

    results["model_comparison"] = table
    return knn_pipe

# ----------------------------------------------------------------------
# 11. Thời gian dự đoán khi train tăng 1x / 5x / 10x
# ----------------------------------------------------------------------
def scaling_time_experiment(X_train, X_test, y_train, best_k, results):
    factors = [1, 5, 10]
    times = []

    for f in factors:
        X_big = pd.concat([X_train] * f, ignore_index=True)
        y_big = pd.concat([y_train] * f, ignore_index=True)

        p = Pipeline([
            ("scale", StandardScaler()),
            ("knn", KNeighborsRegressor(n_neighbors=best_k, weights="distance", n_jobs=-1)),
        ]).fit(X_big, y_big)

        t0 = time.time()
        p.predict(X_test)
        elapsed = (time.time() - t0) / len(X_test) * 1000  # ms/sample
        times.append(elapsed)
        print(f"[11] Train x{f:>2d} ({len(X_big)} dòng) -> "
              f"predict/sample={elapsed:.4f}ms")

    results["scaling_time_ms_per_sample"] = dict(zip(factors, times))

    plt.figure(figsize=(7, 5))
    plt.plot(factors, times, marker="o", color="darkorange")
    plt.xlabel("Hệ số nhân kích thước train")
    plt.ylabel("Thời gian dự đoán / mẫu (ms)")
    plt.title("KNN chậm dần khi dữ liệu train tăng")
    plt.tight_layout()
    plt.savefig(REPORTS / "thoi_gian_predict.png", dpi=150)
    plt.close()

# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    results = {}

    X_train, X_test, y_train, y_test = load_data()
    lin = baseline(X_train, X_test, y_train, y_test, results)
    knn_no_scale(X_train, X_test, y_train, y_test, results)
    pipe_default = knn_scaled_default(X_train, X_test, y_train, y_test, results)
    best_k = sweep_k(X_train, X_test, y_train, y_test, results)
    compare_weights_metric(X_train, X_test, y_train, y_test, best_k, results)
    location_weight_experiment(X_train, X_test, y_train, y_test, best_k, results)

    # Pipeline cuối dùng best_k để lấy ví dụ + so sánh + lưu model
    final_pipe = Pipeline([
        ("scale", StandardScaler()),
        ("knn", KNeighborsRegressor(n_neighbors=best_k, weights="distance", n_jobs=-1)),
    ]).fit(X_train, y_train)

    similar_houses_examples(final_pipe, X_train, X_test, y_train, y_test, results)
    compare_models(X_train, X_test, y_train, y_test, lin, best_k, results)
    scaling_time_experiment(X_train, X_test, y_train, best_k, results)

    joblib.dump(final_pipe, MODELS / "knn_pipeline.joblib")
    (REPORTS / "results_summary.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n=== HOÀN TẤT ===")
    print(f"Model đã lưu: {MODELS / 'knn_pipeline.joblib'}")
    print(f"Kết quả tổng hợp: {REPORTS / 'results_summary.json'}")


if __name__ == "__main__":
    main()