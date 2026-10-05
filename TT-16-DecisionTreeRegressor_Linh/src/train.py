import argparse
import os
import urllib.request

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error)
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree

SEED = 42
N_SAMPLE = 200_000
CHOSEN_DEPTH = 5 
DATA_URL = ("https://d37ci6vzurychx.cloudfront.net/trip-data/"
            "yellow_tripdata_2026-01.parquet")

# Đường dẫn TƯƠNG ĐỐI theo vị trí file -> chạy được trên mọi máy
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA = os.path.join(BASE_DIR, "dataset", "yellow_tripdata_2026-01.parquet")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
MODEL_DIR = os.path.join(BASE_DIR, "models")

FEATURES = ["trip_distance", "passenger_count", "PULocationID",
            "DOLocationID", "hour", "dow", "is_rush_hour"]
LOOKUP_FEATURES = ["trip_distance", "is_rush_hour"]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.environ.get("TAXI_DATA", DEFAULT_DATA))
    p.add_argument("--download", action="store_true",
                   help="tải parquet từ NYC TLC nếu chưa có")
    return p.parse_args()

def rel(path):
    try:
        return os.path.relpath(path, BASE_DIR)
    except ValueError:
        return os.path.basename(path)

def leaf_rules(tree, feature_names):
    t = tree.tree_
    out = {}

    def walk(node, bounds):
        if t.children_left[node] == -1:
            out[node] = dict(bounds)
            return
        f = feature_names[t.feature[node]]
        thr = float(t.threshold[node])
        lo, hi = bounds.get(f, (-np.inf, np.inf))
        walk(t.children_left[node], {**bounds, f: (lo, min(hi, thr))})
        walk(t.children_right[node], {**bounds, f: (max(lo, thr), hi)})

    walk(0, {})
    return out

def build_lookup_table(tree, X_train, feature_names):
    rules = leaf_rules(tree, feature_names)
    counts = pd.Series(tree.apply(X_train)).value_counts()
    rows = []
    for leaf, b in rules.items():
        lo, hi = b.get("trip_distance", (-np.inf, np.inf))
        rlo, rhi = b.get("is_rush_hour", (-np.inf, np.inf))
        if rhi <= 0.5:
            rush = "Không"
        elif rlo >= 0.5:
            rush = "Có"
        else:
            rush = "Cả hai"
        rows.append({
            "leaf_id": leaf,
            "gio_cao_diem": rush,
            "dam_tu_(>)": 0.0 if np.isinf(lo) else round(lo, 2),
            "dam_den_(<=)": "không giới hạn" if np.isinf(hi) else round(hi, 2),
            "gia_de_xuat_usd": round(float(tree.tree_.value[leaf][0][0]), 1),
            "so_chuyen_lam_can_cu": int(counts.get(leaf, 0)),
        })
    df = pd.DataFrame(rows)
    df["_k"] = df["dam_tu_(>)"]
    return df.sort_values(["gio_cao_diem", "_k"]).drop(columns="_k")

def main():
    args = parse_args()
    for d in (OUT_DIR, REPORT_DIR, MODEL_DIR):
        os.makedirs(d, exist_ok=True)

    log_lines = []

    def log(msg=""):
        print(msg)
        log_lines.append(str(msg))

    log("=" * 70)
    log("TT-16 — DECISION TREE REGRESSOR — ĐỊNH GIÁ CƯỚC TAXI (DỮ LIỆU THẬT)")
    log("=" * 70)

    # ---- BƯỚC 1: TẢI DỮ LIỆU ----
    data_path = args.data
    if not os.path.exists(data_path):
        if args.download:
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            log(f"[BƯỚC 1] Đang tải dữ liệu từ {DATA_URL}")
            urllib.request.urlretrieve(DATA_URL, data_path)
        else:
            raise SystemExit(
                f"Không thấy file dữ liệu: {data_path}\n"
                f"Tải tại: {DATA_URL}\n"
                f"hoặc chạy lại với --download (xem README)."
            )
    log(f"\n[BƯỚC 1] Đọc dữ liệu thật từ {rel(data_path)}")
    df = pd.read_parquet(data_path)
    log(f"  -> Số dòng đọc được từ file gốc: {len(df):,}")
    if len(df) > N_SAMPLE:
        df = df.sample(n=N_SAMPLE, random_state=SEED)
    log(f"  -> Số dòng sau khi lấy mẫu: {len(df):,}")

    # ---- BƯỚC 2: LÀM SẠCH ----
    log("\n[BƯỚC 2] Làm sạch dữ liệu")
    n0 = len(df)
    df = df[df["fare_amount"] > 0]
    n1 = len(df)
    log(f"  - Loại vì fare_amount <= 0             : {n0 - n1:>7,} dòng")
    df = df[(df["trip_distance"] > 0) & (df["trip_distance"] <= 100)]
    n2 = len(df)
    log(f"  - Loại vì trip_distance <=0 hoặc >100  : {n1 - n2:>7,} dòng")
    df = df[df["passenger_count"] > 0]
    n3 = len(df)
    log(f"  - Loại vì passenger_count == 0         : {n2 - n3:>7,} dòng")
    log(f"  -> Còn lại sau làm sạch: {n3:,} / {n0:,} dòng ({n3 / n0 * 100:.1f}%)")
    log("\n  [!] RÒ RỈ DỮ LIỆU: KHÔNG đưa tip_amount, tolls_amount, total_amount")
    log("      vào đặc trưng X, vì chúng chỉ biết SAU khi chuyến đi kết thúc.")

    # ---- BƯỚC 3: ĐẶC TRƯNG + CHIA TRAIN/TEST ----
    log("\n[BƯỚC 3] Tạo đặc trưng thời gian")
    df["hour"] = df["tpep_pickup_datetime"].dt.hour
    df["dow"] = df["tpep_pickup_datetime"].dt.dayofweek
    df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
    X, y = df[FEATURES], df["fare_amount"]
    log(f"  -> Đặc trưng sử dụng: {FEATURES}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED)
    log("  -> TEST được 'khoá' đến BƯỚC 11: mọi lựa chọn siêu tham số")
    log("     đều làm bằng cross-validation trên TRAIN.")

    # ---- BƯỚC 4: EDA ----
    log("\n[BƯỚC 4] EDA")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].scatter(df["trip_distance"], df["fare_amount"], s=2, alpha=0.15, color="#2a78d6")
    axes[0].set(xlabel="Quãng đường (dặm)", ylabel="Cước phí (USD)",
                title="Quãng đường vs Cước phí", xlim=(0, 40))
    hourly_avg = df.groupby("hour")["fare_amount"].mean()
    axes[1].bar(hourly_avg.index, hourly_avg.values, color="#eb6834")
    axes[1].set(xlabel="Giờ trong ngày", ylabel="Cước trung bình (USD)",
                title="Cước trung bình theo giờ")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "eda_overview.png"), dpi=130)
    plt.close()
    log("  -> Đã lưu reports/eda_overview.png")

    # ---- BƯỚC 5: BASELINE ----
    log("\n[BƯỚC 5] Baseline")
    dummy = DummyRegressor(strategy="mean").fit(X_train, y_train)
    mae_dummy = mean_absolute_error(y_test, dummy.predict(X_test))
    log(f"  - DummyRegressor (mean)             MAE = {mae_dummy:.2f}")
    mae_manual = mean_absolute_error(y_test, 3.0 + 2.0 * X_test["trip_distance"])
    log(f"  - Công thức thủ công (3 + 2*dặm)    MAE = {mae_manual:.2f}")

    # ---- BƯỚC 6: CÂY KHÔNG GIỚI HẠN -> OVERFIT ----
    log("\n[BƯỚC 6] Cây không giới hạn độ sâu (chứng minh overfit)")
    tree_full = DecisionTreeRegressor(random_state=SEED).fit(X_train, y_train)
    mae_train_full = mean_absolute_error(y_train, tree_full.predict(X_train))
    mae_test_full = mean_absolute_error(y_test, tree_full.predict(X_test))
    log(f"  - Số lá: {tree_full.get_n_leaves():,} | Số tầng: {tree_full.get_depth()}")
    log(f"  - MAE train = {mae_train_full:.3f}  (gần 0 -> cây 'học thuộc' dữ liệu)")
    log(f"  - MAE test  = {mae_test_full:.3f}  (cao hơn nhiều -> OVERFIT rõ ràng)")

    # ---- BƯỚC 7: CHỌN max_depth BẰNG CROSS-VALIDATION TRÊN TRAIN ----
    log("\n[BƯỚC 7] Quét max_depth 1..20 bằng 5-fold CV trên TẬP TRAIN")
    depths = list(range(1, 21))
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_tr, cv_va, cv_se = [], [], []
    for d in depths:
        t = DecisionTreeRegressor(max_depth=d, min_samples_leaf=500, random_state=SEED)
        r = cross_validate(t, X_train, y_train, cv=kf, n_jobs=-1,
                           scoring="neg_mean_absolute_error", return_train_score=True)
        cv_tr.append(-r["train_score"].mean())
        cv_va.append(-r["test_score"].mean())
        cv_se.append(r["test_score"].std(ddof=1) / np.sqrt(len(r["test_score"])))
    i_best = int(np.argmin(cv_va))
    best_depth = depths[i_best]
    # Quy tắc 1-SE: cây đơn giản nhất mà CV MAE không tệ hơn tốt nhất quá 1 sai số chuẩn
    thr = cv_va[i_best] + cv_se[i_best]
    depth_1se = next(d for d, v in zip(depths, cv_va) if v <= thr)
    cv5 = cv_va[CHOSEN_DEPTH - 1]
    log(f"  -> Độ sâu tốt nhất theo CV (MAE val)     : {best_depth} (CV MAE = {cv_va[i_best]:.3f})")
    log(f"  -> Độ sâu theo quy tắc 1-SE (gọn nhất)   : {depth_1se}")
    log(f"  -> max_depth={CHOSEN_DEPTH} (ràng buộc nghiệp vụ): CV MAE = {cv5:.3f}, "
        f"kém tối ưu CV {cv5 - cv_va[i_best]:.3f} USD ({(cv5 / cv_va[i_best] - 1) * 100:.1f}%)")
    log("  -> Chọn depth=5 để tổng đài tra bảng bằng tay; đã ĐO cái giá phải trả")
    log("     bằng CV (không nhìn test), không phải là chọn theo cảm tính.")

    plt.figure(figsize=(7, 4.5))
    plt.plot(depths, cv_tr, marker="o", label="MAE train (CV)", color="#2a78d6")
    plt.plot(depths, cv_va, marker="o", label="MAE validation (CV)", color="#eb6834")
    plt.axvline(CHOSEN_DEPTH, color="gray", linestyle="--", linewidth=1,
                label=f"max_depth={CHOSEN_DEPTH} (chọn)")
    plt.axvline(best_depth, color="green", linestyle=":", linewidth=1,
                label=f"tốt nhất theo CV = {best_depth}")
    plt.xlabel("max_depth")
    plt.ylabel("MAE")
    plt.title("MAE train/validation (5-fold CV) theo độ sâu cây")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "mae_theo_depth.png"), dpi=130)
    plt.close()
    log("  -> Đã lưu reports/mae_theo_depth.png")

    # ---- BƯỚC 8: HÀM BẬC THANG ----
    log("\n[BƯỚC 8] Vẽ hàm dự đoán theo quãng đường (bậc thang)")
    tree5 = DecisionTreeRegressor(max_depth=CHOSEN_DEPTH, min_samples_leaf=500,
                                  criterion="squared_error", random_state=SEED)
    tree5.fit(X_train, y_train)
    dist_range = np.linspace(0.1, 30, 300)
    grid = pd.DataFrame([X_train.iloc[0].copy()] * len(dist_range))
    grid["trip_distance"] = dist_range  # các đặc trưng khác giữ cố định
    plt.figure(figsize=(7, 4.5))
    plt.scatter(X_train["trip_distance"], y_train, s=2, alpha=0.1, color="#888780",
                label="Dữ liệu thật (train)")
    plt.plot(dist_range, tree5.predict(grid), color="#eb6834", linewidth=2.5,
             label="Cây dự đoán (bậc thang)")
    plt.xlim(0, 30)
    plt.xlabel("Quãng đường (dặm)")
    plt.ylabel("Cước phí (USD)")
    plt.title("Hàm dự đoán của cây là hàm BẬC THANG, không phải đường mượt")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "ham_bac_thang.png"), dpi=130)
    plt.close()
    log("  -> Đã lưu reports/ham_bac_thang.png")
    log(f"  -> Số mức giá (lá) tối đa có thể trả ra: {tree5.get_n_leaves()}")

    # ---- BƯỚC 9: EXPORT TEXT + VẼ CÂY ----
    log("\n[BƯỚC 9] Xuất cây dạng text và hình ảnh")
    with open(os.path.join(REPORT_DIR, "cay_quyet_dinh.txt"), "w", encoding="utf-8") as f:
        f.write(export_text(tree5, feature_names=FEATURES))
    log("  -> Đã lưu reports/cay_quyet_dinh.txt")
    plt.figure(figsize=(20, 10))
    plot_tree(tree5, feature_names=FEATURES, filled=True, rounded=True,
              fontsize=7, max_depth=3, proportion=True)
    plt.title("Cây quyết định (hiển thị 3 tầng đầu, đầy đủ 5 tầng trong file text)")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "cay_quyet_dinh.png"), dpi=130)
    plt.close()
    log("  -> Đã lưu reports/cay_quyet_dinh.png")

    # ---- BƯỚC 10: BẢNG TRA CƯỚC (KHÔNG CHỒNG LẤN) ----
    log("\n[BƯỚC 10] Xuất bảng tra cước CSV cho tổng đài")
    log("  Vấn đề cũ: cây tách thêm theo LocationID/giờ nên cùng một khoảng dặm")
    log("  xuất hiện ở nhiều lá -> km_min/km_max chồng lấn, tra nhầm.")
    log(f"  Cách sửa: cây tra cứu riêng chỉ dùng {LOOKUP_FEATURES}; mỗi dòng bảng là")
    log("  (giờ cao điểm?) x (khoảng dặm) với cận lấy từ ĐƯỜNG ĐI của cây.")
    tree_lookup = DecisionTreeRegressor(max_depth=CHOSEN_DEPTH, min_samples_leaf=500,
                                        random_state=SEED)
    tree_lookup.fit(X_train[LOOKUP_FEATURES], y_train)
    bang = build_lookup_table(tree_lookup, X_train[LOOKUP_FEATURES], LOOKUP_FEATURES)
    bang.to_csv(os.path.join(OUT_DIR, "bang_tra_cuoc.csv"), index=False, encoding="utf-8-sig")
    # Kiểm tra không chồng lấn trong từng nhóm giờ
    ok = True
    for _, g in bang.groupby("gio_cao_diem"):
        lo = g["dam_tu_(>)"].astype(float).to_numpy()
        hi = pd.to_numeric(g["dam_den_(<=)"], errors="coerce").fillna(np.inf).to_numpy()
        order = np.argsort(lo)
        ok &= bool(np.all(hi[order][:-1] <= lo[order][1:] + 1e-9))
    log(f"  -> Đã lưu outputs/bang_tra_cuoc.csv ({len(bang)} mức giá) | "
        f"kiểm tra không chồng lấn: {'ĐẠT' if ok else 'KHÔNG ĐẠT'}")
    pred_lookup = tree_lookup.predict(X_test[LOOKUP_FEATURES])
    mae_lookup = mean_absolute_error(y_test, pred_lookup)
    within15_lookup = (np.abs(pred_lookup - y_test) / y_test <= 0.15).mean() * 100
    log(f"  -> Cây tra cứu: MAE test = {mae_lookup:.2f} | ±15% = {within15_lookup:.1f}%")

    # ---- BƯỚC 11: ĐÁNH GIÁ CUỐI TRÊN TEST (DÙNG 1 LẦN) ----
    log("\n[BƯỚC 11] Đánh giá cuối trên TEST (chỉ chạy 1 lần, sau khi đã chốt mô hình)")
    pred_test = tree5.predict(X_test)
    mae5 = mean_absolute_error(y_test, pred_test)
    mape5 = mean_absolute_percentage_error(y_test, pred_test) * 100
    rmse5 = mean_squared_error(y_test, pred_test) ** 0.5
    within_15 = (np.abs(pred_test - y_test) / y_test <= 0.15).mean() * 100
    log(f"  - MAE  = {mae5:.2f}")
    log(f"  - MAPE = {mape5:.2f}%")
    log(f"  - RMSE = {rmse5:.2f}")
    log(f"  - % chuyến sai số trong ±15%: {within_15:.1f}%")

    # ---- BƯỚC 12: SO SÁNH ----
    log("\n[BƯỚC 12] So sánh với Random Forest và Linear Regression")
    lin = LinearRegression().fit(X_train, y_train)
    mae_lin = mean_absolute_error(y_test, lin.predict(X_test))
    rf = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=200,
                               random_state=SEED, n_jobs=-1).fit(X_train, y_train)
    mae_rf = mean_absolute_error(y_test, rf.predict(X_test))
    log(f"  - Decision Tree (depth={CHOSEN_DEPTH})  MAE = {mae5:.2f}")
    log(f"  - Linear Regression        MAE = {mae_lin:.2f}")
    log(f"  - Random Forest            MAE = {mae_rf:.2f}")

    plt.figure(figsize=(6, 4))
    plt.bar(["Dummy", f"Tree d={CHOSEN_DEPTH}", "Linear", "Random\nForest"],
            [mae_dummy, mae5, mae_lin, mae_rf],
            color=["#888780", "#eb6834", "#2a78d6", "#1baf7a"])
    plt.ylabel("MAE")
    plt.title("So sánh MAE giữa các mô hình")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "so_sanh_mo_hinh.png"), dpi=130)
    plt.close()
    log("  -> Đã lưu reports/so_sanh_mo_hinh.png")

    # ---- LƯU MODEL + LOG ----
    joblib.dump(tree5, os.path.join(MODEL_DIR, "tree_reg.joblib"))
    joblib.dump(tree_lookup, os.path.join(MODEL_DIR, "tree_lookup.joblib"))
    log("\n-> Đã lưu models/tree_reg.joblib, models/tree_lookup.joblib")
    log("\n" + "=" * 70)
    log("HOÀN TẤT")
    log("=" * 70)
    with open(os.path.join(REPORT_DIR, "log_chay.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    return {"mae5": mae5, "mape5": mape5, "rmse5": rmse5, "within_15": within_15,
            "best_depth_cv": best_depth, "depth_1se": depth_1se, "cv_mae_depth5": cv5,
            "mae_lookup": mae_lookup, "mae_dummy": mae_dummy, "mae_manual": mae_manual,
            "mae_lin": mae_lin, "mae_rf": mae_rf,
            "mae_train_full": mae_train_full, "mae_test_full": mae_test_full}


if __name__ == "__main__":
    main()