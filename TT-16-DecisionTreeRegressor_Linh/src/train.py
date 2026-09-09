import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import joblib
np.random.seed(42)
REAL_DATA_PATH = "D:\TT_ML\TT-16-DecisionTreeRegressor_Linh\dataset\yellow_tripdata_2026-01.parquet"
N_SAMPLE = 200_000

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "outputs")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
MODEL_DIR = os.path.join(BASE_DIR, "models")

for d in (OUT_DIR, REPORT_DIR, MODEL_DIR):
    os.makedirs(d, exist_ok=True)

# ----------------------------------------------------------------------------
# HÀM CHÍNH
# ----------------------------------------------------------------------------
def main():
    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(str(msg))

    log("=" * 70)
    log("TT-16 — DECISION TREE REGRESSOR — ĐỊNH GIÁ CƯỚC TAXI (DỮ LIỆU THẬT)")
    log("=" * 70)

    # ---- BƯỚC 1: TẢI DỮ LIỆU THẬT + LẤY MẪU ----
    log(f"\n[BƯỚC 1] Đọc dữ liệu thật từ {REAL_DATA_PATH}")
    df = pd.read_parquet(REAL_DATA_PATH)
    log(f"  -> Số dòng đọc được từ file gốc: {len(df):,}")

    if len(df) > N_SAMPLE:
        df = df.sample(n=N_SAMPLE, random_state=42)
    log(f"  -> Số dòng sau khi lấy mẫu: {len(df):,}")

    # ---- BƯỚC 2: LÀM SẠCH (4 bước bắt buộc) ----
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
    log(f"  -> Còn lại sau làm sạch: {n3:,} / {n0:,} dòng ({n3/n0*100:.1f}%)")

    log("\n  [!] RÒ RỈ DỮ LIỆU: KHÔNG đưa tip_amount, tolls_amount, total_amount")
    log("      vào đặc trưng X, vì chúng chỉ biết SAU khi chuyến đi kết thúc.")

    # ---- BƯỚC 3: ĐẶC TRƯNG THỜI GIAN ----
    log("\n[BƯỚC 3] Tạo đặc trưng thời gian")
    df["hour"] = df["tpep_pickup_datetime"].dt.hour
    df["dow"] = df["tpep_pickup_datetime"].dt.dayofweek
    df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)

    features = ["trip_distance", "passenger_count", "PULocationID",
                "DOLocationID", "hour", "dow", "is_rush_hour"]
    X = df[features]
    y = df["fare_amount"]
    log(f"  -> Đặc trưng sử dụng: {features}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ---- BƯỚC 4: EDA ----
    log("\n[BƯỚC 4] EDA")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].scatter(df["trip_distance"], df["fare_amount"], s=2, alpha=0.15, color="#2a78d6")
    axes[0].set_xlabel("Quãng đường (mile)")
    axes[0].set_ylabel("Cước phí (USD)")
    axes[0].set_title("Quãng đường vs Cước phí")
    axes[0].set_xlim(0, 40)

    hourly_avg = df.groupby("hour")["fare_amount"].mean()
    axes[1].bar(hourly_avg.index, hourly_avg.values, color="#eb6834")
    axes[1].set_xlabel("Giờ trong ngày")
    axes[1].set_ylabel("Cước trung bình (USD)")
    axes[1].set_title("Cước trung bình theo giờ")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "eda_overview.png"), dpi=130)
    plt.close()
    log("  -> Đã lưu reports/eda_overview.png")

    # ---- BƯỚC 5: BASELINE ----
    log("\n[BƯỚC 5] Baseline")
    dummy = DummyRegressor(strategy="mean").fit(X_train, y_train)
    mae_dummy = mean_absolute_error(y_test, dummy.predict(X_test))
    log(f"  - DummyRegressor (mean)           MAE = {mae_dummy:.2f}")

    manual_pred = 3.0 + 2.0 * X_test["trip_distance"]
    mae_manual = mean_absolute_error(y_test, manual_pred)
    log(f"  - Công thức thủ công (3 + 2*km)   MAE = {mae_manual:.2f}")

    # ---- BƯỚC 6: CÂY KHÔNG GIỚI HẠN ĐỘ SÂU -> CHỨNG MINH OVERFIT ----
    log("\n[BƯỚC 6] Cây không giới hạn độ sâu (chứng minh overfit)")
    tree_full = DecisionTreeRegressor(random_state=42)
    tree_full.fit(X_train, y_train)
    mae_train_full = mean_absolute_error(y_train, tree_full.predict(X_train))
    mae_test_full = mean_absolute_error(y_test, tree_full.predict(X_test))
    log(f"  - Số lá: {tree_full.get_n_leaves():,} | Số tầng: {tree_full.get_depth()}")
    log(f"  - MAE train = {mae_train_full:.3f}  (gần 0 -> cây 'học thuộc' dữ liệu)")
    log(f"  - MAE test  = {mae_test_full:.3f}  (cao hơn nhiều -> OVERFIT rõ ràng)")

    # ---- BƯỚC 7: QUÉT max_depth 1..20 ----
    log("\n[BƯỚC 7] Quét max_depth từ 1 đến 20")
    depths = list(range(1, 21))
    train_maes, test_maes = [], []
    for d in depths:
        t = DecisionTreeRegressor(max_depth=d, min_samples_leaf=500, random_state=42)
        t.fit(X_train, y_train)
        train_maes.append(mean_absolute_error(y_train, t.predict(X_train)))
        test_maes.append(mean_absolute_error(y_test, t.predict(X_test)))
    best_depth = depths[int(np.argmin(test_maes))]
    log(f"  -> Độ sâu có MAE test thấp nhất: {best_depth} (MAE = {min(test_maes):.3f})")
    log(f"  -> Đề bài chọn max_depth=5 để ƯU TIÊN khả năng tra bảng bằng tay,")
    log(f"     dù có thể không phải điểm tối ưu thống kê tuyệt đối.")

    plt.figure(figsize=(7, 4.5))
    plt.plot(depths, train_maes, marker="o", label="MAE train", color="#2a78d6")
    plt.plot(depths, test_maes, marker="o", label="MAE test", color="#eb6834")
    plt.axvline(5, color="gray", linestyle="--", linewidth=1, label="max_depth=5 (chọn)")
    plt.xlabel("max_depth")
    plt.ylabel("MAE")
    plt.title("MAE train/test theo độ sâu cây")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "mae_theo_depth.png"), dpi=130)
    plt.close()
    log("  -> Đã lưu reports/mae_theo_depth.png")

    # ---- BƯỚC 8: HÀM BẬC THANG ----
    log("\n[BƯỚC 8] Vẽ hàm dự đoán theo quãng đường (bậc thang)")
    tree5 = DecisionTreeRegressor(max_depth=5, min_samples_leaf=500,
                                   criterion="squared_error", random_state=42)
    tree5.fit(X_train, y_train)

    dist_range = np.linspace(0.1, 30, 300)
    sample_row = X_train.iloc[0].copy()
    grid = pd.DataFrame([sample_row] * len(dist_range))
    grid["trip_distance"] = dist_range
    step_pred = tree5.predict(grid)

    plt.figure(figsize=(7, 4.5))
    plt.scatter(X_test["trip_distance"], y_test, s=2, alpha=0.1, color="#888780", label="Dữ liệu thật")
    plt.plot(dist_range, step_pred, color="#eb6834", linewidth=2.5, label="Cây dự đoán (bậc thang)")
    plt.xlim(0, 30)
    plt.xlabel("Quãng đường")
    plt.ylabel("Cước phí")
    plt.title("Hàm dự đoán của cây là hàm BẬC THANG, không phải đường mượt")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "ham_bac_thang.png"), dpi=130)
    plt.close()
    log("  -> Đã lưu reports/ham_bac_thang.png")
    log(f"  -> Số mức giá (lá) tối đa có thể trả ra: {tree5.get_n_leaves()}")

    # ---- BƯỚC 9: EXPORT TEXT + VẼ CÂY ----
    log("\n[BƯỚC 9] Xuất cây dạng text và hình ảnh")
    tree_text = export_text(tree5, feature_names=features)
    with open(os.path.join(REPORT_DIR, "cay_quyet_dinh.txt"), "w", encoding="utf-8") as f:
        f.write(tree_text)
    log("  -> Đã lưu reports/cay_quyet_dinh.txt")

    plt.figure(figsize=(20, 10))
    plot_tree(tree5, feature_names=features, filled=True, rounded=True,
              fontsize=7, max_depth=3, proportion=True)
    plt.title("Cây quyết định (hiển thị 3 tầng đầu, đầy đủ 5 tầng trong file text)")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "cay_quyet_dinh.png"), dpi=130)
    plt.close()
    log("  -> Đã lưu reports/cay_quyet_dinh.png")

    # ---- BƯỚC 10: BẢNG TRA CƯỚC ----
    log("\n[BƯỚC 10] Xuất bảng tra cước CSV cho tổng đài")
    leaf_ids_train = tree5.apply(X_train)
    tmp = X_train.copy()
    tmp["leaf_id"] = leaf_ids_train
    tmp["gia_du_doan"] = tree5.predict(X_train)

    bang_tra_cuoc = (
        tmp.groupby("leaf_id")
        .agg(
            gia_de_xuat=("gia_du_doan", "mean"),
            so_chuyen_lam_can_cu=("gia_du_doan", "count"),
            km_min=("trip_distance", "min"),
            km_max=("trip_distance", "max"),
            ty_le_gio_cao_diem=("is_rush_hour", "mean"),
        )
        .reset_index()
        .sort_values("km_min")
    )
    bang_tra_cuoc["gia_de_xuat"] = bang_tra_cuoc["gia_de_xuat"].round(1)
    bang_tra_cuoc["ty_le_gio_cao_diem"] = (bang_tra_cuoc["ty_le_gio_cao_diem"] * 100).round(0)
    bang_tra_cuoc.to_csv(os.path.join(OUT_DIR, "bang_tra_cuoc.csv"), index=False, encoding="utf-8-sig")
    log(f"  -> Đã lưu outputs/bang_tra_cuoc.csv ({len(bang_tra_cuoc)} mức giá)")

    # ---- BƯỚC 11: % CHUYẾN ĐẠT SAI SỐ ±15% ----
    log("\n[BƯỚC 11] Kiểm tra sai số ±15%")
    pred_test = tree5.predict(X_test)
    mae5 = mean_absolute_error(y_test, pred_test)
    mape5 = mean_absolute_percentage_error(y_test, pred_test) * 100
    rmse5 = mean_squared_error(y_test, pred_test) ** 0.5
    error_pct = np.abs(pred_test - y_test) / y_test
    within_15 = (error_pct <= 0.15).mean() * 100
    log(f"  - MAE  = {mae5:.2f}")
    log(f"  - MAPE = {mape5:.2f}%")
    log(f"  - RMSE = {rmse5:.2f}")
    log(f"  - % chuyến sai số trong ±15%: {within_15:.1f}%")

    # ---- BƯỚC 12: SO SÁNH VỚI RANDOM FOREST VÀ LINEAR REGRESSION ----
    log("\n[BƯỚC 12] So sánh với Random Forest và Linear Regression")
    lin = LinearRegression().fit(X_train, y_train)
    mae_lin = mean_absolute_error(y_test, lin.predict(X_test))

    rf = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=200,
                                random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    mae_rf = mean_absolute_error(y_test, rf.predict(X_test))

    log(f"  - Decision Tree (depth=5)  MAE = {mae5:.2f}")
    log(f"  - Linear Regression        MAE = {mae_lin:.2f}")
    log(f"  - Random Forest            MAE = {mae_rf:.2f}")

    plt.figure(figsize=(6, 4))
    models_cmp = ["Dummy", "Tree d=5", "Linear", "Random\nForest"]
    maes_cmp = [mae_dummy, mae5, mae_lin, mae_rf]
    plt.bar(models_cmp, maes_cmp, color=["#888780", "#eb6834", "#2a78d6", "#1baf7a"])
    plt.ylabel("MAE")
    plt.title("So sánh MAE giữa các mô hình")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "so_sanh_mo_hinh.png"), dpi=130)
    plt.close()
    log("  -> Đã lưu reports/so_sanh_mo_hinh.png")

    # ---- LƯU MODEL ----
    joblib.dump(tree5, os.path.join(MODEL_DIR, "tree_reg.joblib"))
    log(f"\n-> Đã lưu models/tree_reg.joblib")

    # ---- LƯU LOG ----
    with open(os.path.join(REPORT_DIR, "log_chay.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    log("-> Đã lưu reports/log_chay.txt")

    log("\n" + "=" * 70)
    log("HOÀN TẤT")
    log("=" * 70)

    return {
        "n_leaves": tree5.get_n_leaves(),
        "mae5": mae5, "mape5": mape5, "rmse5": rmse5,
        "within_15": within_15, "best_depth": best_depth,
        "mae_dummy": mae_dummy, "mae_manual": mae_manual,
        "mae_lin": mae_lin, "mae_rf": mae_rf,
        "mae_train_full": mae_train_full, "mae_test_full": mae_test_full,
        "n_leaves_full": tree_full.get_n_leaves(),
    }


if __name__ == "__main__":
    main()