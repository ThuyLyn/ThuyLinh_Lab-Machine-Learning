import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

try:
    from xgboost import XGBRegressor
    CO_XGBOOST = True
except ImportError:
    CO_XGBOOST = False

# ---------------------------------------------------------------------------
DATA_PATH = "D:\TT_ML\TT-17-Random-Forest-Regression_Linh\dataset\Clean_Dataset.csv"
REPORTS_DIR = "reports"
MODELS_DIR = "models"
RANDOM_STATE = 42

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

COT_PHAN_LOAI = ["airline", "source_city", "departure_time", "stops",
                 "arrival_time", "destination_city", "class"]
COT_SO = ["duration", "days_left"]


def buoc_1_nap_du_lieu():
    print("\n[BƯỚC 1] Nạp dữ liệu & bỏ cột thừa")
    df = pd.read_csv(DATA_PATH)
    for cot_thua in ["Unnamed: 0", "flight"]:
        if cot_thua in df.columns:
            df = df.drop(columns=[cot_thua])
            print(f"  - Đã bỏ cột '{cot_thua}'")
    print(f"  - Shape sau khi bỏ cột: {df.shape}")
    return df


def buoc_2_eda_days_left(df):
    print("\n[BƯỚC 2] EDA: giá theo days_left")
    gia_tb = df.groupby("days_left")["price"].mean().reset_index()
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=gia_tb, x="days_left", y="price")
    plt.title("Giá vé trung bình theo số ngày còn lại (days_left)")
    plt.xlabel("Số ngày còn lại tới chuyến bay")
    plt.ylabel("Giá vé trung bình")
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/gia_theo_days_left.png", dpi=120)
    plt.close()
    print(f"  - Đã lưu {REPORTS_DIR}/gia_theo_days_left.png")


def buoc_3_eda_boxplot(df):
    print("\n[BƯỚC 3] EDA: boxplot theo class và airline")
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    sns.boxplot(data=df, x="class", y="price", ax=ax[0])
    ax[0].set_title("Giá vé theo hạng vé (class)")
    sns.boxplot(data=df, x="airline", y="price", ax=ax[1])
    ax[1].set_title("Giá vé theo hãng bay (airline)")
    ax[1].tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/boxplot_class_airline.png", dpi=120)
    plt.close()
    print(f"  - Đã lưu {REPORTS_DIR}/boxplot_class_airline.png")


def buoc_4_pipeline():
    print("\n[BƯỚC 4] Xây pipeline tiền xử lý (OneHotEncoder, không scale)")
    return ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), COT_PHAN_LOAI)],
        remainder="passthrough",
    )


def buoc_5_baseline(tien_xu_ly, X_train, X_test, y_train, y_test):
    print("\n[BƯỚC 5] Baseline: Dummy + Linear + 1 cây đơn")
    ket_qua = {}
    mo_hinh_can_thu = [
        ("Dummy", DummyRegressor(strategy="mean")),
        ("Linear Regression", LinearRegression()),
        ("Cây đơn (TT-16)", DecisionTreeRegressor(random_state=RANDOM_STATE)),
    ]
    for ten, model in mo_hinh_can_thu:
        pipe = Pipeline([("prep", tien_xu_ly), ("model", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        r2 = float(r2_score(y_test, pred))
        ket_qua[ten] = {"rmse": rmse, "r2": r2}
        print(f"  - {ten:20s} RMSE={rmse:>12,.0f}  R²={r2:.4f}")
    return ket_qua


def buoc_6_random_forest(tien_xu_ly, X_train, X_test, y_train, y_test):
    print("\n[BƯỚC 6] Random Forest + oob_score_")
    rf = RandomForestRegressor(
        n_estimators=300,
        max_features=1.0,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        oob_score=True,
    )
    pipe_rf = Pipeline([("prep", tien_xu_ly), ("model", rf)])
    pipe_rf.fit(X_train, y_train)

    oob = pipe_rf.named_steps["model"].oob_score_
    pred = pipe_rf.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    r2 = float(r2_score(y_test, pred))

    print(f"  - oob_score_ = {oob:.4f}")
    print(f"  - RMSE test  = {rmse:,.0f}")
    print(f"  - R² test    = {r2:.4f}")

    joblib.dump(pipe_rf, f"{MODELS_DIR}/rf_reg.joblib")
    print(f"  - Đã lưu model: {MODELS_DIR}/rf_reg.joblib")
    return pipe_rf, {"oob_score": float(oob), "rmse": rmse, "r2": r2}


def buoc_7_rmse_theo_so_cay(tien_xu_ly, X_train, X_test, y_train, y_test):
    print("\n[BƯỚC 7] RMSE theo n_estimators (tìm điểm bão hoà)")
    so_cay_list = [10, 30, 50, 100, 200, 300, 500]
    rmse_list = []
    for n in so_cay_list:
        rf_tmp = RandomForestRegressor(
            n_estimators=n, max_features=1.0, min_samples_leaf=2,
            n_jobs=-1, random_state=RANDOM_STATE,
        )
        p = Pipeline([("prep", tien_xu_ly), ("model", rf_tmp)])
        p.fit(X_train, y_train)
        pred = p.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        rmse_list.append(rmse)
        print(f"  - n_estimators={n:4d}  RMSE={rmse:,.0f}")

    plt.figure(figsize=(8, 5))
    plt.plot(so_cay_list, rmse_list, marker="o")
    plt.xlabel("Số cây (n_estimators)")
    plt.ylabel("RMSE")
    plt.title("RMSE theo số cây trong rừng")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/rmse_theo_so_cay.png", dpi=120)
    plt.close()
    print(f"  - Đã lưu {REPORTS_DIR}/rmse_theo_so_cay.png")
    return dict(zip(so_cay_list, [float(x) for x in rmse_list]))


def buoc_8_permutation_importance(pipe_rf, X_test, y_test):
    print("\n[BƯỚC 8] Permutation importance (KHÔNG dùng feature_importances_ mặc định)")
    ket_qua = permutation_importance(
        pipe_rf, X_test, y_test, n_repeats=5,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    df_importance = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": ket_qua.importances_mean,
        "importance_std": ket_qua.importances_std,
    }).sort_values("importance_mean", ascending=False)

    print(df_importance.to_string(index=False))

    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_importance, x="importance_mean", y="feature", color="steelblue")
    plt.title("Permutation Importance")
    plt.xlabel("Mức giảm R² khi xáo trộn cột")
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/permutation_importance.png", dpi=120)
    plt.close()
    print(f"  - Đã lưu {REPORTS_DIR}/permutation_importance.png")
    return df_importance


def buoc_9_pdp_days_left(pipe_rf, X_train):
    print("\n[BƯỚC 9] PDP cho days_left")
    X_train = X_train.copy()
    X_train["days_left"] = X_train["days_left"].astype(float)  # tránh FutureWarning sklearn
    fig, ax = plt.subplots(figsize=(8, 5))
    PartialDependenceDisplay.from_estimator(
        pipe_rf, X_train, features=["days_left"], ax=ax,
    )
    plt.title("Partial Dependence: days_left → price")
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/pdp_days_left.png", dpi=120)
    plt.close()

    # Định lượng: mua sớm 1 tuần (14 -> 7 ngày) tiết kiệm bao nhiêu?
    mau_14 = X_train.copy().iloc[:200].assign(days_left=14)
    mau_7 = X_train.copy().iloc[:200].assign(days_left=7)
    gia_14 = pipe_rf.predict(mau_14).mean()
    gia_7 = pipe_rf.predict(mau_7).mean()
    chenh_lech = gia_7 - gia_14
    print(f"  - Giá trung bình (mô hình) tại days_left=14: {gia_14:,.0f}")
    print(f"  - Giá trung bình (mô hình) tại days_left=7 : {gia_7:,.0f}")
    print(f"  - Kết luận: đợi từ 14 xuống 7 ngày trước bay làm giá "
          f"{'tăng' if chenh_lech > 0 else 'giảm'} trung bình {abs(chenh_lech):,.0f}")
    print(f"  - Đã lưu {REPORTS_DIR}/pdp_days_left.png")
    return {"gia_tb_14_ngay": float(gia_14), "gia_tb_7_ngay": float(gia_7),
            "chenh_lech": float(chenh_lech)}


def buoc_10_khoang_du_bao(pipe_rf, X_test, y_test):
    print("\n[BƯỚC 10] Khoảng dự báo 10-90% từ các cây trong rừng")
    prep = pipe_rf.named_steps["prep"]
    rf_model = pipe_rf.named_steps["model"]
    X_test_transformed = prep.transform(X_test)

    du_doan_tung_cay = np.stack([cay.predict(X_test_transformed) for cay in rf_model.estimators_])
    khoang_thap = np.percentile(du_doan_tung_cay, 10, axis=0)
    khoang_cao = np.percentile(du_doan_tung_cay, 90, axis=0)

    trong_khoang = ((y_test.values >= khoang_thap) & (y_test.values <= khoang_cao)).mean()
    print(f"  - Tỉ lệ giá thật rơi trong khoảng 10-90%: {trong_khoang:.1%} (kỳ vọng ~80%)")

    # Biểu đồ minh hoạ trên 60 mẫu đầu (test set)
    n_hien_thi = 60
    idx = np.arange(n_hien_thi)
    plt.figure(figsize=(12, 5))
    plt.fill_between(idx, khoang_thap[:n_hien_thi], khoang_cao[:n_hien_thi],
                      alpha=0.3, label="Khoảng dự báo 10-90%")
    plt.scatter(idx, y_test.values[:n_hien_thi], color="red", s=15, label="Giá thật")
    plt.legend()
    plt.title(f"Khoảng dự báo 10-90% (60 mẫu đầu) — tỉ lệ phủ thực tế: {trong_khoang:.1%}")
    plt.xlabel("Mẫu")
    plt.ylabel("Giá vé")
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/khoang_du_bao.png", dpi=120)
    plt.close()
    print(f"  - Đã lưu {REPORTS_DIR}/khoang_du_bao.png")
    return {"ti_le_phu": float(trong_khoang)}


def buoc_11_ngoai_suy(pipe_rf, X_test):
    print("\n[BƯỚC 11] Thí nghiệm ngoại suy: days_left = 100")
    max_train = X_test["days_left"].max()
    print(f"  - days_left tối đa trong dữ liệu: {max_train}")

    mau = X_test.iloc[[0]].copy()
    ket_qua = {}
    for dl in [max_train, 60, 100, 500]:
        mau["days_left"] = dl
        gia = pipe_rf.predict(mau)[0]
        ket_qua[int(dl)] = float(gia)
        print(f"  - days_left={dl:>4}: giá dự đoán = {gia:,.0f}")
    return ket_qua


def buoc_12_so_sanh_xgboost(tien_xu_ly, X_train, X_test, y_train, y_test):
    print("\n[BƯỚC 12] So sánh với XGBoost Regressor")
    xgb = XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    pipe_xgb = Pipeline([("prep", tien_xu_ly), ("model", xgb)])
    pipe_xgb.fit(X_train, y_train)
    pred = pipe_xgb.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    r2 = float(r2_score(y_test, pred))
    print(f"  - XGBoost RMSE={rmse:,.0f}  R²={r2:.4f}")
    return {"rmse": rmse, "r2": r2}


def main():
    df = buoc_1_nap_du_lieu()
    buoc_2_eda_days_left(df)
    buoc_3_eda_boxplot(df)

    X = df.drop(columns=["price"])
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    tien_xu_ly = buoc_4_pipeline()
    ket_qua_baseline = buoc_5_baseline(tien_xu_ly, X_train, X_test, y_train, y_test)
    pipe_rf, ket_qua_rf = buoc_6_random_forest(tien_xu_ly, X_train, X_test, y_train, y_test)
    rmse_theo_so_cay = buoc_7_rmse_theo_so_cay(tien_xu_ly, X_train, X_test, y_train, y_test)
    df_importance = buoc_8_permutation_importance(pipe_rf, X_test, y_test)
    ket_qua_pdp = buoc_9_pdp_days_left(pipe_rf, X_train)
    ket_qua_khoang = buoc_10_khoang_du_bao(pipe_rf, X_test, y_test)
    ket_qua_ngoai_suy = buoc_11_ngoai_suy(pipe_rf, X_test)
    ket_qua_xgb = buoc_12_so_sanh_xgboost(tien_xu_ly, X_train, X_test, y_train, y_test)

    tong_ket = {
        "baseline": ket_qua_baseline,
        "random_forest": ket_qua_rf,
        "rmse_theo_so_cay": rmse_theo_so_cay,
        "permutation_importance": df_importance.to_dict(orient="records"),
        "pdp_days_left": ket_qua_pdp,
        "khoang_du_bao": ket_qua_khoang,
        "ngoai_suy": ket_qua_ngoai_suy,
        "xgboost": ket_qua_xgb,
    }
    with open(f"{REPORTS_DIR}/tong_ket_ket_qua.json", "w", encoding="utf-8") as f:
        json.dump(tong_ket, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("HOÀN TẤT. Xem chi tiết số liệu tại reports/tong_ket_ket_qua.json")
    print("=" * 60)


if __name__ == "__main__":
    main()