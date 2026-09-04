import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import json
import os
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, lasso_path
from sklearn.metrics import root_mean_squared_error

RANDOM_STATE = 42
OUT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(OUT_DIR, "reports")
MODELS_DIR = os.path.join(OUT_DIR, "models")

# ---------------------------------------------------------------------------
# 1. Dữ liệu: load_diabetes (10 biến thật) + 190 biến nhiễu -> 200 cột
# ---------------------------------------------------------------------------
print("BƯỚC 1: Nạp dữ liệu và mở rộng thành 200 cột")

X, y = load_diabetes(return_X_y=True, as_frame=True)
BIEN_THAT = list(X.columns)  # 10 tên cột thật, dùng để chấm điểm sau
rng = np.random.default_rng(RANDOM_STATE)
nhieu = pd.DataFrame(
    rng.normal(size=(len(X), 190)),
    columns=[f"chi_so_nhieu_{i:03d}" for i in range(190)],
)
X_full = pd.concat([X, nhieu], axis=1)
print(f"Kích thước X_full: {X_full.shape}  (10 biến thật + 190 biến nhiễu)")
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y, test_size=0.2, random_state=RANDOM_STATE
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ---------------------------------------------------------------------------
# 2. Baseline: Linear Regression trên đủ 200 cột (không phạt) -> overfit
# ---------------------------------------------------------------------------
print("BƯỚC 2: Baseline Linear Regression (200 cột, không regularization)")

lr_baseline = LinearRegression().fit(X_train, y_train)
rmse_lr_train = root_mean_squared_error(y_train, lr_baseline.predict(X_train))
rmse_lr_test = root_mean_squared_error(y_test, lr_baseline.predict(X_test))
print(f"RMSE train: {rmse_lr_train:.2f}")
print(f"RMSE test : {rmse_lr_test:.2f}")
print(f"-> Chênh lệch train/test = {rmse_lr_test - rmse_lr_train:.2f} (overfit rõ rệt)")

# ---------------------------------------------------------------------------
# 3. LassoCV dò alpha tối ưu
# ---------------------------------------------------------------------------
print("BƯỚC 3: LassoCV (dò alpha bằng 5-fold CV)")

pipe_lasso = Pipeline([
    ("scale", StandardScaler()),
    ("lasso", LassoCV(alphas=np.logspace(-4, 1, 100), cv=5,
                       max_iter=50000, random_state=RANDOM_STATE)),
])
pipe_lasso.fit(X_train, y_train)
alpha_opt = pipe_lasso["lasso"].alpha_
print(f"Alpha tối ưu: {alpha_opt:.5f}")
rmse_lasso_train = root_mean_squared_error(y_train, pipe_lasso.predict(X_train))
rmse_lasso_test = root_mean_squared_error(y_test, pipe_lasso.predict(X_test))
print(f"RMSE train: {rmse_lasso_train:.2f}")
print(f"RMSE test : {rmse_lasso_test:.2f}")

# ---------------------------------------------------------------------------
# 4. CHẤM ĐIỂM CHỌN BIẾN
# ---------------------------------------------------------------------------
print("BƯỚC 4: CHẤM ĐIỂM CHỌN BIẾN")

he_so = pd.Series(pipe_lasso["lasso"].coef_, index=X_full.columns)
bien_duoc_giu = he_so[he_so != 0].index.tolist()

giu_dung = sorted(set(bien_duoc_giu) & set(BIEN_THAT))
giu_nham = sorted(set(bien_duoc_giu) - set(BIEN_THAT))
bo_sot = sorted(set(BIEN_THAT) - set(bien_duoc_giu))

print(f"Tổng số biến Lasso giữ lại : {len(bien_duoc_giu)}/200")
print(f"Giữ ĐÚNG (biến thật)       : {len(giu_dung)}/10  -> {giu_dung}")
print(f"Giữ NHẦM (biến nhiễu)      : {len(giu_nham)}     -> {giu_nham}")
print(f"BỎ SÓT (biến thật bị loại) : {len(bo_sot)}       -> {bo_sot}")

bang_diem = pd.DataFrame({
    "Chỉ số": ["Số biến thật được giữ (recall)",
               "Số biến nhiễu bị giữ nhầm (false positive)",
               "Số biến thật bị bỏ sót",
               "Tổng số biến Lasso giữ lại"],
    "Giá trị": [f"{len(giu_dung)}/10", len(giu_nham), len(bo_sot), f"{len(bien_duoc_giu)}/200"],
})
print("\nBảng chấm điểm:")
print(bang_diem.to_string(index=False))
bang_diem.to_csv(os.path.join(REPORTS_DIR, "bang_cham_diem.csv"), index=False)

# ---------------------------------------------------------------------------
# 5. Coefficient path
# ---------------------------------------------------------------------------
print("BƯỚC 5: Vẽ coefficient path")

X_train_scaled = StandardScaler().fit_transform(X_train)
alphas_path, coefs_path, _ = lasso_path(
    X_train_scaled, y_train, alphas=np.logspace(-4, 1, 100), max_iter=50000
)

plt.figure(figsize=(11, 6))
for i, col in enumerate(X_full.columns):
    mau = "tab:red" if col in BIEN_THAT else "lightgray"
    do_day = 2.2 if col in BIEN_THAT else 0.6
    zorder = 3 if col in BIEN_THAT else 1
    plt.plot(np.log10(alphas_path), coefs_path[i], color=mau, linewidth=do_day, zorder=zorder)
plt.axvline(np.log10(alpha_opt), color="black", linestyle="--", label=f"alpha tối ưu={alpha_opt:.4f}")
plt.xlabel("log10(alpha)")
plt.ylabel("Hệ số hồi quy")
plt.title("Lasso Coefficient Path\n(đỏ = 10 biến thật, xám = 190 biến nhiễu)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "lasso_path.png"), dpi=130)
plt.close()
print("Đã lưu reports/lasso_path.png")

# RMSE train/test theo alpha
mse_path = []
for a in alphas_path:
    from sklearn.linear_model import Lasso
    m = Lasso(alpha=a, max_iter=50000).fit(X_train_scaled, y_train)
    X_test_scaled = StandardScaler().fit(X_train).transform(X_test) if False else None

scaler_tmp = StandardScaler().fit(X_train)
X_test_scaled = scaler_tmp.transform(X_test)
rmse_train_list, rmse_test_list = [], []
for a in alphas_path:
    from sklearn.linear_model import Lasso
    m = Lasso(alpha=a, max_iter=50000).fit(X_train_scaled, y_train)
    rmse_train_list.append(root_mean_squared_error(y_train, m.predict(X_train_scaled)))
    rmse_test_list.append(root_mean_squared_error(y_test, m.predict(X_test_scaled)))

plt.figure(figsize=(10, 6))
plt.plot(np.log10(alphas_path), rmse_train_list, label="RMSE train")
plt.plot(np.log10(alphas_path), rmse_test_list, label="RMSE test")
plt.axvline(np.log10(alpha_opt), color="black", linestyle="--", label="alpha tối ưu")
plt.xlabel("log10(alpha)")
plt.ylabel("RMSE")
plt.title("RMSE train/test theo alpha")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "rmse_vs_alpha.png"), dpi=130)
plt.close()
print("Đã lưu reports/rmse_vs_alpha.png")

# ---------------------------------------------------------------------------
# 7. So sánh Ridge vs Lasso
# ---------------------------------------------------------------------------
print("BƯỚC 6: So sánh Ridge vs Lasso")

pipe_ridge = Pipeline([
    ("scale", StandardScaler()),
    ("ridge", RidgeCV(alphas=np.logspace(-4, 4, 100), cv=5)),
]).fit(X_train, y_train)

so_bien_ridge = int((pipe_ridge["ridge"].coef_ != 0).sum())
so_bien_lasso = int((pipe_lasso["lasso"].coef_ != 0).sum())
rmse_ridge_test = root_mean_squared_error(y_test, pipe_ridge.predict(X_test))

bang_sosanh = pd.DataFrame({
    "Model": ["Linear Regression (200 cột)", "Ridge (L2)", "Lasso (L1)"],
    "So_bien_giu": [200, so_bien_ridge, so_bien_lasso],
    "RMSE_test": [rmse_lr_test, rmse_ridge_test, rmse_lasso_test],
})
print(bang_sosanh.to_string(index=False))
bang_sosanh.to_csv(os.path.join(REPORTS_DIR, "bang_so_sanh_ridge_lasso.csv"), index=False)

plt.figure(figsize=(8, 5))
plt.bar(bang_sosanh["Model"], bang_sosanh["So_bien_giu"], color=["gray", "tab:blue", "tab:red"])
plt.ylabel("Số biến còn lại (khác 0)")
plt.title("Số biến giữ lại: Linear vs Ridge vs Lasso")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "ridge_vs_lasso.png"), dpi=130)
plt.close()
print("Đã lưu reports/ridge_vs_lasso.png")

# ---------------------------------------------------------------------------
# 8. Thí nghiệm biến tương quan
# ---------------------------------------------------------------------------
print("BƯỚC 7: Thí nghiệm biến tương quan (nhân đôi cột 'bmi')")

X_corr = X_full.copy()
X_corr["bmi_ban_sao"] = X_full["bmi"] + rng.normal(0, 0.01, size=len(X_full))

ket_qua_seed = []
for seed in [0, 1, 42, 99, 123]:
    Xtr_c, Xte_c, ytr_c, yte_c = train_test_split(X_corr, y, test_size=0.2, random_state=seed)
    p = Pipeline([
        ("scale", StandardScaler()),
        ("lasso", LassoCV(alphas=np.logspace(-4, 1, 60), cv=5, max_iter=50000, random_state=seed)),
    ]).fit(Xtr_c, ytr_c)
    hs = pd.Series(p["lasso"].coef_, index=X_corr.columns)
    giu_bmi = hs["bmi"] != 0
    giu_bmi_copy = hs["bmi_ban_sao"] != 0
    ket_qua_seed.append({"seed": seed, "giu_bmi": giu_bmi, "giu_bmi_ban_sao": giu_bmi_copy,
                          "he_so_bmi": round(hs["bmi"], 4), "he_so_bmi_ban_sao": round(hs["bmi_ban_sao"], 4)})

bang_corr = pd.DataFrame(ket_qua_seed)
print(bang_corr.to_string(index=False))
bang_corr.to_csv(os.path.join(REPORTS_DIR, "thi_nghiem_tuong_quan.csv"), index=False)
print("\n-> Quan sát: việc Lasso chọn 'bmi' hay 'bmi_ban_sao' đổi theo seed -> minh chứng")
print("   cho tính KHÔNG ổn định của Lasso khi có biến tương quan cao.")

plt.figure(figsize=(7, 5))
x_pos = np.arange(len(bang_corr))
plt.bar(x_pos - 0.2, bang_corr["he_so_bmi"], width=0.4, label="bmi")
plt.bar(x_pos + 0.2, bang_corr["he_so_bmi_ban_sao"], width=0.4, label="bmi_ban_sao")
plt.xticks(x_pos, [f"seed={s}" for s in bang_corr["seed"]])
plt.ylabel("Hệ số hồi quy")
plt.title("Hệ số của bmi vs bmi_ban_sao theo từng seed")
plt.axhline(0, color="black", linewidth=0.8)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "chon_bien_score.png"), dpi=130)
plt.close()
print("Đã lưu reports/chon_bien_score.png")

# ---------------------------------------------------------------------------
# 9. Debiased Lasso
# ---------------------------------------------------------------------------
print("BƯỚC 8: Debiased Lasso (Linear Regression chỉ trên biến Lasso chọn)")

lr_debiased = LinearRegression().fit(X_train[bien_duoc_giu], y_train)
rmse_debiased_train = root_mean_squared_error(y_train, lr_debiased.predict(X_train[bien_duoc_giu]))
rmse_debiased_test = root_mean_squared_error(y_test, lr_debiased.predict(X_test[bien_duoc_giu]))
print(f"Số biến sử dụng    : {len(bien_duoc_giu)}")
print(f"RMSE train (debiased): {rmse_debiased_train:.2f}")
print(f"RMSE test  (debiased): {rmse_debiased_test:.2f}")
print(f"RMSE test  (Lasso đầy đủ): {rmse_lasso_test:.2f}")

# ---------------------------------------------------------------------------
# 10. Đề xuất bộ xét nghiệm cuối cùng + tiết kiệm chi phí
# ---------------------------------------------------------------------------
print("BƯỚC 9: Đề xuất bộ xét nghiệm cuối cùng")

GIA_TRUNG_BINH_MOT_CHI_SO = 25_000  # đồng, giả định để minh hoạ
so_chi_so_ban_dau = 200
so_chi_so_de_xuat = len(bien_duoc_giu)

chi_phi_ban_dau = so_chi_so_ban_dau * GIA_TRUNG_BINH_MOT_CHI_SO
chi_phi_de_xuat = so_chi_so_de_xuat * GIA_TRUNG_BINH_MOT_CHI_SO
tiet_kiem_pct = (1 - chi_phi_de_xuat / chi_phi_ban_dau) * 100

print(f"Bộ xét nghiệm đề xuất ({so_chi_so_de_xuat} chỉ số): {bien_duoc_giu}")
print(f"Chi phí ước tính ban đầu (200 chỉ số) : {chi_phi_ban_dau:,.0f} đ/bệnh nhân")
print(f"Chi phí ước tính sau chọn lọc         : {chi_phi_de_xuat:,.0f} đ/bệnh nhân")
print(f"Tiết kiệm                              : {tiet_kiem_pct:.1f}%")

de_xuat = {
    "bo_chi_so_de_xuat": bien_duoc_giu,
    "so_luong": so_chi_so_de_xuat,
    "chi_phi_ban_dau_dong": chi_phi_ban_dau,
    "chi_phi_de_xuat_dong": chi_phi_de_xuat,
    "tiet_kiem_phan_tram": round(tiet_kiem_pct, 1),
    "rmse_lasso_test": round(rmse_lasso_test, 3),
    "rmse_debiased_test": round(rmse_debiased_test, 3),
    "alpha_toi_uu": round(float(alpha_opt), 5),
}
with open(os.path.join(REPORTS_DIR, "de_xuat_cuoi_cung.json"), "w", encoding="utf-8") as f:
    json.dump(de_xuat, f, ensure_ascii=False, indent=2)
print("Đã lưu reports/de_xuat_cuoi_cung.json")

# ---------------------------------------------------------------------------
# 10. Lưu model
# ---------------------------------------------------------------------------
joblib.dump(pipe_lasso, os.path.join(MODELS_DIR, "lasso_pipeline.joblib"))
print(f"\nĐã lưu model: models/lasso_pipeline.joblib")
print("HOÀN TẤT. Xem toàn bộ kết quả trong thư mục reports/")