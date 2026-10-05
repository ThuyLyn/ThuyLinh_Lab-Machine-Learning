import os
import warnings

import matplotlib
matplotlib.use("Agg")  # vẽ ra file, không cần mở cửa sổ
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import seasonal_decompose

from stationarity import load_airpassengers, bang_adf
from train import (mape, chia_train_test, baselines, fit_sarima,
                   quet_luoi, du_bao, ti_le_phu)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # thư mục gốc dự án
REPORTS, MODELS = os.path.join(ROOT, "reports"), os.path.join(ROOT, "models")
os.makedirs(REPORTS, exist_ok=True); os.makedirs(MODELS, exist_ok=True)

def tieu_de(s):
    print("\n" + "=" * 70 + f"\n{s}\n" + "=" * 70)

y = load_airpassengers()
tieu_de(f"1. DỮ LIỆU: {len(y)} tháng, {y.index[0].date()} → {y.index[-1].date()}")

# Phân rã
fig, axes = plt.subplots(4, 2, figsize=(13, 9), sharex="col")
for j, m in enumerate(["additive", "multiplicative"]):
    r = seasonal_decompose(y, model=m, period=12)
    for i, (c, n) in enumerate([(r.observed, "Gốc"), (r.trend, "Trend"),
                                (r.seasonal, "Mùa vụ"), (r.resid, "Phần dư")]):
        axes[i, j].plot(c); axes[i, j].set_ylabel(n)
    axes[0, j].set_title(m)
plt.tight_layout(); plt.savefig(os.path.join(REPORTS, "decompose.png")); plt.close()

tieu_de("2. BẢNG ADF (p < 0,05 mới là DỪNG)")
print(bang_adf(y).to_string(index=False))

# ACF / PACF
y_dung = np.log(y).diff(1).diff(12).dropna()
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
plot_acf(y_dung, lags=36, ax=ax[0]); plot_pacf(y_dung, lags=36, ax=ax[1], method="ywm")
plt.tight_layout(); plt.savefig(os.path.join(REPORTS, "acf_pacf.png")); plt.close()

# Chia dữ liệu + baseline
train, test = chia_train_test(y, 24); train_log = np.log(train)
bl = baselines(train, test)
tieu_de(f"3. BASELINE (train {len(train)} tháng, test {len(test)} tháng)")
for k, v in bl.items():
    print(f"MAPE {k:<16}: {mape(test, v):.2f}%")

# Dò lưới
tieu_de("4. DÒ LƯỚI THEO AIC (top 5)")
luoi = quet_luoi(train_log)
print(luoi.head(5).to_string())
best_order, best_seasonal = luoi.iloc[0]["order"], luoi.iloc[0]["seasonal"]
fit = fit_sarima(train_log, best_order, best_seasonal)
fit_111 = fit_sarima(train_log, (1, 1, 1), (1, 1, 1, 12))
print(f"\nChọn: SARIMA{best_order}{best_seasonal}  (AIC = {fit.aic:.1f})")

# Ljung-Box
tieu_de("5. KIỂM ĐỊNH PHẦN DƯ — Ljung-Box (cần p > 0,05)")
for ten, f in [(f"{best_order}{best_seasonal}", fit), ("(1,1,1)(1,1,1,12)", fit_111)]:
    lb = acorr_ljungbox(f.resid.iloc[13:], lags=[12, 24], return_df=True)
    print(f"{ten}: p(lag12) = {lb['lb_pvalue'].iloc[0]:.3f} | p(lag24) = {lb['lb_pvalue'].iloc[1]:.3f}")
fit.plot_diagnostics(figsize=(12, 8), lags=24); plt.tight_layout()
plt.savefig(os.path.join(REPORTS, "residual_diagnostics.png")); plt.close()

# Dự báo
mean, ci = du_bao(fit, 24)
mean_111, ci_111 = du_bao(fit_111, 24)
tieu_de("6. KẾT QUẢ DỰ BÁO 24 THÁNG")
kq = pd.DataFrame({
    "MAPE (%)": {"Naive": mape(test, bl["Naive"]), "Seasonal naive": mape(test, bl["Seasonal naive"]),
                 "SARIMA (chọn theo AIC)": mape(test, mean), "SARIMA (1,1,1)(1,1,1,12)": mape(test, mean_111)},
    "Tỉ lệ phủ 95% (%)": {"Naive": np.nan, "Seasonal naive": np.nan,
                          "SARIMA (chọn theo AIC)": ti_le_phu(test, ci),
                          "SARIMA (1,1,1)(1,1,1,12)": ti_le_phu(test, ci_111)},
}).round(2)
print(kq.to_string())
pred_1b = np.exp(fit.apply(np.log(y)).get_prediction(start=test.index[0]).predicted_mean)
print(f"\nMAPE dự báo 1 bước (tập test): {mape(test, pred_1b):.2f}%")
print("SARIMA thắng seasonal naive:", mape(test, mean) < mape(test, bl["Seasonal naive"]))

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(y, color="black", label="Thực tế"); ax.plot(bl["Seasonal naive"], "--", color="gray", label="Seasonal naive")
ax.plot(mean, color="crimson", label="SARIMA")
ax.fill_between(ci.index, ci["thap"], ci["cao"], color="crimson", alpha=.18, label="Khoảng tin cậy 95%")
ax.axvline(test.index[0], color="k", ls=":"); ax.legend(); ax.set_title("Dự báo 24 tháng + khoảng tin cậy 95%")
plt.tight_layout(); plt.savefig(os.path.join(REPORTS, "du_bao_khoang.png")); plt.close()

fit.save(os.path.join(MODELS, "sarima_fit.pkl"))
tieu_de("XONG")
print(f"Ảnh : {REPORTS}\nModel: {MODELS}")