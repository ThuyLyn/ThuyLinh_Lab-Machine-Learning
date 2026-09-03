import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import joblib

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ── Đường dẫn output (script chạy từ thư mục gốc project) ──────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS = os.path.join(ROOT, "reports")
MODELS = os.path.join(ROOT, "models")
os.makedirs(REPORTS, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)

def log(msg):
    print(f"\n{'='*60}\n{msg}\n{'='*60}")

# BƯỚC 1-2: Nạp dữ liệu, phát hiện bẫy
log("BƯỚC 1-2: Nạp dữ liệu & phát hiện bẫy")
data = fetch_california_housing(as_frame=True)
df = data.frame
print(f">> Kích thước dữ liệu: {df.shape}")
print(df.describe())

so_dong_cat_ngon = (df["MedHouseVal"] == df["MedHouseVal"].max()).sum()
ty_le_cat_ngon = so_dong_cat_ngon / len(df) * 100
print(f"\n>> Số dòng bị cắt ngọn ở giá trần (5.0 = 500k USD): {so_dong_cat_ngon} "
      f"({ty_le_cat_ngon:.2f}% tổng dữ liệu)")

print(f">> AveOccup max = {df['AveOccup'].max():.1f} (outlier rõ rệt, bình thường ~2-4)")
print(f">> AveRooms max = {df['AveRooms'].max():.1f}")

# Clip outlier theo phân vị 99 cho AveRooms, AveOccup (mục 3, bẫy #2)
for col in ["AveRooms", "AveOccup", "AveBedrms", "Population"]:
    p99 = df[col].quantile(0.99)
    n_clip = (df[col] > p99).sum()
    df[col] = df[col].clip(upper=p99)
    print(f">> Clip {col} tại phân vị 99 ({p99:.2f}): {n_clip} dòng bị ảnh hưởng")

# BƯỚC 3: EDA
log("BƯỚC 3: EDA — scatter, heatmap, bản đồ giá")

plt.figure(figsize=(8, 7))
sc = plt.scatter(df["Longitude"], df["Latitude"], c=df["MedHouseVal"],
                  cmap="viridis", s=6, alpha=0.6)
plt.colorbar(sc, label="Giá nhà (100k USD)")
plt.xlabel("Longitude"); plt.ylabel("Latitude")
plt.title("Bản đồ giá nhà theo toạ độ California")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS, "ban_do_gia.png"), dpi=120)
plt.close()

print(">> Đã lưu: ban_do_gia.png")

# Chia train/test 
X = df.drop(columns="MedHouseVal")
y = df["MedHouseVal"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# BƯỚC 4: Baseline
log("BƯỚC 4: Baseline (DummyRegressor)")
dummy = DummyRegressor(strategy="mean")
dummy.fit(X_train, y_train)
rmse_baseline = np.sqrt(mean_squared_error(y_test, dummy.predict(X_test)))
print(f">> RMSE baseline: {rmse_baseline:.4f}")

#  BƯỚC 5: Linear Regression cơ bản
log("BƯỚC 5: Linear Regression")
pipe = Pipeline([("scale", StandardScaler()), ("lr", LinearRegression())])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f">> RMSE : {rmse:.4f}  (baseline: {rmse_baseline:.4f})")
print(f">> MAE  : {mae:.4f}")
print(f">> R²   : {r2:.4f}")
print(f">> MAPE : {mape:.2f}%")

# BƯỚC 6: Residual plot 
log("BƯỚC 6: Residual plot")
residuals = y_test - y_pred

plt.figure(figsize=(7, 5))
plt.scatter(y_pred, residuals, alpha=0.25, s=10)
plt.axhline(0, color="red", linestyle="--", linewidth=1.5)
plt.xlabel("Giá dự đoán (ŷ)")
plt.ylabel("Phần dư (y - ŷ)")
plt.title("Residual Plot — Linear Regression")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS, "residual_plot.png"), dpi=120)
plt.close()
print(">> Đã lưu: residual_plot.png")

# BƯỚC 7: Q-Q plot
log("BƯỚC 7: Q-Q plot")
plt.figure(figsize=(6, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Q-Q Plot — Phần dư")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS, "qq_plot.png"), dpi=120)
plt.close()
print(">> Đã lưu: qq_plot.png")

#BƯỚC 8: Log-transform
log("BƯỚC 8: Thử dự đoán log(giá)")
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

pipe_log = Pipeline([("scale", StandardScaler()), ("lr", LinearRegression())])
pipe_log.fit(X_train, y_train_log)
y_pred_log = pipe_log.predict(X_test)
y_pred_from_log = np.expm1(y_pred_log)

rmse_log = np.sqrt(mean_squared_error(y_test, y_pred_from_log))
r2_log = r2_score(y_test, y_pred_from_log)
print(f">> RMSE (log-transform, quy về thang gốc): {rmse_log:.4f}  (gốc: {rmse:.4f})")
print(f">> R²   (log-transform, quy về thang gốc): {r2_log:.4f}  (gốc: {r2:.4f})")

residuals_log = y_test_log - y_pred_log
plt.figure(figsize=(7, 5))
plt.scatter(y_pred_log, residuals_log, alpha=0.25, s=10)
plt.axhline(0, color="red", linestyle="--", linewidth=1.5)
plt.xlabel("log(1+giá) dự đoán")
plt.ylabel("Phần dư")
plt.title("Residual Plot — sau log-transform")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS, "residual_plot_log.png"), dpi=120)
plt.close()
print(">> Đã lưu: residual_plot_log.png")

ket_luan_log = ("cải thiện" if rmse_log < rmse else "không cải thiện đáng kể")
print(f">> Kết luận: log-transform {ket_luan_log} so với mô hình gốc.")

# BƯỚC 9: VIF — đa cộng tuyến 
log("BƯỚC 9: Kiểm tra đa cộng tuyến (VIF)")
X_vif = X_train.reset_index(drop=True).copy()
vif_data = pd.DataFrame()
vif_data["feature"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
vif_data = vif_data.sort_values("VIF", ascending=False)
print(vif_data.to_string(index=False))
vif_data.to_csv(os.path.join(REPORTS, "vif.csv"), index=False)
print(">> Đã lưu: vif.csv")
print(">> Ngưỡng cảnh báo thường dùng: VIF > 5 (hoặc > 10) là đáng lo.")

# BƯỚC 10: Feature engineering
log("BƯỚC 10: Feature engineering")
X_train_fe = X_train.copy()
X_test_fe = X_test.copy()

SF = (37.77, -122.42)
LA = (34.05, -118.24)

for Xd in [X_train_fe, X_test_fe]:
    Xd["rooms_per_household"] = Xd["AveRooms"] / Xd["AveOccup"].replace(0, np.nan)
    Xd["rooms_per_household"] = Xd["rooms_per_household"].fillna(Xd["AveRooms"])
    Xd["dist_to_SF"] = np.sqrt((Xd["Latitude"] - SF[0]) ** 2 + (Xd["Longitude"] - SF[1]) ** 2)
    Xd["dist_to_LA"] = np.sqrt((Xd["Latitude"] - LA[0]) ** 2 + (Xd["Longitude"] - LA[1]) ** 2)

pipe_fe = Pipeline([("scale", StandardScaler()), ("lr", LinearRegression())])
pipe_fe.fit(X_train_fe, y_train)
y_pred_fe = pipe_fe.predict(X_test_fe)
rmse_fe = np.sqrt(mean_squared_error(y_test, y_pred_fe))
r2_fe = r2_score(y_test, y_pred_fe)
print(f">> RMSE sau feature engineering: {rmse_fe:.4f}  (gốc: {rmse:.4f})")
print(f">> R²   sau feature engineering: {r2_fe:.4f}  (gốc: {r2:.4f})")

# BƯỚC 11: Bảng hệ số chuẩn hoá 
log("BƯỚC 11: Bảng hệ số đã chuẩn hoá")
he_so = pd.Series(pipe["lr"].coef_, index=X.columns).sort_values(key=abs, ascending=False)
print(he_so.to_string())
he_so.to_csv(os.path.join(REPORTS, "he_so.csv"), header=["coefficient"])

plt.figure(figsize=(8, 5))
he_so.sort_values().plot(kind="barh", color=["#d62728" if v < 0 else "#2ca02c" for v in he_so.sort_values()])
plt.xlabel("Hệ số (đã chuẩn hoá)")
plt.title("Mức ảnh hưởng của từng đặc trưng lên giá nhà")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS, "he_so.png"), dpi=120)
plt.close()
print(">> Đã lưu: he_so.csv, he_so.png")

top3 = he_so.abs().sort_values(ascending=False).head(3)
print("\n>> DIỄN GIẢI 3 YẾU TỐ ẢNH HƯỞNG MẠNH NHẤT (ngôn ngữ nghiệp vụ):")
for feat in top3.index:
    huong = "tăng" if he_so[feat] > 0 else "giảm"
    print(f"   - {feat}: mỗi 1 độ lệch chuẩn tăng thêm ở '{feat}' làm giá nhà "
          f"(giữ các yếu tố khác không đổi) có xu hướng {huong}, hệ số = {he_so[feat]:.4f}. "
          f"(Đây là mối tương quan, KHÔNG khẳng định nhân quả.)")

# Lưu model 
log("LƯU MODEL")
joblib.dump(pipe, os.path.join(MODELS, "lr_pipeline.joblib"))
print(f">> Đã lưu model: {os.path.join(MODELS, 'lr_pipeline.joblib')}")

# Tổng kết
log("TỔNG KẾT")
print(f"RMSE baseline        : {rmse_baseline:.4f}")
print(f"RMSE Linear Regression: {rmse:.4f}")
print(f"RMSE (log-transform)  : {rmse_log:.4f}")
print(f"RMSE (feature eng.)   : {rmse_fe:.4f}")
print(f"R²   Linear Regression: {r2:.4f}")
print(f"Mức tham chiếu README : R² ~0.58-0.61, RMSE ~0.72-0.75")
print(f"\nHạn chế cần nêu: (1) nhãn bị cắt ngọn ở 5.0 ({so_dong_cat_ngon} dòng), "
      f"(2) quan hệ toạ độ-giá là phi tuyến, model tuyến tính bắt kém.")