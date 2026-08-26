import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import joblib
import os

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

os.makedirs("reports", exist_ok=True)
os.makedirs("models", exist_ok=True)

# 1. NAP DU LIEU + GOP NHAN THANH NHI PHAN
COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty",
]

def load_nsl_kdd(path):
    df = pd.read_csv(path, names=COLUMNS)
    df["y"] = (df["label"] != "normal").astype(int)
    df = df.drop(columns=["label", "difficulty"])
    return df

train_path = "D:\TT_ML\TT-09-AdaBoost_Linh\data\KDDTrain+.txt"
test_path = "D:\TT_ML\TT-09-AdaBoost_Linh\data\KDDTest+.txt"

df_train = load_nsl_kdd(train_path)
df_test = load_nsl_kdd(test_path)

print("Train shape:", df_train.shape, "| Attack rate:", df_train["y"].mean().round(3))
print("Test shape :", df_test.shape, "| Attack rate:", df_test["y"].mean().round(3))

X_train_raw = df_train.drop(columns=["y"])
y_train = df_train["y"]
X_test_raw = df_test.drop(columns=["y"])
y_test = df_test["y"]

# 2. ONE-HOT COT PHAN LOAI + SCALE COT SO
cat_cols = ["protocol_type", "service", "flag"]
num_cols = [c for c in X_train_raw.columns if c not in cat_cols]

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", StandardScaler(), num_cols),
])

# 4. BASELINE: DummyClassifier + 1 stump don le
def make_pipeline(estimator):
    return Pipeline([("prep", preprocess), ("clf", estimator)])

dummy = make_pipeline(DummyClassifier(strategy="most_frequent"))
dummy.fit(X_train_raw, y_train)
f1_dummy = f1_score(y_test, dummy.predict(X_test_raw))

single_stump = make_pipeline(DecisionTreeClassifier(max_depth=1, random_state=RANDOM_STATE))
single_stump.fit(X_train_raw, y_train)
f1_stump = f1_score(y_test, single_stump.predict(X_test_raw))

print(f"\n[Baseline] DummyClassifier F1 = {f1_dummy:.3f}")
print(f"[Baseline] 1 stump (depth=1) F1 = {f1_stump:.3f}  <-- rat yeu, nhu du kien")

# 5. ADABOOST 300 STUMP
ada = make_pipeline(AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=300,
    learning_rate=0.5,
    random_state=RANDOM_STATE,
))
ada.fit(X_train_raw, y_train)
f1_ada_test = f1_score(y_test, ada.predict(X_test_raw))
print(f"[AdaBoost 300 stumps] F1 tren test NSL-KDD goc = {f1_ada_test:.3f}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(ada, X_train_raw, y_train, cv=cv, scoring="f1")
print(f"[AdaBoost 300 stumps] F1 cross-validation = {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
print(">> So sanh CV vs test-goc: chenh lech lon la BINH THUONG vi test co tan cong la (zero-day).")

# 6. DUONG F1 THEO n_estimators = 1..300
# Dung staged_predict de khong phai train lai 300 lan
ada_raw = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=300, learning_rate=0.5, random_state=RANDOM_STATE,
)
X_train_enc = preprocess.fit_transform(X_train_raw, y_train)
X_test_enc = preprocess.transform(X_test_raw)
ada_raw.fit(X_train_enc, y_train)

f1_by_round = [
    f1_score(y_test, pred) for pred in ada_raw.staged_predict(X_test_enc)
]

plt.figure(figsize=(8, 5))
plt.plot(range(1, 301), f1_by_round)
plt.xlabel("So vong lap (n_estimators)")
plt.ylabel("F1-score tren tap test")
plt.title("F1 theo so vong lap AdaBoost")
plt.grid(alpha=0.3)
plt.savefig("reports/f1_theo_vong_lap.png", dpi=120, bbox_inches="tight")
plt.close()
print("\n[Saved] reports/f1_theo_vong_lap.png")

# 7. THI NGHIEM NHIEU NHAN: dao nguoc 5% nhan train
def flip_labels(y, frac, seed=RANDOM_STATE):
    y_noisy = y.copy()
    rng = np.random.RandomState(seed)
    n_flip = int(len(y) * frac)
    idx = rng.choice(y.index, size=n_flip, replace=False)
    y_noisy.loc[idx] = 1 - y_noisy.loc[idx]
    return y_noisy

y_train_noisy = flip_labels(y_train, 0.05)

ada_noisy = make_pipeline(AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=300, learning_rate=0.5, random_state=RANDOM_STATE,
))
ada_noisy.fit(X_train_raw, y_train_noisy)
f1_ada_noisy = f1_score(y_test, ada_noisy.predict(X_test_raw))

rf_clean = make_pipeline(RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE))
rf_clean.fit(X_train_raw, y_train)
f1_rf_clean = f1_score(y_test, rf_clean.predict(X_test_raw))

rf_noisy = make_pipeline(RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE))
rf_noisy.fit(X_train_raw, y_train_noisy)
f1_rf_noisy = f1_score(y_test, rf_noisy.predict(X_test_raw))

noise_table = pd.DataFrame({
    "Model": ["AdaBoost", "RandomForest"],
    "F1_sach": [f1_ada_test, f1_rf_clean],
    "F1_nhieu_5%": [f1_ada_noisy, f1_rf_noisy],
})
noise_table["Tut_diem"] = noise_table["F1_sach"] - noise_table["F1_nhieu_5%"]
print("\n=== THI NGHIEM NHIEU NHAN (5%) ===")
print(noise_table.to_string(index=False))
print(">> Ky vong: AdaBoost tut nhieu hon Random Forest (do co che tang trong so mau sai).")

plt.figure(figsize=(6, 4))
x = np.arange(2)
plt.bar(x - 0.15, noise_table["F1_sach"], width=0.3, label="F1 sach")
plt.bar(x + 0.15, noise_table["F1_nhieu_5%"], width=0.3, label="F1 nhieu 5%")
plt.xticks(x, noise_table["Model"])
plt.ylabel("F1-score")
plt.title("Anh huong cua nhieu nhan: AdaBoost vs Random Forest")
plt.legend()
plt.savefig("reports/thi_nghiem_nhieu.png", dpi=120, bbox_inches="tight")
plt.close()
print("[Saved] reports/thi_nghiem_nhieu.png")

# 8. SO SANH ADABOOST vs GRADIENT BOOSTING vs RANDOM FOREST
from sklearn.ensemble import GradientBoostingClassifier

gb = make_pipeline(GradientBoostingClassifier(
    n_estimators=200, max_depth=3, random_state=RANDOM_STATE))
gb.fit(X_train_raw, y_train)
f1_gb = f1_score(y_test, gb.predict(X_test_raw))

compare_table = pd.DataFrame({
    "Model": ["AdaBoost (300 stump)", "GradientBoosting", "RandomForest"],
    "F1_test_goc": [f1_ada_test, f1_gb, f1_rf_clean],
})
print("\n=== SO SANH 3 THUAT TOAN ENSEMBLE ===")
print(compare_table.to_string(index=False))

plt.figure(figsize=(6, 4))
plt.bar(compare_table["Model"], compare_table["F1_test_goc"])
plt.ylabel("F1-score (tap test goc)")
plt.title("So sanh AdaBoost vs GradientBoosting vs RandomForest")
plt.xticks(rotation=15)
plt.savefig("reports/so_sanh_ensemble.png", dpi=120, bbox_inches="tight")
plt.close()
print("[Saved] reports/so_sanh_ensemble.png")

# 10. MA TRAN NHAM LAN + UOC TINH BAO DONG GIA / NGAY
y_pred = ada.predict(X_test_raw)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print("\nMa tran nham lan (AdaBoost, tap test goc):")
print(cm)
print(classification_report(y_test, y_pred, target_names=["normal", "attack"]))

packets_per_minute = 100_000  
fp_rate = fp / (fp + tn)
false_alarms_per_day = fp_rate * packets_per_minute * 60 * 24
print(f"\nTy le bao dong gia (FPR) = {fp_rate:.4%}")
print(f"Uoc tinh so bao dong gia/ngay (voi {packets_per_minute:,} goi/phut) = {false_alarms_per_day:,.0f}")

#Luu model
joblib.dump(ada, "models/adaboost.joblib")
print("\n[Saved] models/adaboost.joblib")
print("\nHOAN TAT. Xem cac file trong reports/ va models/.")