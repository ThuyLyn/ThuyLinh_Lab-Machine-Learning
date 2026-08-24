import os

try:
    from IPython.display import display
except ImportError:
    display = print

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier, HistGradientBoostingClassifier,
    RandomForestClassifier, AdaBoostClassifier,
)
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, log_loss
import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
DATA_DIR = "D:\TT_ML\TT-07-Gradient_Boosting-Linh\dataset"  
pd.set_option('display.max_columns', 50)

COLS = ["age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country",
        "income"]

def clean_raw(df):
    str_cols = df.select_dtypes(include="object").columns
    for c in str_cols:
        df[c] = df[c].str.strip()
    df = df.replace("?", np.nan)
    df = df.drop(columns=["education", "fnlwgt"])
    df["income"] = df["income"].str.replace(".", "", regex=False)
    return df

train_df = pd.read_csv(f"{DATA_DIR}/adult.data", names=COLS, sep=r",\s*",
                        engine="python", na_values="?")
test_df = pd.read_csv(f"{DATA_DIR}/adult.test", names=COLS, sep=r",\s*",
                       engine="python", na_values="?", skiprows=1)

train_df, test_df = clean_raw(train_df), clean_raw(test_df)
print("Train:", train_df.shape, "| Test:", test_df.shape)
train_df.head()
import joblib

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

train_df.groupby("education-num")["income"].apply(lambda s: (s == ">50K").mean()).plot(
    kind="bar", ax=axes[0], title="Tỉ lệ >50K theo education-num", color="#4C72B0")

train_df.assign(hours_bin=pd.cut(train_df["hours-per-week"], [0,20,40,60,100])).groupby(
    "hours_bin", observed=True)["income"].apply(lambda s: (s==">50K").mean()).plot(
    kind="bar", ax=axes[1], title="Tỉ lệ >50K theo giờ làm/tuần", color="#DD8452")

train_df.groupby("marital-status")["income"].apply(lambda s: (s==">50K").mean()).plot(
    kind="bar", ax=axes[2], title="Tỉ lệ >50K theo hôn nhân", color="#55A868")

for ax in axes:
    ax.set_ylabel("Tỉ lệ >50K")
plt.tight_layout()
plt.show()

print((train_df["capital-gain"] == 0).mean(), "giá trị bằng 0")
train_df["capital-gain"].plot(kind="hist", bins=50, title="Phân phối capital-gain (lệch mạnh)")
plt.show()

y_train = (train_df["income"] == ">50K").astype(int)
X_train = train_df.drop(columns=["income"])
y_test = (test_df["income"] == ">50K").astype(int)
X_test = test_df.drop(columns=["income"])

def build_pipeline(X, model):
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = X.select_dtypes(exclude="object").columns.tolist()
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", "passthrough", num_cols),
    ])
    return Pipeline([("pre", pre), ("model", model)])

baselines = {
    "Dummy (most_frequent)": DummyClassifier(strategy="most_frequent"),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
}
rows = []
for name, model in baselines.items():
    pipe = build_pipeline(X_train, model)
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    rows.append({"model": name, "roc_auc": roc_auc_score(y_test, proba),
                 "pr_auc": average_precision_score(y_test, proba)})
pd.DataFrame(rows)

gb = GradientBoostingClassifier(
    n_estimators=500, learning_rate=0.05, max_depth=3,
    subsample=0.8, validation_fraction=0.1, n_iter_no_change=20,
    random_state=RANDOM_STATE,
)
gb_pipe = build_pipeline(X_train, gb)
gb_pipe.fit(X_train, y_train)

proba = gb_pipe.predict_proba(X_test)[:, 1]
print("ROC-AUC:", roc_auc_score(y_test, proba))
print("PR-AUC :", average_precision_score(y_test, proba))

pre = gb_pipe.named_steps["pre"]
model = gb_pipe.named_steps["model"]
Xtr, Xte = pre.transform(X_train), pre.transform(X_test)

train_loss = [log_loss(y_train, p) for p in model.staged_predict_proba(Xtr)]
test_loss = [log_loss(y_test, p) for p in model.staged_predict_proba(Xte)]
best_iter = int(np.argmin(test_loss))

plt.figure(figsize=(7,4.5))
plt.plot(train_loss, label="Train loss")
plt.plot(test_loss, label="Validation loss")
plt.axvline(best_iter, color="red", ls="--", label=f"Tốt nhất tại cây #{best_iter}")
plt.xlabel("Số cây"); plt.ylabel("Log loss"); plt.legend()
plt.title("Overfit xuất hiện khi validation loss bắt đầu tăng trở lại")
plt.show()
print(f"Sau cây #{best_iter}, train loss tiếp tục giảm nhưng validation loss tăng -> overfit.")

lrs, n_ests = [0.3, 0.1, 0.05], [50, 200, 500]
grid = np.zeros((len(lrs), len(n_ests)))
for i, lr in enumerate(lrs):
    for j, n in enumerate(n_ests):
        m = GradientBoostingClassifier(n_estimators=n, learning_rate=lr, max_depth=3,
                                        random_state=RANDOM_STATE)
        p = build_pipeline(X_train, m)
        p.fit(X_train, y_train)
        grid[i, j] = average_precision_score(y_test, p.predict_proba(X_test)[:, 1])

plt.figure(figsize=(6,5))
plt.imshow(grid, cmap="YlGnBu")
plt.xticks(range(len(n_ests)), n_ests); plt.yticks(range(len(lrs)), lrs)
plt.xlabel("n_estimators"); plt.ylabel("learning_rate")
for i in range(len(lrs)):
    for j in range(len(n_ests)):
        plt.text(j, i, f"{grid[i,j]:.3f}", ha="center", va="center")
plt.title("PR-AUC: lr nhỏ + nhiều cây thường tốt/ổn định hơn")
plt.colorbar()
plt.show()

import time
compare = {
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
    "Gradient Boosting": gb,
    "AdaBoost": AdaBoostClassifier(n_estimators=300, learning_rate=0.5, random_state=RANDOM_STATE),
}
rows = []
for name, model in compare.items():
    pipe = build_pipeline(X_train, model)
    t0 = time.time()
    pipe.fit(X_train, y_train)
    dt = time.time() - t0
    proba = pipe.predict_proba(X_test)[:, 1]
    rows.append({"model": name, "pr_auc": average_precision_score(y_test, proba),
                 "roc_auc": roc_auc_score(y_test, proba), "train_time_s": dt})
pd.DataFrame(rows).sort_values("pr_auc", ascending=False)

hgb = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05,
                                      early_stopping=True, random_state=RANDOM_STATE)
hgb_pipe = build_pipeline(X_train, hgb)

t0 = time.time(); gb_pipe.fit(X_train, y_train); t_gb = time.time() - t0
t0 = time.time(); hgb_pipe.fit(X_train, y_train); t_hgb = time.time() - t0

print(f"GradientBoosting:     {t_gb:.2f}s")
print(f"HistGradientBoosting: {t_hgb:.2f}s  ({t_gb/max(t_hgb,1e-9):.1f}x nhanh hơn)")

def bias_table(pipe, X_test, y_test, col):
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = pipe.predict(X_test)
    out = []
    for g in X_test[col].dropna().unique():
        mask = X_test[col] == g
        if mask.sum() < 20:
            continue
        out.append({
            "group": g, "n": int(mask.sum()),
            "predicted_positive_rate": pred[mask].mean(),
            "actual_positive_rate": y_test[mask].mean(),
            "avg_predicted_probability": proba[mask].mean(),
        })
    return pd.DataFrame(out)

display(bias_table(gb_pipe, X_test, y_test, "sex"))
display(bias_table(gb_pipe, X_test, y_test, "race"))
bt = bias_table(gb_pipe, X_test, y_test, "sex")
plt.figure(figsize=(5,4))
plt.bar(bt["group"], bt["predicted_positive_rate"], color=["#4C72B0", "#DD8452"])
plt.ylabel("Tỉ lệ dự đoán income > 50K")
plt.title("Thiên lệch dự đoán theo giới tính")
plt.show()

# ============================================================
# LƯU MODEL SAU KHI TRAIN
# ============================================================
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)
joblib.dump(gb_pipe, "models/gb_pipeline.joblib")
print("\nĐã lưu model tại: models/gb_pipeline.joblib")
