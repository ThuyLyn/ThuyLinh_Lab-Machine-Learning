import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance

# Load model
data = joblib.load("models/random_forest_results.joblib")
pipeline = data["model"]
preprocessor = pipeline.named_steps["preprocessing"]
model = pipeline.named_steps["random_forest"]

# Data
df = pd.read_csv("dataset/bank-additional-full.csv", sep=";")
df["y"] = df["y"].map({"no": 0, "yes": 1})
df["pdays_missing"] = (df["pdays"] == 999).astype(int)
df = df.drop(columns="duration")

X = df.drop(columns="y")
y = df["y"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

X_train = preprocessor.transform(X_train)
X_test = preprocessor.transform(X_test)

os.makedirs("reports", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# AUC theo số cây
trees = [10, 50, 100, 150, 200, 300, 400, 500]
auc = []

for n in trees:
    rf = RandomForestClassifier(
        n_estimators=n,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    prob = rf.predict_proba(X_test)[:, 1]
    score = roc_auc_score(y_test, prob)
    auc.append(score)
    print(f"{n} cây -> AUC = {score:.4f}")

plt.plot(trees, auc, marker="o")
plt.xlabel("n_estimators")
plt.ylabel("ROC-AUC")
plt.title("AUC theo số lượng cây")
plt.grid()
plt.savefig("reports/auc_theo_so_cay.png", dpi=300)
plt.close()

# Feature importance
features = preprocessor.get_feature_names_out()

importance = pd.DataFrame({
    "feature": features,
    "feature_importance": model.feature_importances_
})

perm = permutation_importance(
    model, X_test, y_test,
    n_repeats=5,
    random_state=42,
    n_jobs=-1
)

importance["permutation"] = perm.importances_mean

top = importance.nlargest(10, "permutation")

print("\n===== TOP 10 FEATURES =====")
print(top)

top.plot(
    x="feature",
    y=["feature_importance", "permutation"],
    kind="bar",
    figsize=(10, 5)
)

plt.title("Feature Importance vs Permutation Importance")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("reports/permutation_importance.png", dpi=300)
plt.close()

# Precision@5000 và lift
prob = model.predict_proba(X_test)[:, 1]

result = pd.DataFrame({
    "actual": y_test.values,
    "probability": prob
}).sort_values("probability", ascending=False)

top5000 = result.head(5000)

precision5000 = top5000["actual"].mean()
random_rate = y_test.mean()
lift = precision5000 / random_rate

print("\n===== PRECISION@5000 =====")
print(f"Precision@5000: {precision5000:.4f}")
print(f"Random rate:    {random_rate:.4f}")
print(f"Lift:           {lift:.2f}")

top5000.to_csv(
    "outputs/danh_sach_goi_top5000.csv",
    index=False
)

# Lift/ cumulative gain
result["gain"] = result["actual"].cumsum() / result["actual"].sum()
result["percentage"] = (result.index + 1) / len(result)

plt.plot(
    result["percentage"],
    result["gain"],
    label="Random Forest"
)

plt.plot(
    [0, 1], [0, 1],
    "--",
    label="Random"
)

plt.xlabel("Tỷ lệ khách hàng được chọn")
plt.ylabel("Tỷ lệ khách hàng mục tiêu")
plt.title("Cumulative Gain Curve")
plt.legend()
plt.grid()

plt.savefig(
    "reports/lift_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nĐã lưu toàn bộ kết quả!")