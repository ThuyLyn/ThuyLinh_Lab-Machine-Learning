import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

#Load Random Forest đã train
rf_pipeline = joblib.load(
    "models/random_forest_pipeline.joblib"
)

print("Đã load Random Forest!")

# Đọc và xử lý dữ liệu
df = pd.read_csv(
    "dataset/bank-additional-full.csv",
    sep=";"
)

df["y"] = df["y"].map({
    "no": 0,
    "yes": 1
})

df["pdays_missing"] = (
    df["pdays"] == 999
).astype(int)

df = df.drop(columns="duration")

X = df.drop(columns="y")
y = df["y"]
# Chia Train/Test

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# DummyClassifier
dummy = DummyClassifier(
    strategy="prior"
)

dummy.fit(
    X_train,
    y_train
)

dummy_prob = dummy.predict_proba(
    X_test
)[:, 1]

dummy_auc = roc_auc_score(
    y_test,
    dummy_prob
)
# Decision Tree

# Dùng preprocessing có sẵn trong Random Forest
preprocessor = rf_pipeline.named_steps[
    "preprocessing"
]

X_train_encoded = preprocessor.fit_transform(
    X_train,
    y_train
)

X_test_encoded = preprocessor.transform(
    X_test
)

tree = DecisionTreeClassifier(
    random_state=42,
    class_weight="balanced"
)

tree.fit(
    X_train_encoded,
    y_train
)

tree_prob = tree.predict_proba(
    X_test_encoded
)[:, 1]

tree_auc = roc_auc_score(
    y_test,
    tree_prob
)
# random Forest

rf_prob = rf_pipeline.predict_proba(
    X_test
)[:, 1]

rf_auc = roc_auc_score(
    y_test,
    rf_prob
)
# Kết quả

print("\n===== BASELINE =====")

print(f"DummyClassifier : {dummy_auc:.4f}")
print(f"Decision Tree   : {tree_auc:.4f}")
print(f"Random Forest   : {rf_auc:.4f}")