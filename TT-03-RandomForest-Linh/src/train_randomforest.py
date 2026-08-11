import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
# Đọc dữ liệu

df = pd.read_csv(
    "dataset/bank-additional-full.csv",
    sep=";"
)

print("Kích thước dữ liệu:", df.shape)
# Xử lý target

df["y"] = df["y"].map({
    "no": 0,
    "yes": 1
})
# Xử lý pdays

df["pdays_missing"] = (
    df["pdays"] == 999
).astype(int)
# Bỏ duration

df = df.drop(
    columns="duration"
)
# Tách X, y

X = df.drop(columns="y")
y = df["y"]
# Chia train/test

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Train:", X_train.shape)
print("Test :", X_test.shape)
# xác định categorical

categorical_columns = X_train.select_dtypes(
    include=["object", "string"]
).columns

print("\nCác cột categorical:")
print(list(categorical_columns))
# One - hot encoder

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_columns
        )
    ],
    remainder="passthrough"
)
# Random Forest

model = RandomForestClassifier(
    n_estimators=200,
    max_features="sqrt",
    class_weight="balanced_subsample",
    oob_score=True,
    random_state=42,
    n_jobs=-1
)
# Pipeline

pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("random_forest", model)
])
# Train model

print("\nĐang train Random Forest...")

pipeline.fit(
    X_train,
    y_train
)

print("Train hoàn thành!")

# Tính AUC

y_probability = pipeline.predict_proba(
    X_test
)[:, 1]

auc = roc_auc_score(
    y_test,
    y_probability
)
# Lấy OOB Score

oob_score = model.oob_score_
# Kết quả

print("\n===== KẾT QUẢ TRAIN =====")

print(f"ROC-AUC  : {auc:.4f}")
print(f"OOB Score: {oob_score:.4f}")
# Lưu model và kết quả

results = {
    "model": pipeline,
    "auc": auc,
    "oob_score": oob_score
}

joblib.dump(
    results,
    "models/random_forest_results.joblib"
)

print("\nĐã lưu:")
print(
    "models/random_forest_results.joblib"
)