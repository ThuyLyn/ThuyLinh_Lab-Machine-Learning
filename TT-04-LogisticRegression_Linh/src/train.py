import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Đọc dữ liệu
df = pd.read_csv("dataset/heart.csv")

# Xóa dữ liệu trùng
df = df.drop_duplicates()

#  Xác định biến
categorical_columns = [
    "cp",
    "restecg",
    "slope",
    "thal",
    "ca"
]

numeric_columns = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak"
]

X = df.drop(columns=["target"])
y = df["target"]

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
# Preprocessing

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        ),
        (
            "numeric",
            StandardScaler(),
            numeric_columns
        )
    ]
)

# Tạo Logistic Regression
model = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                random_state=42
            )
        )
    ]
)
# Train
model.fit(X_train, y_train)
# Lưu model
joblib.dump(
    model,
    "models/logreg_pipeline.joblib"
)

print("Đã train và lưu model thành công!")