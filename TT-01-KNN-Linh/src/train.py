import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Đường dẫn

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "dataset", "diabetes.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "knn_pipeline.joblib")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# Đọc dữ liệu

df = pd.read_csv(DATA_PATH)

df[
    ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
] = df[
    ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
].replace(0, np.nan)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Chia dữ liệu

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Pipeline

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier())
])

# Grid Search

param_grid = {
    "knn__n_neighbors": range(1, 32, 2),
    "knn__weights": ["uniform", "distance"],
    "knn__metric": ["euclidean", "manhattan"]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="recall",
    n_jobs=-1
)

print("Đang huấn luyện...")

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# Dự đoán

y_pred = best_model.predict(X_test)

# Đánh giá

print("\nBest parameters:", grid.best_params_)

print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1-score : {f1_score(y_test, y_pred):.4f}")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Không bệnh", "Mắc bệnh"]
    )
)

# Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Không bệnh", "Mắc bệnh"]
).plot(cmap="Blues")

plt.tight_layout()

plt.savefig(
    os.path.join(REPORT_DIR, "confusion_matrix.png"),
    dpi=300
)

plt.close()

# Recall theo K

k_values = range(1, 32, 2)
recalls = []

for k in k_values:

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k))
    ])

    model.fit(X_train, y_train)

    recalls.append(
        recall_score(y_test, model.predict(X_test))
    )

plt.plot(k_values, recalls, marker="o")

plt.xlabel("K")
plt.ylabel("Recall")
plt.grid()

plt.savefig(
    os.path.join(REPORT_DIR, "recall_theo_K.png"),
    dpi=300
)

plt.close()

# Lưu model

joblib.dump(best_model, MODEL_PATH)

print("\nĐã lưu model và các biểu đồ.")