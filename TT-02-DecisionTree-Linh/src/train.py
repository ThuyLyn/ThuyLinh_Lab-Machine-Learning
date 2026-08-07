import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score
)

# Tạo thư mục

os.makedirs("reports", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Đọc dữ liệu

df = pd.read_csv(
    "D:\TT_ML\TT-02-DecisionTree-Linh\dataset\WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

# Xóa cột không cần thiết

df.drop(
    columns=[
        "EmployeeCount",
        "StandardHours",
        "Over18",
        "EmployeeNumber"
    ],
    inplace=True
)

# Chia dữ liệu

X = df.drop("Attrition", axis=1)
y = df["Attrition"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Tiền xử lý

cat_cols = X.select_dtypes(
    include="object"
).columns

num_cols = X.select_dtypes(
    exclude="object"
).columns

preprocessor = ColumnTransformer([
    (
        "cat",
        OneHotEncoder(handle_unknown="ignore"),
        cat_cols
    ),
    (
        "num",
        "passthrough",
        num_cols
    )
])

# Pipeline

pipeline = Pipeline([
    ("prep", preprocessor),
    ("model", DecisionTreeClassifier(random_state=42))
])

# Grid Search

param_grid = {
    "model__criterion": ["gini", "entropy"],
    "model__max_depth": [3, 4, 5, 6],
    "model__min_samples_leaf": [10, 20, 30],
    "model__class_weight": ["balanced"]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="f1_weighted",
    n_jobs=-1
)

print("Đang huấn luyện...")

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# Dự đoán

y_pred = best_model.predict(X_test)

# Đánh giá

print("\nBest parameters:")

print(grid.best_params_)

print(
    classification_report(
        y_test,
        y_pred
    )
)

print(
    "F1-score:",
    round(
        f1_score(
            y_test,
            y_pred,
            average="weighted"
        ),
        4
    )
)

# Overfit theo depth

depths = range(1, 16)

train_scores = []
test_scores = []

for depth in depths:

    model = Pipeline([
        ("prep", preprocessor),
        (
            "model",
            DecisionTreeClassifier(
                max_depth=depth,
                random_state=42
            )
        )
    ])

    model.fit(X_train, y_train)

    train_scores.append(
        accuracy_score(
            y_train,
            model.predict(X_train)
        )
    )

    test_scores.append(
        accuracy_score(
            y_test,
            model.predict(X_test)
        )
    )

plt.figure(figsize=(8, 5))

plt.plot(
    depths,
    train_scores,
    marker="o",
    label="Train"
)

plt.plot(
    depths,
    test_scores,
    marker="s",
    label="Test"
)

plt.xlabel("Max depth")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()

plt.savefig(
    "reports/overfit_theo_depth.png",
    dpi=300
)

plt.close()

# Feature importance

tree = best_model.named_steps["model"]

feature_names = (
    best_model.named_steps["prep"]
    .get_feature_names_out()
)

importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": tree.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
).head(10)

plt.figure(figsize=(10, 6))

plt.barh(
    importance["Feature"][::-1],
    importance["Importance"][::-1]
)

plt.xlabel("Importance")

plt.tight_layout()

plt.savefig(
    "reports/feature_importance.png",
    dpi=300
)

plt.close()

# Cây quyết định

plt.figure(figsize=(22, 10))

plot_tree(
    tree,
    feature_names=feature_names,
    class_names=["No", "Yes"],
    filled=True,
    rounded=True,
    fontsize=7
)

plt.savefig(
    "reports/cay_quyet_dinh.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# Lưu model

joblib.dump(
    best_model,
    "models/tree.joblib"
)

print("\nĐã lưu model và biểu đồ.")