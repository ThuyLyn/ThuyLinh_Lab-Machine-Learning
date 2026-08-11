import pandas as pd
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
# Chuyển y: yes/no sang 1/0
df["y"] = df["y"].map({
    "no": 0,
    "yes": 1
})

# Hàm train Random Forest
def train_model(data, name):
    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    X = data.drop(columns="y")
    y = data["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

# Tìm các cột dạng chữ
    categorical_columns = X_train.select_dtypes(
        include="object"
    ).columns

# Chuyển dữ liệu chữ thành số
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "category",
                OneHotEncoder(handle_unknown="ignore"),
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
            random_state=42,
            n_jobs=-1
        )   
# Ghép xử lý dữ liệu + model
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("random_forest", model)
    ])

 # Train
    pipeline.fit(X_train, y_train)

    # Dự đoán xác suất thuộc lớp YES
    probability = pipeline.predict_proba(X_test)[:, 1]

    # Tính ROC-AUC
    auc = roc_auc_score(y_test, probability)

    print(f"ROC-AUC: {auc:.4f}")

    return auc

# model có duration
auc_with_duration = train_model(
    df,
    "MODEL 1 - CÓ DURATION"
)
#model không có duration
df_no_duration = df.drop(
    columns="duration"
)

auc_without_duration = train_model(
    df_no_duration,
    "MODEL 2 - KHÔNG CÓ DURATION"
)
# So sánh kqua
print("\n" + "=" * 50)
print("KẾT QUẢ DATA LEAKAGE")
print("=" * 50)

print(f"Có duration    : {auc_with_duration:.4f}")
print(f"Không duration : {auc_without_duration:.4f}")

print(
    f"Chênh lệch     : "
    f"{auc_with_duration - auc_without_duration:.4f}"
)
