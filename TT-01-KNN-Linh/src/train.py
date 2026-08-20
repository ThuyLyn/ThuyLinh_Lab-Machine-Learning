import os
import json
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
    ConfusionMatrixDisplay,
)
# Đường dẫn
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "diabetes.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "knn_pipeline.joblib")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
METRICS_PATH = os.path.join(REPORT_DIR, "metrics.json")

ZERO_AS_NAN_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
PARAM_GRID = {
    "knn__n_neighbors": list(range(1, 32, 2)),
    "knn__weights": ["uniform", "distance"],
    "knn__metric": ["euclidean", "manhattan"],
}
# Các hàm xử lý
def load_data(path: str):
    df = pd.read_csv(path)
    df[ZERO_AS_NAN_COLS] = df[ZERO_AS_NAN_COLS].replace(0, np.nan)
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    return X, y


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier()),
    ])


def train_model(X_train, y_train) -> GridSearchCV:
    pipeline = build_pipeline()
    grid = GridSearchCV(
        pipeline,
        PARAM_GRID,
        cv=5,
        scoring="recall",
        n_jobs=-1,
    )
    print("Đang huấn luyện...")
    grid.fit(X_train, y_train)
    return grid


def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "classification_report": classification_report(
            y_test, y_pred,
            target_names=["Không bệnh", "Mắc bệnh"],
            output_dict=True,
        ),
    }

    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1_score']:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Không bệnh", "Mắc bệnh"]))

    return metrics, y_pred


def save_confusion_matrix(y_test, y_pred, out_path: str):
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Không bệnh", "Mắc bệnh"],
    ).plot(cmap="Blues")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_recall_vs_k(grid: GridSearchCV, out_path: str):
    results = pd.DataFrame(grid.cv_results_)
    best_params = grid.best_params_

    mask = (
        (results["param_knn__weights"] == best_params["knn__weights"])
        & (results["param_knn__metric"] == best_params["knn__metric"])
    )
    subset = results[mask].sort_values("param_knn__n_neighbors")

    k_values = subset["param_knn__n_neighbors"]
    recalls = subset["mean_test_score"]

    plt.plot(k_values, recalls, marker="o")
    plt.xlabel("K")
    plt.ylabel("Recall (5-fold CV trên tập train)")
    plt.title(f"weights={best_params['knn__weights']}, metric={best_params['knn__metric']}")
    plt.grid()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_metrics(metrics: dict, best_params: dict, out_path: str):
    payload = {
        "best_params": best_params,
        **{k: v for k, v in metrics.items() if k != "classification_report"},
        "classification_report": metrics["classification_report"],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
# Main
def main():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    X, y = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    grid = train_model(X_train, y_train)
    best_model = grid.best_estimator_
    print("\nBest parameters:", grid.best_params_)

    metrics, y_pred = evaluate_model(best_model, X_test, y_test)

    save_confusion_matrix(y_test, y_pred, os.path.join(REPORT_DIR, "confusion_matrix.png"))
    save_recall_vs_k(grid, os.path.join(REPORT_DIR, "recall_theo_K.png"))
    save_metrics(metrics, grid.best_params_, METRICS_PATH)

    joblib.dump(best_model, MODEL_PATH)

    print(f"\nĐã lưu model tại: {MODEL_PATH}")
    print(f"Đã lưu biểu đồ và metrics tại: {REPORT_DIR}")


if __name__ == "__main__":
    main()