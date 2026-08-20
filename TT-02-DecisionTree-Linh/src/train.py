import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    make_scorer,
)
# Đường dẫn 

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "WA_Fn-UseC_-HR-Employee-Attrition.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "tree.joblib")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
METRICS_PATH = os.path.join(REPORT_DIR, "metrics.json")
RULES_PATH = os.path.join(REPORT_DIR, "luat_cay_quyet_dinh.txt")

DROP_COLS = ["EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"]
SENSITIVE_COLS = ["Gender", "MaritalStatus"]  # dùng để cảnh báo đạo đức, không tự động xóa

PARAM_GRID = {
    "model__criterion": ["gini", "entropy"],
    "model__max_depth": [3, 4, 5, 6],
    "model__min_samples_leaf": [10, 20, 30],
    "model__class_weight": ["balanced"],
}
# Các hàm xử lý
def load_data(path: str):
    df = pd.read_csv(path)  #Đọc file CSV 
    df = df.drop(columns=DROP_COLS) #Loại bỏ 4 cột không nghĩa
    X = df.drop("Attrition", axis=1)
    y = df["Attrition"]
    return X, y
#Tiền xử lý
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = X.select_dtypes(include="object").columns #Lấy ds cột kiểu chữ
    num_cols = X.select_dtypes(exclude="object").columns #Lấy ds cột kiểu số
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),# cột chữ mã hóa onehot bỏ qua giá trị lạ
        ("num", "passthrough", num_cols),# cột số giữu nguyên không scale
    ]) 

def build_pipeline(preprocessor: ColumnTransformer) -> Pipeline:
    return Pipeline([
        ("prep", preprocessor), #Tiền xử lý dữ liệu
        ("model", DecisionTreeClassifier(random_state=42)), #Model cây quyết định
    ])

def train_model(X_train, y_train, preprocessor: ColumnTransformer) -> GridSearchCV:
    pipeline = build_pipeline(preprocessor)
    f1_yes_scorer = make_scorer(f1_score, pos_label="Yes")

    grid = GridSearchCV(
        pipeline,
        PARAM_GRID,
        cv=5, #chia dữ liệu train thành 5 fold để kiểm chứng chéo
        scoring=f1_yes_scorer,
        n_jobs=-1, #dùng tất cả CPU để chạy, nhanh hơn
    )
    print("Đang huấn luyện...")
    grid.fit(X_train, y_train)
    return grid


def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_yes": f1_score(y_test, y_pred, pos_label="Yes"),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }

    print(f"Accuracy         : {metrics['accuracy']:.4f}")
    print(f"F1 (lop 'Yes')   : {metrics['f1_yes']:.4f}")
    print(f"F1 weighted      : {metrics['f1_weighted']:.4f}")
    print(classification_report(y_test, y_pred))

    return metrics, y_pred


def save_overfit_curve(X_train, y_train, X_test, y_test, preprocessor, best_params, out_path):
    depths = range(1, 16)
    train_scores, test_scores = [], []

    for depth in depths:
        model = Pipeline([
            ("prep", preprocessor),
            ("model", DecisionTreeClassifier(
                max_depth=depth,
                criterion=best_params["model__criterion"],
                min_samples_leaf=best_params["model__min_samples_leaf"],
                class_weight=best_params["model__class_weight"],
                random_state=42,
            )),
        ])
        model.fit(X_train, y_train)
        train_scores.append(accuracy_score(y_train, model.predict(X_train)))
        test_scores.append(accuracy_score(y_test, model.predict(X_test)))

    plt.figure(figsize=(8, 5))
    plt.plot(depths, train_scores, marker="o", label="Train")
    plt.plot(depths, test_scores, marker="s", label="Test")
    plt.xlabel("Max depth")
    plt.ylabel("Accuracy")
    plt.title(f"criterion={best_params['model__criterion']}, "
              f"min_samples_leaf={best_params['model__min_samples_leaf']}")
    plt.legend()
    plt.grid()
    plt.savefig(out_path, dpi=300)
    plt.close()

def save_feature_importance(best_model: Pipeline, out_path: str) -> pd.DataFrame:
    tree = best_model.named_steps["model"]
    feature_names = best_model.named_steps["prep"].get_feature_names_out()

    importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": tree.feature_importances_,
    }).sort_values(by="Importance", ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    plt.barh(importance["Feature"][::-1], importance["Importance"][::-1])
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    return importance

def save_tree_plot(best_model: Pipeline, out_path: str):
    tree = best_model.named_steps["model"]
    feature_names = best_model.named_steps["prep"].get_feature_names_out()

    plt.figure(figsize=(22, 10))
    plot_tree(
        tree,
        feature_names=feature_names,
        class_names=["No", "Yes"],
        filled=True,
        rounded=True,
        fontsize=7,
    )
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def save_rules_text(best_model: Pipeline, out_path: str):
    """Xuat luat dang van ban (export_text) - san pham quan trong nhat
    cua bai toan nay theo yeu cau de bai: cay phai cho ra QUY TAC doc
    duoc, khong chi mot file model."""
    tree = best_model.named_steps["model"]
    feature_names = best_model.named_steps["prep"].get_feature_names_out()

    rules = export_text(tree, feature_names=list(feature_names))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(rules)
    return rules

def check_sensitive_features(importance: pd.DataFrame):
    flagged = [
        row["Feature"] for _, row in importance.iterrows()
        if any(row["Feature"].split("_")[1:2] == [col] or col in row["Feature"] for col in SENSITIVE_COLS)
    ]
    if flagged:
        print(f"\n[CANH BAO DAO DUC] Cay dang dung dac trung nhay cam trong top-importance: {flagged}")
        print("Can can nhac loai bo de tranh phan biet doi xu.")


def save_metrics(metrics: dict, best_params: dict, out_path: str):
    payload = {
        "best_params": best_params,
        "accuracy": metrics["accuracy"],
        "f1_yes": metrics["f1_yes"],
        "f1_weighted": metrics["f1_weighted"],
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

    preprocessor = build_preprocessor(X)
    grid = train_model(X_train, y_train, preprocessor)
    best_model = grid.best_estimator_
    print("\nBest parameters:", grid.best_params_)

    metrics, y_pred = evaluate_model(best_model, X_test, y_test)

    save_overfit_curve(
        X_train, y_train, X_test, y_test, preprocessor, grid.best_params_,
        os.path.join(REPORT_DIR, "overfit_theo_depth.png"),
    )
    importance = save_feature_importance(best_model, os.path.join(REPORT_DIR, "feature_importance.png"))
    save_tree_plot(best_model, os.path.join(REPORT_DIR, "cay_quyet_dinh.png"))
    rules_text = save_rules_text(best_model, RULES_PATH)
    check_sensitive_features(importance)
    save_metrics(metrics, grid.best_params_, METRICS_PATH)

    joblib.dump(best_model, MODEL_PATH)

    print(f"\nĐã luuw model tại: {MODEL_PATH}")
    print(f"Đã lưu biểu đồ, metrics.json và luật tại: {REPORT_DIR}")
    print(rules_text[:500], "...")


if __name__ == "__main__":
    main()