from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent        
CSV_NAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


class ChurnDataset(Dataset):

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def find_csv(path=None):
    candidates = [Path(path)] if path else [ROOT / "data" / CSV_NAME, Path("data") / CSV_NAME, Path(CSV_NAME)]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Không thấy {CSV_NAME}. Đặt vào thư mục data/ hoặc truyền --data <đường dẫn>.")


def build_preprocessor(cat_cols):
    ct = ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])
    return Pipeline([("pre", ct)])


def prepare_data(csv_path=None, seed=42):
    df = pd.read_csv(find_csv(csv_path))
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)  # 11 dòng rỗng
    df = df.drop(columns="customerID")
    y = (df.pop("Churn") == "Yes").astype(int).values
    cat_cols = [c for c in df.columns if c not in NUM_COLS]

    X_tmp, X_test, y_tmp, y_test = train_test_split(df, y, test_size=0.2, stratify=y, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=seed)

    pre = build_preprocessor(cat_cols)
    X_train = pre.fit_transform(X_train).astype("float32")     
    X_val = pre.transform(X_val).astype("float32")
    X_test = pre.transform(X_test).astype("float32")
    return {"X_train": X_train, "X_val": X_val, "X_test": X_test,
            "y_train": y_train, "y_val": y_val, "y_test": y_test,
            "n_features": X_train.shape[1], "preprocessor": pre}

def make_loaders(data, batch_size=64):
    train_loader = DataLoader(ChurnDataset(data["X_train"], data["y_train"]), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(ChurnDataset(data["X_val"], data["y_val"]), batch_size=256)
    test_loader = DataLoader(ChurnDataset(data["X_test"], data["y_test"]), batch_size=256)
    return train_loader, val_loader, test_loader

if __name__ == "__main__":     
    d = prepare_data()
    tr, va, te = make_loaders(d)
    xb, yb = next(iter(tr))
    print("n_features:", d["n_features"], "| batch:", tuple(xb.shape), tuple(yb.shape))
    print("tỉ lệ churn train/val/test:", d["y_train"].mean().round(3), d["y_val"].mean().round(3), d["y_test"].mean().round(3))