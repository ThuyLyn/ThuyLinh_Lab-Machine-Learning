from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error


# --------------------------------------------------------------------------- #
# 1. Nạp dữ liệu
# --------------------------------------------------------------------------- #
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["dteday"])
    df = df.sort_values(["dteday", "hr"]).reset_index(drop=True)
    return df

# --------------------------------------------------------------------------- #
# 2. Chứng minh rò rỉ (BẪY 1)
# --------------------------------------------------------------------------- #
def prove_leakage(df: pd.DataFrame) -> dict:
    from xgboost import XGBRegressor

    leak_cols = [c for c in ["casual", "registered"] if c in df.columns]
    if not leak_cols:
        raise ValueError("Không tìm thấy cột casual/registered để chứng minh rò rỉ.")

    drop_cols = ["cnt", "dteday"] + ([ "instant"] if "instant" in df.columns else [])
    X = df.drop(columns=drop_cols)
    y = df["cnt"]

    # Chỉ cần train/test đơn giản (không cần time-split ở bước chứng minh này,
    # vì mục tiêu là minh hoạ rò rỉ đại số, không phải đánh giá model thật)
    n = len(df)
    split = int(n * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = XGBRegressor(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    result = {
        "leak_columns_used": leak_cols,
        "r2_with_leak": r2_score(y_test, pred),
        "rmse_with_leak": float(np.sqrt(mean_squared_error(y_test, pred))),
    }
    return result


def drop_leaky_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [c for c in ["casual", "registered"] if c in df.columns]
    return df.drop(columns=cols_to_drop)


# --------------------------------------------------------------------------- #
# 3. Chia dữ liệu THEO THỜI GIAN (BẪY 2)
# --------------------------------------------------------------------------- #
def time_based_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["yr"] == 0].copy()
    val = df[(df["yr"] == 1) & (df["mnth"] <= 9)].copy()
    test = df[(df["yr"] == 1) & (df["mnth"] > 9)].copy()
    return train, val, test


# --------------------------------------------------------------------------- #
# 5. Mã hoá chu kỳ sin/cos
# --------------------------------------------------------------------------- #
def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cycles = {"hr": 24, "mnth": 12, "weekday": 7}
    for col, period in cycles.items():
        df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / period)
        df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / period)
    return df


# --------------------------------------------------------------------------- #
# 6. Baseline naive: "cùng giờ tuần trước"
# --------------------------------------------------------------------------- #
def naive_weekly_baseline(df: pd.DataFrame, target_col: str = "cnt") -> pd.Series:
    return df[target_col].shift(168)


def evaluate(y_true, y_pred, label: str = "") -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    print(f"[{label}] RMSE = {rmse:.2f} | R^2 = {r2:.4f}")
    return {"label": label, "rmse": rmse, "r2": r2}