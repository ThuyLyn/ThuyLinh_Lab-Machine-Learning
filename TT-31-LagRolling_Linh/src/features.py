import numpy as np
import pandas as pd

try:
    import holidays
except ImportError:          # không bắt buộc; thiếu thì bỏ cột ngày lễ
    holidays = None

LAGS = [1, 2, 3, 7, 14, 30]
WINDOWS = [7, 14, 30]

def make_features(d: pd.DataFrame, leak: bool = False) -> pd.DataFrame:
    d = d.copy()
    y = d["phu_tai"]

    # --- Lag: giá trị quá khứ ---
    for lag in LAGS:
        d[f"lag_{lag}"] = y.shift(lag)

    # --- Rolling: thống kê cửa sổ quá khứ (shift(1) TRƯỚC rolling) ---
    base = y if leak else y.shift(1)
    for w in WINDOWS:
        r = base.rolling(w)
        d[f"tb_{w}"] = r.mean()
        d[f"std_{w}"] = r.std()
        d[f"max_{w}"] = r.max()
        d[f"min_{w}"] = r.min()

    # --- Lịch + mã hoá chu kỳ ---
    d["thu"] = d.index.dayofweek
    d["cuoi_tuan"] = (d["thu"] >= 5).astype(int)
    d["thang_sin"] = np.sin(2 * np.pi * d.index.month / 12)
    d["thang_cos"] = np.cos(2 * np.pi * d.index.month / 12)

    # --- Ngày lễ (dữ liệu UCI là của Pháp) ---
    if holidays is not None:
        fr = holidays.France(years=range(d.index.year.min(), d.index.year.max() + 1))
        d["la_ngay_le"] = [int(x in fr) for x in d.index.date]
    return d


def feature_columns(d: pd.DataFrame) -> list:
    return [c for c in d.columns if c not in ("phu_tai", "bi_thieu")]
