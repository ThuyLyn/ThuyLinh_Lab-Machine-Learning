import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


def load_airpassengers() -> pd.Series:
    df = sm.datasets.get_rdataset("AirPassengers").data   # cần internet
    return pd.Series(
        df["value"].values,
        index=pd.date_range("1949-01-01", periods=len(df), freq="MS"),
        name="khach",
    )


def bang_adf(y: pd.Series) -> pd.DataFrame:
    y_log = np.log(y)
    phien_ban = {
        "Gốc": y,
        "Log": y_log,
        "Log + diff(1)": y_log.diff(1),
        "Log + diff(1) + diff(12)": y_log.diff(1).diff(12),
    }
    rows = []
    for ten, s in phien_ban.items():
        stat, p, *_ = adfuller(s.dropna())
        rows.append([ten, round(stat, 3), round(p, 4),
                     "DỪNG" if p < 0.05 else "KHÔNG DỪNG"])
    return pd.DataFrame(rows, columns=["Phiên bản", "ADF stat", "p-value", "Kết luận"])