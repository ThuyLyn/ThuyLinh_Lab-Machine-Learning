import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

MAX_GAP = 6
CONT_COLS = ["traffic_volume", "temp", "rain_log", "snow_1h", "clouds_all"]

def load_raw(path):
    raw = pd.read_csv(path)
    raw["date_time"] = pd.to_datetime(raw["date_time"])
    return raw

def clean(raw):
    """Khử trùng date_time, lọc ngoại lai, lan cờ ngày lễ. Trả về (df, hol_by_date)."""
    # pandas bản mới đọc "None" thành NaN -> coi cả NaN lẫn "None" là không phải ngày lễ
    raw = raw.copy()
    raw["holiday_flag"] = (raw["holiday"].notna() & (raw["holiday"] != "None")).astype(int)
    df = (raw.sort_values("date_time").groupby("date_time")
             .agg(traffic_volume=("traffic_volume", "first"), temp=("temp", "first"),
                  rain_1h=("rain_1h", "first"), snow_1h=("snow_1h", "first"),
                  clouds_all=("clouds_all", "first"), weather_main=("weather_main", "first"),
                  holiday_flag=("holiday_flag", "max")))
    df.loc[df["temp"] < 200, "temp"] = np.nan        # Kelvin: <200K là vô lý (kể cả 0K)
    df.loc[df["rain_1h"] > 100, "rain_1h"] = np.nan  # >100 mm/giờ là vô lý
    hol_by_date = df.groupby(df.index.normalize())["holiday_flag"].max()
    return df, hol_by_date

def to_hourly(df, hol_by_date, max_gap=MAX_GAP):
    h = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="h"))
    h["imputed"] = h["traffic_volume"].isna().astype(int)
    h["is_holiday"] = h.index.normalize().map(hol_by_date).fillna(0)
    num = ["traffic_volume", "temp", "rain_1h", "snow_1h", "clouds_all"]
    isna = h["traffic_volume"].isna()
    run_len = isna.groupby((isna != isna.shift()).cumsum()).transform("sum")
    short = isna & (run_len <= max_gap)
    interp = h[num].interpolate(limit_area="inside")
    for c in num:
        h.loc[short, c] = interp.loc[short, c]
    for c in ["temp", "rain_1h", "snow_1h", "clouds_all"]:
        h[c] = h[c].interpolate(limit_area="inside").ffill().bfill()
    h["weather_main"] = h["weather_main"].ffill().bfill()
    return h


def build_features(h):
    d = h[h["traffic_volume"].notna()].copy()
    hr, dw = d.index.hour, d.index.dayofweek
    d["hour_sin"], d["hour_cos"] = np.sin(2*np.pi*hr/24), np.cos(2*np.pi*hr/24)
    d["dow_sin"], d["dow_cos"] = np.sin(2*np.pi*dw/7), np.cos(2*np.pi*dw/7)
    d["rain_log"] = np.log1p(d["rain_1h"])
    wm = pd.get_dummies(d["weather_main"], prefix="w").astype(float)
    d = pd.concat([d, wm], axis=1)
    feats = CONT_COLS + ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_holiday", "imputed"] + list(wm.columns)
    return d, feats

def split_and_scale(d, ratios=(0.70, 0.15)):
    """Chia THEO THỜI GIAN 70/15/15; scaler CHỈ fit trên train. Trả về (d, d_s, scaler)."""
    n = len(d); i1, i2 = int(n*ratios[0]), int(n*(ratios[0]+ratios[1]))
    d = d.copy()
    d["split"] = ["train"]*i1 + ["val"]*(i2-i1) + ["test"]*(n-i2)
    scaler = StandardScaler().fit(d.loc[d.split == "train", CONT_COLS])
    d_s = d.copy()
    d_s[CONT_COLS] = scaler.transform(d[CONT_COLS])
    return d, d_s, scaler
print("OK")