import argparse, os, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data import load_raw, clean, to_hourly, build_features, split_and_scale
from sequences import make_dataset

DATA_PATH = r"D:\TT_ML\TT-27-RNN-LSTM_Linh\data\Metro_Interstate_Traffic_Volume.csv"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
REPORTS = os.path.join(ROOT, "reports")
MODELS = os.path.join(ROOT, "models")

H = lambda n: pd.Timedelta(hours=n)

def mae_on(T, y_true, y_pred, common):
    m = pd.DatetimeIndex(T).isin(common)
    return float(np.mean(np.abs(y_true[m] - y_pred[m])))

class Ctx:
    def __init__(self, path):
        df, hol = clean(load_raw(path))
        self.h = to_hourly(df, hol)
        self.d, self.feats = build_features(self.h)
        self.d, self.d_s, self.scaler = split_and_scale(self.d)
        self.mu, self.sd = self.scaler.mean_[0], self.scaler.scale_[0]
        self.sr = self.h["traffic_volume"]
        self._common = {}

    def ds(self, window, horizon):
        return make_dataset(self.d, self.d_s, self.feats, window, horizon)

    def baseline(self, T):
        T = pd.DatetimeIndex(T)
        return (self.sr.reindex(T - H(24)).values, self.sr.reindex(T - H(168)).values)

    def common(self, window, horizon):
        """Các giờ test mà cả 2 baseline đều có giá trị -> mọi mô hình chấm trên CÙNG tập."""
        if (window, horizon) not in self._common:
            te = self.ds(window, horizon)["test"]
            T = te["t0"] + H(horizon)
            a, b = self.baseline(T)
            self._common[(window, horizon)] = T[~np.isnan(a) & ~np.isnan(b)]
        return self._common[(window, horizon)]

# Bước 3: EDA
def eda(ctx):
    real = ctx.h[ctx.h.imputed == 0]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    real.groupby(real.index.hour)["traffic_volume"].mean().plot(ax=ax[0], marker="o"); ax[0].set_title("TB theo giờ trong ngày")
    real.groupby(real.index.dayofweek)["traffic_volume"].mean().plot(ax=ax[1], kind="bar"); ax[1].set_title("TB theo thứ (0=T2)")
    real.groupby(real.index.month)["traffic_volume"].mean().plot(ax=ax[2], kind="bar"); ax[2].set_title("TB theo tháng")
    plt.tight_layout(); plt.savefig(os.path.join(REPORTS, "eda_theo_gio.png"), dpi=130); plt.close()

# Bước 7: XGBoost + lag
def xgb_frame(ctx, horizon):
    h = ctx.h; tv = h["traffic_volume"]
    X = pd.DataFrame(index=h.index)                     # index = t0
    for k in [0, 1, 2, 3, 6, 12, 24, 48]:
        X[f"lag_{k}"] = tv.shift(k)
    X["seas_day"] = tv.shift(24 - horizon) if horizon <= 24 else np.nan   # = giá trị tại T-24h
    X["seas_week"] = tv.shift(168 - horizon)                              # = giá trị tại T-168h
    T = h.index + H(horizon)
    X["hour_sin"], X["hour_cos"] = np.sin(2*np.pi*T.hour/24), np.cos(2*np.pi*T.hour/24)
    X["dow"] = T.dayofweek
    X["hol_T"] = pd.Series(h["is_holiday"].values, index=h.index).shift(-horizon)
    for c in ["temp", "rain_1h", "snow_1h", "clouds_all"]:
        X[c] = h[c]
    return X, tv.shift(-horizon)

def run_xgb(ctx, horizon, test_t0):
    import xgboost as xgb
    X, y = xgb_frame(ctx, horizon)
    spl = ctx.d["split"].reindex(X.index + H(horizon)).values
    ok = y.notna().values
    tr, va = (spl == "train") & ok, (spl == "val") & ok
    t = time.time()
    m = xgb.XGBRegressor(n_estimators=800, learning_rate=0.05, max_depth=6, subsample=0.8,
                         colsample_bytree=0.8, early_stopping_rounds=30, eval_metric="mae", n_jobs=-1)
    m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
    return m.predict(X.loc[test_t0]), time.time() - t

# Bước 8: RNN / LSTM / GRU
def static_feats(ctx, t0, horizon):
    assert horizon <= 24, "static_feats chỉ hợp lệ khi horizon <= 24"
    T = pd.DatetimeIndex(t0) + H(horizon)
    hol = ctx.h["is_holiday"].reindex(T).fillna(0).values
    def z(v):
        v = np.asarray(v, dtype="float64"); m = np.isnan(v)
        out = (v - ctx.mu) / ctx.sd; out[m] = 0.0
        return out, m.astype("float64")
    dz, dm = z(ctx.sr.reindex(T - H(24)).values)
    wz, wm = z(ctx.sr.reindex(T - H(168)).values)
    return np.column_stack([np.sin(2*np.pi*T.hour/24), np.cos(2*np.pi*T.hour/24),
                            np.sin(2*np.pi*T.dayofweek/7), np.cos(2*np.pi*T.dayofweek/7),
                            hol, dz, dm, wz, wm]).astype("float32")

def build_model(kind, window, n_features, n_static=0):
    import tensorflow as tf
    from tensorflow.keras import layers
    Rnn = {"SimpleRNN": layers.SimpleRNN, "LSTM": layers.LSTM, "GRU": layers.GRU}[kind]
    if n_static == 0:
        m = tf.keras.Sequential([
            layers.Input(shape=(window, n_features)),
            Rnn(64, return_sequences=True), layers.Dropout(0.2),
            Rnn(32), layers.Dropout(0.2),
            layers.Dense(1),                      # KHÔNG activation (hồi quy)
        ])
    else:                                         # 2 đầu vào: chuỗi quá khứ + đặc trưng giờ đích
        seq = layers.Input(shape=(window, n_features)); st = layers.Input(shape=(n_static,))
        x = Rnn(64, return_sequences=True)(seq); x = layers.Dropout(0.2)(x)
        x = Rnn(32)(x); x = layers.Dropout(0.2)(x)
        s = layers.Dense(16, activation="relu")(st)
        z = layers.Dense(32, activation="relu")(layers.Concatenate()([x, s]))
        m = tf.keras.Model([seq, st], layers.Dense(1)(z))
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return m

def train_eval(ctx, kind, window=24, horizon=1, common=None, epochs=40, seed=42, extra=False):
    import tensorflow as tf
    ds = ctx.ds(window, horizon)
    tf.keras.backend.clear_session(); np.random.seed(seed); tf.random.set_seed(seed)
    n_static = 9 if extra else 0
    m = build_model(kind, window, len(ctx.feats), n_static)
    inp = lambda p: [ds[p]["X"], static_feats(ctx, ds[p]["t0"], horizon)] if extra else ds[p]["X"]
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    t = time.time()
    hist = m.fit(inp("train"), ds["train"]["y"], validation_data=(inp("val"), ds["val"]["y"]),
                 epochs=epochs, batch_size=128, shuffle=True, callbacks=[es], verbose=0)
    dt = time.time() - t
    te = ds["test"]
    pred = m.predict(inp("test"), verbose=0).ravel() * ctx.sd + ctx.mu
    y = te["y"] * ctx.sd + ctx.mu
    T = te["t0"] + H(horizon)
    return dict(kind=kind, window=window, horizon=horizon, model=m, pred=pred, y=y, T=T,
                params=m.count_params(), time=dt, epochs=len(hist.history["loss"]),
                val_mae=min(hist.history["val_mae"]) * ctx.sd,
                test_mae=mae_on(T, y, pred, common))

DOW = ["T2", "T3", "T4", "T5", "T6", "T7", "CN"]
def main(path, window=24, seeds=1, extra=False):
    os.makedirs(REPORTS, exist_ok=True); os.makedirs(MODELS, exist_ok=True)
    print("Kết quả sẽ được lưu vào:", REPORTS)
    ctx = Ctx(path)
    print("Đặc trưng:", len(ctx.feats), "| giờ dữ liệu:", len(ctx.d))

    # --- bước 3
    eda(ctx)

    # --- bước 6: baseline naive
    WMAX = max(48, window)
    COMMON = ctx.common(WMAX, 1)                     # giao của mọi cửa sổ <= WMAX
    te = ctx.ds(window, 1)["test"]
    T1 = te["t0"] + H(1); y1 = te["y"] * ctx.sd + ctx.mu
    b24, b168 = ctx.baseline(T1)
    mae_naive, mae_seas = mae_on(T1, y1, b24, COMMON), mae_on(T1, y1, b168, COMMON)

    # --- bước 7: XGBoost + lag
    xp, xgb_time = run_xgb(ctx, 1, te["t0"])
    mae_xgb = mae_on(T1, y1, xp, COMMON)

    # --- bước 8: SimpleRNN vs LSTM vs GRU (chạy `seeds` lần với seed khác nhau để đo độ dao động)
    res, runs = {}, {}
    for k in ["SimpleRNN", "LSTM", "GRU"]:
        runs[k] = [train_eval(ctx, k, window, 1, COMMON, seed=42 + i) for i in range(seeds)]
        res[k] = runs[k][0]                          # lần đầu (seed 42) dùng để vẽ biểu đồ
    rows = []
    for k, rr in runs.items():
        maes = np.array([r["test_mae"] for r in rr])
        row = {"Mô hình": k, "Tham số": rr[0]["params"], "Thời gian train (s)": round(np.mean([r["time"] for r in rr]), 1),
               "Epoch": rr[0]["epochs"], "Val MAE": round(np.mean([r["val_mae"] for r in rr]), 1),
               "Test MAE": round(maes.mean(), 1)}
        if seeds > 1: row["Std"] = round(maes.std(ddof=1), 1)
        rows.append(row)
    if extra:                                        # LSTM + đặc trưng mùa vụ của giờ đích (kiểm chứng giả thuyết)
        res["LSTM+mùa vụ"] = train_eval(ctx, "LSTM", window, 1, COMMON, extra=True)
        r = res["LSTM+mùa vụ"]
        rows.append({"Mô hình": "LSTM + đặc trưng mùa vụ", "Tham số": r["params"], "Thời gian train (s)": round(r["time"], 1),
                     "Epoch": r["epochs"], "Val MAE": round(r["val_mae"], 1), "Test MAE": round(r["test_mae"], 1)})
    for name, mae, t in [("Naive (hôm qua)", mae_naive, 0), ("Seasonal naive (tuần trước)", mae_seas, 0),
                         ("XGBoost + lag", mae_xgb, xgb_time)]:
        rows.append({"Mô hình": name, "Tham số": "-", "Thời gian train (s)": round(t, 1),
                     "Epoch": "-", "Val MAE": "-", "Test MAE": round(mae, 1)})
    tab = pd.DataFrame(rows).fillna("-")
    tab.to_csv(os.path.join(REPORTS, "bang_so_sanh_mo_hinh.csv"), index=False, encoding="utf-8-sig")
    print(f"\n== Bước 6-8: so sánh mô hình (Test MAE, xe/giờ, dự báo 1h, {seeds} seed) =="); print(tab.to_string(index=False))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(tab["Mô hình"], tab["Test MAE"].astype(float)); ax[0].set_title("Test MAE"); ax[0].tick_params(axis="x", rotation=60)
    for k, r in res.items(): ax[1].scatter(r["time"], r["test_mae"], s=80, label=k)
    ax[1].set_xlabel("Thời gian train (s)"); ax[1].set_ylabel("Test MAE"); ax[1].legend()
    plt.tight_layout(); plt.savefig(os.path.join(REPORTS, "rnn_lstm_gru.png"), dpi=130); plt.close()

    # --- bước 9: khảo sát độ dài cửa sổ (LSTM)
    win = {w: (res["LSTM"] if w == window else train_eval(ctx, "LSTM", w, 1, COMMON)) for w in [6, 12, 24, 48]}
    wt = pd.DataFrame([{"Cửa sổ (giờ)": w, "Test MAE": round(r["test_mae"], 1), "Thời gian train (s)": round(r["time"], 1)}
                       for w, r in win.items()])
    wt.to_csv(os.path.join(REPORTS, "bang_do_dai_cua_so.csv"), index=False, encoding="utf-8-sig")
    print("\n== Bước 9: độ dài cửa sổ =="); print(wt.to_string(index=False))
    plt.figure(figsize=(6, 4)); plt.plot(wt["Cửa sổ (giờ)"], wt["Test MAE"], marker="o")
    plt.axhline(mae_seas, ls="--", c="gray", label="Seasonal naive"); plt.axhline(mae_xgb, ls=":", c="green", label="XGBoost")
    plt.xlabel("Độ dài cửa sổ (giờ)"); plt.ylabel("Test MAE"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(REPORTS, "do_dai_cua_so.png"), dpi=130); plt.close()

    # --- bước 10: dự báo vs thực tế 7 ngày cuối test
    L = res["LSTM"]
    s_act = pd.Series(L["y"], index=L["T"])
    sel = s_act.index >= s_act.index.max() - pd.Timedelta(days=7)
    plt.figure(figsize=(15, 4.5))
    plt.plot(s_act[sel], label="Thực tế", c="black", lw=2)
    plt.plot(pd.Series(b168, index=L["T"])[sel], label="Seasonal naive", c="gray", alpha=.7)
    plt.plot(pd.Series(xp, index=L["T"])[sel], label="XGBoost", c="green", alpha=.8)
    plt.plot(pd.Series(L["pred"], index=L["T"])[sel], label="LSTM", c="crimson", alpha=.9)
    plt.title("Dự báo 1 giờ tới — 7 ngày cuối của tập test"); plt.ylabel("xe/giờ"); plt.legend(ncol=4)
    plt.tight_layout(); plt.savefig(os.path.join(REPORTS, "du_bao_vs_thuc_te.png"), dpi=130); plt.close()
    L["model"].save(os.path.join(MODELS, "lstm_traffic.keras"))

    # --- bước 11: dự báo nhiều bước. MỌI horizon chấm trên tập chung tính theo cửa sổ WMAX (giống bước 8);
    #     horizon 1 dùng lại kết quả bước 8 nên số liệu khớp bảng ở trên.
    rows = []
    for hz in [1, 3, 6]:
        com = ctx.common(WMAX, hz)
        r = res["LSTM"] if hz == 1 else train_eval(ctx, "LSTM", window, hz, com)
        if hz == 1: xph = xp
        else: xph, _ = run_xgb(ctx, hz, ctx.ds(window, hz)["test"]["t0"])
        _, bw = ctx.baseline(r["T"])
        row = {"Horizon (giờ)": hz, "LSTM MAE": round(mae_on(r["T"], r["y"], r["pred"], com), 1),
               "XGBoost MAE": round(mae_on(r["T"], r["y"], xph, com), 1),
               "Seasonal naive MAE": round(mae_on(r["T"], r["y"], bw, com), 1)}
        if extra:
            rx = res["LSTM+mùa vụ"] if hz == 1 else train_eval(ctx, "LSTM", window, hz, com, extra=True)
            row["LSTM+mùa vụ MAE"] = round(mae_on(rx["T"], rx["y"], rx["pred"], com), 1)
        rows.append(row)
    mh = pd.DataFrame(rows)
    mh.to_csv(os.path.join(REPORTS, "bang_nhieu_buoc.csv"), index=False, encoding="utf-8-sig")
    print("\n== Bước 11: dự báo nhiều bước =="); print(mh.to_string(index=False))

    # --- bước 12: phân tích lỗi
    err = pd.DataFrame({"abs_err": np.abs(L["y"] - L["pred"])}, index=L["T"])
    err = err[err.index.isin(COMMON)]
    by_hour = err.groupby(err.index.hour)["abs_err"].mean()
    by_dow = err.groupby(err.index.dayofweek)["abs_err"].mean()
    by_day = err.groupby(err.index.date)["abs_err"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    by_hour.plot(kind="bar", ax=ax[0]); ax[0].set_title("MAE LSTM theo giờ")
    by_dow.plot(kind="bar", ax=ax[1]); ax[1].set_title("MAE LSTM theo thứ (0=T2)")
    plt.tight_layout(); plt.savefig(os.path.join(REPORTS, "phan_tich_loi.png"), dpi=130); plt.close()
    print("\n== Bước 12: phân tích lỗi ==")
    print(f"Giờ sai nhiều nhất: {by_hour.idxmax()}h (MAE {by_hour.max():.0f}) | thứ sai nhiều nhất: {DOW[by_dow.idxmax()]}")
    by_day.head(10).rename("MAE").to_csv(os.path.join(REPORTS, "top10_ngay_sai_nhieu.csv"), encoding="utf-8-sig")
    print("10 ngày sai nhiều nhất (ngày, thứ, MAE):")
    for day, v in by_day.head(10).items():
        print(f"  {day}  {DOW[pd.Timestamp(day).dayofweek]}  {v:.0f}")

    # --- kết luận trung thực theo tiêu chí đề
    lm = float(np.mean([r["test_mae"] for r in runs["LSTM"]]))
    lt = float(np.mean([r["time"] for r in runs["LSTM"]])); speed = lt / max(xgb_time, 1e-9)
    print("\n== Kết luận ==")
    print(f"LSTM {lm:.0f} | Seasonal naive {mae_seas:.0f} | XGBoost {mae_xgb:.0f} | Naive {mae_naive:.0f} (MAE, xe/giờ)")
    print("LSTM thắng seasonal naive -> deep learning có giá trị." if lm < mae_seas else
          "LSTM KHÔNG thắng seasonal naive -> kết luận trung thực: bài này không cần deep learning.")
    print(f"XGBoost train nhanh hơn LSTM ~{speed:.0f} lần; chênh MAE (LSTM - XGB) = {lm - mae_xgb:+.0f} xe/giờ.")
    if seeds > 1:
        sd_l = np.std([r["test_mae"] for r in runs["LSTM"]], ddof=1)
        print(f"Độ dao động LSTM giữa {seeds} seed: std = {sd_l:.1f} xe/giờ -> chênh lệch nhỏ hơn mức này không có ý nghĩa.")
    if abs(lm - mae_xgb) < 0.05 * mae_xgb and speed >= 5:
        print("XGBoost tương đương mà nhanh hơn nhiều -> hãy ghi kết luận này vào README.")
    print("\nĐã lưu vào", REPORTS, ":", ", ".join(sorted(os.listdir(REPORTS))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="link hoặc đường dẫn file (bỏ trống thì dùng DATA_PATH ở đầu file)")
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--seeds", type=int, default=1, help="số lần huấn luyện (seed khác nhau) cho mỗi mô hình ở bước 8; >1 để đo độ dao động")
    ap.add_argument("--extra", action="store_true", help="thêm thí nghiệm LSTM + đặc trưng mùa vụ của giờ đích")
    a = ap.parse_args()
    path = a.data or DATA_PATH
    main(path, a.window, a.seeds, a.extra)