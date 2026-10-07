import argparse
import itertools
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from features import feature_columns, make_features

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent          
REP, MOD = ROOT / "reports", ROOT / "models"
TRAIN_END, VAL_END = "2008-12-31", "2009-12-31"         
GIA_DIEN, N_HO = 2000, 1_000_000                        
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .3})

# tiện ích
def mape(y, p): return float(np.mean(np.abs((y - p) / y)) * 100)
def score(y, p): return {"MAE": float(MAE(y, p)), "MAPE": mape(y, p)}
def bias(y, p): return float(np.mean(p - y))

def load_daily(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, sep=";", na_values="?", low_memory=False)
    raw["dt"] = pd.to_datetime(raw["Date"] + " " + raw["Time"], format="%d/%m/%Y %H:%M:%S")
    s = raw.set_index("dt")["Global_active_power"].astype(float)
    return s.resample("D").mean().to_frame("phu_tai").asfreq("D")

def fill_short_gaps(x: pd.Series, max_len: int = 3) -> pd.Series:
    isna = x.isna()
    run_id = (isna != isna.shift()).cumsum()
    run_len = isna.groupby(run_id).transform("sum")
    filled = x.interpolate(limit_area="inside")
    return x.where(~(isna & (run_len <= max_len)), filled)

def split_by_time(df: pd.DataFrame):
    idx = df.index
    return (df[idx <= TRAIN_END],
            df[(idx > TRAIN_END) & (idx <= VAL_END)],
            df[idx > VAL_END])

def find_data(arg):
    if arg:
        return Path(arg)
    for cand in [ROOT / "data" / "household_power_consumption.txt",
                 *(ROOT / "data").glob("household_power_consumption*")]:
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"Không thấy file dữ liệu trong {ROOT / 'data'}. Dùng --data <đường_dẫn>.")

def main(data_path=None):
    REP.mkdir(exist_ok=True); MOD.mkdir(exist_ok=True)
    R = {}

    # ---- Bước 1-2: đọc + vá lỗ hổng ----
    path = find_data(data_path)
    print(f"[B1] Đọc {path}")
    daily = load_daily(path)
    n, n_thieu = len(daily), int(daily["phu_tai"].isna().sum())
    daily["bi_thieu"] = daily["phu_tai"].isna().astype(int)
    daily["phu_tai"] = fill_short_gaps(daily["phu_tai"], 3)
    R.update(so_ngay=n, ngay_thieu_goc=n_thieu, pct_thieu_goc=n_thieu / n * 100,
             ngay_con_thieu_sau_va=int(daily["phu_tai"].isna().sum()))
    print(f"[B2] {n} ngày | thiếu gốc {n_thieu} ({n_thieu / n:.2%}) | còn thiếu sau vá lỗ ngắn {R['ngay_con_thieu_sau_va']}")

    # ---- Bước 3: EDA ----
    fig, ax = plt.subplots(1, 3, figsize=(17, 4))
    daily["phu_tai"].groupby(daily.index.month).mean().plot(kind="bar", ax=ax[0], color="#4C78A8")
    ax[0].set(title="Phụ tải TB theo tháng (mùa vụ)", xlabel="Tháng", ylabel="kW")
    byd = daily["phu_tai"].groupby(daily.index.dayofweek).mean(); byd.index = ["T2", "T3", "T4", "T5", "T6", "T7", "CN"]
    byd.plot(kind="bar", ax=ax[1], color=["#4C78A8"] * 5 + ["#E45756"] * 2)
    ax[1].set(title="Theo thứ trong tuần (đỏ = cuối tuần)", xlabel="", ylabel="kW")
    daily["phu_tai"].plot(ax=ax[2], lw=.7, color="#333")
    daily["phu_tai"].rolling(30, min_periods=15).mean().plot(ax=ax[2], color="#E45756", label="TB 30 ngày")
    ax[2].set(title="Toàn chuỗi", xlabel="", ylabel="kW"); ax[2].legend()
    plt.tight_layout(); plt.savefig(REP / "eda_mua_vu.png"); plt.close()
    wk = daily["phu_tai"][daily.index.dayofweek >= 5].mean(); wd = daily["phu_tai"][daily.index.dayofweek < 5].mean()
    R.update(tb_cuoi_tuan=float(wk), tb_ngay_thuong=float(wd))

    # ---- Bước 4-6: đặc trưng, bỏ NaN, chia theo thời gian ----
    feat = make_features(daily)
    FEATS = feature_columns(feat)
    data = feat.dropna(subset=FEATS + ["phu_tai"])
    R.update(so_dac_trung=len(FEATS), so_dong_sau_dropna=len(data))
    train, val, test = split_by_time(data)
    trval = pd.concat([train, val])
    Xtr, ytr, Xva, yva = train[FEATS], train["phu_tai"], val[FEATS], val["phu_tai"]
    Xtv, ytv, Xte, yte = trval[FEATS], trval["phu_tai"], test[FEATS], test["phu_tai"]
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError("Một tập rỗng — kiểm tra TRAIN_END/VAL_END và phạm vi ngày của dữ liệu.")
    R["kich_thuoc"] = dict(train=len(train), val=len(val), test=len(test))
    print(f"[B6] train {len(train)} | val {len(val)} | test {len(test)} | {len(FEATS)} đặc trưng")

    # ---- Bước 7: 2 baseline ----
    res = {"Naive (hôm qua)": score(yte, Xte["lag_1"]),
           "Seasonal naive (tuần trước)": score(yte, Xte["lag_7"])}

    # ---- Bước 8: 3 thuật toán, chọn siêu tham số trên VAL ----
    builders = {
        "Ridge": (lambda alpha: make_pipeline(StandardScaler(), Ridge(alpha=alpha)),
                  {"alpha": [0.1, 1, 10, 100]}),
        "RandomForest": (lambda min_samples_leaf, max_features: RandomForestRegressor(
            300, min_samples_leaf=min_samples_leaf, max_features=max_features, n_jobs=-1, random_state=0),
            {"min_samples_leaf": [3, 5], "max_features": [0.5, 1.0]}),
        "XGBoost": (lambda max_depth, n_estimators: XGBRegressor(
            n_estimators=n_estimators, learning_rate=0.03, max_depth=max_depth, subsample=.8,
            colsample_bytree=.8, random_state=0, n_jobs=-1),
            {"max_depth": [2, 3, 4], "n_estimators": [300, 600]}),
    }

    def grid_search(build, grid):
        best, bp = np.inf, None
        for vals in itertools.product(*grid.values()):
            p = dict(zip(grid, vals)); e = MAE(yva, build(**p).fit(Xtr, ytr).predict(Xva))
            if e < best: best, bp = e, p
        return bp, best

    best_params, models, preds = {}, {}, {}
    for name, (b, g) in builders.items():
        bp, ev = grid_search(b, g); best_params[name] = bp
        models[name] = b(**bp).fit(Xtv, ytv)               # refit train+val; test chỉ chạm một lần
        preds[name] = pd.Series(models[name].predict(Xte), index=yte.index)
        res[name] = score(yte, preds[name])
        print(f"[B8] {name}: {bp} | MAE val {ev:.4f}")
    R.update(tham_so_tot_nhat=best_params, bang_so_sanh=res)
    tab = pd.DataFrame(res).T.round(4)
    print("\n", tab, "\n")
    ml = ["Ridge", "RandomForest", "XGBoost"]
    best_name = min(ml, key=lambda k: res[k]["MAE"]); R["model_tot_nhat"] = best_name
    sn = res["Seasonal naive (tuần trước)"]["MAE"]
    R["thang_seasonal_naive"] = {k: bool(res[k]["MAE"] < sn) for k in ml}
    print("Thắng seasonal naive:", R["thang_seasonal_naive"])
    models["XGBoost"].save_model(str(MOD / "xgb_power.json"))
    tab.to_csv(REP / "bang_so_sanh.csv")

    # ---- Bước 9: thí nghiệm rò rỉ ----
    xp = best_params["XGBoost"]
    mk = lambda: XGBRegressor(n_estimators=xp["n_estimators"], learning_rate=.03, max_depth=xp["max_depth"],
                              subsample=.8, colsample_bytree=.8, random_state=0, n_jobs=-1)
    leaky = make_features(daily, leak=True).dropna(subset=FEATS + ["phu_tai"])
    l_tr, l_va, l_te = split_by_time(leaky)
    l_tv = pd.concat([l_tr, l_va])
    lk = score(l_te["phu_tai"], mk().fit(l_tv[FEATS], l_tv["phu_tai"]).predict(l_te[FEATS]))
    giam = (1 - lk["MAE"] / res["XGBoost"]["MAE"]) * 100
    R["ro_ri"] = dict(dung=res["XGBoost"], ro_ri=lk, giam_MAE_pct=giam)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, key, ttl in zip(ax, ["MAE", "MAPE"], ["MAE (kW)", "MAPE (%)"]):
        v = [res["XGBoost"][key], lk[key]]
        bars = a.bar(["Đúng\n(shift(1))", "RÒ RỈ\n(thiếu shift)"], v, color=["#54A24B", "#E45756"]); a.set_title(ttl)
        for b_, x in zip(bars, v): a.text(b_.get_x() + b_.get_width() / 2, x, f"{x:.3f}", ha="center", va="bottom")
    fig.suptitle(f"Thí nghiệm rò rỉ: sai số giảm giả tạo {giam:.0f}% (cùng XGBoost, cùng dữ liệu)")
    plt.tight_layout(); plt.savefig(REP / "thi_nghiem_ro_ri.png"); plt.close()
    print(f"[B9] MAE đúng {res['XGBoost']['MAE']:.4f} vs rò rỉ {lk['MAE']:.4f} (giảm giả tạo {giam:.1f}%)")

    # ---- Bước 10: feature importance ----
    imp = pd.Series(models["XGBoost"].feature_importances_, index=FEATS).sort_values()
    plt.figure(figsize=(7, 6)); imp.plot(kind="barh", color="#4C78A8"); plt.title("Feature importance — XGBoost")
    plt.tight_layout(); plt.savefig(REP / "feature_importance.png"); plt.close()
    R["top5_importance"] = {k: float(v) for k, v in imp.sort_values(ascending=False).head(5).items()}
    print("[B10] top5:", {k: round(v, 3) for k, v in R["top5_importance"].items()})

    # ---- Bước 11: thí nghiệm ngoại suy (tuyệt đối vs chênh lệch) ----
    tags = [("tuyệt đối", None), ("chênh lệch lag_1", "lag_1"), ("chênh lệch lag_7", "lag_7")]
    ext = {}
    for mname in ["XGBoost", "RandomForest"]:
        for tag, base in tags:
            m = builders[mname][0](**best_params[mname])
            if base is None:
                m.fit(Xtv, ytv); p = pd.Series(m.predict(Xte), index=yte.index)
            else:
                m.fit(Xtv, ytv - Xtv[base]); p = pd.Series(m.predict(Xte), index=yte.index) + Xte[base]
            ext[(mname, tag)] = p
    ext_tab = {f"{k[0]} | {k[1]}": {**score(yte, p), "bias": bias(yte, p)} for k, p in ext.items()}
    R["ngoai_suy"] = ext_tab
    R["xu_huong"] = dict(tb_train=float(ytr.mean()), tb_val=float(yva.mean()), tb_test=float(yte.mean()))
    print("[B11]\n", pd.DataFrame(ext_tab).T.round(4).to_string())
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.2))
    w = yte.rolling(14, min_periods=7).mean(); ax[0].plot(w.index, w, "k", lw=2, label="Thực tế (TB 14 ngày)")
    for (tag, _), c in zip(tags, ["#E45756", "#54A24B", "#4C78A8"]):
        ax[0].plot(w.index, ext[("XGBoost", tag)].rolling(14, min_periods=7).mean(), c=c, label=f"XGB {tag}")
    ax[0].set(title="Năm test: dự báo tuyệt đối vs chênh lệch", ylabel="kW"); ax[0].legend(fontsize=8)
    x = np.arange(3)
    for i, mn in enumerate(["XGBoost", "RandomForest"]):
        ax[1].bar(x + (i - .5) * .38, [ext_tab[f"{mn} | {t}"]["MAE"] for t, _ in tags], .38, label=mn)
    ax[1].set_xticks(x); ax[1].set_xticklabels([t for t, _ in tags]); ax[1].set(title="MAE năm test", ylabel="kW"); ax[1].legend()
    plt.tight_layout(); plt.savefig(REP / "ngoai_suy.png"); plt.close()

    # ---- Bước 12: 60 ngày cuối ----
    plt.figure(figsize=(12, 4))
    plt.plot(yte.index[-60:], yte.iloc[-60:], "k-o", ms=3, label="Thực tế")
    plt.plot(yte.index[-60:], preds[best_name].iloc[-60:], "r-o", ms=3, label=f"Dự báo ({best_name})")
    plt.plot(yte.index[-60:], Xte["lag_7"].iloc[-60:], "--", c="gray", alpha=.7, label="Seasonal naive")
    plt.title("Dự báo vs thực tế — 60 ngày cuối"); plt.ylabel("kW"); plt.legend()
    plt.tight_layout(); plt.savefig(REP / "du_bao_60_ngay.png"); plt.close()

    # ---- Bước 13: TimeSeriesSplit (KHÔNG dùng KFold thường) ----
    folds = []
    for tr, va in TimeSeriesSplit(n_splits=5).split(Xtv):
        m = mk().fit(Xtv.iloc[tr], ytv.iloc[tr]); folds.append(float(MAE(ytv.iloc[va], m.predict(Xtv.iloc[va]))))
    R["timeseries_split"] = dict(folds=folds, mean=float(np.mean(folds)), std=float(np.std(folds)))
    print(f"[B13] TimeSeriesSplit MAE {np.mean(folds):.4f} ± {np.std(folds):.4f}  {np.round(folds, 4)}")

    # ---- Bước 14: quy đổi ra tiền ----
    kwh = res[best_name]["MAE"] * 24
    R["tien"] = dict(gia_dien=GIA_DIEN, mae_kw=res[best_name]["MAE"], sai_so_kwh_ngay=kwh,
                     vnd_ngay_1_ho=kwh * GIA_DIEN, vnd_nam_1_ho=kwh * GIA_DIEN * 365, n_ho=N_HO,
                     ty_vnd_nam_he_thong=kwh * GIA_DIEN * N_HO * 365 / 1e9)
    print(f"[B14] {best_name}: {kwh:.2f} kWh/ngày/hộ ≈ {kwh * GIA_DIEN:,.0f} VND/ngày/hộ; "
          f"{N_HO:,} hộ ≈ {R['tien']['ty_vnd_nam_he_thong']:.1f} tỷ VND/năm (giả định)")

    with open(REP / "ketqua.json", "w", encoding="utf-8") as f:
        json.dump(R, f, ensure_ascii=False, indent=2)
    print(f"\nXong. Kết quả ở {REP} và {MOD}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="household_power_consumption.txt")
    main(ap.parse_args().data)