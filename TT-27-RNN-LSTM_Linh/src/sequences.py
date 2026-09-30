import numpy as np
import pandas as pd

def tao_chuoi(X, y, do_dai=24, buoc_du_bao=1):
    Xs, ys = [], []
    for i in range(len(X) - do_dai - buoc_du_bao + 1):
        Xs.append(X[i:i+do_dai])
        ys.append(y[i+do_dai+buoc_du_bao-1])
    return np.array(Xs), np.array(ys)

def make_dataset(d, d_s, feats, window=24, horizon=1):
    Fv = d_s[feats].values.astype("float32")
    yv = d_s["traffic_volume"].values.astype("float32")
    run_id = (d.index.to_series().diff() != pd.Timedelta("1h")).cumsum().values   # đoạn liên tục
    key = np.char.add(run_id.astype(str), np.char.add("_", d["split"].values.astype(str)))
    bk = {s: {"X": [], "y": [], "t0": []} for s in ["train", "val", "test"]}
    for k in pd.unique(key):
        idx = np.where(key == k)[0]
        if len(idx) < window + horizon:
            continue
        Xs, ys = tao_chuoi(Fv[idx], yv[idx], window, horizon)
        t0 = d.index[idx][window-1: window-1+len(ys)]
        s = k.split("_")[1]
        bk[s]["X"].append(Xs); bk[s]["y"].append(ys); bk[s]["t0"].append(t0.values)
    return {s: {"X": np.concatenate(b["X"]), "y": np.concatenate(b["y"]),
                "t0": pd.DatetimeIndex(np.concatenate(b["t0"]))} for s, b in bk.items()}

print("OK")