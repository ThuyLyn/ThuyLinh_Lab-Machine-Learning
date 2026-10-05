import itertools
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")


def mape(thuc, dubao) -> float:
    thuc, dubao = np.asarray(thuc), np.asarray(dubao)
    return float(np.mean(np.abs((thuc - dubao) / thuc)) * 100)


def chia_train_test(y: pd.Series, n_test: int = 24):
    return y.iloc[:-n_test], y.iloc[-n_test:]


def baselines(train: pd.Series, test: pd.Series) -> dict:
    naive = pd.Series(train.iloc[-1], index=test.index)
    reps = int(np.ceil(len(test) / 12))
    snaive = pd.Series(np.tile(train.iloc[-12:].values, reps)[: len(test)], index=test.index)
    return {"Naive": naive, "Seasonal naive": snaive}


def fit_sarima(y_log, order, seasonal_order):
    # Giữ mặc định enforce_stationarity/invertibility=True: tắt chúng làm AIC bị méo
    return SARIMAX(y_log, order=order, seasonal_order=seasonal_order).fit(disp=False)


def quet_luoi(y_log, d=1, D=1, s=12, rng_pq=(0, 1, 2), rng_PQ=(0, 1)) -> pd.DataFrame:
    ket_qua = []
    for p, q, P, Q in itertools.product(rng_pq, rng_pq, rng_PQ, rng_PQ):
        try:
            f = fit_sarima(y_log, (p, d, q), (P, D, Q, s))
            ket_qua.append([(p, d, q), (P, D, Q, s), f.aic])
        except Exception:
            continue
    return (pd.DataFrame(ket_qua, columns=["order", "seasonal", "AIC"])
            .sort_values("AIC").reset_index(drop=True))


def du_bao(fit, steps=24, alpha=0.05):
    kq = fit.get_forecast(steps=steps)
    mean = np.exp(kq.predicted_mean)
    ci = np.exp(kq.conf_int(alpha=alpha))
    ci.columns = ["thap", "cao"]
    return mean, ci


def ti_le_phu(test, ci) -> float:
    return float(((test >= ci["thap"]) & (test <= ci["cao"])).mean() * 100)