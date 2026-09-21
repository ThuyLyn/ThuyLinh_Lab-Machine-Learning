import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


def make_preprocess_pipeline() -> Pipeline:
    return Pipeline([
        ("log", FunctionTransformer(np.log1p, validate=True)),
        ("scale", StandardScaler()),
    ])


def scan_k(X_scaled: np.ndarray, k_range=range(2, 13), random_state: int = 42) -> pd.DataFrame:
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        rows.append({"K": k, "inertia": km.inertia_, "silhouette": sil})
    return pd.DataFrame(rows)


def n_init_experiment(X_scaled: np.ndarray, k: int, seeds=(0, 1, 2, 3, 4)) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        km1 = KMeans(n_clusters=k, n_init=1, random_state=seed)
        labels1 = km1.fit_predict(X_scaled)
        rows.append({
            "seed": seed, "n_init": 1,
            "inertia": km1.inertia_,
            "silhouette": silhouette_score(X_scaled, labels1),
        })
        km10 = KMeans(n_clusters=k, n_init=10, random_state=seed)
        labels10 = km10.fit_predict(X_scaled)
        rows.append({
            "seed": seed, "n_init": 10,
            "inertia": km10.inertia_,
            "silhouette": silhouette_score(X_scaled, labels10),
        })
    return pd.DataFrame(rows)

def fit_final_kmeans(X_scaled: np.ndarray, k: int, random_state: int = 42):
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = km.fit_predict(X_scaled)
    return km, labels

def pca_2d(X_scaled: np.ndarray, random_state: int = 42):
    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(X_scaled)
    return coords, pca

def run_dbscan(X_scaled: np.ndarray, eps: float = 0.8, min_samples: int = 10):
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X_scaled)
    n_noise = int((labels == -1).sum())
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return labels, n_clusters, n_noise

def describe_clusters(sp: pd.DataFrame, label_col: str = "cum") -> pd.DataFrame:
    tong_doanh_thu_all = sp["tong_doanh_thu"].sum()
    g = sp.groupby(label_col).agg(
        so_ma_hang=("StockCode", "count"),
        doanh_thu=("tong_doanh_thu", "sum"),
        gia_tb=("gia_trung_binh", "mean"),
        don_hang_tb=("so_don_hang", "mean"),
        ty_le_mua_lai_tb=("ty_le_mua_lai", "mean"),
    ).reset_index()
    g["ty_le_doanh_thu_%"] = (g["doanh_thu"] / tong_doanh_thu_all * 100).round(1)
    return g.sort_values("doanh_thu", ascending=False).reset_index(drop=True)