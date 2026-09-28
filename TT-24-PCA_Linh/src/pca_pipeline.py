from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

NGUONG_PHUONG_SAI = [0.80, 0.90, 0.95, 0.99]
# Lưới K mịn hơn, có các mốc 26/63/102/179 (ngưỡng phương sai) và 200
K_LIST = [2, 5, 10, 15, 20, 26, 30, 40, 50, 63, 80, 102, 125, 150, 179, 200, 561]
K_SO_SANH_SELECTKBEST = [10, 26, 63, 102, 179]
N_SPLITS = 5
N_CLASSES = 6
RANDOM_STATE = 42

# ----------------------------------------------------------------------
# Nạp dữ liệu
# ----------------------------------------------------------------------
def _read_features(data_dir: Path) -> list[str]:
    feats = pd.read_csv(data_dir / "features.txt", sep=r"\s+", header=None, names=["idx", "name"])
    seen, unique_names = {}, []
    for n in feats["name"].tolist():
        if n in seen:
            seen[n] += 1
            unique_names.append(f"{n}__{seen[n]}")
        else:
            seen[n] = 0
            unique_names.append(n)
    return unique_names


def _read_activity_labels(data_dir: Path) -> dict[int, str]:
    labels = pd.read_csv(data_dir / "activity_labels.txt", sep=r"\s+", header=None,
                         names=["id", "activity"])
    return dict(zip(labels["id"], labels["activity"]))


def load_har_data(data_dir: str | Path) -> dict:
    data_dir = Path(data_dir)
    feature_names = _read_features(data_dir)
    activity_map = _read_activity_labels(data_dir)

    def _load_split(split: str):
        X = pd.read_csv(data_dir / split / f"X_{split}.txt", sep=r"\s+", header=None)
        X.columns = feature_names
        y = pd.read_csv(data_dir / split / f"y_{split}.txt", header=None).iloc[:, 0]
        subj = pd.read_csv(data_dir / split / f"subject_{split}.txt", header=None).iloc[:, 0]
        return X, y, subj, y.map(activity_map)

    X_train, y_train, subj_train, y_train_name = _load_split("train")
    X_test, y_test, subj_test, y_test_name = _load_split("test")

    overlap = set(subj_train.unique()) & set(subj_test.unique())
    assert not overlap, f"RÒ RỈ DỮ LIỆU: subject {overlap} xuất hiện ở cả train và test!"

    return {
        "X_train": X_train, "y_train": y_train, "y_train_name": y_train_name,
        "subject_train": subj_train,
        "X_test": X_test, "y_test": y_test, "y_test_name": y_test_name,
        "subject_test": subj_test,
        "feature_names": feature_names, "activity_map": activity_map,
    }


# ----------------------------------------------------------------------
# Bước 2-5: chuẩn hoá, PCA đầy đủ, scree, phương sai tích luỹ, bảng ngưỡng
# ----------------------------------------------------------------------
def fit_scaler_and_full_pca(X_train: pd.DataFrame):
    scaler = StandardScaler().fit(X_train)  # fit CHỈ trên train
    X_train_s = scaler.transform(X_train)
    pca_full = PCA(random_state=RANDOM_STATE).fit(X_train_s)
    return scaler, pca_full, X_train_s


def bang_nguong_phuong_sai(pca_full: PCA, n_features_total: int) -> pd.DataFrame:
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    rows = []
    for nguong in NGUONG_PHUONG_SAI:
        k = int(np.argmax(cum_var >= nguong) + 1)
        rows.append({"nguong_phuong_sai": f"{nguong:.0%}", "so_chieu_can": k,
                     "phan_tram_giam_chieu": f"{(1 - k / n_features_total):.1%}"})
    return pd.DataFrame(rows)


def ve_scree_plot(pca_full: PCA, out_path: Path, top_n: int = 50):
    plt.figure(figsize=(8, 5))
    v = pca_full.explained_variance_ratio_[:top_n]
    plt.plot(range(1, len(v) + 1), v, marker="o", markersize=3)
    plt.xlabel("Thành phần chính (PC)")
    plt.ylabel("Tỉ lệ phương sai giải thích")
    plt.title(f"Scree Plot ({top_n} thành phần đầu)")
    plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


def ve_phuong_sai_tich_luy(pca_full: PCA, out_path: Path):
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cum_var) + 1), cum_var)
    for nguong in NGUONG_PHUONG_SAI:
        k = int(np.argmax(cum_var >= nguong) + 1)
        plt.axhline(nguong, color="gray", linestyle="--", linewidth=0.8)
        plt.axvline(k, color="gray", linestyle="--", linewidth=0.8)
        plt.scatter([k], [nguong], color="red", zorder=5)
        plt.annotate(f"{nguong:.0%} @ k={k}", (k, nguong), textcoords="offset points",
                     xytext=(5, -10), fontsize=8)
    plt.xlabel("Số thành phần (K)"); plt.ylabel("Phương sai tích luỹ")
    plt.title("Phương sai tích luỹ theo số thành phần")
    plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


# ----------------------------------------------------------------------
# Bước 6 : chọn K bằng GroupKFold theo subject trên TRAIN
# ----------------------------------------------------------------------
def _make_pipe(k: int, n_features: int, method: str = "pca") -> Pipeline:
    if method == "pca":
        red = PCA(n_components=None if k >= n_features else k, random_state=RANDOM_STATE)
    else:
        red = SelectKBest(f_classif, k=min(k, n_features))
    return Pipeline([
        ("scale", StandardScaler()),   # nằm TRONG pipeline -> mỗi fold tự fit lại, không rò rỉ
        ("red", red),
        ("clf", LinearSVC(dual=False, max_iter=5000, random_state=RANDOM_STATE)),
    ])


def cv_theo_K(X_train: np.ndarray, y_train: np.ndarray, groups: np.ndarray,
              k_list: list[int], method: str = "pca") -> pd.DataFrame:
    gkf = GroupKFold(n_splits=N_SPLITS)
    n_features = X_train.shape[1]
    rows = []
    for k in k_list:
        r = cross_validate(_make_pipe(k, n_features, method), X_train, y_train,
                           groups=groups, cv=gkf, scoring="accuracy", n_jobs=-1)
        acc = r["test_score"]
        rows.append({"K": k, "cv_acc_mean": acc.mean(), "cv_acc_std": acc.std(ddof=1),
                     "cv_acc_se": acc.std(ddof=1) / np.sqrt(len(acc)),
                     "fit_time_s": r["fit_time"].mean()})
        print(f"  [{method}] K={k:>4d} | CV acc={acc.mean():.4f} ± {acc.std(ddof=1):.4f}"
              f" | fit={r['fit_time'].mean():.2f}s")
    return pd.DataFrame(rows)


def chon_K_1se(df_cv: pd.DataFrame) -> dict:
    """K nhỏ nhất có CV-accuracy >= (best_mean - SE của best). Không phụ thuộc test."""
    i_best = df_cv["cv_acc_mean"].idxmax()
    best = df_cv.loc[i_best]
    nguong = best["cv_acc_mean"] - best["cv_acc_se"]
    ung_vien = df_cv[df_cv["cv_acc_mean"] >= nguong]
    return {"K_tot_nhat": int(best["K"]), "acc_tot_nhat": float(best["cv_acc_mean"]),
            "nguong_1se": float(nguong), "K_chon": int(ung_vien["K"].min())}

# ----------------------------------------------------------------------
# Ràng buộc thiết bị: footprint mô hình theo K
# ----------------------------------------------------------------------
def footprint_kb(k: int, n_features: int, bytes_per: int, method: str = "pca") -> float:
    """Số tham số phải lưu trên thiết bị (đã gộp scaler vào bước sau nó)."""
    if method == "pca":
        # ma trận chiếu K x n + bias K (gộp scaler+mean) ; SVM: 6 x K + 6
        n_params = k * n_features + k + N_CLASSES * k + N_CLASSES
    else:
        # SelectKBest: K chỉ số (int16) + scale/mean K ; SVM 6 x K + 6
        n_params = 2 * k + N_CLASSES * k + N_CLASSES
        return (k * 2 + (n_params - k * 2) * bytes_per) / 1024
    return n_params * bytes_per / 1024


def bang_footprint(df_cv: pd.DataFrame, n_features: int, ram_kb: float) -> pd.DataFrame:
    df = df_cv[["K", "cv_acc_mean"]].copy()
    df["pca_fp32_KB"] = [footprint_kb(k, n_features, 4) for k in df["K"]]
    df["pca_int8_KB"] = [footprint_kb(k, n_features, 1) for k in df["K"]]
    df["vua_fp32"] = df["pca_fp32_KB"] <= ram_kb
    df["vua_int8"] = df["pca_int8_KB"] <= ram_kb
    return df

# ----------------------------------------------------------------------
# Đánh giá TEST (CHỈ để báo cáo, không dùng chọn K)
# ----------------------------------------------------------------------
def danh_gia_test(X_train, y_train, X_test, y_test, k: int, method: str = "pca") -> dict:
    pipe = _make_pipe(k, X_train.shape[1], method)
    t0 = time.time(); pipe.fit(X_train, y_train); t = time.time() - t0
    return {"K": k, "test_acc": float(pipe.score(X_test, y_test)), "train_time_s": t}


def ve_danh_doi_K(df_cv: pd.DataFrame, k_chon: int, out_path: Path):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.errorbar(df_cv["K"], df_cv["cv_acc_mean"], yerr=df_cv["cv_acc_se"], fmt="o-",
                 color="tab:blue", label="CV accuracy (GroupKFold, ±SE)")
    ax1.set_xlabel("Số chiều K"); ax1.set_ylabel("CV accuracy", color="tab:blue")
    ax1.axvline(k_chon, color="red", linestyle="--", linewidth=1, label=f"K chọn (1-SE) = {k_chon}")
    ax2 = ax1.twinx()
    ax2.plot(df_cv["K"], df_cv["fit_time_s"], "s--", color="tab:orange", label="Thời gian fit (s)")
    ax2.set_ylabel("Thời gian fit (giây)", color="tab:orange")
    l1, b1 = ax1.get_legend_handles_labels(); l2, b2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, b1 + b2, loc="center right")
    plt.title("Đánh đổi K vs CV accuracy & thời gian fit")
    fig.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


def ve_pca_vs_selectkbest(df_pca: pd.DataFrame, df_skb: pd.DataFrame, out_path: Path):
    plt.figure(figsize=(8, 5))
    plt.plot(df_pca["K"], df_pca["cv_acc_mean"], "o-", label="PCA")
    plt.plot(df_skb["K"], df_skb["cv_acc_mean"], "s-", label="SelectKBest (f_classif)")
    plt.xscale("log"); plt.xlabel("Số chiều K"); plt.ylabel("CV accuracy")
    plt.title("PCA vs SelectKBest"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


# ----------------------------------------------------------------------
# Bước 7-8-9-10
# ----------------------------------------------------------------------
def ve_scatter_pc1_pc2(X_train_s, y_train_name: pd.Series, out_path: Path):
    pca2 = PCA(n_components=2, random_state=RANDOM_STATE).fit(X_train_s)
    X2 = pca2.transform(X_train_s)
    plt.figure(figsize=(8, 6))
    for act in sorted(y_train_name.unique()):
        m = (y_train_name == act).values
        plt.scatter(X2[m, 0], X2[m, 1], s=8, alpha=0.5, label=act)
    plt.xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]:.1%} phương sai)")
    plt.ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]:.1%} phương sai)")
    plt.title("Chiếu dữ liệu lên PC1-PC2, tô màu theo hoạt động")
    plt.legend(markerscale=2, fontsize=8); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


def phan_tich_pc1(pca_full: PCA, feature_names: list[str], top_n: int = 10) -> pd.DataFrame:
    w = pca_full.components_[0]
    idx = np.argsort(np.abs(w))[::-1][:top_n]
    return pd.DataFrame({"dac_trung_goc": [feature_names[i] for i in idx], "trong_so_PC1": w[idx]})


def sai_so_tai_tao_theo_K(X_test_s, pca_full: PCA, k_list: list[int], out_path: Path) -> pd.DataFrame:
    """Dùng lại pca_full (đã fit trên train) và cắt K thành phần đầu - nhanh hơn fit lại từng K."""
    mu, V = pca_full.mean_, pca_full.components_
    rows = []
    for k in k_list:
        if k >= V.shape[0]:
            continue
        Z = (X_test_s - mu) @ V[:k].T
        X_rec = Z @ V[:k] + mu
        rows.append({"K": k, "mse_tai_tao": float(np.mean((X_test_s - X_rec) ** 2))})
    df = pd.DataFrame(rows)
    plt.figure(figsize=(8, 5))
    plt.plot(df["K"], df["mse_tai_tao"], "o-")
    plt.xlabel("Số chiều K"); plt.ylabel("MSE tái tạo (test)")
    plt.title("Sai số tái tạo ngược theo K"); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
    return df


# ----------------------------------------------------------------------
# Báo cáo tự động
# ----------------------------------------------------------------------
def _md(df: pd.DataFrame, floatfmt: str | None = None) -> str:
    """Bảng markdown thuần pandas (không cần thư viện tabulate)."""
    def fmt(v):
        if isinstance(v, (float, np.floating)) and floatfmt:
            return format(v, floatfmt)
        return str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def ghi_bao_cao(path: Path, m: dict, bang_nguong: pd.DataFrame, df_fp: pd.DataFrame,
                df_cmp: pd.DataFrame, df_pc1: pd.DataFrame, df_rec: pd.DataFrame):
    md = f"""# Báo cáo kết quả (sinh tự động từ pca_pipeline.py)

## 1. Bảng ngưỡng phương sai
{_md(bang_nguong)}

## 2. Chọn K (GroupKFold theo subject, trên TRAIN)
- K tốt nhất theo CV: **{m['K_tot_nhat']}** (CV acc = {m['acc_tot_nhat']:.4f})
- Quy tắc 1-SE -> ngưỡng {m['nguong_1se']:.4f} -> **K chọn = {m['K_chon']}**
- Giảm chiều thực sự: {m['phan_tram_giam_chieu']}
- Accuracy TEST tại K chọn (chỉ để báo cáo): **{m['test_acc_K_chon']:.4f}**;
  baseline K=561: {m['test_acc_baseline']:.4f}

## 3. Footprint mô hình vs RAM {m['ram_kb']} KB
{_md(df_fp, '.3f')}

- K lớn nhất vừa {m['ram_kb']} KB ở fp32: {m['K_max_vua_fp32']}, ở int8: {m['K_max_vua_int8']}

## 4. PCA vs SelectKBest (CV accuracy)
{_md(df_cmp, '.4f')}

## 5. 10 đặc trưng đóng góp nhiều nhất vào PC1
{_md(df_pc1, '.4f')}

## 6. Sai số tái tạo (test)
{_md(df_rec, '.5f')}
"""
    path.write_text(md, encoding="utf-8")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main(data_dir: str, out_dir: str, ram_kb: float):
    out_dir = Path(out_dir)
    reports, models = out_dir / "reports", out_dir / "models"
    reports.mkdir(parents=True, exist_ok=True); models.mkdir(parents=True, exist_ok=True)

    print("== Bước 1: Nạp dữ liệu ==")
    d = load_har_data(data_dir)
    X_train, X_test = d["X_train"], d["X_test"]
    y_train, y_test = d["y_train"].values.ravel(), d["y_test"].values.ravel()
    groups = d["subject_train"].values
    n_features = len(d["feature_names"])
    print(f"  X_train {X_train.shape} | X_test {X_test.shape} | "
          f"{len(set(groups))} subject train, không trùng test (assert OK)")

    print("\n== Bước 2-5: PCA đầy đủ, scree, phương sai tích luỹ, bảng ngưỡng ==")
    scaler, pca_full, X_train_s = fit_scaler_and_full_pca(X_train)
    X_test_s = scaler.transform(X_test)
    ve_scree_plot(pca_full, reports / "scree_plot.png")
    ve_phuong_sai_tich_luy(pca_full, reports / "variance_tich_luy.png")
    bang_nguong = bang_nguong_phuong_sai(pca_full, n_features)
    print(bang_nguong.to_string(index=False))

    print("\n== Bước 6: Chọn K bằng GroupKFold (subject) trên TRAIN ==")
    df_cv = cv_theo_K(X_train.values, y_train, groups, K_LIST, "pca")
    sel = chon_K_1se(df_cv)
    k_chon = sel["K_chon"]
    print(f"  -> K tốt nhất={sel['K_tot_nhat']} | ngưỡng 1-SE={sel['nguong_1se']:.4f} | K chọn={k_chon}")
    if k_chon >= n_features:
        print("  !! CẢNH BÁO: K chọn = 561, tức KHÔNG giảm chiều. Hãy xem lại bảng CV.")
    ve_danh_doi_K(df_cv, k_chon, reports / "danh_doi_chieu_accuracy.png")
    df_cv.to_csv(reports / "cv_theo_K.csv", index=False)

    print("\n== Ràng buộc RAM thiết bị ==")
    df_fp = bang_footprint(df_cv, n_features, ram_kb)
    print(df_fp.to_string(index=False))
    k_max_fp32 = int(df_fp.loc[df_fp["vua_fp32"], "K"].max()) if df_fp["vua_fp32"].any() else None
    k_max_int8 = int(df_fp.loc[df_fp["vua_int8"], "K"].max()) if df_fp["vua_int8"].any() else None
    print(f"  K lớn nhất vừa {ram_kb} KB: fp32={k_max_fp32}, int8={k_max_int8}")

    print("\n== Đánh giá TEST (chỉ báo cáo) ==")
    res_chon = danh_gia_test(X_train.values, y_train, X_test.values, y_test, k_chon)
    res_base = danh_gia_test(X_train.values, y_train, X_test.values, y_test, n_features)
    print(f"  K={k_chon}: test acc={res_chon['test_acc']:.4f} | K=561: {res_base['test_acc']:.4f}")

    print("\n== So sánh với SelectKBest ==")
    df_skb = cv_theo_K(X_train.values, y_train, groups,
                       sorted(set(K_SO_SANH_SELECTKBEST + [k_chon])), "skb")
    ve_pca_vs_selectkbest(df_cv[df_cv["K"].isin(df_skb["K"])], df_skb,
                          reports / "pca_vs_selectkbest.png")
    df_cmp = (df_cv[df_cv["K"].isin(df_skb["K"])][["K", "cv_acc_mean"]]
              .rename(columns={"cv_acc_mean": "pca_cv_acc"})
              .merge(df_skb[["K", "cv_acc_mean"]].rename(columns={"cv_acc_mean": "selectkbest_cv_acc"}),
                     on="K"))
    print(df_cmp.to_string(index=False))

    print("\n== Bước 7-8: scatter PC1-PC2, loadings PC1 ==")
    ve_scatter_pc1_pc2(X_train_s, d["y_train_name"], reports / "pc1_pc2_scatter.png")
    df_pc1 = phan_tich_pc1(pca_full, d["feature_names"])
    print(df_pc1.to_string(index=False))

    print("\n== Bước 10: Sai số tái tạo ==")
    df_rec = sai_so_tai_tao_theo_K(X_test_s, pca_full,
                                   sorted(set([2, 10, 25, 50, k_chon, 100, 200])),
                                   reports / "reconstruction_error.png")
    print(df_rec.to_string(index=False))

    print("\n== Lưu pipeline cuối + metrics ==")
    final_pipe = _make_pipe(k_chon, n_features, "pca").fit(X_train.values, y_train)
    joblib.dump(final_pipe, models / "pca_pipeline.joblib")

    metrics = {
        **sel, "n_features": n_features,
        "phan_tram_giam_chieu": f"{(1 - k_chon / n_features):.1%}",
        "test_acc_K_chon": res_chon["test_acc"], "test_acc_baseline": res_base["test_acc"],
        "ram_kb": ram_kb, "K_max_vua_fp32": k_max_fp32, "K_max_vua_int8": k_max_int8,
        "footprint_K_chon_fp32_KB": footprint_kb(k_chon, n_features, 4),
        "footprint_K_chon_int8_KB": footprint_kb(k_chon, n_features, 1),
        "bang_nguong": bang_nguong.to_dict(orient="records"),
    }
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    ghi_bao_cao(reports / "bao_cao_ket_qua.md", metrics, bang_nguong,
                df_fp, df_cmp, df_pc1, df_rec)
    print("  Đã ghi models/pca_pipeline.joblib, reports/metrics.json, reports/bao_cao_ket_qua.md")


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent  # gốc dự án (cha của src/)
    p = argparse.ArgumentParser(description="PCA trên UCI HAR Dataset")
    p.add_argument("--data-dir", default=str(ROOT / "dataset" / "UCI HAR Dataset"),
                   help="Thư mục 'UCI HAR Dataset' đã giải nén")
    p.add_argument("--out-dir", default=str(ROOT), help="Thư mục gốc để ghi models/ và reports/")
    p.add_argument("--ram-kb", type=float, default=64.0, help="RAM thiết bị đích (KB)")
    a = p.parse_args()

    data_path = Path(a.data_dir)
    if not (data_path / "features.txt").exists():
        # Trường hợp giải nén bị lồng 2 lớp: UCI HAR Dataset/UCI HAR Dataset/
        nested = data_path / "UCI HAR Dataset"
        if (nested / "features.txt").exists():
            data_path = nested
        else:
            raise SystemExit(
                f"Không tìm thấy features.txt trong: {data_path}\n"
                f"Hãy truyền đúng đường dẫn bằng --data-dir \"...\\UCI HAR Dataset\""
            )
    main(str(data_path), a.out_dir, a.ram_kb)