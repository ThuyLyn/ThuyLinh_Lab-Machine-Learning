from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# ----------------------------------------------------------------------
# Cấu hình mặc định
# ----------------------------------------------------------------------
NGUONG_PHUONG_SAI = [0.80, 0.90, 0.95, 0.99]
K_LIST = [2, 10, 25, 50, 100, 200, 561]
RANDOM_STATE = 42

def _read_features(data_dir: Path) -> list[str]:
    """Đọc tên 561 đặc trưng từ features.txt, đảm bảo tên cột không trùng nhau."""
    feat_path = data_dir / "features.txt"
    feats = pd.read_csv(feat_path, sep=r"\s+", header=None, names=["idx", "name"])
    names = feats["name"].tolist()
    seen = {}
    unique_names = []
    for n in names:
        if n in seen:
            seen[n] += 1
            unique_names.append(f"{n}__{seen[n]}")
        else:
            seen[n] = 0
            unique_names.append(n)
    return unique_names


def _read_activity_labels(data_dir: Path) -> dict[int, str]:
    labels = pd.read_csv(
        data_dir / "activity_labels.txt", sep=r"\s+", header=None,
        names=["id", "activity"]
    )
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
        y_name = y.map(activity_map)
        return X, y, subj, y_name

    X_train, y_train, subj_train, y_train_name = _load_split("train")
    X_test, y_test, subj_test, y_test_name = _load_split("test")

    # Kiểm tra bất biến quan trọng: không người nào xuất hiện ở cả train và test
    overlap = set(subj_train.unique()) & set(subj_test.unique())
    if overlap:
        raise AssertionError(
            f"RÒ RỈ DỮ LIỆU: các subject {overlap} xuất hiện ở cả train và test!"
        )

    return {
        "X_train": X_train, "y_train": y_train, "y_train_name": y_train_name,
        "subject_train": subj_train,
        "X_test": X_test, "y_test": y_test, "y_test_name": y_test_name,
        "subject_test": subj_test,
        "feature_names": feature_names,
        "activity_map": activity_map,
    }


# ========================================================================
# Bước 2-3-4-5: chuẩn hoá, PCA đầy đủ, scree plot, phương sai tích luỹ, bảng ngưỡng
# ========================================================================
def fit_scaler_and_full_pca(X_train: pd.DataFrame):
    scaler = StandardScaler().fit(X_train)          # fit CHỈ trên train
    X_train_s = scaler.transform(X_train)
    pca_full = PCA(random_state=RANDOM_STATE).fit(X_train_s)
    return scaler, pca_full, X_train_s

def bang_nguong_phuong_sai(pca_full: PCA, n_features_total: int) -> pd.DataFrame:
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    rows = []
    for nguong in NGUONG_PHUONG_SAI:
        k = int(np.argmax(cum_var >= nguong) + 1)
        rows.append({
            "nguong_phuong_sai": f"{nguong:.0%}",
            "so_chieu_can": k,
            "phan_tram_giam_chieu": f"{(1 - k / n_features_total):.1%}",
        })
    return pd.DataFrame(rows)


def ve_scree_plot(pca_full: PCA, out_path: Path, top_n: int = 50):
    plt.figure(figsize=(8, 5))
    var_ratio = pca_full.explained_variance_ratio_[:top_n]
    plt.plot(range(1, len(var_ratio) + 1), var_ratio, marker="o", markersize=3)
    plt.xlabel("Thành phần chính (PC)")
    plt.ylabel("Tỉ lệ phương sai giải thích")
    plt.title(f"Scree Plot ({top_n} thành phần đầu)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def ve_phuong_sai_tich_luy(pca_full: PCA, out_path: Path):
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cum_var) + 1), cum_var)
    for nguong in NGUONG_PHUONG_SAI:
        k = int(np.argmax(cum_var >= nguong) + 1)
        plt.axhline(nguong, color="gray", linestyle="--", linewidth=0.8)
        plt.axvline(k, color="gray", linestyle="--", linewidth=0.8)
        plt.scatter([k], [nguong], color="red", zorder=5)
        plt.annotate(f"{nguong:.0%} @ k={k}", (k, nguong),
                      textcoords="offset points", xytext=(5, -10), fontsize=8)
    plt.xlabel("Số thành phần (K)")
    plt.ylabel("Phương sai tích luỹ")
    plt.title("Phương sai tích luỹ theo số thành phần")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ----------------------------------------------------------------------
# Bước 6 ( thí nghiệm chính): đánh đổi K vs accuracy + thời gian train
# ----------------------------------------------------------------------
def thi_nghiem_danh_doi_K(X_train_s: np.ndarray, y_train, X_test_s: np.ndarray, y_test,
                            k_list: list[int] = K_LIST) -> pd.DataFrame:
    rows = []
    for k in k_list:
        n_components = None if k >= X_train_s.shape[1] else k
        pipe = Pipeline([
            ("pca", PCA(n_components=n_components, random_state=RANDOM_STATE)),
            ("clf", LinearSVC(max_iter=5000, random_state=RANDOM_STATE)),
        ])
        t0 = time.time()
        pipe.fit(X_train_s, y_train)
        train_time = time.time() - t0
        acc = pipe.score(X_test_s, y_test)
        rows.append({"K": k, "accuracy": acc, "thoi_gian_train_s": train_time})
        print(f"  K={k:>4d} | accuracy={acc:.4f} | thời gian train={train_time:.2f}s")
    return pd.DataFrame(rows)


def tim_diem_ngot(df_ketqua: pd.DataFrame, nguong_giam_toi_da: float = 0.01) -> int:
    """Điểm ngọt = K nhỏ nhất mà accuracy giảm dưới `nguong_giam_toi_da` so với K=561 (baseline)."""
    acc_baseline = df_ketqua.loc[df_ketqua["K"] == df_ketqua["K"].max(), "accuracy"].iloc[0]
    du_dieu_kien = df_ketqua[acc_baseline - df_ketqua["accuracy"] < nguong_giam_toi_da]
    return int(du_dieu_kien["K"].min())


def ve_danh_doi_K(df_ketqua: pd.DataFrame, diem_ngot: int, out_path: Path):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(df_ketqua["K"], df_ketqua["accuracy"], "o-", color="tab:blue", label="Accuracy")
    ax1.set_xlabel("Số chiều K")
    ax1.set_ylabel("Accuracy", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.axvline(diem_ngot, color="red", linestyle="--", linewidth=1,
                label=f"Điểm ngọt K={diem_ngot}")

    ax2 = ax1.twinx()
    ax2.plot(df_ketqua["K"], df_ketqua["thoi_gian_train_s"], "s--", color="tab:orange",
              label="Thời gian train (s)")
    ax2.set_ylabel("Thời gian train (giây)", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    plt.title("Đánh đổi: số chiều (K) vs Accuracy & thời gian train")
    fig.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# ----------------------------------------------------------------------
# Bước 7: scatter PC1-PC2 tô màu theo hoạt động
# ----------------------------------------------------------------------
def ve_scatter_pc1_pc2(X_train_s: np.ndarray, y_train_name: pd.Series, out_path: Path):
    pca2 = PCA(n_components=2, random_state=RANDOM_STATE).fit(X_train_s)
    X2 = pca2.transform(X_train_s)

    plt.figure(figsize=(8, 6))
    for act in sorted(y_train_name.unique()):
        mask = (y_train_name == act).values
        plt.scatter(X2[mask, 0], X2[mask, 1], s=8, alpha=0.5, label=act)
    plt.xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]:.1%} phương sai)")
    plt.ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]:.1%} phương sai)")
    plt.title("Chiếu dữ liệu lên PC1-PC2, tô màu theo hoạt động")
    plt.legend(markerscale=2, fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return pca2

# ----------------------------------------------------------------------
# Bước 8: phân tích ý nghĩa PC1 — 10 đặc trưng gốc đóng góp nhiều nhất
# ----------------------------------------------------------------------
def phan_tich_pc1(pca_full: PCA, feature_names: list[str], top_n: int = 10) -> pd.DataFrame:
    pc1_weights = pca_full.components_[0]
    top_idx = np.argsort(np.abs(pc1_weights))[::-1][:top_n]
    return pd.DataFrame({
        "dac_trung_goc": [feature_names[i] for i in top_idx],
        "trong_so_PC1": pc1_weights[top_idx],
    })

# ----------------------------------------------------------------------
# Bước 9: đo dung lượng gốc vs sau PCA
# ----------------------------------------------------------------------
def do_dung_luong(X_train: pd.DataFrame, k: int) -> dict:
    mb_goc = X_train.memory_usage(deep=True).sum() / 1e6
    mb_sau_pca = X_train.shape[0] * k * 8 / 1e6  # float64 = 8 bytes
    return {
        "dung_luong_goc_MB": round(mb_goc, 2),
        "dung_luong_sau_pca_MB": round(mb_sau_pca, 2),
        "phan_tram_tiet_kiem": f"{(1 - mb_sau_pca / mb_goc):.1%}",
    }
# ----------------------------------------------------------------------
# Bước 10: tái tạo ngược (inverse_transform), đo sai số theo K
# ----------------------------------------------------------------------
def sai_so_tai_tao_theo_K(X_test_s: np.ndarray, X_train_s: np.ndarray,
                            k_list: list[int], out_path: Path) -> pd.DataFrame:
    rows = []
    for k in k_list:
        if k >= X_train_s.shape[1]:
            continue
        pca_k = PCA(n_components=k, random_state=RANDOM_STATE).fit(X_train_s)
        X_rec = pca_k.inverse_transform(pca_k.transform(X_test_s))
        mse = float(np.mean((X_test_s - X_rec) ** 2))
        rows.append({"K": k, "mse_tai_tao": mse})

    df = pd.DataFrame(rows)
    plt.figure(figsize=(8, 5))
    plt.plot(df["K"], df["mse_tai_tao"], "o-")
    plt.xlabel("Số chiều K")
    plt.ylabel("MSE tái tạo (trên tập test)")
    plt.title("Sai số tái tạo ngược theo số chiều K")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return df
# ----------------------------------------------------------------------
# MAIN: chạy toàn bộ pipeline
# ----------------------------------------------------------------------
def main(data_dir: str, out_dir: str):
    out_dir = Path(out_dir)
    reports_dir = out_dir / "reports"
    models_dir = out_dir / "models"
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("== Bước 1: Nạp dữ liệu (giữ nguyên chia theo người) ==")
    d = load_har_data(data_dir)
    X_train, y_train, y_train_name = d["X_train"], d["y_train"], d["y_train_name"]
    X_test, y_test = d["X_test"], d["y_test"]
    feature_names = d["feature_names"]
    n_features = len(feature_names)
    print(f"  X_train: {X_train.shape} | X_test: {X_test.shape}")

    print("\n== Bước 2-3: Chuẩn hoá (fit trên train) + PCA đầy đủ ==")
    scaler, pca_full, X_train_s = fit_scaler_and_full_pca(X_train)
    X_test_s = scaler.transform(X_test)

    print("\n== Bước 3: Scree plot ==")
    ve_scree_plot(pca_full, reports_dir / "scree_plot.png")

    print("== Bước 4: Phương sai tích luỹ ==")
    ve_phuong_sai_tich_luy(pca_full, reports_dir / "variance_tich_luy.png")

    print("\n== Bước 5: Bảng ngưỡng phương sai -> số chiều ==")
    bang_nguong = bang_nguong_phuong_sai(pca_full, n_features)
    print(bang_nguong.to_string(index=False))

    print("\n== Bước 6: Đánh đổi K vs accuracy ==")
    df_ketqua = thi_nghiem_danh_doi_K(X_train_s, y_train.values.ravel(),
                                        X_test_s, y_test.values.ravel())
    diem_ngot = tim_diem_ngot(df_ketqua)
    print(f"  -> Điểm ngọt: K = {diem_ngot} (accuracy giảm < 1% so với K=561)")
    ve_danh_doi_K(df_ketqua, diem_ngot, reports_dir / "danh_doi_chieu_accuracy.png")

    print("\n== Bước 7: Scatter PC1-PC2 ==")
    ve_scatter_pc1_pc2(X_train_s, y_train_name, reports_dir / "pc1_pc2_scatter.png")

    print("\n== Bước 8: Phân tích PC1 ==")
    df_pc1 = phan_tich_pc1(pca_full, feature_names)
    print(df_pc1.to_string(index=False))

    print("\n== Bước 9: Dung lượng gốc vs sau PCA (tại điểm ngọt) ==")
    dung_luong = do_dung_luong(X_train, diem_ngot)
    print(dung_luong)

    print("\n== Bước 10: Sai số tái tạo ngược theo K ==")
    k_list_tai_tao = sorted(set([2, 10, 25, 50, diem_ngot, 100, 200]))
    df_tai_tao = sai_so_tai_tao_theo_K(
        X_test_s, X_train_s, k_list_tai_tao,
        reports_dir / "reconstruction_error.png",
    )
    print(df_tai_tao.to_string(index=False))

    print("\n== Lưu pipeline cuối (Scaler + PCA(k tối ưu) + LinearSVC) ==")
    final_pipe = Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=diem_ngot, random_state=RANDOM_STATE)),
        ("clf", LinearSVC(max_iter=5000, random_state=RANDOM_STATE)),
    ])
    final_pipe.fit(X_train, y_train.values.ravel())
    acc_final = final_pipe.score(X_test, y_test.values.ravel())
    joblib.dump(final_pipe, models_dir / "pca_pipeline.joblib")
    print(f"  Đã lưu models/pca_pipeline.joblib | accuracy trên test = {acc_final:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TT-24 PCA trên UCI HAR Dataset")
    parser.add_argument("--data-dir", type=str, default="dataset/UCI HAR Dataset",
                         help="Đường dẫn tới thư mục 'UCI HAR Dataset' đã giải nén")
    parser.add_argument("--out-dir", type=str, default=".",
                         help="Thư mục gốc để ghi models/ và reports/")
    args = parser.parse_args()
    main(args.data_dir, args.out_dir)