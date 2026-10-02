
import argparse
import itertools
import time
import warnings
import matplotlib
matplotlib.use("Agg")                       # lưu ảnh, không mở cửa sổ
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from preprocess import (MODELS, REMOVE, REPORTS, SEED, ensure_dirs, eda,
                        load_data, make_vectorizer, split_validation)

warnings.filterwarnings("ignore")
THRESHOLD = 0.6
THS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def show(title, df):
    print(f"\n=== {title} ===")
    print(df.round(4).to_string(index=False))

# ---------------------------------------------------------------- Bước 1
def buoc1_ro_ri():
    cau_hinh = {
        "Không remove gì (rò rỉ nặng)": (),
        "Chỉ bỏ headers": ("headers",),
        "Bỏ headers + footers": ("headers", "footers"),
        "Bỏ cả 3 (đúng chuẩn)": ("headers", "footers", "quotes"),
    }

    def chay(remove):
        tr, te = load_data(remove)
        pipe = make_pipeline(make_vectorizer(ngram_range=(1, 1), max_features=None),
                             LinearSVC(random_state=SEED))
        pipe.fit(tr.data, tr.target)          # vectorizer fit CHỈ trên train
        return accuracy_score(te.target, pipe.predict(te.data))

    leak = pd.DataFrame({"cau_hinh": list(cau_hinh),
                         "accuracy_test": [chay(v) for v in cau_hinh.values()]})
    show("buoc1_ro_ri — accuracy theo mức remove", leak)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(leak.cau_hinh[::-1], leak.accuracy_test[::-1],
            color=["#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"])
    for i, v in enumerate(leak.accuracy_test[::-1]):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Accuracy (test)")
    ax.set_title("Rò rỉ nhãn làm accuracy tăng ảo")
    plt.tight_layout()
    plt.savefig(REPORTS / "leakage_comparison.png", dpi=150)
    plt.close(fig)
    return leak

# ---------------------------------------------------------------- Bước 3–4
def buoc3_baseline(train, test):
    dummy = DummyClassifier(strategy="most_frequent").fit(train.data, train.target)
    nb = make_pipeline(CountVectorizer(min_df=3), MultinomialNB()).fit(train.data, train.target)
    base = pd.DataFrame({
        "model": ["Dummy (lớp phổ biến nhất)", "CountVectorizer + NaiveBayes"],
        "accuracy_test": [accuracy_score(test.target, dummy.predict(test.data)),
                          accuracy_score(test.target, nb.predict(test.data))]})
    show("buoc3_baseline", base)
    return base

def buoc4_tfidf_svc(train, test):
    pipe = make_pipeline(make_vectorizer(), LinearSVC(C=1.0, random_state=SEED))
    pipe.fit(train.data, train.target)
    acc = accuracy_score(test.target, pipe.predict(test.data))
    print(f"\n=== Bước 4 — TF-IDF + LinearSVC: accuracy test = {acc:.4f} "
          f"(mức tham chiếu ~0,69–0,73) ===")
    return acc

# ---------------------------------------------------------------- Bước 5
def buoc5_khao_sat(train):
    Xa, Xv, ya, yv = split_validation(train)
    rows = []
    for ng, mdf, sub in itertools.product([(1, 1), (1, 2)], [1, 3, 5], [True, False]):
        vec = make_vectorizer(ngram_range=ng, min_df=mdf, sublinear_tf=sub, max_features=None)
        A, V = vec.fit_transform(Xa), vec.transform(Xv)
        clf = LinearSVC(random_state=SEED).fit(A, ya)
        rows.append(dict(ngram_range=str(ng), min_df=mdf, sublinear_tf=sub,
                         so_dac_trung=A.shape[1],
                         acc_val=accuracy_score(yv, clf.predict(V))))
    grid = pd.DataFrame(rows).sort_values("acc_val", ascending=False)
    show("buoc5_khao_sat — tham số vectorizer (trên validation)", grid)
    return grid

# ---------------------------------------------------------------- Bước 6
def buoc6_so_sanh(train, test):
    vec = make_vectorizer()
    Xtr = vec.fit_transform(train.data)       # fit CHỈ trên train
    Xte = vec.transform(test.data)
    models = {
        "NaiveBayes": MultinomialNB(alpha=0.1),
        "LogReg": LogisticRegression(C=10, max_iter=1000),
        "LinearSVC": LinearSVC(C=1.0, random_state=SEED),
    }
    rows = []
    for name, m in models.items():
        t0 = time.perf_counter()
        m.fit(Xtr, train.target)
        t = time.perf_counter() - t0
        rows.append(dict(model=name,
                         acc_test=accuracy_score(test.target, m.predict(Xte)),
                         thoi_gian_train_s=t))
    show("buoc6_so_sanh — 3 bộ phân loại trên cùng TF-IDF",
         pd.DataFrame(rows).sort_values("acc_test", ascending=False))
    return vec, models, Xtr, Xte

# ---------------------------------------------------------------- Bước 7
def buoc7_top_tu(svc, vec, names, k=15):
    feat = np.array(vec.get_feature_names_out())
    top_words = {}
    for i, c in enumerate(names):
        idx = np.argsort(svc.coef_[i])[-k:][::-1]
        top_words[c] = (feat[idx], svc.coef_[i][idx])

    print(f"\n=== Bước 7 — Top {k} từ mỗi lớp ===")
    for c, (w, _) in top_words.items():
        print(f"{c:26s}", ", ".join(w))

    nghi_van = {"edu", "com", "writes", "article", "subject", "lines", "organization",
                "nntp", "host", "posting", "reply", "distribution"}
    lot = {c: sorted(nghi_van & set(w)) for c, (w, _) in top_words.items() if nghi_van & set(w)}
    print("Từ đáng ngờ lọt vào top:", lot if lot else "không có")

    fig, axes = plt.subplots(5, 4, figsize=(22, 24))
    for ax, (c, (w, s)) in zip(axes.ravel(), top_words.items()):
        ax.barh(w[::-1], s[::-1], color="#264653")
        ax.set_title(c, fontsize=11)
        ax.tick_params(labelsize=8)
    plt.tight_layout()
    plt.savefig(REPORTS / "top_tu_moi_lop.png", dpi=120)
    plt.close(fig)
    return top_words

# ---------------------------------------------------------------- Bước 8
def buoc8_nham_lan(svc, Xte, y_test, names):
    y_pred = svc.predict(Xte)
    fig, ax = plt.subplots(figsize=(14, 14))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=names, normalize="true", xticks_rotation=90,
        values_format=".1f", ax=ax, colorbar=False, cmap="Blues")
    plt.tight_layout()
    plt.savefig(REPORTS / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    cm = confusion_matrix(y_test, y_pred)
    np.fill_diagonal(cm, 0)
    flat = np.argsort(cm.ravel())[::-1][:10]
    pairs = pd.DataFrame([dict(that=names[i], bi_doan_thanh=names[j], so_van_ban=cm[i, j])
                          for i, j in zip(*np.unravel_index(flat, cm.shape))])
    show("buoc8_nham_lan — 10 cặp lớp hay nhầm nhất", pairs)

# ---------------------------------------------------------------- Bước 9
def bang_nguong(conf, pred, y, ths=THS):
    rows = []
    for th in ths:
        m = conf >= th
        rows.append(dict(nguong=th, pct_tu_dong=m.mean(),
                         acc_phan_tu_dong=(pred[m] == y[m]).mean() if m.any() else np.nan,
                         pct_can_nguoi=1 - m.mean()))
    return pd.DataFrame(rows)

def buoc9_nguong(train, test, Xtr, Xte):
    # (a) validation để CHỌN ngưỡng
    Xa, Xv, ya, yv = split_validation(train)
    vec_v = make_vectorizer().fit(Xa)
    cal_v = CalibratedClassifierCV(LinearSVC(random_state=SEED), cv=3).fit(vec_v.transform(Xa), ya)
    pv = cal_v.predict_proba(vec_v.transform(Xv))
    show("buoc9_nguong_validation", bang_nguong(pv.max(1), pv.argmax(1), yv))

    # (b) test để BÁO CÁO
    cal = CalibratedClassifierCV(LinearSVC(random_state=SEED), cv=3).fit(Xtr, train.target)
    proba = cal.predict_proba(Xte)
    conf, pred = proba.max(1), proba.argmax(1)
    bang = bang_nguong(conf, pred, test.target)
    show("buoc9_nguong_test", bang)
    print(f"Accuracy toàn bộ (không ngưỡng): {(pred == test.target).mean():.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].plot(bang.pct_tu_dong * 100, bang.acc_phan_tu_dong * 100, "o-")
    for _, r in bang.iterrows():
        ax[0].annotate(f"{r.nguong}", (r.pct_tu_dong * 100, r.acc_phan_tu_dong * 100),
                       textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax[0].set_xlabel("% ticket xử lý tự động")
    ax[0].set_ylabel("Accuracy phần tự động (%)")
    ax[0].set_title("Đánh đổi: tự động hoá vs độ chính xác")
    ax[1].hist(conf[pred == test.target], bins=30, alpha=.7, label="dự đoán đúng")
    ax[1].hist(conf[pred != test.target], bins=30, alpha=.7, label="dự đoán sai")
    ax[1].axvline(THRESHOLD, color="k", ls="--")
    ax[1].legend()
    ax[1].set_xlabel("max(proba)")
    ax[1].set_title("Phân bố độ tin cậy")
    plt.tight_layout()
    plt.savefig(REPORTS / "nguong_tin_cay.png", dpi=150)
    plt.close(fig)
    return conf, pred

# ---------------------------------------------------------------- Bước 10
def buoc10_thoi_gian(train, test):
    final = make_pipeline(make_vectorizer(),
                          CalibratedClassifierCV(LinearSVC(C=1.0, random_state=SEED), cv=3))
    t0 = time.perf_counter()
    final.fit(train.data, train.target)
    t_train = time.perf_counter() - t0

    doc = [test.data[0]]
    final.predict_proba(doc)                  # khởi động
    ts = []
    for _ in range(300):
        t0 = time.perf_counter()
        final.predict_proba(doc)
        ts.append((time.perf_counter() - t0) * 1000)
    ms = float(np.mean(ts))
    print("\n=== Bước 10 — Thời gian ===")
    print(f"Train toàn bộ: {t_train:.1f} s")
    print(f"Dự đoán 1 văn bản: trung bình {ms:.2f} ms | p95 {np.percentile(ts, 95):.2f} ms")
    print("ĐẠT < 5 ms" if ms < 5 else "CHƯA ĐẠT — thử giảm max_features hoặc bỏ bigram")
    return final

# ---------------------------------------------------------------- Bước 11
def buoc11_ca_sai(test, names, pred, conf):
    n_words = np.array([len(t.split()) for t in test.data])
    dung = pred == test.target
    so_sanh = pd.DataFrame({
        "nhom": ["dự đoán đúng", "dự đoán sai"],
        "conf_trung_binh": [conf[dung].mean(), conf[~dung].mean()],
        "so_tu_trung_vi": [np.median(n_words[dung]), np.median(n_words[~dung])]})
    show("buoc11_ca_sai — so sánh ca đúng/sai", so_sanh)

    rng = np.random.default_rng(SEED)
    sai = np.where(~dung)[0]
    print("\n--- 10 ca sai (hãy phân loại nguyên nhân từng ca) ---")
    for i in rng.choice(sai, min(10, len(sai)), replace=False):
        print(f"#{i}  THẬT: {names[test.target[i]]}  →  ĐOÁN: {names[pred[i]]} "
              f"(conf={conf[i]:.2f}, {n_words[i]} từ)")
        print("   ", test.data[i][:300].replace("\n", " "), "\n")

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-leakage", action="store_true", help="bỏ Bước 1")
    ap.add_argument("--skip-grid", action="store_true", help="bỏ Bước 5")
    args = ap.parse_args()
    ensure_dirs()

    if not args.skip_leakage:
        buoc1_ro_ri()                                              # Bước 1

    train, test = load_data(REMOVE)                                # từ đây CHỈ dùng bản đã remove
    names = train.target_names
    eda(train, REPORTS / "eda.png")                                # Bước 2
    base = buoc3_baseline(train, test)                             # Bước 3
    acc_svc = buoc4_tfidf_svc(train, test)                         # Bước 4
    if not args.skip_grid:
        buoc5_khao_sat(train)                                      # Bước 5
    vec, models, Xtr, Xte = buoc6_so_sanh(train, test)             # Bước 6
    svc = models["LinearSVC"]
    buoc7_top_tu(svc, vec, names)                                  # Bước 7
    buoc8_nham_lan(svc, Xte, test.target, names)                   # Bước 8
    conf, pred = buoc9_nguong(train, test, Xtr, Xte)               # Bước 9
    final = buoc10_thoi_gian(train, test)                          # Bước 10
    buoc11_ca_sai(test, names, pred, conf)                         # Bước 11

    joblib.dump(final, MODELS / "tfidf_pipeline.joblib")           # Bước 12
    print("\nĐã lưu model:", MODELS / "tfidf_pipeline.joblib")
    show("buoc12_tong_ket", pd.DataFrame([
        ("Dummy", base.accuracy_test[0]),
        ("CountVectorizer + NB", base.accuracy_test[1]),
        ("TF-IDF + LinearSVC", acc_svc),
        ("TF-IDF + LinearSVC (calibrated)", (pred == test.target).mean()),
    ], columns=["model", "accuracy_test"]))

if __name__ == "__main__":
    main()