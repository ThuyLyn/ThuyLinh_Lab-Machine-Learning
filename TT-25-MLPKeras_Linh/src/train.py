import argparse, json, os, time
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve

from data import load_and_split_onehot, load_and_split_embedding, compute_class_weight_dict
from model import build_mlp_basic, build_mlp_regularized, build_mlp_embedding, get_callbacks

os.makedirs("reports", exist_ok=True)
os.makedirs("models", exist_ok=True)
RESULTS = []

def precision_at_k(y_true, y_score, k=3000):
    idx = np.argsort(y_score)[::-1][:k]
    return float(y_true[idx].mean())

def evaluate(name, y_val, y_pred, seconds, notes=""):
    pr_auc = average_precision_score(y_val, y_pred)
    p3000 = precision_at_k(y_val, y_pred)
    RESULTS.append({"model": name, "pr_auc": round(pr_auc, 4),
                     "precision_at_3000": round(p3000, 4),
                     "train_seconds": round(seconds, 1), "notes": notes})
    print(f"[{name}] PR-AUC={pr_auc:.4f}  P@3000={p3000:.4f}  ({seconds:.1f}s)  {notes}")
    return pr_auc

def fit_keras(model, X_train, y_train, X_val, y_val, checkpoint, class_weight=None, epochs=150):
    t0 = time.time()
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                         epochs=epochs, batch_size=256, verbose=0, class_weight=class_weight,
                         callbacks=get_callbacks(f"models/{checkpoint}.keras"))
    y_pred = model.predict(X_val, verbose=0).ravel()
    return history, y_pred, time.time() - t0

def plot_history(history, title, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric, label in [(axes[0], "loss", "Loss"), (axes[1], "pr_auc", "PR-AUC")]:
        ax.plot(history[metric], label="train")
        ax.plot(history[f"val_{metric}"], label="val")
        ax.set_title(f"{label} — {title}"); ax.set_xlabel("epoch"); ax.legend()
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()

def plot_bar(labels, values, ylabel, title, path):
    plt.figure(figsize=(6, 4))
    plt.bar(labels, values); plt.ylabel(ylabel); plt.title(title)
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()

def plot_pr_curves(y_val, preds, path):
    plt.figure(figsize=(6, 5))
    for name, y_pred in preds.items():
        p, r, _ = precision_recall_curve(y_val, y_pred)
        plt.plot(r, p, label=name)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curve")
    plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()

def plot_embedding_vs_onehot(onehot_model, onehot_pr_auc, emb_model, emb_pr_auc, path):
    labels = ["One-hot (tốt nhất)", "Embedding"]
    pr_aucs = [onehot_pr_auc, emb_pr_auc]
    params = [onehot_model.count_params(), emb_model.count_params()]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(labels, pr_aucs)
    axes[0].set_title("PR-AUC"); axes[0].set_ylabel("PR-AUC")

    axes[1].bar(labels, params, color="orange")
    axes[1].set_title("Số tham số"); axes[1].set_ylabel("Số tham số")

    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


def main(data_path):
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, cols = load_and_split_onehot(data_path)
    print(f"Train {X_train.shape}  Val {X_val.shape}  Response=1: {y_train.mean():.3f}")
    cw = compute_class_weight_dict(y_train)
    preds = {}

    # Bước 4: baseline LightGBM
    import lightgbm as lgb
    t0 = time.time()
    lgb_clf = lgb.LGBMClassifier(n_estimators=500, class_weight="balanced", random_state=42)
    lgb_clf.fit(X_train, y_train)
    preds["LightGBM"] = lgb_clf.predict_proba(X_val)[:, 1]
    evaluate("LightGBM (baseline)", y_val, preds["LightGBM"], time.time() - t0, "class_weight=balanced")

    # Bước 5: MLP cơ bản (không Dropout/BN)
    basic = build_mlp_basic(X_train.shape[1])
    _, y_pred, secs = fit_keras(basic, X_train, y_train, X_val, y_val, "mlp_basic", epochs=100)
    evaluate("MLP cơ bản", y_val, y_pred, secs)

    # Bước 6-7: MLP + Dropout/BN (kiến trúc mặc định) + learning curve
    default_arch = (128, 64)
    reg = build_mlp_regularized(X_train.shape[1], hidden_units=default_arch)
    history, y_pred, secs = fit_keras(reg, X_train, y_train, X_val, y_val, "mlp_128_64", class_weight=cw)
    evaluate(f"MLP + Dropout/BN {default_arch}", y_val, y_pred, secs, "class_weight=balanced")
    plot_history(history.history, str(default_arch), "reports/learning_curves.png")

    # Bước 8: so sánh kiến trúc (tái dùng kết quả (128,64) ở trên, không train lại)
    arch_results = {default_arch: (reg, history.history)}
    for arch in [(64,), (256, 128, 64)]:
        m = build_mlp_regularized(X_train.shape[1], hidden_units=arch)
        h, yp, s = fit_keras(m, X_train, y_train, X_val, y_val, f"mlp_{'_'.join(map(str, arch))}", class_weight=cw)
        evaluate(f"MLP + Dropout/BN {arch}", y_val, yp, s, "class_weight=balanced")
        arch_results[arch] = (m, h.history)

    plot_bar([str(a) for a in arch_results], [max(h["val_pr_auc"]) for _, h in arch_results.values()],
              "Best val PR-AUC", "So sánh kiến trúc MLP", "reports/kien_truc_comparison.png")

    best_arch = max(arch_results, key=lambda a: max(arch_results[a][1]["val_pr_auc"]))
    best_model = arch_results[best_arch][0]
    preds["MLP tốt nhất"] = best_model.predict(X_val, verbose=0).ravel()
    best_onehot_pr_auc = average_precision_score(y_val, preds["MLP tốt nhất"])
    print(f"Kiến trúc tốt nhất: {best_arch}")

    # Bước 9: class_weight có/không, trên kiến trúc tốt nhất
    m_no_cw = build_mlp_regularized(X_train.shape[1], hidden_units=best_arch)
    _, y_pred, secs = fit_keras(m_no_cw, X_train, y_train, X_val, y_val, "mlp_best_no_cw", class_weight=None)
    evaluate(f"MLP + Dropout/BN {best_arch} (không class_weight)", y_val, y_pred, secs, "class_weight=None")

    # Bước 10: MLP + Embedding
    data, n_cat, n_num, _, _ = load_and_split_embedding(data_path)
    Xtr_e, ytr_e = data["train"]; Xval_e, yval_e = data["val"]
    emb = build_mlp_embedding(n_num, n_cat["region"], n_cat["channel"])
    _, y_pred_emb, secs = fit_keras(emb, Xtr_e, ytr_e, Xval_e, yval_e, "mlp_embedding",
                                     class_weight=compute_class_weight_dict(ytr_e))
    preds["MLP + Embedding"] = y_pred_emb
    emb_pr_auc = evaluate("MLP + Embedding", yval_e, y_pred_emb, secs, f"{emb.count_params()} tham số")

    plot_embedding_vs_onehot(best_model, best_onehot_pr_auc, emb, emb_pr_auc,
                              "reports/embedding_vs_onehot.png")

    # Bước 11-12: PR curve + bảng kết luận
    plot_pr_curves(y_val, preds, "reports/pr_curve.png")

    df = pd.DataFrame(RESULTS)
    print(df.to_string(index=False))
    df.to_csv("reports/ket_qua_so_sanh.csv", index=False)
    with open("reports/ket_luan.json", "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, ensure_ascii=False, indent=2)

    best = df.loc[df["pr_auc"].idxmax()]
    lgb_row = df[df["model"].str.contains("LightGBM")].iloc[0]
    print(f"\nModel tốt nhất: {best['model']} (PR-AUC={best['pr_auc']})")
    if best["model"] != lgb_row["model"] and best["pr_auc"] - lgb_row["pr_auc"] < 0.01:
        print("=> Chênh lệch với LightGBM không đáng kể — LightGBM vẫn thực dụng hơn "
              "(train nhanh, ít cần tinh chỉnh).")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/train.csv")
    main(p.parse_args().data)