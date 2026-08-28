import time
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

RANDOM_STATE = 42
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "reports")
MODEL_DIR = os.path.join(BASE, "models")
os.makedirs(OUT, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# 1. LOAD DATA
from sklearn.datasets import fetch_openml
 
print("Đang tải MNIST từ OpenML (lần đầu sẽ cache vào ~/scikit_learn_data, mất vài phút)...")
X_all, y_all = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
y_all = y_all.astype(int)
 
IMG_SIZE = 28          # MNIST: ảnh 28x28 = 784 pixel
PIXEL_MAX = 255.0      # MNIST: pixel 0..255 (load_digits là 0..16)
 
X_raw, y = X_all, y_all
X_norm = X_raw / PIXEL_MAX
 
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
X_train_norm, X_test_norm, _, _ = train_test_split(
    X_norm, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
 
print(f"Dữ liệu: {X_raw.shape[0]} mẫu, {X_raw.shape[1]} pixel/ảnh ({IMG_SIZE}x{IMG_SIZE}), {len(np.unique(y))} lớp")
print(f"Train: {len(y_train)} | Test: {len(y_test)}\n")

# 3. BASELINE — Logistic Regression
lr = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
lr.fit(X_train_norm, y_train)
acc_lr = accuracy_score(y_test, lr.predict(X_test_norm))
print(f"[3] Baseline Logistic Regression: acc = {acc_lr:.4f}")

# 4 vs 5. MLP KHÔNG chuẩn hoá vs CÓ chuẩn hoá
def make_mlp(**kw):
    base = dict(hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
                alpha=1e-4, batch_size=128, learning_rate_init=1e-3, max_iter=300,
                early_stopping=True, n_iter_no_change=10, random_state=RANDOM_STATE)
    base.update(kw)
    return MLPClassifier(**base)

t0 = time.time()
mlp_raw = make_mlp()
mlp_raw.fit(X_train_raw, y_train)
acc_raw = accuracy_score(y_test, mlp_raw.predict(X_test_raw))
print(f"[4] MLP KHÔNG chuẩn hoá: acc = {acc_raw:.4f}  (iters={mlp_raw.n_iter_}, time={time.time()-t0:.1f}s)")

t0 = time.time()
mlp_norm = make_mlp()
mlp_norm.fit(X_train_norm, y_train)
acc_norm = accuracy_score(y_test, mlp_norm.predict(X_test_norm))
print(f"[5] MLP CÓ chuẩn hoá:   acc = {acc_norm:.4f}  (iters={mlp_norm.n_iter_}, time={time.time()-t0:.1f}s)\n")

# 6. So sánh 4 kiến trúc
architectures = [(64,), (128,), (128, 64), (256, 128, 64)]
def count_params(sizes, n_in=64, n_out=10):
    layers = [n_in] + list(sizes) + [n_out]
    return sum(layers[i]*layers[i+1] + layers[i+1] for i in range(len(layers)-1))

arch_results = []
for arch in architectures:
    t0 = time.time()
    m = make_mlp(hidden_layer_sizes=arch)
    m.fit(X_train_norm, y_train)
    acc = accuracy_score(y_test, m.predict(X_test_norm))
    dt = time.time() - t0
    n_params = count_params(arch)
    arch_results.append((arch, acc, n_params, dt))
    print(f"[6] Kiến trúc {arch}: acc={acc:.4f}, params={n_params}, time={dt:.1f}s")

best_acc, best_arch = -1, None
for arch, acc, n_params, dt in arch_results:
    if acc > best_acc:
        best_acc, best_arch = acc, arch
print(f"\n→ Kiến trúc tốt nhất trên tập test: {best_arch} (acc={best_acc:.4f})\n")

# biểu đồ so sánh kiến trúc: accuracy vs số tham số
fig, ax1 = plt.subplots(figsize=(7, 4.5))
labels = [str(a) for a, *_ in arch_results]
accs = [a for _, a, *_ in arch_results]
params = [p for *_, p, _ in arch_results]
x = np.arange(len(labels))
bars = ax1.bar(x, accs, color="#4C72B0", alpha=0.85, label="Accuracy")
ax1.set_ylabel("Accuracy"); ax1.set_ylim(0.85, 1.0)
ax1.set_xticks(x); ax1.set_xticklabels(labels)
ax1.set_xlabel("Kiến trúc (hidden_layer_sizes)")
for i, v in enumerate(accs):
    ax1.text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=9)
ax2 = ax1.twinx()
ax2.plot(x, params, color="#C44E52", marker="o", label="Số tham số")
ax2.set_ylabel("Số tham số")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")
plt.title("So sánh kiến trúc: Accuracy vs Số tham số")
plt.tight_layout()
plt.savefig(f"{OUT}/kien_truc_comparison.png", dpi=120)
plt.close()
print("→ Đã lưu kien_truc_comparison.png\n")

# dùng lại mlp_norm (128,64) làm model chính cho các bước sau
main_model = mlp_norm

# 7. Loss curve
plt.figure(figsize=(6,4))
plt.plot(main_model.loss_curve_)
plt.title("Loss curve — MLP (128,64), chuẩn hoá")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/loss_curves.png", dpi=120)
plt.close()
print("[7] Đã lưu loss_curves.png")

# 8. So sánh 3 activation
plt.figure(figsize=(6,4))
act_results = {}
for act in ["relu", "tanh", "logistic"]:
    m = make_mlp(activation=act, max_iter=300)
    m.fit(X_train_norm, y_train)
    acc = accuracy_score(y_test, m.predict(X_test_norm))
    act_results[act] = acc
    plt.plot(m.loss_curve_, label=f"{act} (acc={acc:.3f})")
    print(f"[8] activation={act}: acc={acc:.4f}, n_iter={m.n_iter_}")
plt.title("So sánh activation function")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/activation_comparison.png", dpi=120)
plt.close()
print("→ Đã lưu activation_comparison.png\n")

# 9. So sánh 3 learning rate
plt.figure(figsize=(6,4))
lr_results = {}
for lrate in [1e-2, 1e-3, 1e-4]:
    m = make_mlp(learning_rate_init=lrate, max_iter=300)
    m.fit(X_train_norm, y_train)
    acc = accuracy_score(y_test, m.predict(X_test_norm))
    lr_results[lrate] = acc
    plt.plot(m.loss_curve_, label=f"lr={lrate} (acc={acc:.3f})")
    print(f"[9] learning_rate={lrate}: acc={acc:.4f}")
plt.title("So sánh learning rate")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/learning_rate_comparison.png", dpi=120)
plt.close()
print("→ Đã lưu learning_rate_comparison.png\n")

# 10. Ma trận nhầm lẫn 10x10
y_pred = main_model.predict(X_test_norm)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
plt.imshow(cm, cmap="Blues")
plt.colorbar()
plt.xticks(range(10)); plt.yticks(range(10))
plt.xlabel("Dự đoán"); plt.ylabel("Thực tế")
plt.title("Ma trận nhầm lẫn 10x10")
for i in range(10):
    for j in range(10):
        if cm[i,j] > 0:
            plt.text(j, i, cm[i,j], ha="center", va="center",
                      color="white" if cm[i,j] > cm.max()/2 else "black", fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT}/confusion_10x10.png", dpi=120)
plt.close()

cm_off = cm.copy()
np.fill_diagonal(cm_off, 0)
idx = np.unravel_index(np.argmax(cm_off), cm_off.shape)
print(f"[10] Cặp số hay bị nhầm nhất: thực tế={idx[0]} → dự đoán={idx[1]} ({cm_off[idx]} lần)")
print("→ Đã lưu confusion_10x10.png\n")

# 11. Hiển thị các ảnh dự đoán sai
wrong_idx = np.where(y_pred != y_test)[0][:20]
n_show = len(wrong_idx)
if n_show > 0:
    cols = 5
    rows = (n_show + cols - 1) // cols
    plt.figure(figsize=(cols*2, rows*2.2))
    for i, widx in enumerate(wrong_idx):
        plt.subplot(rows, cols, i+1)
        plt.imshow(X_test_raw[widx].reshape(28,28), cmap="gray")
        plt.title(f"thật:{y_test[widx]} đoán:{y_pred[widx]}", fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{OUT}/anh_sai.png", dpi=120)
    plt.close()
print(f"[11] Có {n_show} ảnh sai trong batch đầu, đã lưu anh_sai.png\n")

# 12. Human-in-the-loop: ngưỡng tin cậy 99%
proba = main_model.predict_proba(X_test_norm)
confidence = proba.max(axis=1)
pred = proba.argmax(axis=1)

THRESH = 0.99
auto_mask = confidence >= THRESH
n_auto = auto_mask.sum()
n_manual = (~auto_mask).sum()
acc_auto = accuracy_score(y_test[auto_mask], pred[auto_mask]) if n_auto > 0 else float("nan")
acc_overall = accuracy_score(y_test, pred)

print("[12] HUMAN-IN-THE-LOOP (ngưỡng 99%)")
print(f"     Tổng test: {len(y_test)}")
print(f"     Tự động xử lý: {n_auto} ({100*n_auto/len(y_test):.1f}%)  — accuracy trong nhóm này: {acc_auto:.4f}")
print(f"     Chuyển người kiểm tra: {n_manual} ({100*n_manual/len(y_test):.1f}%)")
print(f"     Accuracy tổng thể (không lọc): {acc_overall:.4f}\n")

# 13. So sánh với CNN (TT-26)
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models

    # CNN cần input dạng ảnh 2D (H, W, 1), không trải phẳng như MLP
    X_train_img = X_train_norm.reshape(-1, 8, 8, 1)
    X_test_img = X_test_norm.reshape(-1, 8, 8, 1)

    cnn = models.Sequential([
        layers.Input(shape=(8, 8, 1)),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(10, activation="softmax"),
    ])
    cnn.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    t0 = time.time()
    history = cnn.fit(X_train_img, y_train, validation_split=0.1, epochs=30,
                       batch_size=32, verbose=0)
    time_cnn = time.time() - t0

    test_loss, acc_cnn = cnn.evaluate(X_test_img, y_test, verbose=0)
    n_params_cnn = cnn.count_params()

    print(f"[13] CNN: acc = {acc_cnn:.4f}, params={n_params_cnn}, time={time_cnn:.1f}s")
    print(f"     So với MLP (128,64) chuẩn hoá: {acc_norm:.4f}")
    print(f"     Chênh lệch: {acc_cnn - acc_norm:+.4f}")

    # biểu đồ so sánh loss/accuracy CNN theo epoch
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title("CNN — Loss theo epoch")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].bar(["MLP (128,64)", "CNN"], [acc_norm, acc_cnn], color=["#4C72B0", "#55A868"])
    axes[1].set_ylim(0.9, 1.0)
    axes[1].set_title("So sánh accuracy: MLP vs CNN")
    for i, v in enumerate([acc_norm, acc_cnn]):
        axes[1].text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{OUT}/mlp_vs_cnn.png", dpi=120)
    plt.close()
    print("→ Đã lưu mlp_vs_cnn.png\n")
    cnn_available = True
except Exception as e:
    print(f"[13] Bỏ qua so sánh CNN (lỗi: {e})\n")
    acc_cnn, n_params_cnn, time_cnn = None, None, None
    cnn_available = False

# LƯU MODEL PIPELINE (StandardScaler + MLP) — dùng khi triển khai thật
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", make_mlp(hidden_layer_sizes=best_arch)),
])
pipeline.fit(X_train_raw, y_train)  # scaler tự học mean/std, không cần chia 16 thủ công
acc_pipeline = accuracy_score(y_test, pipeline.predict(X_test_raw))
joblib.dump(pipeline, f"{MODEL_DIR}/mlp_pipeline.joblib")
print(f"[Model] Pipeline (StandardScaler + MLP {best_arch}) acc={acc_pipeline:.4f}")
print(f"→ Đã lưu models/mlp_pipeline.joblib\n")

# TỔNG KẾT
print("="*60)
print("TỔNG KẾT")
print("="*60)
print(f"Baseline Logistic Regression : {acc_lr:.4f}")
print(f"MLP không chuẩn hoá          : {acc_raw:.4f}")
print(f"MLP có chuẩn hoá (128,64)    : {acc_norm:.4f}")
print(f"Kiến trúc tốt nhất           : {best_arch} → {best_acc:.4f}")
print(f"Human-in-the-loop @99%       : {100*n_auto/len(y_test):.1f}% tự động, acc={acc_auto:.4f}")
if cnn_available:
    print(f"CNN vs MLP                   : CNN={acc_cnn:.4f} | MLP={acc_norm:.4f} | chênh lệch={acc_cnn-acc_norm:+.4f}")