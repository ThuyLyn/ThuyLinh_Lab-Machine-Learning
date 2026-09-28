import argparse
import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf

from src.data_pipeline import build_datasets, print_dataset_report
from src.model import build_baseline_cnn, build_transfer_model, unfreeze_for_finetune


def plot_history(histories: dict, out_path: str):
    """histories: {"baseline": history, "transfer_frozen": history, "finetune": history}"""
    metrics = ["loss", "recall", "auc"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4))
    for ax, m in zip(axes, metrics):
        for name, hist in histories.items():
            if m in hist.history:
                ax.plot(hist.history[m], label=f"{name} (train)")
            if f"val_{m}" in hist.history:
                ax.plot(hist.history[f"val_{m}"], "--", label=f"{name} (val)")
        ax.set_title(m)
        ax.set_xlabel("epoch")
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Đã lưu learning curves -> {out_path}")

def select_threshold_for_recall(y_true, y_prob, target_recall=0.97):
    thresholds = np.linspace(0.01, 0.99, 99)
    best_t = 0.5
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        recall = tp / max(tp + fn, 1)
        if recall >= target_recall:
            best_t = t  # giữ ngưỡng CAO NHẤT vẫn thoả recall (giảm báo động giả)
    return best_t

def evaluate_at_threshold(y_true, y_prob, threshold, class_names, out_dir, tag="test"):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)

    tn, fp, fn, tp = cm.ravel()
    print(f"\n=== Kết quả trên {tag} (ngưỡng={threshold:.2f}) ===")
    print(report)
    print(f" SỐ CA VIÊM PHỔI BỊ BỎ SÓT (False Negative): {fn} / {fn + tp}")
    if tn == 0 or tp == 0:
        print(" CẢNH BÁO: model đoán TOÀN BỘ ảnh cùng một lớp -> model suy biến, kết quả KHÔNG hợp lệ. Xem lại quá trình train.")
    print(f"    Số báo động giả (False Positive): {fp} / {fp + tn}")

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(class_names)
    ax.set_yticks([0, 1]); ax.set_yticklabels(class_names)
    ax.set_xlabel("Dự đoán"); ax.set_ylabel("Thực tế")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(f"Ma trận nhầm lẫn ({tag}, ngưỡng={threshold:.2f})")
    plt.colorbar(im)
    plt.tight_layout()
    cm_path = os.path.join(out_dir, f"confusion_matrix_{tag}.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Đã lưu ma trận nhầm lẫn -> {cm_path}")

    return {"threshold": float(threshold), "fn": int(fn), "fp": int(fp), "tp": int(tp), "tn": int(tn)}


def get_labels_and_probs(model, dataset):
    y_true, y_prob = [], []
    for x, y in dataset:
        p = model.predict(x, verbose=0)
        y_true.append(y.numpy())
        y_prob.append(p)
    return np.concatenate(y_true).flatten(), np.concatenate(y_prob).flatten()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Thư mục chest_xray/ (chứa train/, val/, test/)")
    parser.add_argument("--out_dir", default="reports")
    parser.add_argument("--models_dir", default="models")
    parser.add_argument("--epochs_baseline", type=int, default=15)
    parser.add_argument("--epochs_frozen", type=int, default=10)
    parser.add_argument("--epochs_finetune", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)

    # Bước 1: đếm ảnh -> phát hiện 3 vấn đề
    print_dataset_report(args.data_dir)

    # Bước 2-4: dataset + class_weight + augmentation (không lật ngang)
    train_ds, val_ds, test_ds, class_weight, class_names = build_datasets(args.data_dir)
    print(f"class_names = {class_names}, class_weight = {class_weight}")

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=4, restore_best_weights=True
    )

    histories = {}

    # Bước 5: Baseline CNN train từ đầu
    print("\n### GIAI ĐOẠN 0: Baseline CNN train từ đầu ###")
    baseline = build_baseline_cnn()
    hist_baseline = baseline.fit(
        train_ds, validation_data=val_ds, epochs=args.epochs_baseline,
        class_weight=class_weight, callbacks=[early_stop],
    )
    histories["baseline"] = hist_baseline

    # Bước 6: Transfer learning giai đoạn 1 (đóng băng base)
    print("\n### GIAI ĐOẠN 1: Transfer Learning (base đóng băng) ###")
    model, base = build_transfer_model()
    hist_frozen = model.fit(
        train_ds, validation_data=val_ds, epochs=args.epochs_frozen,
        class_weight=class_weight, callbacks=[early_stop],
    )
    histories["transfer_frozen"] = hist_frozen

    # Bước 7: Fine-tuning giai đoạn 2 (lr nhỏ hơn 100 lần)
    print("\n### GIAI ĐOẠN 2: Fine-tuning (mở khoá 30 layer cuối, lr=1e-5) ###")
    model = unfreeze_for_finetune(model, base, n_last_trainable=30, lr=1e-5)
    hist_finetune = model.fit(
        train_ds, validation_data=val_ds, epochs=args.epochs_finetune,
        class_weight=class_weight, callbacks=[early_stop],
    )
    histories["finetune"] = hist_finetune

    # Bước 8: learning curves
    plot_history(histories, os.path.join(args.out_dir, "learning_curves.png"))

    # Bước 9: chọn ngưỡng trên VALIDATION để đạt recall >= 0.97
    y_val_true, y_val_prob = get_labels_and_probs(model, val_ds)
    threshold = select_threshold_for_recall(y_val_true, y_val_prob, target_recall=0.97)
    print(f"\nNgưỡng chọn trên validation để recall >= 0.97: {threshold:.3f}")

    # Bước 10: đánh giá TEST 1 LẦN DUY NHẤT ở ngưỡng đã chọn
    y_test_true, y_test_prob = get_labels_and_probs(model, test_ds)
    test_result = evaluate_at_threshold(
        y_test_true, y_test_prob, threshold, class_names, args.out_dir, tag="test"
    )

    model.save(os.path.join(args.models_dir, "best_model.keras"))

    with open(os.path.join(args.out_dir, "final_results.json"), "w") as f:
        json.dump({"threshold": threshold, "test": test_result, "class_names": class_names}, f, indent=2)

    print("\n Hoàn tất. Xem reports/ để lấy learning_curves.png, confusion_matrix_test.png, final_results.json")
    print(" Tiếp theo: chạy src/gradcam.py để sinh ảnh Grad-CAM (bước 11) và")
    print("    src/evaluate.py để phân tích 10 ca dự đoán sai (bước 12).")


if __name__ == "__main__":
    main()