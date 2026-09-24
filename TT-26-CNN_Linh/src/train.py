import argparse
import json
import os

import matplotlib.pyplot as plt
import tensorflow as tf

from data_pipeline import prepare_all
from model import (
    build_baseline_cnn,
    build_transfer_model,
    unfreeze_for_finetune,
    get_default_callbacks,
)

def plot_history(histories: dict, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    offset = 0
    for name, hist in histories.items():
        epochs = range(offset, offset + len(hist["loss"]))
        axes[0].plot(epochs, hist["loss"], label=f"{name} - train")
        axes[0].plot(epochs, hist["val_loss"], "--", label=f"{name} - val")
        axes[1].plot(epochs, hist["recall"], label=f"{name} - train")
        axes[1].plot(epochs, hist["val_recall"], "--", label=f"{name} - val")
        offset += len(hist["loss"])

    axes[0].set_title("Loss theo epoch")
    axes[0].set_xlabel("epoch")
    axes[0].legend(fontsize=8)

    axes[1].set_title("Recall theo epoch")
    axes[1].set_xlabel("epoch")
    axes[1].axhline(0.97, color="red", linestyle=":", label="mục tiêu 0.97")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Đã lưu learning curves tại: {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "--data_dir",
    default="data/chest_xray",
    help="thư mục chứa train/val/test"
)
    parser.add_argument("--epochs_baseline", type=int, default=15)
    parser.add_argument("--epochs_stage1", type=int, default=10)
    parser.add_argument("--epochs_stage2", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_unfreeze", type=int, default=30)
    parser.add_argument("--out_dir", default=".")
    args = parser.parse_args()

    models_dir = os.path.join(args.out_dir, "models")
    reports_dir = os.path.join(args.out_dir, "reports")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Bước 1-4: dữ liệu
    # ------------------------------------------------------------------
    ds = prepare_all(args.data_dir, batch_size=args.batch_size)

    all_histories = {}
    results_summary = {}

    # ------------------------------------------------------------------
    # Bước 5: Baseline CNN train từ đầu
    # ------------------------------------------------------------------
    print("\n=== [1/3] Baseline CNN (train từ đầu) ===")
    baseline = build_baseline_cnn()
    baseline_ckpt = os.path.join(models_dir, "baseline_cnn.keras")
    hist_baseline = baseline.fit(
        ds.train_ds, validation_data=ds.val_ds,
        epochs=args.epochs_baseline,
        class_weight=ds.class_weight,
        callbacks=get_default_callbacks(baseline_ckpt),
    )
    all_histories["baseline"] = hist_baseline.history
    results_summary["baseline"] = {
        k: float(v[-1]) for k, v in hist_baseline.history.items() if k.startswith("val_")
    }

    # ------------------------------------------------------------------
    # Bước 6: Transfer Learning giai đoạn 1 (đóng băng base)
    # ------------------------------------------------------------------
    print("\n=== [2/3] Transfer Learning - Giai đoạn 1 (base đóng băng) ===")
    transfer_model = build_transfer_model()
    stage1_ckpt = os.path.join(models_dir, "transfer_stage1.keras")
    hist_stage1 = transfer_model.fit(
        ds.train_ds, validation_data=ds.val_ds,
        epochs=args.epochs_stage1,
        class_weight=ds.class_weight,
        callbacks=get_default_callbacks(stage1_ckpt),
    )
    all_histories["stage1_frozen"] = hist_stage1.history
    results_summary["stage1_frozen"] = {
        k: float(v[-1]) for k, v in hist_stage1.history.items() if k.startswith("val_")
    }

    # ------------------------------------------------------------------
    # Bước 7: Fine-tuning giai đoạn 2 (lr nhỏ hơn 100 lần)
    # ------------------------------------------------------------------
    print("\n=== [3/3] Fine-tuning - Giai đoạn 2 (lr=1e-5) ===")
    transfer_model = unfreeze_for_finetune(transfer_model, n_unfreeze=args.n_unfreeze, lr=1e-5)
    best_ckpt = os.path.join(models_dir, "best_model.keras")
    hist_stage2 = transfer_model.fit(
        ds.train_ds, validation_data=ds.val_ds,
        epochs=args.epochs_stage2,
        class_weight=ds.class_weight,
        callbacks=get_default_callbacks(best_ckpt),
    )
    all_histories["stage2_finetune"] = hist_stage2.history
    results_summary["stage2_finetune"] = {
        k: float(v[-1]) for k, v in hist_stage2.history.items() if k.startswith("val_")
    }

    # ------------------------------------------------------------------
    # Bước 8: Learning curves + bảng so sánh
    # ------------------------------------------------------------------
    plot_history(all_histories, os.path.join(reports_dir, "learning_curves.png"))

    with open(os.path.join(reports_dir, "training_summary.json"), "w", encoding="utf-8") as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)

    print("\n=== SO SÁNH 3 MÔ HÌNH (trên val, epoch cuối) ===")
    for name, metrics in results_summary.items():
        print(f"{name:18s} | val_recall={metrics.get('val_recall', 0):.3f} | "
              f"val_auc={metrics.get('val_auc', 0):.3f} | "
              f"val_accuracy={metrics.get('val_accuracy', 0):.3f}")

    print(f"\nModel tốt nhất (dùng để đánh giá test) đã lưu tại: {best_ckpt}")
    print("Tiếp theo: chạy evaluate.py để chọn ngưỡng recall>=0.97 và đánh giá trên tập test.")


if __name__ == "__main__":
    main()