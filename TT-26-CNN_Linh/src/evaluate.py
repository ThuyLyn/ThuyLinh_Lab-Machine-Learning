import argparse
import os
import json

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def collect_predictions(model, data_dir, img_size=(224, 224)):
    test_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "test"),
        labels="inferred", label_mode="binary", color_mode="rgb",
        image_size=img_size, batch_size=32, shuffle=False,
    )
    class_names = test_ds.class_names

    images, y_true, y_prob = [], [], []
    for x, y in test_ds:
        x_norm = x
        p = model.predict(x_norm, verbose=0)
        images.append(x.numpy())
        y_true.append(y.numpy())
        y_prob.append(p)

    images = np.concatenate(images)
    y_true = np.concatenate(y_true).flatten()
    y_prob = np.concatenate(y_prob).flatten()
    return images, y_true, y_prob, class_names

def show_misclassified(images, y_true, y_prob, threshold, class_names, out_dir, n=10):
    y_pred = (y_prob >= threshold).astype(int)
    wrong_idx = np.where(y_pred != y_true)[0]

    # Ưu tiên false negative (viêm phổi bị bỏ sót) lên trước vì nguy hiểm hơn
    fn_idx = [i for i in wrong_idx if y_true[i] == 1 and y_pred[i] == 0]
    fp_idx = [i for i in wrong_idx if y_true[i] == 0 and y_pred[i] == 1]
    ordered = fn_idx + fp_idx
    ordered = ordered[:n]

    if not ordered:
        print("Không có ca sai nào trong tập test ở ngưỡng này (hiếm, kiểm tra lại).")
        return

    cols = 5
    rows = int(np.ceil(len(ordered) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.5 * rows))
    axes = np.array(axes).reshape(-1)

    for ax_i, idx in enumerate(ordered):
        ax = axes[ax_i]
        ax.imshow(images[idx].astype("uint8"))
        true_label = class_names[int(y_true[idx])]
        pred_label = class_names[int(y_pred[idx])]
        error_type = "BỎ SÓT (nguy hiểm)" if (y_true[idx] == 1 and y_pred[idx] == 0) else "báo động giả"
        ax.set_title(f"Thật:{true_label} | Đoán:{pred_label}\np={y_prob[idx]:.2f} ({error_type})", fontsize=8)
        ax.axis("off")

    for ax_i in range(len(ordered), len(axes)):
        axes[ax_i].axis("off")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "ca_du_doan_sai.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Đã lưu {len(ordered)} ca dự đoán sai -> {out_path}")
    print(f"  Trong đó: {len(fn_idx)} ca BỎ SÓT viêm phổi (false negative), "
          f"{len(fp_idx)} ca báo động giả (false positive)")
    print(
        "\ Gợi ý phân tích nguyên nhân :\n"
        "  - Ảnh mờ, chụp lệch tư thế, hoặc chất lượng thấp?\n"
        "  - Vùng tổn thương quá nhỏ/mờ so với độ phân giải 224x224?\n"
        "  - Nhãn gốc trong dataset có khả năng bị gán sai (label noise)?\n"
        "  - Model có bị 'đánh lừa' bởi chi tiết không liên quan (đối chiếu với Grad-CAM)?"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.5,
                         help="Dùng đúng ngưỡng đã chọn ở bước 9 (train.py in ra / lưu trong final_results.json)")
    parser.add_argument("--out_dir", default="reports")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model = tf.keras.models.load_model(args.model)
    images, y_true, y_prob, class_names = collect_predictions(model, args.data_dir)
    show_misclassified(images, y_true, y_prob, args.threshold, class_names, args.out_dir, n=10)


if __name__ == "__main__":
    main()