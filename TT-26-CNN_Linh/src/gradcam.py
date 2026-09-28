import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2


def find_last_conv_layer(model: tf.keras.Model, base_layer_name_hint="efficientnet") -> str:
    # Trường hợp model có 1 layer con là base network (Functional API)
    for layer in model.layers:
        if hasattr(layer, "layers"):  # đây là sub-model (vd EfficientNetB0)
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    return layer.name, sub_layer.name
    # Trường hợp model phẳng (baseline CNN)
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return None, layer.name
    raise ValueError("Không tìm thấy layer Conv2D nào trong model.")

def make_gradcam_heatmap(img_array, model, base_name, last_conv_name, pred_index=None):
    if base_name is not None:
        base_model = model.get_layer(base_name)
        # Cần build lại forward pass đầy đủ vì base là sub-model độc lập
        with tf.GradientTape() as tape:
            conv_out = base_model.get_layer(last_conv_name).output
            grad_model = tf.keras.models.Model(base_model.inputs, [conv_out, base_model.output])
            conv_output, base_output = grad_model(img_array)
            # Tiếp tục forward qua phần đầu phân loại (các layer sau base trong model gốc)
            x = base_output
            for layer in model.layers:
                if layer.name == base_name:
                    continue
                if layer.__class__.__name__ in ("InputLayer",):
                    continue
                x = layer(x)
            preds = x
            if pred_index is None:
                pred_index = 0
            class_channel = preds[:, pred_index]
        grads = tape.gradient(class_channel, conv_output)
    else:
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_name).output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = 0
            class_channel = preds[:, pred_index]
        grads = tape.gradient(class_channel, conv_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_heatmap(img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(img * 255) if img.max() <= 1.0 else np.uint8(img)
    superimposed = cv2.addWeighted(overlay, 1 - alpha, heatmap_color, alpha, 0)
    return superimposed

def run_gradcam_report(model_path, data_dir, out_dir, n_images=6, img_size=(224, 224)):
    model = tf.keras.models.load_model(model_path)
    base_name, last_conv_name = find_last_conv_layer(model)
    print(f"Dùng base_name={base_name}, last_conv_layer={last_conv_name}")

    test_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "test"),
        labels="inferred", label_mode="binary", color_mode="rgb",
        image_size=img_size, batch_size=1, shuffle=True, seed=123,
    )
    class_names = test_ds.class_names

    fig, axes = plt.subplots(2, n_images, figsize=(3 * n_images, 6))
    count = 0
    for x, y in test_ds.take(n_images * 3):
        if count >= n_images:
            break
        x_norm = x
        heatmap = make_gradcam_heatmap(x_norm, model, base_name, last_conv_name)
        img_np = x[0].numpy()
        overlay = overlay_heatmap(img_np, heatmap)

        axes[0, count].imshow(img_np.astype("uint8"))
        axes[0, count].set_title(f"Gốc: {class_names[int(y.numpy()[0][0])]}")
        axes[0, count].axis("off")

        axes[1, count].imshow(overlay)
        axes[1, count].set_title("Grad-CAM")
        axes[1, count].axis("off")

        count += 1

    plt.tight_layout()
    out_path = os.path.join(out_dir, "gradcam_examples.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Đã lưu Grad-CAM -> {out_path}")
    print(
        "\n KIỂM TRA BẰNG MẮT: vùng nóng (đỏ/vàng) có nằm TRONG vùng phổi không?\n"
        "   Nếu heatmap tập trung vào góc ảnh, chữ, hoặc viền phim -> model đang\n"
        "   'gian lận' bằng manh mối giả, KHÔNG học đặc trưng y khoa thật -> phải\n"
        "   ghi rõ điều này trong báo cáo và cân nhắc lại augmentation/crop ảnh."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", default="reports")
    parser.add_argument("--n_images", type=int, default=6)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    run_gradcam_report(args.model, args.data_dir, args.out_dir, args.n_images)

print("OK")