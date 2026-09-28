import os
from pathlib import Path
from collections import Counter

import numpy as np
import tensorflow as tf

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
VAL_SPLIT = 0.15  # tự tách từ train, KHÔNG dùng thư mục val/ gốc (chỉ 16 ảnh)


def count_images(data_dir: str) -> dict:
    data_dir = Path(data_dir)
    counts = {}
    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
        counts[split] = {}
        for cls in ["NORMAL", "PNEUMONIA"]:
            cls_dir = split_dir / cls
            if cls_dir.exists():
                n = len([f for f in cls_dir.iterdir() if f.suffix.lower() in (".jpeg", ".jpg", ".png")])
                counts[split][cls] = n
    return counts

def print_dataset_report(data_dir: str) -> None:
    counts = count_images(data_dir)
    print("=== BÁO CÁO SỐ LƯỢNG ẢNH ===")
    for split, cls_counts in counts.items():
        total = sum(cls_counts.values())
        line = " | ".join(f"{k}: {v}" for k, v in cls_counts.items())
        print(f"{split:6s} (n={total:5d}): {line}")

    if "val" in counts and sum(counts["val"].values()) < 50:
        print("\n  Tập val gốc quá nhỏ (< 50 ảnh) -> KHÔNG dùng để chọn model.")
    if "train" in counts:
        n_normal = counts["train"].get("NORMAL", 0)
        n_pneu = counts["train"].get("PNEUMONIA", 0)
        if n_normal and n_pneu:
            ratio = n_pneu / (n_normal + n_pneu)
            print(f"  Mất cân bằng lớp trong train: PNEUMONIA chiếm {ratio:.1%}.")
    if "train" in counts and "test" in counts:
        train_ratio = counts["train"].get("PNEUMONIA", 0) / max(sum(counts["train"].values()), 1)
        test_ratio = counts["test"].get("PNEUMONIA", 0) / max(sum(counts["test"].values()), 1)
        print(f"  Phân phối lệch: train PNEUMONIA={train_ratio:.1%} vs test PNEUMONIA={test_ratio:.1%}.")


def build_datasets(data_dir: str, img_size=IMG_SIZE, batch_size=BATCH_SIZE, val_split=VAL_SPLIT, seed=SEED):
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="binary",
        color_mode="rgb",
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        validation_split=val_split,
        subset="training",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="binary",
        color_mode="rgb",
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,   # PHẢI True (cùng seed với train) thì 2 tập mới là phần bù của nhau
        seed=seed,
        validation_split=val_split,
        subset="validation",
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="binary",
        color_mode="rgb",
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False,
    )

    class_names = train_ds.class_names  # ['NORMAL', 'PNEUMONIA'] theo thứ tự alphabet

    # Kiểm tra validation có đủ cả 2 lớp (tránh lỗi tập val chỉ chứa 1 lớp)
    val_labels = np.concatenate([y.numpy() for _, y in val_ds], axis=0).flatten().astype(int)
    n_norm, n_pneu = int((val_labels == 0).sum()), int((val_labels == 1).sum())
    print(f"Validation tự tách: {len(val_labels)} ảnh — NORMAL={n_norm}, PNEUMONIA={n_pneu}")
    assert n_norm > 0 and n_pneu > 0, "Validation chỉ có 1 lớp -> kết quả val/threshold vô nghĩa!"

    # Bước 2b: class_weight để bù mất cân bằng (tính từ nhãn thật trong train_ds)
    labels = np.concatenate([y.numpy() for _, y in train_ds], axis=0).flatten()
    counter = Counter(labels.astype(int))
    total = sum(counter.values())
    class_weight = {cls: total / (len(counter) * cnt) for cls, cnt in counter.items()}

    # Augmentation CHỈ áp dụng cho train, và KHÔNG lật ngang (RandomFlip("horizontal"))
    # vì lật ngang sai giải phẫu (tim nằm bên trái).
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(10 / 360),      # ±10 độ
        tf.keras.layers.RandomTranslation(0.10, 0.10),  # dịch ±10%
        tf.keras.layers.RandomZoom(0.10),               # phóng to/thu nhỏ ±10%
        tf.keras.layers.RandomContrast(0.10),           # tương phản nhẹ
    ], name="medical_safe_augmentation")

    # KHÔNG chia 255 ở đây: EfficientNetB0 đã có lớp chuẩn hoá bên trong, nhận ảnh thang 0-255.
    # Baseline CNN tự chia 255 bằng lớp Rescaling nằm trong model.

    def prep_train(x, y):
        x = augmentation(x, training=True)
        return x, y

    def prep_eval(x, y):
        return x, y

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.map(prep_train, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    val_ds = val_ds.map(prep_eval, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    test_ds = test_ds.map(prep_eval, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_weight, class_names


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/chest_xray"
    print_dataset_report(data_dir)