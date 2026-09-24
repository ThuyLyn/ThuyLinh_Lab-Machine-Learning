import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

IMG_SIZE = (224, 224)
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]  # 0 = NORMAL, 1 = PNEUMONIA

# --------------------------------------------------------------------------
# 1. Đếm ảnh và phát hiện vấn đề của bộ dữ liệu
# --------------------------------------------------------------------------
def count_images(data_dir: str) -> pd.DataFrame:
    data_dir = pathlib.Path(data_dir)
    rows = []
    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
        for cls in CLASS_NAMES:
            cls_dir = split_dir / cls
            n = len(list(cls_dir.glob("*.jpeg"))) + len(list(cls_dir.glob("*.jpg"))) + \
                len(list(cls_dir.glob("*.png")))
            rows.append({"split": split, "class": cls, "count": n})

    df = pd.DataFrame(rows)
    print(df.pivot(index="split", columns="class", values="count"))

    val_total = df[df["split"] == "val"]["count"].sum()
    if 0 < val_total <= 20:
        print(f"\n CẢNH BÁO: tập val gốc chỉ có {val_total} ảnh -> quá nhỏ để tin cậy. "
              f"Sẽ tự tách lại validation từ train (xem split_dataframe()).")

    train_df = df[df["split"] == "train"]
    if len(train_df) == 2:
        n0, n1 = train_df["count"].values
        ratio = max(n0, n1) / (n0 + n1)
        if ratio > 0.65:
            print(f" CẢNH BÁO: mất cân bằng lớp trong train ({ratio:.0%} là lớp đa số) "
                  f"-> dùng class_weight khi train.")

    return df

# --------------------------------------------------------------------------
# 2. Gộp dữ liệu + tự tách lại validation
# --------------------------------------------------------------------------
def build_dataframe(data_dir: str, splits=("train", "val")) -> pd.DataFrame:
    data_dir = pathlib.Path(data_dir)
    records = []
    for split in splits:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
        for label_idx, cls in enumerate(CLASS_NAMES):
            for ext in ("*.jpeg", "*.jpg", "*.png"):
                for fp in (split_dir / cls).glob(ext):
                    records.append({"filepath": str(fp), "label": label_idx, "label_name": cls})
    return pd.DataFrame(records)

def build_test_dataframe(data_dir: str) -> pd.DataFrame:
    return build_dataframe(data_dir, splits=("test",))

def split_dataframe(df: pd.DataFrame, val_size: float = 0.15, random_state: int = 42):
    train_df, val_df = train_test_split(
        df, test_size=val_size, stratify=df["label"], random_state=random_state
    )
    print(f"Train mới: {len(train_df)} ảnh | Val mới: {len(val_df)} ảnh")
    print("Phân bố train:\n", train_df["label_name"].value_counts())
    print("Phân bố val:\n", val_df["label_name"].value_counts())
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

# --------------------------------------------------------------------------
# 3. class_weight cho mất cân bằng lớp
# --------------------------------------------------------------------------
def get_class_weights(labels) -> dict:
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
    print("class_weight:", class_weight)
    return class_weight


# --------------------------------------------------------------------------
# 4. tf.data pipeline
# --------------------------------------------------------------------------
def _load_and_preprocess(filepath, label, img_size=IMG_SIZE):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, img_size)
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

# Tạo augmentation layer MỘT LẦN ở ngoài tf.data.map()
rotation = tf.keras.layers.RandomRotation(
    factor=10 / 360,
    fill_mode="constant"
)

translation = tf.keras.layers.RandomTranslation(
    0.1,
    0.1,
    fill_mode="constant"
)

zoom = tf.keras.layers.RandomZoom(
    0.1,
    fill_mode="constant"
)


def _augment(img, label):
    # Thay đổi độ sáng ngẫu nhiên
    img = tf.image.random_brightness(img, max_delta=0.1)

    # Thay đổi độ tương phản ngẫu nhiên
    img = tf.image.random_contrast(
        img,
        lower=0.9,
        upper=1.1
    )

    # Xoay ảnh ngẫu nhiên ±10 độ
    img = rotation(img, training=True)

    # Dịch chuyển ảnh
    img = translation(img, training=True)

    # Zoom ảnh
    img = zoom(img, training=True)

    # Giới hạn pixel trong khoảng [0, 1]
    img = tf.clip_by_value(img, 0.0, 1.0)

    return img, label

def make_dataset(df: pd.DataFrame, batch_size=32, img_size=IMG_SIZE,
                  augment=False, shuffle=False) -> tf.data.Dataset:
    filepaths = df["filepath"].values
    labels = df["label"].values.astype("float32")

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=42)

    ds = ds.map(lambda fp, lb: _load_and_preprocess(fp, lb, img_size),
                num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# --------------------------------------------------------------------------
# 5. Hiển thị ảnh mẫu (dùng trong notebook, bước 3 của README)
# --------------------------------------------------------------------------
def show_samples(df: pd.DataFrame, n_per_class: int = 8, save_path: str = None):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, n_per_class, figsize=(2 * n_per_class, 4.5))
    for row_idx, cls in enumerate(CLASS_NAMES):
        subset = df[df["label_name"] == cls].sample(n_per_class, random_state=1)
        for col_idx, fp in enumerate(subset["filepath"]):
            img = tf.io.read_file(fp)
            img = tf.image.decode_jpeg(img, channels=3)
            axes[row_idx, col_idx].imshow(img.numpy())
            axes[row_idx, col_idx].axis("off")
            if col_idx == 0:
                axes[row_idx, col_idx].set_ylabel(cls)
        axes[row_idx, 0].set_title(cls, loc="left", fontsize=12, fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

@dataclass
class Datasets:
    train_ds: tf.data.Dataset
    val_ds: tf.data.Dataset
    test_ds: tf.data.Dataset
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    class_weight: dict

def prepare_all(data_dir: str, batch_size: int = 32, val_size: float = 0.15) -> Datasets:
    count_images(data_dir)

    full_df = build_dataframe(data_dir, splits=("train", "val"))
    train_df, val_df = split_dataframe(full_df, val_size=val_size)
    test_df = build_test_dataframe(data_dir)

    class_weight = get_class_weights(train_df["label"].values)

    train_ds = make_dataset(train_df, batch_size=batch_size, augment=True, shuffle=True)
    val_ds = make_dataset(val_df, batch_size=batch_size, augment=False, shuffle=False)
    test_ds = make_dataset(test_df, batch_size=batch_size, augment=False, shuffle=False)

    return Datasets(train_ds, val_ds, test_ds, train_df, val_df, test_df, class_weight)


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/chest_xray"
    prepare_all(data_dir)