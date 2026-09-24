import tensorflow as tf
INPUT_SHAPE = (224, 224, 3)
def _metrics():
    return [
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
    ]

# --------------------------------------------------------------------------
# 1. Baseline CNN tự xây (train từ đầu)
# --------------------------------------------------------------------------
def build_baseline_cnn(input_shape=INPUT_SHAPE) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),

        tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ], name="baseline_cnn")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=_metrics(),
    )
    return model

# --------------------------------------------------------------------------
# 2. Transfer Learning - EfficientNetB0
# --------------------------------------------------------------------------
def build_transfer_model(input_shape=INPUT_SHAPE) -> tf.keras.Model:
    base = tf.keras.applications.EfficientNetB0(
        input_shape=input_shape, include_top=False, weights="imagenet"
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Lambda(lambda t: t * 255.0)(inputs)
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs, name="efficientnetb0_transfer")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=_metrics(),
    )
    # lưu tham chiếu tới base để dùng ở giai đoạn fine-tune
    model.base_model = base
    return model


def unfreeze_for_finetune(model: tf.keras.Model, n_unfreeze: int = 30, lr: float = 1e-5):
    base = model.base_model
    base.trainable = True
    for layer in base.layers[:-n_unfreeze]:
        layer.trainable = False

    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=_metrics(),
    )
    print(f"Đã mở khoá {n_unfreeze} tầng cuối của base. lr fine-tune = {lr}")
    return model


def get_default_callbacks(checkpoint_path: str, monitor="val_recall", patience=6):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor=monitor, mode="max",
            save_best_only=True, save_weights_only=False
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor, mode="max", patience=patience,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7
        ),

 ]


print("OK")