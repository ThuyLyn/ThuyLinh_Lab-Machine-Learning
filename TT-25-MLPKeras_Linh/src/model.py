import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow.keras import layers, models, Model

def _compile(model: Model) -> Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.Precision(name="precision"),
        ],
    )
    return model

def build_mlp_basic(n_features: int) -> Model:
    model = models.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ], name="mlp_basic")
    return _compile(model)


def build_mlp_regularized(n_features: int, hidden_units=(128, 64),
                           dropout_rates=None) -> Model:
    if dropout_rates is None:
        base_rates = [0.3, 0.2, 0.1]
        dropout_rates = [base_rates[min(i, len(base_rates) - 1)] for i in range(len(hidden_units))]

    inputs = layers.Input(shape=(n_features,))
    x = inputs
    for units, rate in zip(hidden_units, dropout_rates):
        x = layers.Dense(units, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(rate)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    name = "mlp_" + "_".join(str(u) for u in hidden_units)
    model = Model(inputs, outputs, name=name)
    return _compile(model)


def build_mlp_embedding(n_numeric: int, n_region: int, n_channel: int,
                         region_dim: int = 8, channel_dim: int = 12,
                         hidden_units=(128, 64)) -> Model:
    region_input = layers.Input(shape=(1,), name="region")
    region_emb = layers.Embedding(n_region, region_dim, name="region_embedding")(region_input)
    region_emb = layers.Flatten()(region_emb)

    channel_input = layers.Input(shape=(1,), name="channel")
    channel_emb = layers.Embedding(n_channel, channel_dim, name="channel_embedding")(channel_input)
    channel_emb = layers.Flatten()(channel_emb)

    numeric_input = layers.Input(shape=(n_numeric,), name="numeric")

    x = layers.Concatenate()([region_emb, channel_emb, numeric_input])
    for units in hidden_units:
        x = layers.Dense(units, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = Model(
        inputs=[region_input, channel_input, numeric_input],
        outputs=outputs,
        name="mlp_embedding",
    )
    return _compile(model)


def get_callbacks(checkpoint_path: str = "models/best.keras"):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_pr_auc", mode="max", patience=10, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5
        ),
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor="val_pr_auc", mode="max", save_best_only=True
        ),
    ]