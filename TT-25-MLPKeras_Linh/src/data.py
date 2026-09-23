import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

RANDOM_STATE = 42

NUMERIC_COLS = ["Age", "Annual_Premium_log", "Vintage"]
BINARY_COLS = ["Driving_License", "Previously_Insured"]
CATEGORICAL_ONEHOT_COLS = ["Gender", "Vehicle_Age", "Vehicle_Damage"]
HIGH_CARD_COLS = ["Region_Code", "Policy_Sales_Channel"]  # nhiều mức -> one-hot hoặc embedding
TARGET_COL = "Response"


def _basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Annual_Premium_log"] = np.log1p(df["Annual_Premium"])
    for col in CATEGORICAL_ONEHOT_COLS:
        if df[col].dtype == object:
            df[col] = df[col].str.strip()
    return df

def _split_three_way(X: pd.DataFrame, y: pd.Series):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def load_and_split_onehot(csv_path: str):
    df = pd.read_csv(csv_path)
    df = _basic_clean(df)

    cat_cols_all = CATEGORICAL_ONEHOT_COLS + HIGH_CARD_COLS
    df_encoded = pd.get_dummies(df, columns=cat_cols_all, drop_first=True)

    feature_cols = [c for c in df_encoded.columns if c not in (TARGET_COL, "id", "Annual_Premium")]
    X = df_encoded[feature_cols]
    y = df_encoded[TARGET_COL]

    X_train, X_val, X_test, y_train, y_val, y_test = _split_three_way(X, y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    return (
        X_train_s, X_val_s, X_test_s,
        y_train.to_numpy(), y_val.to_numpy(), y_test.to_numpy(),
        scaler, feature_cols,
    )


def load_and_split_embedding(csv_path: str):
    df = pd.read_csv(csv_path)
    df = _basic_clean(df)

    encoders = {}
    for col in HIGH_CARD_COLS:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_ONEHOT_COLS, drop_first=True)

    numeric_feature_cols = [
        c for c in df_encoded.columns
        if c not in (TARGET_COL, "id", "Annual_Premium", *HIGH_CARD_COLS,
                     f"{HIGH_CARD_COLS[0]}_enc", f"{HIGH_CARD_COLS[1]}_enc")
    ]

    y = df_encoded[TARGET_COL]
    region = df_encoded[f"{HIGH_CARD_COLS[0]}_enc"]
    channel = df_encoded[f"{HIGH_CARD_COLS[1]}_enc"]
    numeric = df_encoded[numeric_feature_cols]

    idx_train, idx_temp = train_test_split(
        df_encoded.index, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=0.50, stratify=y.loc[idx_temp], random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    numeric_train = scaler.fit_transform(numeric.loc[idx_train])
    numeric_val = scaler.transform(numeric.loc[idx_val])
    numeric_test = scaler.transform(numeric.loc[idx_test])

    def pack(idx, numeric_arr):
        return {
            "region": region.loc[idx].to_numpy(),
            "channel": channel.loc[idx].to_numpy(),
            "numeric": numeric_arr,
        }

    data = {
        "train": (pack(idx_train, numeric_train), y.loc[idx_train].to_numpy()),
        "val": (pack(idx_val, numeric_val), y.loc[idx_val].to_numpy()),
        "test": (pack(idx_test, numeric_test), y.loc[idx_test].to_numpy()),
    }

    n_categories = {
        "region": df[f"{HIGH_CARD_COLS[0]}_enc"].nunique(),
        "channel": df[f"{HIGH_CARD_COLS[1]}_enc"].nunique(),
    }
    n_numeric = len(numeric_feature_cols)
    return data, n_categories, n_numeric, scaler, encoders


def compute_class_weight_dict(y_train: np.ndarray) -> dict:
    from sklearn.utils.class_weight import compute_class_weight

    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    return {0: weights[0], 1: weights[1]}


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/train.csv"
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, cols = load_and_split_onehot(path)
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"Tỉ lệ Response=1 (train): {y_train.mean():.3f}")
    print(f"Số cột sau one-hot: {len(cols)}")
    