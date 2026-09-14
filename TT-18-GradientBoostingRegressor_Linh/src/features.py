import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

# ---------------------------------------------------------------------------
# 1. Danh sách cột — theo data_description.txt của cuộc thi Kaggle
# ---------------------------------------------------------------------------
# NaN ở các cột này = "nhà KHÔNG CÓ tiện ích đó" -> điền chuỗi 'None'
NONE_COLS = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "MasVnrType",
]
# NaN ở các cột số này cũng có nghĩa "không có" -> điền 0
ZERO_COLS = [
    "GarageYrBlt", "GarageArea", "GarageCars", "MasVnrArea",
    "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath",
]
# Các cột biến CHẤT LƯỢNG có thứ tự rõ ràng -> OrdinalEncoder, KHÔNG one-hot
QUAL_COLS = [
    "ExterQual", "ExterCond", "BsmtQual", "BsmtCond",
    "HeatingQC", "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond",
]
QUAL_ORDER = ["None", "Po", "Fa", "TA", "Gd", "Ex"]

TARGET_COL = "SalePrice"
ID_COL = "Id"
# ---------------------------------------------------------------------------
# 2. Làm sạch dữ liệu (xử lý NaN)
# ---------------------------------------------------------------------------

def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing_before = df.isnull().sum()
    missing_before = missing_before[missing_before > 0]
    n_none_type = len([c for c in NONE_COLS + ZERO_COLS if c in missing_before.index])
    print(f"[load_and_clean] Số cột có giá trị thiếu: {len(missing_before)}")
    print(f"[load_and_clean] Trong đó 'thiếu vì không có tiện ích': {n_none_type} cột")
    print(f"[load_and_clean] 'Thiếu thật' còn lại: {len(missing_before) - n_none_type} cột")

    # Nhóm 1: NaN nghĩa là "không có" (categorical) -> 'None'
    for col in NONE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    # Nhóm 2: NaN nghĩa là "không có" (numeric) -> 0
    for col in ZERO_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Nhóm 3: thiếu thật -> điền hợp lý theo từng cột
    if "LotFrontage" in df.columns:
        df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
            lambda x: x.fillna(x.median())
        )
        df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())

    if "Electrical" in df.columns:
        df["Electrical"] = df["Electrical"].fillna(df["Electrical"].mode()[0])

    if "MSZoning" in df.columns:
        df["MSZoning"] = df["MSZoning"].fillna(df["MSZoning"].mode()[0])

    if "Functional" in df.columns:
        df["Functional"] = df["Functional"].fillna("Typ")

    if "SaleType" in df.columns:
        df["SaleType"] = df["SaleType"].fillna(df["SaleType"].mode()[0])

    if "KitchenQual" in df.columns:
        df["KitchenQual"] = df["KitchenQual"].fillna(df["KitchenQual"].mode()[0])

    if "Exterior1st" in df.columns:
        df["Exterior1st"] = df["Exterior1st"].fillna(df["Exterior1st"].mode()[0])
    if "Exterior2nd" in df.columns:
        df["Exterior2nd"] = df["Exterior2nd"].fillna(df["Exterior2nd"].mode()[0])

    # Bất kỳ cột object nào còn sót NaN -> 'None'; cột số còn sót -> median
    for col in df.select_dtypes(include="object").columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna("None")
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    remaining = df.isnull().sum().sum()
    print(f"[load_and_clean] Giá trị thiếu còn lại sau xử lý: {remaining}")
    return df

# ---------------------------------------------------------------------------
# 3. Feature engineering
# ---------------------------------------------------------------------------
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if {"1stFlrSF", "2ndFlrSF", "TotalBsmtSF"}.issubset(df.columns):
        df["TotalSF"] = df["1stFlrSF"] + df["2ndFlrSF"] + df["TotalBsmtSF"]

    if {"YrSold", "YearBuilt"}.issubset(df.columns):
        df["TuoiNha"] = df["YrSold"] - df["YearBuilt"]
        df["TuoiNha"] = df["TuoiNha"].clip(lower=0)  # phòng dữ liệu lỗi (bán trước khi xây)

    if {"YearRemodAdd", "YearBuilt"}.issubset(df.columns):
        df["DaSuaChua"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int)

    if {"FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"}.issubset(df.columns):
        df["TotalBath"] = (
            df["FullBath"] + 0.5 * df["HalfBath"]
            + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"]
        )

    return df
# ---------------------------------------------------------------------------
# 4. Encoding (ordinal + one-hot) — fit trên train, transform nhất quán trên test
# ---------------------------------------------------------------------------

class FeatureEncoder:
    def __init__(self):
        self.ordinal_encoders = {}
        self.dummy_columns = None  # danh sách cột sau one-hot lúc fit

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Ordinal encode các cột chất lượng
        for col in QUAL_COLS:
            if col in df.columns:
                oe = OrdinalEncoder(
                    categories=[QUAL_ORDER],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                )
                df[col] = oe.fit_transform(df[[col]])
                self.ordinal_encoders[col] = oe

        # One-hot các cột danh mục còn lại
        cat_cols = [c for c in df.select_dtypes(include="object").columns]
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

        self.dummy_columns = df.columns.tolist()
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        for col, oe in self.ordinal_encoders.items():
            if col in df.columns:
                df[col] = oe.transform(df[[col]])

        cat_cols = [c for c in df.select_dtypes(include="object").columns]
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

        # Đồng bộ cột với lúc fit (thêm cột thiếu = 0, bỏ cột thừa)
        df = df.reindex(columns=self.dummy_columns, fill_value=0)
        return df

# ---------------------------------------------------------------------------
# 5. Hàm tiện ích tổng hợp toàn bộ pipeline
# ---------------------------------------------------------------------------

def build_feature_pipeline(df: pd.DataFrame, encoder: "FeatureEncoder" = None, fit: bool = True):
    df = add_engineered_features(df)

    y_log = None
    if TARGET_COL in df.columns:
        y = df[TARGET_COL]
        y_log = np.log1p(y)
        df = df.drop(columns=[TARGET_COL])

    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])

    if encoder is None:
        encoder = FeatureEncoder()

    X = encoder.fit_transform(df) if fit else encoder.transform(df)
    return X, y_log, encoder