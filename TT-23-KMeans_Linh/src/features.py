import numpy as np
import pandas as pd
NON_PRODUCT_CODES = {
    "POST", "M", "BANK CHARGES", "DOT", "ADJUST", "ADJUST2",
    "C2", "D", "S", "AMAZONFEE", "TEST001", "TEST002", "PADS", "CRUK",
}

MIN_ORDERS_PER_PRODUCT = 5  # loại sản phẩm bán quá ít (< 5 đơn hàng)


def load_raw(path_xlsx: str) -> pd.DataFrame:
    """Đọc và gộp 2 sheet (2009-2010, 2010-2011) của file Online Retail II."""
    sheets = pd.read_excel(path_xlsx, sheet_name=None)
    df = pd.concat(sheets.values(), ignore_index=True)
    return df


def clean_transactions(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    log = []
    n0 = len(df)
    df = df.copy()
    df["Invoice"] = df["Invoice"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str).str.upper().str.strip()

    # 1. Loại hoá đơn huỷ (Invoice bắt đầu bằng 'C')
    n_before = len(df)
    df = df[~df["Invoice"].str.startswith("C")]
    log.append(("Loại hoá đơn huỷ (Invoice bắt đầu 'C')", n_before - len(df)))

    # 2. Loại Quantity <= 0 và Price <= 0
    n_before = len(df)
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
    log.append(("Loại Quantity <= 0 hoặc Price <= 0", n_before - len(df)))

    # 3. Loại StockCode không phải sản phẩm
    n_before = len(df)
    df = df[~df["StockCode"].isin(NON_PRODUCT_CODES)]
    log.append(("Loại StockCode không phải sản phẩm (POST, M, DOT, ...)", n_before - len(df)))

    # 4. Loại dòng thiếu Description hoàn toàn (không đủ để mô tả sản phẩm)
    n_before = len(df)
    df = df[df["Description"].notna()]
    log.append(("Loại dòng thiếu Description", n_before - len(df)))

    df["ThanhTien"] = df["Quantity"] * df["Price"]

    if verbose:
        print(f"Số dòng ban đầu: {n0:,}")
        for step, removed in log:
            print(f"  - {step}: -{removed:,} dòng")
        print(f"Số dòng còn lại sau làm sạch giao dịch: {len(df):,}")

    return df.reset_index(drop=True)


def build_product_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    n_products_before = df["StockCode"].nunique()

    sp = df.groupby("StockCode").agg(
        tong_so_luong=("Quantity", "sum"),
        tong_doanh_thu=("ThanhTien", "sum"),
        gia_trung_binh=("Price", "mean"),
        so_don_hang=("Invoice", "nunique"),
        so_khach_mua=("Customer ID", "nunique"),
        do_lech_sl=("Quantity", "std"),
    ).reset_index()

    # Mô tả sản phẩm phổ biến nhất (để đọc hiểu, không dùng làm đặc trưng)
    mo_ta = (
        df.groupby("StockCode")["Description"]
        .agg(lambda x: x.value_counts().idxmax())
        .reset_index()
        .rename(columns={"Description": "Mo_ta"})
    )
    sp = sp.merge(mo_ta, on="StockCode", how="left")

    sp["do_lech_sl"] = sp["do_lech_sl"].fillna(0)
    sp["sl_moi_don"] = sp["tong_so_luong"] / sp["so_don_hang"]
    sp["ty_le_mua_lai"] = sp["so_don_hang"] / sp["so_khach_mua"].replace(0, np.nan)
    sp["ty_le_mua_lai"] = sp["ty_le_mua_lai"].fillna(sp["ty_le_mua_lai"].median())

    # Loại sản phẩm bán quá ít
    n_before = len(sp)
    sp = sp[sp["so_don_hang"] >= MIN_ORDERS_PER_PRODUCT].reset_index(drop=True)

    if verbose:
        print(f"Số sản phẩm (StockCode) trước tổng hợp: {n_products_before:,}")
        print(f"Loại sản phẩm có < {MIN_ORDERS_PER_PRODUCT} đơn hàng: -{n_before - len(sp):,}")
        print(f"Số sản phẩm còn lại để gom cụm: {len(sp):,}")

    return sp


FEATURE_COLS = [
    "tong_so_luong",
    "tong_doanh_thu",
    "gia_trung_binh",
    "so_don_hang",
    "so_khach_mua",
    "do_lech_sl",
    "sl_moi_don",
    "ty_le_mua_lai",
]