import pandas as pd
# Đọc dữ liệu

df = pd.read_csv(
    "dataset/bank-additional-full.csv",
    sep=";"
)

print("Dữ liệu ban đầu:")
print(df.shape)

# Tạo cột pdays_mising

df["pdays_missing"] = (
    df["pdays"] == 999
).astype(int)

# Kết quả

print("\nSố khách chưa từng được liên hệ:")

print(
    df["pdays_missing"].value_counts()
)

print("\nMột số dòng kiểm tra:")

print(
    df[["pdays", "pdays_missing"]].head(10)
)

# Kiểm tra giá trị unknown
print("\nSố lượng 'unknown':")

print("job:")
print((df["job"] == "unknown").sum())

print("\neducation:")
print((df["education"] == "unknown").sum())

print("\nhousing:")
print((df["housing"] == "unknown").sum())