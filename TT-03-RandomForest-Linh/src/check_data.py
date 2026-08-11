import pandas as pd
df = pd.read_csv(
    "dataset/bank-additional-full.csv",
    sep=";"
)

print("Kích thước dữ liệu: ")
print(df.shape)

print("\nDanh sách cột: ")
print(list(df.columns))

print("\n5 dòng đầu tiên: ")
print(df.head())

print("\nThông tin dữ liệu: ")
print(df.info())

print("\nPhân bố target: ")
print(df["y"].value_counts())