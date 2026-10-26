import pandas as pd


df = pd.read_csv(
    "dataset/oa_features.csv"
)


df["looking_away"] = (
    df["label"]
    .isin([
        "looking_away",
        "combined"
    ])
    .astype(int)
)


df["phone"] = (
    df["label"]
    .isin([
        "phone",
        "combined"
    ])
    .astype(int)
)


df["multiple_faces"] = (
    df["label"]
    .isin([
        "multiple_faces",
        "combined"
    ])
    .astype(int)
)


df.to_csv(
    "dataset/behavior_features.csv",
    index=False
)


print("======================================")
print("BEHAVIOR DATASET CREATED")
print("======================================")

print()

print("Dataset shape:")
print(df.shape)

print()

print("Looking away:")
print(df["looking_away"].value_counts())

print()

print("Phone:")
print(df["phone"].value_counts())

print()

print("Multiple faces:")
print(df["multiple_faces"].value_counts())

print()

print(
    "Saved to:"
)

print(
    "dataset/behavior_features.csv"
)