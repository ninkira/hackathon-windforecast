import pandas as pd, os

data_dir = "./data"
train = pd.read_parquet(os.path.join(data_dir, "windfeels_train.parquet"))
test  = pd.read_parquet(os.path.join(data_dir, "windfeels_test.parquet"))

print(train.shape, test.shape)
print(train.dtypes)
print(train.head(10))
print("\ncols only in train:", set(train.columns) - set(test.columns))
print("missing frac:\n", train.isna().mean().sort_values(ascending=False).head(10))