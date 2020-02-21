import pandas as pd
import numpy as np

directory = "../../datasets/sentiment140/"

df = pd.read_csv(directory + "sentiment140_clean.csv", index_col="id")

print(df.head())

print()
print(df.info())

print()
print(df[df.isnull().any(axis=1)].head())

print()
print(np.sum(df.isnull().any(axis=1)))

print()
print(df.isnull().any(axis=0))

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
df.index.name = "id"

print()
print(df.head())