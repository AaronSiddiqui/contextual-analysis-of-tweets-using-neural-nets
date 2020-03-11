import pandas as pd

def find_class_ratios(df, cls):
    total_rows = len(df)
    ratios = {}

    for c in df[cls].unique().tolist():
        num = len(df.loc[df[cls] == c])
        ratios[c] = num/total_rows

    return ratios

def reduce_dataset(df, cls, ratios, n):
    if sum([v for v in ratios.values()]) != 1:
        raise ValueError("Ratios must equal 1")

    reduced_dfs = []

    for k, v in ratios.items():
        num_rows = int(n * v)
        reduced_df = df.loc[df[cls] == k]

        reduced_dfs.append(reduced_df.head(num_rows))

    return pd.concat(reduced_dfs)
