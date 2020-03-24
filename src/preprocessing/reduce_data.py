import pandas as pd

"""Finds and returns a dictionary of the ratios for the classes in a feature"""


def find_feature_ratios(df, feat):
    total_rows = len(df)
    ratios = {}

    # A
    for cls in df[feat].unique().tolist():
        num = len(df.loc[df[feat] == cls])
        ratios[cls] = num / total_rows

    return ratios


"""Reduces the dataset to n based on a dictionary of ratios for the classes in 
   a feature and returns the result"""


def reduce_dataset(df, feat, ratios, n):
    if sum([v for v in ratios.values()]) != 1:
        raise ValueError("Ratios must equal 1")

    reduced_dfs = []

    for cls, ratio in ratios.items():
        num_rows = int(n * ratio)
        reduced_df = df.loc[df[feat] == cls]

        reduced_dfs.append(reduced_df.head(num_rows))

    return pd.concat(reduced_dfs)
