import pandas as pd

directory = "../../datasets/sentiment140/"
cols = ["sentiment", "id", "date", "query_string", "user", "text"]

# Reads in the 1.6 million tweets
df = pd.read_csv(directory + "sentiment140_original.csv", encoding="utf-8",
                 header=None, names=cols)

# Creates a dataframe that contains the first 50,000 negative tweets
neg_df = df.loc[df["sentiment"] == 0]
neg_df = neg_df.head(100000)

# Creates a dataframe that contains the first 50,000 positive tweets
pos_df = df.loc[df["sentiment"] == 4]
pos_df = pos_df.head(100000)
pos_df.sentiment = 1

# Concatenates to create a dataframe of 100,000 tweets, 50:50 positive, negative
reduced_df = pd.concat([neg_df, pos_df])

# Drops unnecessary features
reduced_df.drop(["id", "date", "query_string", "user"], axis=1, inplace=True)

# Creates a new index column
reduced_df.reset_index(drop=True, inplace=True)
reduced_df.index.name = "id"

print(reduced_df.head())

# Writes dataframe to a new file
reduced_df.to_csv(directory + "sentiment140_reduced.csv")