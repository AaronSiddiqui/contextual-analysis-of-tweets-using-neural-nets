import pandas as pd
import matplotlib.pyplot as plt
from src.clean_tweet import clean_tweet

directory = "../../datasets/sentiment140/"

df = pd.read_csv(directory + "sentiment140_reduced.csv", index_col="id")

# Adds a column that contains the length of each tweet before it's cleaned
df['length'] = [len(t) for t in df.text]

# Shows a box plot of the length of the tweets
fig, ax = plt.subplots(figsize=(5, 5))
plt.boxplot(df.length)
plt.show()

# Prints the first 10 tweets that are greater than 140 characters
df[df.length > 140].head(10)

# Prints examples of each type of tweet that is cleaned
print("Before cleaning...")
print("URL Example:", df.text[0])
print("@mention Example:", df.text[343])
print("Hashtag Example:", df.text[175])
print("HTML Encoding Example:", df.text[279])
print("UTF-8 BOM Example:", df.text[226])

# Cleans each of the tweets
clean_df = df.drop(["length"], axis=1)

for i in df.index:
    clean_df.at[i, "text"] = clean_tweet(clean_df.at[i, "text"])

print()
print("After cleaning...")
print("URL Example:", clean_df.text[0])
print("@mention Example:", clean_df.text[343])
print("Hashtag Example:", clean_df.text[175])
print("HTML Encoding Example:", clean_df.text[279])
print("UTF-8 BOM Example:", clean_df.text[226])

# Prints tweets that are null after the data is processed
# e.g. A tweet that only had a url
print()
print(df[df.isnull().any(axis=1)].head())

# Drop these tweets
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
df.index.name = "id"

# Writes the dataframe to a new file
clean_df.to_csv(directory + "sentiment140_clean.csv")