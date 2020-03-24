import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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