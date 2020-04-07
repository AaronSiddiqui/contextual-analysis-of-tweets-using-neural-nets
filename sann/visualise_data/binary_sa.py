import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

directory = "../../datasets/sentiment140/"

df = pd.read_csv(directory + "sentiment140_reduced.csv", index_col="id")

# Adds a column that contains the length of each tweet before it's cleaned
df['length'] = [len(t) for t in df.text]

# Shows a box plot of the length of the tweets
plt.boxplot(df.length)
plt.ylabel("Number of Characters")
plt.title("Length of Tweets in Reduced Dataset for Binary Sentiment Analysis")
plt.legend()

# Prints examples of each type of tweet that isn't cleaned
print("Before cleaning...")
print("URL Example:", df.text[0])
print("@mention Example:", df.text[343])
print("Hashtag Example:", df.text[175])
print("HTML Encoding Example:", df.text[279])
print()

df = pd.read_csv(directory + "sentiment140_clean.csv", index_col="id")

# Prints examples of each type of tweet that is cleaned
print("After cleaning...")
print("URL Example:", df.text[0])
print("@mention Example:", df.text[340])
print("Hashtag Example:", df.text[175])
print("HTML Encoding Example:", df.text[279])

# Creates a word cloud for the sentiment 140 dataset
neg_tweets = df[df.sentiment == 0]
neg_string = []

for t in neg_tweets.text:
    neg_string.append(t)

neg_string = pd.Series(neg_string).str.cat(sep=' ')

wordcloud = WordCloud(width=1600, height=800,max_font_size=200).generate(neg_string)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

pic_dir = "C:/Users/aaron/Dropbox/Final Year Project/figures/data_processing/"
plt.savefig(pic_dir + "wordcloud_bsa.png")

plt.show()