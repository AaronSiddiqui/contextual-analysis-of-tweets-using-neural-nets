from html import unescape
import re
import random
import string

"""Parses and cleans the tweet with a variety of options"""
def clean_tweet(tweet, remove_urls=True, remove_mentions=True,
                replace_mentions=False, remove_hashtags=True, decode_html=True,
                replace_nonprintable_chars=True, convert_lowercase=True,
                remove_whitespace=True):
    # The tweet can either remove @mentions or replace them
    if remove_mentions and replace_mentions:
        print("You can only choose one of them")
        return

    # Remove the urls
    if remove_urls:
        tweet = re.sub(r"https?:\S+", "", tweet)

    # Removes or replaces the @mentions
    if remove_mentions:
        tweet = re.sub(r"@\S+", "", tweet)
    elif replace_mentions:
        names = ["Tom", "Tim", "Sam", "Matthew", "John"]
        return re.sub(r"@\S+", random.choice(names), tweet)

    # Removes the hashtags
    if remove_hashtags:
        tweet = re.sub(r"#\S+", "", tweet)

    # Decodes the html
    if decode_html:
        tweet = unescape(tweet)

    # Replaces not printable character with "?" i.e. UTF-8 BOM characters
    if replace_nonprintable_chars:
        for char in tweet:
            if char not in string.printable:
                tweet = tweet.replace(char, "?")

    # Converts to lower case
    if convert_lowercase:
        tweet = tweet.lower()

    # Removes white space
    if remove_whitespace:
        tweet = " ".join(tweet.split())

    return tweet
