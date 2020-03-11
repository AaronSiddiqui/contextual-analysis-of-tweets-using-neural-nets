from html import unescape
import re
import random
import string


"""Parses and cleans the tweet"""
def clean_tweet(tweet, rem_ellipsis=True, rem_urls=True, rem_mentions=True,
                rem_texemojis=True, repl_mentions=False, rem_htags=True,
                dec_html=True, conv_lcase=True, expand_abbrev=True,
                rem_punc=True, repl_nonprint_chars=True, rem_wspace=True):
    # The tweet can either remove @mentions or replace them
    if rem_mentions and repl_mentions:
        print("You can only choose one of them")
        return

    # Remove ellipsis i.e. "..."
    # Started with this as there is often problems with processing ellipses, so
    # it's best to remove them first
    if rem_ellipsis:
        tweet = re.sub(r"\.{2,}", " ", tweet)

    # Remove the urls
    if rem_urls:
        tweet = re.sub(r"((https?:)|(www.))\S+", " ", tweet)

    # Removes or replaces the @mentions
    if rem_mentions:
        tweet = re.sub(r"@\S+", " ", tweet)
    elif repl_mentions:
        names = ["Tom", "John", "Emma", "Sam", "Rachel"]
        tweet = re.sub(r"@\S+", " " + random.choice(names) + " ", tweet)

    # Remove basic text-based emojis
    if rem_texemojis:
        tweet = re.sub(r" [:;Xx=][038DdPpSsL<>/\\\(\)\[\]\{\}\-]{1,2}", " ",
                       tweet)

    # Removes the hashtags
    if rem_htags:
        tweet = re.sub(r"#\S+", " ", tweet)

    # Decodes the html
    if dec_html:
        tweet = unescape(tweet)

    # Converts to lower case
    if conv_lcase:
        tweet = tweet.lower()

    # Expands abbreviations
    if expand_abbrev:
        abbrevs = {"can't": "can not", "won't": "will not", "n't": " not",
                   "'ve": " have", "'re": " are", " it's": " it is",
                   "i'm": "i am", " he's": " he is", " she's": " she is",
                   "'ll": " will", "'d": " would", "&": " and ",
                   "%": " percent"}
        for k, v in abbrevs.items():
            tweet = re.sub(r"" + k, v, tweet)

    # Removes punctuation
    if rem_punc:
        tweet = re.sub(r"[" + string.punctuation + "]", " ", tweet)
        #tweet = tweet.translate(str.maketrans("", "", string.punctuation))

    # Replaces not printable character with "?" i.e. UTF-8 BOM characters
    if repl_nonprint_chars:
        for char in tweet:
            if char not in string.printable:
                tweet = tweet.replace(char, "?")

    # Removes whitespace
    if rem_wspace:
        tweet = re.sub(r"(\s{2,})|(\t+)", " ", tweet)
        tweet = re.sub(r"(^\s+)|(\s+$)", "", tweet)

    return tweet