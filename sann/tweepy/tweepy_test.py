import tweepy as tp
from src.tweepy.tweepy_credentials import *

# Creates the OAuth authentication request to use the API
auth = tp.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

api = tp.API(auth)

# Test to print the first 10 tweets of Funhaus
count = 0
for status in tp.Cursor(api.user_timeline, id="FunhausTeam").items():
    print(status.text)
    print("---------------------------------------------------------")

    if count is 10:
        break

    count += 1