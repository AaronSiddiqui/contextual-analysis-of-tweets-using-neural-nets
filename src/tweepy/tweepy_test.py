import tweepy as tp
from src.tweepy.tweepy_credentials import *

auth = tp.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

api = tp.API(auth)

count = 0
for status in tp.Cursor(api.user_timeline, id="FunhausTeam").items():
    print(status.text)
    print("---------------------------------------------------------")

    if count is 50:
        break

    count += 1