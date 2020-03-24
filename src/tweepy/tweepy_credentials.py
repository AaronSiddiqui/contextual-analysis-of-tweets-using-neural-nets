# Access tokens and consumer keys for the tweepy API
ACCESS_TOKEN = ""
ACCESS_TOKEN_SECRET = ""
CONSUMER_KEY = ""
CONSUMER_SECRET = ""
cred_dir = "../credentials"

# Reads them in from the credentials directory
with open(cred_dir + "/access_tokens.txt") as f:
    ACCESS_TOKEN = f.readline().split("\t")[1].rstrip()
    ACCESS_TOKEN_SECRET = f.readline().split("\t")[1].rstrip()

with open(cred_dir + "/consumer_api_keys.txt") as f:
    CONSUMER_KEY = f.readline().split("\t")[1].rstrip()
    CONSUMER_SECRET = f.readline().split("\t")[1].rstrip()