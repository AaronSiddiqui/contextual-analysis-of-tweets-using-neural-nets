ACCESS_TOKEN = ""
ACCESS_TOKEN_SECRET = ""
CONSUMER_KEY = ""
CONSUMER_SECRET = ""
filename = "../credentials/"

with open(filename + "access_tokens.txt") as f:
    ACCESS_TOKEN = f.readline().split("\t")[1].rstrip()
    ACCESS_TOKEN_SECRET = f.readline().split("\t")[1].rstrip()

with open(filename + "consumer_api_keys.txt") as f:
    CONSUMER_KEY = f.readline().split("\t")[1].rstrip()
    CONSUMER_SECRET = f.readline().split("\t")[1].rstrip()