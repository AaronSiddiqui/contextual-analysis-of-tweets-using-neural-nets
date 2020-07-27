# Project directory (replace backslashes with forward slashes if it's on
# Windows)
PROJ_DIR = "C:/Users/aaron/SANN".replace("\\", "/") + "/"

# Access tokens and consumer keys for the tweepy API
ACCESS_TOKEN = ""
ACCESS_TOKEN_SECRET = ""
CONSUMER_KEY = ""
CONSUMER_SECRET = ""

cred_dir = PROJ_DIR + "credentials/"

# Reads them in from the credentials directory
with open(cred_dir + "access_tokens") as f:
    ACCESS_TOKEN = f.readline().split("\t")[1].rstrip()
    ACCESS_TOKEN_SECRET = f.readline().split("\t")[1].rstrip()

with open(cred_dir + "consumer_api_keys") as f:
    CONSUMER_KEY = f.readline().split("\t")[1].rstrip()
    CONSUMER_SECRET = f.readline().split("\t")[1].rstrip()

BINARY_SA_ENCODER = {"negative": 0, "positive": 1}
BINARY_SA_DECODER = {0: "negative", 1: "positive"}

EMOTION_ENCODER = {"joy": 0, "fear": 1, "anger": 2, "sadness": 3, "disgust": 4,
                   "shame": 5, "guilt": 6}
EMOTION_DECODER = {0: "joy", 1: "fear", 2: "anger", 3: "sadness", 4: "disgust",
                   5: "shame", 6: "guilt"}
