import pickle
import tweepy as tp
import constants as ct
from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from sann.preprocessing.clean_data import clean_tweet

app = Flask(__name__)

binary_sa_model = load_model(ct.PROJ_DIR + "models/binary_sentiment_analysis/"
                                           "neural_networks/""cnn/"
                                           "cnn_01_w2v_sg.h5")
binary_sa_tokenizer = None

with open(ct.PROJ_DIR + "models/tokenizers/bsa_tokenizer.pickle", "rb") as handle:
    binary_sa_tokenizer = pickle.load(handle)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/results", methods=["GET", "POST"])
def results():
    username = request.form["twitter_username"]
    num_tweets = int(request.form["num_recent_tweets"])
    count = 0
    tweets = []

    auth = tp.OAuthHandler(ct.CONSUMER_KEY, ct.CONSUMER_SECRET)
    auth.set_access_token(ct.ACCESS_TOKEN, ct.ACCESS_TOKEN_SECRET)
    api = tp.API(auth)

    for status in tp.Cursor(api.user_timeline, tweet_mode="extended",
                            id=username).items():
        if count >= num_tweets:
            break

        if not hasattr(status, "retweeted_status"):
            tweets.append({
                "dirty_text": status.full_text,
                "clean_text": "",
                "no_clean_text": False,
                "date": status.created_at,
                "id": status.id,
                "binary_sentiment": "",
                "fine_grained_sentiment": "",
                "emotion": ""
            })
            count += 1

    for twt in tweets:
        cln_twt = clean_tweet(twt["dirty_text"], rem_htags=False)

        if not cln_twt.strip():
            twt["no_clean_text"] = True

        twt["clean_text"] = cln_twt

    clean_tweets = [twt["clean_text"] for twt in tweets]

    seq = binary_sa_tokenizer.texts_to_sequences(clean_tweets)
    padded_seq = pad_sequences(seq, maxlen=140)

    preds = binary_sa_model.predict_classes(padded_seq)
    results = []

    for pred in preds:
        results.append(ct.BINARY_SA_ENCODER[pred[0]])

    for i in range(len(tweets)):
        twt = tweets[i]

        if tweets[i]["no_clean_text"] is False:
            twt["binary_sentiment"] = results[i]
            twt["fine_grained_sentiment"] = results[i]
            twt["emotion"] = results[i]

    for twt in tweets:
        print(twt["dirty_text"], ":", twt["binary_sentiment"])

    return render_template("results.html")


if __name__ == "__main__":
    app.run(debug=True)
