import pickle

import numpy as np
import tensorflow as tf
import tweepy as tp
from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

import constants as ct
from sann.preprocessing.clean_data import clean_tweet

# Fix for this error related to loading models:
# "Failed to get convolution algorithm. This is probably because cuDNN failed
# to initialize, so try looking to see if a warning log message was printed
# above"
physical_devices = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

app = Flask(__name__)


def load_models():
    global binary_sa_model, binary_sa_tokenizer

    binary_sa_model = load_model(ct.PROJ_DIR +
                                 "models/binary_sentiment_analysis/"
                                 "neural_networks/cnn/cnn_01_w2v_sg.h5")

    with open(ct.PROJ_DIR + "models/tokenizers/binary_sa_tokenizer.pickle",
              "rb") as handle:
        binary_sa_tokenizer = pickle.load(handle)

    global emotion_detection_model, emotion_detection_tokenizer

    emotion_detection_model = load_model(ct.PROJ_DIR +
                                         "models/emotion_detection_2/"
                                         "neural_networks/cnn/cnn_01_emb.h5")

    with open(ct.PROJ_DIR + "models/tokenizers/emotion_detection_2_tokenizer"
                            ".pickle", "rb") as handle:
        emotion_detection_tokenizer = pickle.load(handle)


def predict(cln_twts, tokenizer, model, decoder, num_classes):
    seq = tokenizer.texts_to_sequences(cln_twts)
    padded_seq = pad_sequences(seq, maxlen=140)

    preds = model.predict_classes(padded_seq)

    results = []
    num_each_class = [0] * num_classes

    for pred in preds:
        if type(pred) is np.ndarray:
            pred = pred[0]

        results.append(decoder[pred])
        num_each_class[pred] += 1

    return results, num_each_class


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
                            id=username).items(num_tweets):
        if count >= num_tweets:
            break

        if not hasattr(status, "retweeted_status"):
            tweets.append({
                "dirty_text": status.full_text,
                "clean_text": "",
                "no_clean_text": False,
                "date": status.created_at,
                "id": status.id,
                "binary_sentiment": "N/A",
                "emotion": "N/A"
            })
            count += 1

    num_sentiments, num_emotions = [], []

    if tweets:
        for twt in tweets:
            cln_twt = clean_tweet(twt["dirty_text"], rem_htags=False)

            if not cln_twt.strip():
                twt["no_clean_text"] = True

            twt["clean_text"] = cln_twt

        clean_tweets = [twt["clean_text"] for twt in tweets]

        binary_sa_results, num_sentiments = \
            predict(clean_tweets, binary_sa_tokenizer, binary_sa_model,
                    ct.BINARY_SA_DECODER, 2)

        emotion_detection_results, num_emotions = \
            predict(clean_tweets, emotion_detection_tokenizer,
                    emotion_detection_model, ct.EMOTION_DECODER, 7)

        for i in range(len(tweets)):
            twt = tweets[i]

            if tweets[i]["no_clean_text"] is False:
                twt["binary_sentiment"] = binary_sa_results[i]
                twt["emotion"] = emotion_detection_results[i]

    return render_template("results.html", username=username,
                           num_sentiments=num_sentiments,
                           num_emotions=num_emotions, tweets=tweets)


if __name__ == "__main__":
    print("Loading the models...")
    load_models()

    print("Starting the Flask server...")
    app.run(debug=True)
