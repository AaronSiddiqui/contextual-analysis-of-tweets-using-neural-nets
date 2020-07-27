import pickle

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

from constants import PROJ_DIR, BINARY_SA_DECODER, EMOTION_DECODER
from src.preprocessing.clean_data import clean_tweet

binary_sa_model = load_model(PROJ_DIR + "models/binary_sentiment_analysis/"
                                        "neural_networks/cnn/cnn_01_w2v_sg.h5")

with open(PROJ_DIR + "models/tokenizers/binary_sa_tokenizer.pickle", "rb") as \
        handle:
    binary_sa_tokenizer = pickle.load(handle)

emotion_detection_model = load_model(PROJ_DIR + "models/emotion_detection_2/"
                                                "neural_networks/cnn/cnn_01_emb.h5")

with open(PROJ_DIR + "models/tokenizers/emotion_detection_2_tokenizer.pickle",
          "rb") as handle:
    emotion_detection_tokenizer = pickle.load(handle)

while True:
    text = input("Enter text that is 140 words or less (CTRL-C to escape):\n")
    cln_text = [clean_tweet(text)]

    seq = binary_sa_tokenizer.texts_to_sequences(cln_text)
    padded_seq = pad_sequences(seq, maxlen=140)
    pred = binary_sa_model.predict_classes(padded_seq)
    result = BINARY_SA_DECODER[pred[0][0]]
    print("Binary Sentiment:", result)

    seq = emotion_detection_tokenizer.texts_to_sequences(cln_text)
    padded_seq = pad_sequences(seq, maxlen=140)
    pred = emotion_detection_model.predict_classes(padded_seq)
    result = EMOTION_DECODER[pred[0]]
    print("Emotion Detection:", result)

    print()
