import math

import matplotlib.pyplot as plt

from constants import PROJ_DIR


def get_accuracies(file_path):
    accs = []

    with open(file_path, "r") as file:
        for line in file:
            acc = line.split(":")[2].strip()
            accs.append(float(acc) * 100)

    return accs


model_dir = PROJ_DIR + "models/binary_sentiment_analysis/neural_networks/"
# model_dir = PROJ_DIR + "models/emotion_detection_2/neural_networks/"

cnn_dir = model_dir + "cnn/"
mlp_dir = model_dir + "mlp/"
rnn_dir = model_dir + "rnn/"

cnn_01_emb_accs = get_accuracies(cnn_dir + "cnn_01_emb_evaluation.txt")
cnn_01_cbow_accs = get_accuracies(cnn_dir + "cnn_01_w2v_cbow_evaluation.txt")
cnn_01_sg_accs = get_accuracies(cnn_dir + "cnn_01_w2v_sg_evaluation.txt")

cnn_02_emb_accs = get_accuracies(cnn_dir + "cnn_02_emb_evaluation.txt")
cnn_02_cbow_accs = get_accuracies(cnn_dir + "cnn_02_w2v_cbow_evaluation.txt")
cnn_02_sg_accs = get_accuracies(cnn_dir + "cnn_02_w2v_sg_evaluation.txt")

mlp_01_emb_accs = get_accuracies(mlp_dir + "mlp_01_emb_evaluation.txt")
mlp_01_cbow_accs = get_accuracies(mlp_dir + "mlp_01_w2v_cbow_evaluation.txt")
mlp_01_sg_accs = get_accuracies(mlp_dir + "mlp_01_w2v_sg_evaluation.txt")

rnn_01_emb_accs = get_accuracies(rnn_dir + "rnn_01_emb_evaluation.txt")
rnn_01_cbow_accs = get_accuracies(rnn_dir + "rnn_01_w2v_cbow_evaluation.txt")
rnn_01_sg_accs = get_accuracies(rnn_dir + "rnn_01_w2v_sg_evaluation.txt")

num_epochs = 5
epochs = list(range(1, num_epochs + 1))

plt.plot(epochs, cnn_01_emb_accs, label="CNN 01 with Basic Embedding", linewidth=1)
plt.plot(epochs, cnn_01_cbow_accs, label="CNN 01 with Word2Vec CBOW", linewidth=1)
plt.plot(epochs, cnn_01_sg_accs, label="CNN 01 with Word2Vec SG", linewidth=1)
plt.xlim([1, num_epochs])
plt.xticks(range(math.floor(min(epochs)), math.ceil(max(epochs)) + 1))
plt.xlabel("Number of Epochs Trained")
plt.ylabel("Validation Accuracy (%)")
plt.title("CNN 01 Embeddings for Binary Sentiment Analysis")
plt.legend()

pic_dir = "C:/Users/aaron/Dropbox/Final Year Project/figures/results/"
plt.savefig(pic_dir + "cnn_01_embeddings_bsa.png")

plt.show()
