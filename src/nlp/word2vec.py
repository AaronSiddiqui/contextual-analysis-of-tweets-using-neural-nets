from tqdm import tqdm
from sklearn import utils
from gensim.models import Word2Vec
import multiprocessing
import nltk


class W2V:
    def __init__(self, corpus):
        self.corpus = [nltk.word_tokenize(sen) for sen in tqdm(corpus)]

    def create_model(self, sg, epochs, learning_rate):
        default_opts = {"size": 64,
                        "negative": 5,
                        "window": 2,
                        "min_count": 1,
                        "workers": multiprocessing.cpu_count(),
                        "alpha": 0.065,
                        "min_alpha": 0.065}

        model = Word2Vec(sg=sg, **default_opts)
        model.build_vocab(self.corpus)

        for _ in tqdm(range(epochs)):
            model.train(utils.shuffle(self.corpus),
                        total_examples=len(self.corpus), epochs=1)
            model.min_alpha = model.alpha - learning_rate

        return model
