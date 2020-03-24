from tqdm import tqdm
from sklearn import utils
from gensim.models import Word2Vec
import multiprocessing
import nltk


class W2V:
    def __init__(self, corpus):
        self.corpus = [nltk.word_tokenize(sen) for sen in tqdm(corpus)]

    def create_model(self, model_path, sg=0, size=64, window=2, min_count=2,
                     epochs=10):
        default_opts = {"negative": 5,
                        "workers": multiprocessing.cpu_count()}

        model = Word2Vec(sg=sg, size=size, window=window, min_count=min_count,
                         **default_opts)

        model.build_vocab(self.corpus)
        model.train(self.corpus, total_examples=len(self.corpus),
                    epochs=epochs)
        model.save(model_path)

        return model
