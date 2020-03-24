import multiprocessing
import nltk
from gensim.models import Word2Vec
# Used to display a CLI progress bar
from tqdm import tqdm

"""Class for a basic Word2Vec implementation"""


class W2V:
    def __init__(self, corpus):
        # Separates/tokenises the sentences for the corpus list
        self.corpus = [nltk.word_tokenize(sen) for sen in tqdm(corpus)]

    """Creates and saves a model with some default arguments"""

    def create_model(self, model_path, sg=0, size=64, window=2, min_count=2,
                     epochs=10):
        # Uses negative sampling to reduce training time because only a small
        # percentage of weights are modified, opposed to all of them
        # Also uses all the cpu's cores for training
        default_opts = {"negative": 5,
                        "workers": multiprocessing.cpu_count()}

        model = Word2Vec(sg=sg, size=size, window=window, min_count=min_count,
                         **default_opts)

        model.build_vocab(self.corpus)
        model.train(self.corpus, total_examples=len(self.corpus),
                    epochs=epochs)
        model.save(model_path)

        return model
