from tqdm import tqdm
from sklearn import utils
from gensim.models import Word2Vec
import nltk


class W2V:
    def __init__(self, corpus, **opts):
        self.corpus = [nltk.word_tokenize(sen) for sen in tqdm(corpus)]
        self.model = Word2Vec(**opts)

    def create_model(self, epochs, learning_rate):
        self.model.build_vocab(self.corpus)

        for _ in tqdm(range(epochs)):
            self.model.train(utils.shuffle(self.corpus),
                             total_examples=len(self.corpus), epochs=1)
            self.model.min_alpha = self.model.alpha - learning_rate

        return self.model

    def save_model(self, name):
        self.model.save(name)
        return self.model
