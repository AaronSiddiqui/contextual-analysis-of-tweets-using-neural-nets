from tqdm import tqdm
from sklearn import utils
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import multiprocessing


class TaggedCorpus:
    def __init__(self, text):
        self.text = text
        self.sentences = [TaggedDocument(t.split(), str(i)) for i, t in enumerate(text)]

    def __iter__(self):
        for i, s in enumerate(self.text):
            yield TaggedDocument(s.split(), str(i))

    def to_array(self):
        return self.sentences


class D2V:
    def __init__(self, corpus, **opts):
        self.corpus = TaggedCorpus(corpus)

    def create_model(self, dm, epochs, learning_rate):
        default_opts = {"vector_size": 64,
                        "negative": 5,
                        "min_count": 1,
                        "workers": multiprocessing.cpu_count(),
                        "alpha": 0.065,
                        "min_alpha": 0.065}

        model = Doc2Vec(dm=dm, **default_opts)
        model.build_vocab(self.corpus)

        for _ in tqdm(range(epochs)):
            model.train(utils.shuffle(self.corpus.to_array()),
                        total_examples=len(self.corpus.to_array()),
                        epochs=1)
            model.min_alpha = model.alpha - learning_rate

        return model
