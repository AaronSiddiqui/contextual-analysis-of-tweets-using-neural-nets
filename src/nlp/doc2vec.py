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

    def create_model(self, dm=0, v_size=64, epochs=30, l_rate=0.002):
        default_opts = {"negative": 5,
                        "min_count": 1,
                        "workers": multiprocessing.cpu_count(),
                        "alpha": 0.065,
                        "min_alpha": 0.065}

        model = Doc2Vec(dm=dm, vector_size=v_size, **default_opts)
        model.build_vocab(self.corpus)

        for _ in tqdm(range(epochs)):
            model.train(utils.shuffle(self.corpus.to_array()),
                        total_examples=len(self.corpus.to_array()),
                        epochs=1)
            model.min_alpha = model.alpha - l_rate

        return model
