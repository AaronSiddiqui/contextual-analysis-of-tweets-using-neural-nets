from tqdm import tqdm
from sklearn import utils
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


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
        self.model = Doc2Vec(**opts)

    def create_model(self, epochs, learning_rate):
        self.model.build_vocab(self.corpus)

        for _ in tqdm(range(epochs)):
            self.model.train(utils.shuffle(self.corpus.to_array()),
                             total_examples=len(self.corpus.to_array()),
                             epochs=1)
            self.model.min_alpha = self.model.alpha - learning_rate

        return self.model

    def save_model(self, name):
        self.model.save(name)
        return self.model
