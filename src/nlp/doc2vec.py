import multiprocessing

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
# Used to display a CLI progress bar
from tqdm import tqdm

"""Tags a corpus of text for Doc2Vec"""


class TaggedCorpus:
    def __init__(self, text):
        self.text = text
        self.sentences = [TaggedDocument(t.split(), str(i))
                          for i, t in enumerate(tqdm(text))]

    """Returns the corpus as a list of iterators for building the vocab"""

    def __iter__(self):
        for i, s in enumerate(self.text):
            yield TaggedDocument(s.split(), str(i))

    """Return the corpus as an array for training as it's a list of iterators 
       by default"""

    def to_array(self):
        return self.sentences


"""Class for a basic Doc2Vec implementation"""


class D2V:
    def __init__(self, corpus, **opts):
        # Doc2Vec uses a slightly different corpus to Word2Vec
        # Here, each of the sentences have to be "tagged"
        self.corpus = TaggedCorpus(corpus)

    """Creates and saves a model with some default arguments"""

    def create_model(self, model_path, dm=0, vec_size=64, window=2, min_count=2,
                     epochs=10):
        # Uses negative sampling to reduce training time because only a small
        # percentage of weights are modified, opposed to all of them
        # Also uses all the cpu's cores for training
        default_opts = {"negative": 5,
                        "workers": multiprocessing.cpu_count()}

        model = Doc2Vec(dm=dm, vector_size=vec_size, window=window,
                        min_count=min_count, **default_opts)

        model.build_vocab(self.corpus)
        model.train(self.corpus.to_array(),
                    total_examples=len(self.corpus.to_array()),
                    epochs=epochs)
        model.save(model_path)

        return model
