from tqdm import tqdm
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import multiprocessing


class TaggedCorpus:
    def __init__(self, text):
        self.text = text
        self.sentences = [TaggedDocument(t.split(), str(i)) for i, t in enumerate(tqdm(text))]

    def __iter__(self):
        for i, s in enumerate(self.text):
            yield TaggedDocument(s.split(), str(i))

    def to_array(self):
        return self.sentences


class D2V:
    def __init__(self, corpus, **opts):
        self.corpus = TaggedCorpus(corpus)

    def create_model(self, model_path, dm=0, vec_size=64, window=2, min_count=2,
                     epochs=10):
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
