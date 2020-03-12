import numpy as np


def create_embedding_index(word_emb_map1, word_emb_map2):
    emb_index = {}

    for w in word_emb_map1.vocab.keys():
        emb_index[w] = np.append(word_emb_map1[w], word_emb_map2[w])

    return emb_index


def create_embedding_matrix(num_words, vec_size, word_index, emb_index):
    emb_matrix = np.zeros((num_words, vec_size))

    for w, i in word_index.items():
        if i < num_words:
            emb_vec = emb_index.get(w)

            if emb_vec is not None:
                emb_matrix[i] = emb_vec

    return emb_matrix
