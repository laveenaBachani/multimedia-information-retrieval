import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD, LatentDirichletAllocation
import pandas as pd
import warnings

warnings.simplefilter('ignore', RuntimeWarning)

np.seterr(divide='ignore', invalid='ignore')
np.random.seed(1)


def read_text_descriptor_files(file_name):
    with open(file_name, 'r') as fp:
        line = fp.readline()
        d = {}
        while line:
            tokens = line.split(' ')
            d[tokens[0]] = {}
            for i in range(1, len(tokens), 4):
                if i + 1 < len(tokens):
                    d[tokens[0]][tokens[i]] = int(tokens[i + 1])
            line = fp.readline()
        return d


def tDictionary_to_vector(dictionary):
    texts = []
    for x in dictionary:
        texts.extend(dictionary[x].keys())
    texts = sorted(list(set(texts)))
    array = []
    keys = dictionary.keys()
    for x in keys:
        vector = []
        for text in texts:
            if text in dictionary[x]:
                vector.append(dictionary[x][text])
            else:
                vector.append(0)
        array.append(vector)
    return np.array(array), texts, keys


def normalize_vector(vector):
    minimum, maximum = np.min(vector), np.max(vector)
    ans = (vector - minimum)
    diff = maximum - minimum
    return ans / diff


def get_PCA(vector, k):
    pca = PCA(n_components=k)
    new_vector = pca.fit_transform(vector)
    return new_vector, pca.components_


def get_SVD(vector, k):
    svd = TruncatedSVD(algorithm='arpack', n_components=k)
    new_vector = svd.fit_transform(vector)
    return new_vector, svd.components_


def get_LDA(vector, k):
    lda = LatentDirichletAllocation(n_components=k, learning_method='batch')
    new_vector = lda.fit_transform(vector)
    return new_vector, lda.components_


# ## This requires changes
# def print_pca_dataframe(pca, columns):
#     print(pd.DataFrame(pca.components_, columns=columns))


def consine_similarity(vector, individual_vector):
    numerator = np.sum(vector * individual_vector, axis=1)
    p1 = np.sum(individual_vector ** 2) ** 0.5
    p2 = np.sum(vector ** 2, axis=1) ** 0.5
    denominator = p1 * p2
    return numerator / denominator


def eucledian_distance(vector, individual_vector):
    return np.sum((vector - individual_vector) ** 2, axis=1) ** 0.5


def chi_squared(vector, individual_vector):
    return np.sum(((vector - individual_vector) ** 2) / np.abs(individual_vector), axis=1)


def return_max_k(distances, k):
    ind = np.argpartition(distances, -k)[-k:]
    ind = ind[np.argsort(distances[ind])[::-1]]
    return ind


def return_min_k(distances, k):
    ind = np.argpartition(distances, k)[:k]
    ind = ind[np.argsort(distances[ind])]
    return ind
