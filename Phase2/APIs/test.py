from Phase2.APIs.generic_apis import *


def test1():
    file_name = '../Data/devset_textTermsPerUser.txt'
    read_text_descriptor_files(file_name)


def test2():
    file_name = '../Data/devset_textTermsPerUser.txt'
    tDictionary_to_vector(read_text_descriptor_files(file_name))


def test3():
    file_name = '../Data/devset_textTermsPerUser.txt'
    normalize_vector(tDictionary_to_vector(read_text_descriptor_files(file_name))[0])


def test4():
    print('#################This is pca testing#############################')
    file_name = '../Data/devset_textTermsPerUser.txt'
    vector, columns, keys = tDictionary_to_vector(read_text_descriptor_files(file_name))
    vector = normalize_vector(vector)
    new_vector, components = get_PCA(vector, 10)
    distance_matrix = consine_similarity(new_vector, new_vector[0])
    print(return_max_k(distance_matrix, 5))


def test5():
    print('#################This is svd testing#############################')
    file_name = '../Data/devset_textTermsPerUser.txt'
    vector, columns, keys = tDictionary_to_vector(read_text_descriptor_files(file_name))
    vector = normalize_vector(vector)
    new_vector, components = get_SVD(vector, 10)
    distance_matrix = consine_similarity(new_vector, new_vector[0])
    print(return_max_k(distance_matrix, 5))


def test6():
    print('#################This is LDA testing#############################')
    file_name = '../Data/devset_textTermsPerUser.txt'
    vector, columns, keys = tDictionary_to_vector(read_text_descriptor_files(file_name))
    vector = normalize_vector(vector)
    new_vector, components = get_LDA(vector, 10)
    distance_matrix = consine_similarity(new_vector, new_vector[0])
    print(return_max_k(distance_matrix, 5))
