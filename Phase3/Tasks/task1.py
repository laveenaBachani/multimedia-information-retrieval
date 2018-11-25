import argparse
from Phase3.APIs.generic_apis import *
import numpy as np
import os
from joblib import Parallel, delayed
from Phase3.Tasks.task5 import validate_data
from sklearn.preprocessing import normalize


'''
Task 1: Implement a program which, given a value k, creates an image-image similarity graph, 
such that from each image, there are k outgoing edges to k most similar/related images to it.

Code for creating an adjacency matrix for either textual or visual descriptors is below, although we
strictly use visual models for all future tasks.
'''

# related to textual descriptor
vector = []
columns = []
rows = []

# Argument Parser
def parse_args_process():
    argument_parse = argparse.ArgumentParser()
    argument_parse.add_argument('--k', help='The number semantics you want to find', type=int, required=True)
    args = argument_parse.parse_args()
    return args


def task1():
    args = parse_args_process()
    cwd = os.getcwd()
    matrix_fileName = "adjMatrix"

    if 0:
        # Will NOT be used, strictly for testing purposes
        print("Textual Descriptors\nLoading Image Space...")
        imageFile = '../../Phase2/Data/devset_textTermsPerImage.txt'
        # imageIds by terms
        global vector
        global columns
        global rows

        v, c, r = tDictionary_to_vector(read_text_descriptor_files(imageFile))
        vector = v
        columns = c
        rows = r
        vector = normalize_vector(vector)
        np.save('vector.npy', vector)
        np.save('columns.npy', columns)
        np.save('rows.npy', rows)

        # vector = np.load('vector.npy')
        # columns = np.load('columns.npy')
        # rows = np.load('rows.npy')

        dim = len(rows)
        print("Num images: ", str(dim))
        print("Creating Adjacency Matrix...")

        def topKSimilar(imageIndex):
            print("Starting img ", str(imageIndex))

            individual_vector = vector[imageIndex]
            distances = consine_similarity(vector, individual_vector)
            k = args.k

            ind = return_max_k(distances, k + 1)  # indices of k-most similar images
            # Remove itself/current from being 1 of k most similar
            sim = [z for z in ind if z != imageIndex]
            if len(sim) > k:
                sim = sim[0:k]

            print("Ended img ", str(imageIndex))
            return list(sim)

        kSimilarPerImage = Parallel(n_jobs=num_cores, backend='loky')(delayed(topKSimilar)(i) for i in range(len(rows)))

        print("Length of Rows:", str(dim), "\tLength of Return:", str(len(kSimilarPerImage)))

        adj_matrix = np.zeros((dim, dim))

        for ind, similarity in enumerate(kSimilarPerImage):
            np.put(adj_matrix[ind, :], similarity, 1)

        np.save(matrix_fileName, adj_matrix)
    else:
        print("Using Visual Descriptors...")
        matrix_fileName += '_visual_k' + str(args.k) + '.npy'
        print(matrix_fileName)
        if os.path.exists(cwd + '/' + matrix_fileName):
            print("Edge matrix already exists, aborting...")
            exit(0)

        ans, image_names, names_sorted = validate_data()

        ans = normalize(ans,norm='max',axis=0)  # normalize column-wise
        matrix = []
        for i in range(len(ans)):
            print(i)
            dst = np.argpartition(eucledian_distance(ans, ans[i].reshape(1, ans.shape[1])),args.k + 1)[:args.k + 1]
            # print("dst:", len(dst))
            zeros = np.zeros(ans.shape[0])
            zeros[dst] = 1
            zeros[i] = 0
            # print(np.count_nonzero(zeros))
            matrix.append(zeros)

        matrix = np.array(matrix)
        np.save(matrix_fileName, matrix)
        print(matrix.shape)


if __name__ == '__main__':
    task1()