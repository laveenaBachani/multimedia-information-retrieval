from os import listdir
import argparse
from Phase3.APIs.generic_apis import *
import numpy as np
import os
from joblib import Parallel, delayed
import multiprocessing
import psutil
from Phase3.Tasks.task5 import validate_data
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.preprocessing import normalize


'''
Task 1: Implement a program which, given a value k, creates an image-image similarity graph, 
such that from each image, there are k outgoing edges to k most similar/related images to it.
'''

vector = []
columns = []
rows = []

# Argument Parser
def parse_args_process():
    argument_parse = argparse.ArgumentParser()
    argument_parse.add_argument('--k', help='The number semantics you want to find', type=int, required=True)
    args = argument_parse.parse_args()
    return args


# def algotype(modelname):  # function to select similarity algorithm
#     if modelname in ["CM3x3", "GLRLM", "CN3x3", "CN", "CM"]:
#         algo = eucledian_distance_1D
#     elif modelname in ["LBP", "GLRLM3x3", "LBP3x3"]:
#         algo = chi2_kernel
#     elif modelname in ["HOG", "CSD"]:
#         algo = cosine_distances
#     return algo


def task1():
    args = parse_args_process()
    cwd = os.getcwd()
    num_cores = multiprocessing.cpu_count()
    matrix_fileName = "adjMatrix"

    if 0:
        print("Textual Descriptors\nLoading Image Space...")
        imageFile = '../../Phase2/Data/devset_textTermsPerImage.txt'
        # imageIds by terms
        global vector
        global columns
        global rows

        # v, c, r = tDictionary_to_vector(read_text_descriptor_files(imageFile))
        # vector = v
        # columns = c
        # rows = r
        # vector = normalize_vector(vector)
        # np.save('vector.npy', vector)
        # np.save('columns.npy', columns)
        # np.save('rows.npy', rows)

        vector = np.load('vector.npy')
        columns = np.load('columns.npy')
        rows = np.load('rows.npy')

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
            # TODO: Sometimes cosine_sim isn't returning itself/current as most similar
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
        # location = '../Data/img'
        # models = ["CM", "CM3x3", "CN", "CN3x3", "CSD", "GLRLM", "GLRLM3x3", "HOG", "LBP", "LBP3x3"]
        # modelToLength = {model: 0 for model in models}
        # for y in models:
        #     x = 'acropolis_athens'
        #     file_to_read = '{0}/{1} {2}'.format(location, x, y + '.csv')
        #     with open(file_to_read, 'r') as f:
        #         line = f.readline()
        #         modelToLength[y] = len(line.split(',')[1:])

        ans = normalize(ans,norm='max',axis=0)  # normalize column-wise
        matrix = []
        for i in range(len(ans)):
            print(i)
            dst = np.argpartition(eucledian_distance(ans, ans[i].reshape(1, ans.shape[1])),args.k)[:args.k]
            # print(len(dst))
            zeros = np.zeros(ans.shape[0])
            zeros[dst] = 1
            # print(np.count_nonzero(zeros))
            matrix.append(zeros)

        matrix = np.array(matrix)
        np.save(matrix_fileName, matrix)
        print(matrix.shape)

        # print("Finding similarity between image pairs...")
        # num_images = ans.shape[0]
        # print("Number of images: ", str(num_images))
        # simMatrix = np.full((num_images, num_images), None)
        # # ***Populating "LHS" of symmetric simMatrix***
        # # - Using varied similarity func. per feature type
        # # for i in range(num_images):
        # #     print("Finished row ", str(i), " of simMatrix")
        # #     for j in range(i + 1):
        # #         simScore = 0.0
        # #         ptr = 0  # index of where current model/featureType starts
        # #         for m in models:
        # #             featTypeLength = modelToLength[m]
        # #             # Calculate cumulative siilarity score between 0-1
        # #             simScore += algotype(m)(ans[j, ptr:ptr+featTypeLength], ans[i, ptr:ptr+featTypeLength])
        # #             ptr += featTypeLength
        # #
        # #         simMatrix[i, j] = simScore / 10.0
        #
        # # ***Populating "LHS" of symmetric simMatrix***
        # # - Using varied similarity func. per feature type
        # def simMatrixSlice(i):
        #     print("Finished row ", str(i), " of simMatrix")
        #     for j in range(i + 1):
        #         simScore = 0.0
        #         ptr = 0  # index of where current model/featureType starts
        #         for m in models:
        #             featTypeLength = modelToLength[m]
        #             # Calculate cumulative siilarity score between 0-1
        #             x = ans[j, ptr:ptr + featTypeLength].reshape(1,-1)
        #             y = ans[i, ptr:ptr + featTypeLength].reshape(1,-1)
        #             simScore += 1 - algotype(m)(x, y)[0][0]  # for use w/ sklearn metrics
        #             # simScore += algotype(m)(ans[j, ptr:ptr + featTypeLength], ans[i, ptr:ptr + featTypeLength])[0][0]
        #             ptr += featTypeLength
        #
        #         simMatrix[i, j] = simScore / 10.0
        #
        # Parallel(n_jobs=num_cores - 3, backend='loky')(delayed(simMatrixSlice)(i) for i in range(num_images))
        #
        # simMatrixFileName = 'simMatrix_vis_k' + str(args.k) + '.npy'
        # np.save(simMatrixFileName, simMatrix)
        #
        # # For each image/row, have k 1's that signify outgoing edges to k-most similar images
        # adj_matrix = np.zeros((num_images, num_images))
        # k = args.k
        #
        # for i in range(num_images):
        #     print("Finished row ", str(i), " of adj_matrix")
        #     # Create full distance vector (1 by num_images) for current image using simMatrix
        #     distances = np.zeros((1, num_images))
        #     distances[:i + 1] = simMatrix[i, :i + 1]
        #     distances[i+1:] = simMatrix[i+1:, i]
        #
        #     ind = return_max_k(distances, k + 1)  # indices of k-most + 1 similar images
        #     # Remove itself/current from being 1 of k most similar
        #     sim = [z for z in ind if z != i]
        #     # TODO: Sometimes cosine_sim isn't returning itself/current as most similar
        #     if len(sim) > k:
        #         sim = sim[0:k]
        #
        #     np.put(adj_matrix[i, :], sim, 1)  # place 1's at indices of k-most similar images
        #
        # np.save(matrix_fileName, adj_matrix)


if __name__ == '__main__':
    task1()