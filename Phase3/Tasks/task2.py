import argparse
import numpy as np
from sklearn.cluster import KMeans
import multiprocessing

from Phase3.APIs.visualization_apis import *


'''
Given the image-image graph, identify c clusters (for a user supplied c) using two distinct algorithms. 
You can use the graph partitioning/clustering algorithms of your choice for this task.
Visualize the resulting image clusters.
'''

# Argument Parser
def parse_args_process():
    argument_parse = argparse.ArgumentParser()
    argument_parse.add_argument('--c', help='The number clusters you want to find', type=int, required=True)
    args = argument_parse.parse_args()
    return args



def task2():
    # args = parse_args_process()
    # num_cores = multiprocessing.cpu_count()
    # simMatrix = np.load('adjMatrix.npy')

    # # # Sample input # # #
    graph = np.load('adjMatrix_new.npy')[0:200, 0:200]
    num_nodes = graph.shape[0]
    labels = np.random.randint(10, size=num_nodes)
    print(str(np.amax(labels)))
    visualize_groups(graph, labels, "cluster")

    # # # Sample input # # #
    # graph = np.load('adjMatrix_new.npy')[0:200, 0:200]
    # queryImageIds = ['103568891', '1001205877']
    # mostSim = list(queryImageIds)
    # mostSim.extend(['10660461066', '10754361846'])
    # visualize_similar(graph, queryImageIds, mostSim)


    # kmeans_model = KMeans(n_clusters=args.c, n_jobs=num_cores - 1).fit(simMatrix)
    # kmeans = kmeans_model.transform(simMatrix)



if __name__ == '__main__':
    task2()
