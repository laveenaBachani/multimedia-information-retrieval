import argparse
from Phase3.APIs.visualization_apis import *
from Phase3.Modules import MaxAMinPartitioning as mp
from Phase3.Modules import SpectralPartitioning as sp
from Phase3.Modules import ViewCreator as vc

import time


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
    args = parse_args_process()
    num_clus = args.c
    graph = np.load('adjMatrix_new.npy')#[0:600,0:600]
    num_nodes = graph.shape[0]
    taskId = 2
    visualizer = vc.Visualizer(taskId)
    visualizer.clean_ouput()

    max_a_min_partition = mp.MaxAMinPartitioning()
    clusters = max_a_min_partition.get_clusters(graph, num_clus)
    visualizer.visualize_clusters(clusters, "Max_A_Min")

    spectra_partition = sp.SpectralPartitioning()
    clusters = spectra_partition.get_clusters(graph, num_clus)
    visualizer.visualize_clusters(clusters, "Spectral_Partitioning")

if __name__ == '__main__':
    task2()
