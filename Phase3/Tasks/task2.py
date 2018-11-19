import argparse
from Phase3.APIs.visualization_apis import *
from numpy import linalg as LA


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
    graph = np.load('adjMatrix_new.npy')[0:20, 0:20]
    num_nodes = graph.shape[0]
    labels = spectral_partitioning(graph,4)
    visualize_groups(graph, labels, "cluster")

    # # # Sample input # # #
    # graph = np.load('adjMatrix_new.npy')[0:200, 0:200]
    # queryImageIds = ['103568891', '1001205877']
    # mostSim = list(queryImageIds)
    # mostSim.extend(['10660461066', '10754361846'])
    # visualize_similar(graph, queryImageIds, mostSim)

def spectral_partitioning(graph, k):

    num_nodes = graph.shape[0]
    labels = np.random.randint(1, size=num_nodes)

    # to make adjanceny matrix symmetric
    for i in range(graph.shape[0]):
        if i %1000 == 0:
            print(i)
        for j in range(graph.shape[1]):
            if graph[i][j] == 1:
                graph[j][i] = 1

    clusters = [list(range(graph.shape[0]))]

    for i in range(k):
        # get largest cluster
        tuple = max(enumerate(clusters), key=lambda x: len(x[1]))
        max_len_cluster = tuple[1]
        max_len_cluster_index = tuple[0]

        # adjacency matrix for cluster nodes
        cluster_adj_mat = graph[:, max_len_cluster]
        cluster_adj_mat = cluster_adj_mat[max_len_cluster, :]

        # creating new clusters from previous large cluster
        new_clusters = get_spectral_clusters(cluster_adj_mat, max_len_cluster)

        del clusters[max_len_cluster_index]
        clusters += new_clusters
        print(clusters)
        print()

    for i,v in enumerate(clusters):
        labels[v] = i
    return labels

def get_spectral_clusters(adj_mat , nodeIds):

    # degree of all nodes of graph
    degrees = np.sum(adj_mat, axis=1)

    # Laplacian matrix of graph
    lap_mat = np.diag(degrees) - adj_mat

    e_vals, e_vecs = LA.eig(lap_mat)

    # As all eigen_values of Laplacian matrix are greater than or equal to 0 clipping any small negative value
    e_vals = e_vals.clip(0)

    sorted_evals_index = np.argsort(e_vals)

    # ss_index to store index of second smallest eigen value
    ss_index = 0
    for index in sorted_evals_index:
        if (e_vals[index] > 0):
            ss_index = index
            break

    # eigen vector of second smallest eigen value
    ss_evec = e_vecs[:, ss_index]

    clusters = [[],[]]
    for i, v in enumerate(ss_evec):
        if v >= 0:
            clusters[0].append(nodeIds[i])
        else:
            clusters[1].append(nodeIds[i])

    return clusters

if __name__ == '__main__':
    task2()
