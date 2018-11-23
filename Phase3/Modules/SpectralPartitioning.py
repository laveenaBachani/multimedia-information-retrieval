from numpy import linalg as LA
import numpy as np
import time

class SpectralPartitioning:

    def get_clusters(self,graph, k):

        num_nodes = graph.shape[0]
        # labels = np.random.randint(1, size=num_nodes)

        graph = self.getSymmetricGraph(graph)
        clusters = [list(range(graph.shape[0]))]

        for i in range(k-1):
            # get largest cluster
            tuple = max(enumerate(clusters), key=lambda x: len(x[1]))
            max_len_cluster = tuple[1]
            max_len_cluster_index = tuple[0]

            # adjacency matrix for cluster nodes
            cluster_adj_mat = graph[:, max_len_cluster]
            cluster_adj_mat = cluster_adj_mat[max_len_cluster, :]

            # creating new clusters from previous large cluster
            new_clusters = self.spectral_partition(cluster_adj_mat, max_len_cluster)

            del clusters[max_len_cluster_index]
            clusters += new_clusters
            # print(clusters)
            # print("creating cluster ", (i + 1))
            # print()

        # for i,v in enumerate(clusters):
        #    labels[v] = i
        clusters_dict = {}
        for i, v in enumerate(clusters):
            cluster_label = "cluster_"+ str(i+1)
            clusters_dict[cluster_label] = v
        return clusters_dict

    def spectral_partition(self, adj_mat, nodeIds):

        # degree of all nodes of graph
        #print("Calculating degree")
        degrees = np.sum(adj_mat, axis=1)

        # Laplacian matrix of graph
        #print("Calculating lap_mat")
        lap_mat = np.diag(degrees) - adj_mat

        #print("Calculating eval and evecs")
        st = time.time()
        e_vals, e_vecs = LA.eig(lap_mat)
        tt = time.time() - st
        #print("Time to calculate eval of shape:", adj_mat.shape[0], " is:", tt)

        # As all eigen_values of Laplacian matrix are greater than or equal to 0 clipping any small negative value
        e_vals = e_vals.clip(0)

        #print("sorting eval")
        sorted_evals_index = np.argsort(e_vals)

        # ss_index to store index of second smallest eigen value
        ss_index = 0
        for index in sorted_evals_index:
            if (e_vals[index] > 0):
                ss_index = index
                break

        # eigen vector of second smallest eigen value
        ss_evec = e_vecs[:, ss_index]

        #print("creating 2 clus")
        clusters = [[], []]
        for i, v in enumerate(ss_evec):
            if v >= 0:
                clusters[0].append(nodeIds[i])
            else:
                clusters[1].append(nodeIds[i])

        #print("returing clus")
        return clusters

    @staticmethod
    def getSymmetricGraph(graph):
        for i in range(graph.shape[0]):
            if i % 1000 == 0:
                print("symmetric:", i)
            for j in range(graph.shape[1]):
                if graph[i][j] == 1:
                    graph[j][i] = 1
        return graph