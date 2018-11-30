from numpy import linalg as LA
import numpy as np
import time
import sys

class SpectralPartitioning:

    def get_clusters(self, graph, k):

        num_nodes = graph.shape[0]
        # labels = np.random.randint(1, size=num_nodes)

        clusters = [list(range(graph.shape[0]))]

        for i in range(k-1):

            # get largest cluster
            #tuple = max(enumerate(clusters), key=lambda x: len(x[1]))
            #max_len_cluster = tuple[1]
            #max_len_cluster_index = tuple[0]

            # to store index of cluster which will be divided in this iteration
            old_cluster_index = 0
            # to store second smallest eigen value of old cluster which will be divided in this iteration
            old_cluster_ss_eval = sys.maxsize
            new_clusters = [clusters[old_cluster_index]]
            # newclusterfound = False
            # creating new clusters from previous large cluster
            for cluster_index, cluster in enumerate(clusters):
                # adjacency matrix for cluster nodes
                if len(cluster) <= 1:
                    continue
                cluster_adj_mat = graph[:, cluster]
                cluster_adj_mat = cluster_adj_mat[cluster, :]
                possible_new_clusters, ss_eval = self.spectral_partition(cluster_adj_mat, cluster)
                #if ss_eval <= old_cluster_ss_eval and len(possible_new_clusters[0])>0 and len(possible_new_clusters[1])>0:
                if ss_eval <= old_cluster_ss_eval:
                    old_cluster_ss_eval = ss_eval
                    old_cluster_index = cluster_index
                    new_clusters = possible_new_clusters


            del clusters[old_cluster_index]
            clusters += new_clusters

            # print("newclusters:",clusters)
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
        e_vals, e_vecs = LA.eigh(lap_mat)
        #print("lap_mat:",lap_mat)
        #print("evals:", e_vals)
        #print("evec:", e_vecs)
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

        # second smallest eigen value
        ss_eval = e_vals[ss_index]

        clusters = self.getClusterFromEvec(ss_evec,nodeIds)
        #print(clusters)
        if(len(clusters[0]) ==0 or len(clusters[1]) ==0):
            clusters = self.getClusterFromEvec(e_vecs[:, 1], nodeIds)
            ss_eval = e_vals[1]
            #print("inside:",clusters)
            #print("evec:",e_vecs)
        #print("recieved clus:",nodeIds)
        #print("returing clus:",clusters)
        return clusters, ss_eval

    def getClusterFromEvec(self,ss_evec,nodeIds):
        # print("creating 2 clus")
        clusters = [[], []]
        for i, v in enumerate(ss_evec):
            if v >= 0:
                clusters[0].append(nodeIds[i])
            else:
                clusters[1].append(nodeIds[i])
        return clusters

    def getComponentsFromEvec(self,ss_evec,nodeIds):
        # print("creating 2 clus")
        clusters = [[], []]
        for i, v in enumerate(ss_evec):
            if v == 0:
                clusters[0].append(nodeIds[i])
            else:
                clusters[1].append(nodeIds[i])
        return clusters

