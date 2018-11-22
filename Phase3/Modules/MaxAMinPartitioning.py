import numpy as np
import random
import sys

class MaxAMinPartitioning:

    def __init__(self):
        self.max_dist = sys.maxsize

    # function to find the vertex with minimum distance from src node among nodes
    # which are not yet added
    def minDistance(self, dist, sptSet):
        numNodes = len(dist)

        # Initialize min value
        min = self.max_dist
        min_index = -1

        for v in range(numNodes):
            if sptSet[v] == False and dist[v] <= min:
                min = dist[v]
                min_index = v

        return min_index

    def dijkstra(self, graph, srcNodeId):
        numNodes = graph.shape[0]
        maxDist = self.max_dist
        # dist[i] will hold the shortest distance from srcNodeId to i
        dist = [maxDist for i in range(numNodes)]

        # sptSet[i] will be true if vertex i is included in shortest
        # path tree or shortest distance from src to i is finalized
        sptSet = [False for i in range(numNodes)]

        # Distance of source vertex from itself is always 0
        dist[srcNodeId] = 0
        # Find shortest path for all vertices
        for count in range(numNodes):
            # Pick the minimum distance vertex from the set of vertices not
            # yet processed. u is always equal to src in the first iteration.
            u = self.minDistance(dist, sptSet)
            if u == -1:
                continue

            # Mark the picked vertex as processed
            sptSet[u] = True

            # Update dist value of the adjacent vertices of the picked vertex.
            for v in range(numNodes):

                # Update dist[v] only if is not in sptSet, there is an edge from
                # u to v, and total weight of path from src to  v through u is
                # smaller than current value of dist[v]
                if (sptSet[v] is False) and (dist[u] != maxDist) and (graph[u][v] > 0) and (dist[u]+graph[u][v] < dist[v]):
                    dist[v] = dist[u] + graph[u][v]

        return dist


    def get_clusters(self, graph, k):
        leaders = []
        #leaders_dist = [np.zeros(graph.shape[0])]  #np.array(leaders)
        #leaders_dist = np.array(leaders_dist)
        numNodes = graph.shape[0]

        firstLeader = random.randint(0, numNodes-1)
        leaders.append(firstLeader)

        firstLeader_dist = self.dijkstra(graph, firstLeader)
        leaders_dist = np.array([firstLeader_dist])
        # print("newLeader_dist:", firstLeader_dist)
        # print("leaders_dist:", leaders_dist)
        allNodes = list(range(numNodes))
        for i in range(k-1):
            avg_dist = np.sum(leaders_dist,axis=0)/leaders_dist.shape[0]
            # print("avg_dist:", avg_dist)

            availableNodes = [x for x in allNodes if x not in leaders]
            new_leader = random.choice(availableNodes)
            new_leader_dist = avg_dist[new_leader]

            for node , node_dist in enumerate(avg_dist):
                if (node not in leaders) and node_dist > new_leader_dist:
                    new_leader = node
                    new_leader_dist = node_dist

            leaders.append(new_leader)
            newLeader_dist = self.dijkstra(graph, new_leader)
            leaders_dist = np.append(leaders_dist, np.array([newLeader_dist]), axis=0)
            # print("newLeader_dist:", newLeader_dist)
            # print("leaders_dist:",leaders_dist)


        print(leaders)



graph = [
[0, 4, 0, 0, 0, 0, 0, 8, 0],
[4, 0, 8, 0, 0, 0, 0, 11, 0],
[0, 8, 0, 7, 0, 4, 0, 0, 2],
[0, 0, 7, 0, 9, 14, 0, 0, 0],
[0, 0, 0, 9, 0, 10, 0, 0, 0],
[0, 0, 4, 14, 10, 0, 2, 0, 0],
[0, 0, 0, 0, 0, 2, 0, 1, 6],
[8, 11, 0, 0, 0, 0, 1, 0, 7],
[0, 0, 2, 0, 0, 0, 6, 7, 0]
]
graph = np.array(graph)
obj = MaxAMinPartitioning()
obj.dijkstra(graph,0)
obj.get_clusters(graph,5)