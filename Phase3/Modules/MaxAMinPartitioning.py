import numpy as np
import random
import sys
import time

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
            if sptSet[v] == False and dist[v] < min:
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
        #print("firstiter:",self.getLessCount(dist))
        #print("id:",srcNodeId," connectedNodes:", self.getDirectlyConnectedNodes(graph,srcNodeId))
        for count in range(numNodes):
            # Pick the minimum distance vertex from the set of vertices not
            # yet processed. u is always equal to src in the first iteration.
            u = self.minDistance(dist, sptSet)
            #print("id:", u, "dist:", dist[u], " connectedNodes:", self.getDirectlyConnectedNodes(graph, u), " lessNodes:",self.getConnectedComponentNodes(dist))
            if u == -1:
                break

            # Mark the picked vertex as processed
            sptSet[u] = True

            # Update dist value of the adjacent vertices of the picked vertex.
            for v in range(numNodes):

                # Update dist[v] only if is not in sptSet, there is an edge from
                # u to v, and total weight of path from src to  v through u is
                # smaller than current value of dist[v]
                if (sptSet[v] is False) and (dist[u] != maxDist) and (graph[u][v] > 0) and (dist[u]+graph[u][v] < dist[v]):
                    dist[v] = dist[u] + graph[u][v]

            #print("iter:",count," count:", self.getLessCount(dist))

        return dist


    def get_clusters(self, graph, k):
        leaders = []
        numNodes = graph.shape[0]
        graph = self.getSymmetricGraph(graph)

        firstLeader = random.randint(0, numNodes-1)
        leaders.append(firstLeader)
        firstLeader_dist = self.dijkstra(graph, firstLeader)
        leaders_dist = np.array([firstLeader_dist])
        allNodes = list(range(numNodes))
        availableNodes = [x for x in allNodes]

        for i in range(k-1):

            avg_dist = np.sum(leaders_dist,axis=0)/leaders_dist.shape[0]
            new_leader = random.choice(availableNodes)
            new_leader_dist = avg_dist[new_leader]

            for node in availableNodes:
                if new_leader_dist >= self.max_dist:
                    break
                elif avg_dist[node] > new_leader_dist:
                    new_leader = node
                    new_leader_dist = avg_dist[node]

            leaders.append(new_leader)
            new_leader_all_dist = self.dijkstra(graph, new_leader)
            leaders_dist = np.append(leaders_dist, np.array([new_leader_all_dist]), axis=0)
            availableNodes = [x for x in availableNodes if x not in leaders]

        print(leaders)
        print("total connected components:", len(leaders))
        leaders_cluster = {}
        for leader in leaders:
            leaders_cluster[leader] = []

        for node in allNodes:
            nearestLeader = leaders[0]
            nearestLeaderDist = leaders_dist[0][node]
            nearestLeaderClusterSize = len(leaders_cluster[nearestLeader])

            for leaderIndex, leader in enumerate(leaders):
                distToLeader = leaders_dist[leaderIndex][node]
                leaderClusterSize = len(leaders_cluster[leader])
                if distToLeader < nearestLeaderDist:
                    nearestLeader = leader
                    nearestLeaderDist = distToLeader
                elif distToLeader == nearestLeaderDist and leaderClusterSize < nearestLeaderClusterSize:
                    nearestLeader = leader
                    nearestLeaderDist = distToLeader
            leaderIndex = leaders.index(nearestLeader)
            leaders_cluster = self.insertInCluster(leaders_cluster,node,nearestLeader,leaders_dist, leaderIndex)

        print(leaders_cluster)


    def insertInCluster(self,leaders_cluster, insertNode, leader, leaders_dist, leaderIndex):
        inserted = False
        insertNodeDist = leaders_dist[leaderIndex][insertNode]
        for i, node in enumerate(leaders_cluster[leader]):

            nodeDist = leaders_dist[leaderIndex][node]
            if insertNodeDist < nodeDist:
                leaders_cluster[leader].insert(i, insertNode)
                inserted = True
                break

        if inserted is False:
            leaders_cluster[leader].append(insertNode)

        return leaders_cluster

    def getLessCount(self, arr):
        count = 0
        for i in arr:
            if (i < self.max_dist):
                count += 1
        #print("count:", count)
        return count

    def getConnectedComponentNodes(self, arr):
        list = []
        for i,val in enumerate(arr):
            if (val < self.max_dist):
                 list.append(i)
        return list

    def getDirectlyConnectedNodes(self, graph, u):
        list = []
        for i in range(graph.shape[1]):
            if graph[u][i] >= 1:
                list.append(i)
        return list

    def getSymmetricGraph(self,graph):
        for i in range(graph.shape[0]):
            if i % 1000 == 0:
                print("symmetric:",i)
            for j in range(graph.shape[1]):
                if graph[i][j] == 1:
                    graph[j][i] = 1
        return graph


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
#obj = MaxAMinPartitioning()
#x=obj.dijkstra(graph,0)
#print(x)
#obj.get_clusters(graph,5)