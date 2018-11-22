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
        #print("id:",srcNodeId," connectedNodes:", self.getConnectedNodes(graph,srcNodeId))
        for count in range(numNodes):
            # Pick the minimum distance vertex from the set of vertices not
            # yet processed. u is always equal to src in the first iteration.
            u = self.minDistance(dist, sptSet)
            #print("id:", u, "dist:", dist[u], " connectedNodes:", self.getConnectedNodes(graph, u), " lessNodes:",self.getLessNodes(dist))
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
        #leaders_dist = [np.zeros(graph.shape[0])]  #np.array(leaders)
        #leaders_dist = np.array(leaders_dist)

        for i in range(graph.shape[0]):
            if i % 1000 == 0:
                print(i)
            for j in range(graph.shape[1]):
                if graph[i][j] == 1:
                    graph[j][i] = 1
        numNodes = graph.shape[0]

        firstLeader = random.randint(0, numNodes-1)
        #firstLeader = 29
        leaders.append(firstLeader)
        print(np.sum(graph,axis=1))
        firstLeader_dist = self.dijkstra(graph, firstLeader)
        #print(firstLeader_dist)
        leaderConnectedNodes = self.getLessNodes(firstLeader_dist)
        print("leaderId:",firstLeader, "clusterSize:",self.getLessCount(firstLeader_dist),  "leaderConnectedNodes:",leaderConnectedNodes)
        leaders_dist = np.array([firstLeader_dist])
        leaderConnectedNodes = self.getLessNodes(firstLeader_dist)
        allNodes = list(range(numNodes))
        availableNodes = [x for x in allNodes]
        # print("newLeader_dist:", firstLeader_dist)
        #print("leaders_dist:", leaders_dist)

        #for i in range(k-1):
        i=0
        while(len(availableNodes)>0):
            i += 1
            st = time.time()
            #ast = time.time()
            avg_dist = np.sum(leaders_dist,axis=0)/leaders_dist.shape[0]
            #at = time.time()-ast
            #lst = time.time()
            allleaderallConnectedNodes = [i for i,v in enumerate(avg_dist) if v < self.max_dist]
            #lt = time.time()-lst
            # print("avg_dist:", avg_dist)
            #aast = time.time()


            #aat = time.time()-aast
            #nst = time.time()
            new_leader = random.choice(availableNodes)
            new_leader_dist = avg_dist[new_leader]

            for node in availableNodes:
                if new_leader_dist >= self.max_dist:
                    break
                if avg_dist[node] > new_leader_dist:
                    new_leader = node
                    new_leader_dist = avg_dist[node]


            leaders.append(new_leader)
            #nt = time.time()-nst
            #dst = time.time()
            new_leader_all_dist = self.dijkstra(graph, new_leader)
            leaderConnectedNodes = self.getLessNodes(new_leader_all_dist)
            #dt = time.time()-dst
            leaders_dist = np.append(leaders_dist, np.array([new_leader_all_dist]), axis=0)
            # print("newLeader_dist:", newLeader_dist)
            # print("leaders_dist:",leaders_dist)
            for x in leaderConnectedNodes:
                if x not in availableNodes:
                    print("Some error occured for node x:",x)
                    availableNodes = [x for x in availableNodes if x not in leaderConnectedNodes]
                    print("i:", i, " leaderId:", new_leader, "clusterSize:", self.getLessCount(new_leader_all_dist),
                          "t:", tt, "totalConnectedNodees:",
                          len(allleaderallConnectedNodes), "availableNodes:", len(availableNodes))
                    exit()
            availableNodes = [x for x in availableNodes if x not in leaderConnectedNodes]
            tt = time.time()- st

            #print("leaderId:", new_leader, "clusterSize:", self.getLessCount(new_leader_all_dist),"t:",tt,"at:",at,"lt:",lt,"aat:",aat,"nt:",nt,"dt:",dt, "totalConnectedNodees:",len(allleaderallConnectedNodes))
            print("i:",i," leaderId:", new_leader, "clusterSize:", self.getLessCount(new_leader_all_dist), "t:", tt, "totalConnectedNodees:",
                  len(allleaderallConnectedNodes),"availableNodes:",len(availableNodes),  "leaderConnectedNodes:",leaderConnectedNodes)

        #print(avg_dist)
        print(leaders)
        print("total connected components:",len(leaders))

    def getLessCount(self, arr):
        count = 0
        for i in arr:
            if (i < self.max_dist):
                count += 1
        #print("count:", count)
        return count

    def getLessNodes(self, arr):
        list = []
        for i,val in enumerate(arr):
            if (val < self.max_dist):
                 list.append(i)
        #print("count:", count)
        return list

    def getConnectedNodes(self, graph, u):
        list = []
        for i in range(graph.shape[1]):
            if graph[u][i] >= 1:
                list.append(i)
        return list


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