import os
import csv
import time
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import chi2_kernel
import xmltodict
from Phase2.APIs.generic_apis import *
from scipy.sparse import csc_matrix
import sys

class Task5:
    locations = []
    with open("devset_topics.xml") as fd:
        doc = xmltodict.parse(fd.read())
        for topic in doc['topics']['topic']:
            locations.append(topic['title'])
    Models = ["CM", "CM3x3", "CN", "CN3x3", "CSD", "GLRLM", "GLRLM3x3", "HOG", "LBP", "LBP3x3"]

    def searchFirstFile(self, LocationId, Model_name):
        t = Task5()
        list = []
        first_file_name = t.locations[LocationId - 1] + " " + Model_name + ".csv"
        for file in os.listdir("./img"):
            if file.startswith(first_file_name):
                try:
                    f = open("./img/" + first_file_name)
                    reader = csv.reader(f)
                    for row in reader:
                        list.append(row)
                except:
                    print("file not found")
        return list

    def createMatrixLocLoc(self):
        matrix = []
        t = Task5()
        for i in range(1, len(t.locations)+1):
            listofmax = []
            for x in t.Models:
                listofmaximages = []
                for j in range(1, len(t.locations)+1):
                    list1 = t.searchFirstFile(i, x)
                    list2 = t.searchFirstFile(j, x)
                    if (len(list1)==0 or len(list2)==0):
                        listofmaximages.append(0)
                        continue
                    df = np.array(list1)
                    df2 = np.array(list2)
                    if x in ["CM3x3", "GLRLM", "CN3x3", "CN", "CM"]:
                        dist = euclidean_distances(df[:, 1:], df2[:, 1:])
                    elif x in ["LBP", "GLRLM3x3", "LBP3x3"]:
                        dist = chi2_kernel(df[:, 1:], df2[:, 1:])
                    elif x in ["HOG", "CSD"]:
                        dist = cosine_distances(df[:, 1:], df2[:, 1:])
                    min = dist.min(axis=1)
                    s = np.sum(np.array(min))
                    listofmaximages.append(s)
                listofmax.append(listofmaximages)
            listofm = np.array(listofmax)
            indexes = np.argsort(listofm,axis=1)
            locations = np.argsort(indexes,axis=1)
            sumlocation = np.sum(locations, axis=0)
            matrix.append(sumlocation)
            print(i)
        return matrix

def main():
    start = time.time()
    t1 = Task5()
    k = int(sys.argv[1])
    matrix = t1.createMatrixLocLoc()
    arr = csc_matrix(matrix)

    arr = arr.astype(float)
    arr = np.negative(arr)
    # np.savetxt("data_task6.csv", arr, delimiter=",")
    reduced, n_comp = get_SVD(arr, k)
    for j in range(k):
        for i in range(30):
            index = np.argsort(-reduced[:, j])
            sorted = -np.sort(-reduced[:, j])
        for i in range(30):
            print(t1.locations[index[i]], end = "")
            print(sorted[i])
        print("")

if __name__ == "__main__":
    main()

