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
import argparse


def parse_args_process():
    argument_parse = argparse.ArgumentParser()
    argument_parse.add_argument('--k', help='The number semantics you want to find', type=int, required=True)
    args = argument_parse.parse_args()
    return args


class Task5:
    locations = []
    with open("../Data/devset_topics.xml") as fd:
        doc = xmltodict.parse(fd.read())
        for topic in doc['topics']['topic']:
            locations.append(topic['title'])
    Models = ["CM", "CM3x3", "CN", "CN3x3", "CSD", "GLRLM", "GLRLM3x3", "HOG", "LBP", "LBP3x3"]

    def searchFirstFile(self, LocationId, Model_name):
        t = Task5()
        list = []
        first_file_name = t.locations[LocationId - 1] + " " + Model_name + ".csv"
        for file in os.listdir("../Data/img"):
            if file.startswith(first_file_name):
                try:
                    f = open("../Data/img/" + first_file_name)
                    reader = csv.reader(f)
                    for row in reader:
                        list.append(row)
                except:
                    print("file not found")
        return list

    def createMatrixLocLoc(self):
        matrix = []
        t = Task5()
        for i in range(1, len(t.locations) + 1):
            listofmax = []
            for x in t.Models:
                listofmaximages = []
                for j in range(1, len(t.locations) + 1):
                    list1 = t.searchFirstFile(i, x)
                    list2 = t.searchFirstFile(j, x)
                    if (len(list1) == 0 or len(list2) == 0):
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
            indexes = np.argsort(listofm, axis=1)
            locations = np.argsort(indexes, axis=1)
            sumlocation = np.sum(locations, axis=0)
            matrix.append(sumlocation)
            print(i)
        return matrix


def main(args):
    start = time.time()
    t1 = Task5()
    task_6_data = 'task_6_data'

    k = int(args.k)
    task_6_data = task_6_data + '_' + str(k)
    if os.path.exists(task_6_data + '.npy'):
        arr = np.load(task_6_data + '.npy')
    else:
        matrix = t1.createMatrixLocLoc()
        arr = -np.array(matrix)

        # print(arr.shape, arr)
        # np.savetxt(task_6_data, arr.tolist(), fmt='%f')
        np.save(task_6_data, arr)

    arr = np.array(arr, dtype=np.float64)
    #
    # np.savetxt("data_task6.csv", arr, delimiter=",")
    reduced, n_comp = get_SVD(arr, k)
    with open('task6_output.txt', 'w') as f:
        for j in range(k):
            f.write('Latent Semantics' + str(j + 1) + ':\n')
            for i in range(30):
                index = np.argsort(-reduced[:, j])
                sorted = np.take(reduced[:, j], index)

            for i in range(30):
                f.write(str(t1.locations[index[i]]) + ' : ' + str(sorted[i]) + '\n')
            f.write('\n\n')
        f.flush()
        f.close()


if __name__ == "__main__":
    main(parse_args_process())
