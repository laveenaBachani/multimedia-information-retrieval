import pandas as pd
import sys
import os
import csv
import numpy as np
import operator
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import chi2_kernel
import xmltodict


#create a map that maps from the location id to the location
#search for the location the name in as first.csv
# run a loop for all 29 location
# multiple and save the euclidean distance
# select top 100 euclidean distance and take out averga of them
#endter the location name and average on the image


class Task5:
    locations = []
    with open("devset_topics.xml") as fd:
        doc = xmltodict.parse(fd.read())
        for topic in doc['topics']['topic']:
            locations.append(topic['title'])
    Models = ["CM", "CM3x3", "CN", "CN3x3","CSD","GLRLM", "GLRLM3x3","HOG","LBP", "LBP3x3" ]
    def searchFirstFile(self,LocationId,Model_name):
        t = Task5()
        first_file_name = t.locations[LocationId- 1] + " " + Model_name + ".csv"
        for file in os.listdir("./img"):
            if file.startswith(first_file_name):
                f = open("./img/" + first_file_name)
                reader = csv.reader(f)
                list = []
                for row in reader:
                    list.append(row)
        return list

    def similarityEuclidean(self, LocationId,k):
        t = Task5()
        listofmax= []
        distSum = {}
        listofmaximages = []

        for x in t.Models:
            for j in range(1,31):
                if j != LocationId:
                    list1 = t.searchFirstFile(LocationId, x)
                    list2 = t.searchFirstFile(j,x)
                    df = np.array(list1)
                    df2 = np.array(list2)
                    if x in ["CM3x3", "GLRLM", "CN3x3", "CN","CM"]:
                        dist = euclidean_distances(df[:, 1:], df2[:, 1:])
                    elif x in [ "LBP", "GLRLM3x3", "LBP3x3"]:
                        dist = chi2_kernel(df[:, 1:], df2[:, 1:])
                    elif x in ["HOG", "CSD"]:
                        dist = cosine_distances(df[:, 1:], df2[:, 1:])
                    max = dist.argmin(axis=1)
                    s=0
                    for i in range(len(df)):
                        s += dist[i, max[i]]
                    distSum = {}
                    distSum["value"] = s
                    distSum["Model"] = x
                    distSum["Location"] = j
                    listofmaximages.append(distSum)
            listofmax.append(sorted(listofmaximages, key=lambda k: k["value"]))
            listofmaximages = []

        map = {}
        for i in range(31):
            if i!=LocationId:
                rank = 0
                for list in listofmax:
                    count = 1
                    for x in list:
                        if i == x["Location"]:
                            rank+=count
                            break
                        else:
                            count+=1
                map[i+1] = rank
        sorted_dict = sorted(map.items(), key=operator.itemgetter(1))
        for i in range(k):
            print("Mached Locations")
            print(t.locations[sorted_dict[i][0]])
            for list in listofmax:
                for x in list:
                    if x["Location"] == sorted_dict[i][0]:
                        print("Model: " , end="")
                        print(x["Model"])
                        print("Value", end = "")
                        print(x["value"])


def main():
    t1 =Task5()
    LocationId = sys.argv[1]
    k = sys.argv[2]
    t1.similarityEuclidean(int(LocationId), int(k))


if __name__ == "__main__":
    main()