import locationInfoParser
import numpy as np
import json
import time
import get_latent_features as glf
from APIs import generic_apis

class VisDescParser:

    ALGO_EUCLIDEAN_DISTANCE = "euclidean_distance"

    ALGO_COSINE_SIMILARITY = "cosine_similarity"

    ALGO_CHI_SQUARE_DISTANCE = "chi_square_distance"

    RELATIVE_DEV_SET_PATH = "../devset/"

    TOPICS_INFO_FILE = "devset_topics.xml"

    @staticmethod
    def getFilePath(visDesBasePath, modelName, locationTitle):
        filePath = visDesBasePath + locationTitle + ' ' + modelName + '.csv'
        return filePath

    def getAllLocationDetails(self):
        topicsFilePath = self.RELATIVE_DEV_SET_PATH + self.TOPICS_INFO_FILE
        locInfoParser = locationInfoParser.LocationInfoParser()
        allLocationDetails = locInfoParser.parse_xml_topic(topicsFilePath)
        return allLocationDetails

    def getTask5Items(self, dimRedAlgo, numLatSemFeat, givlocId, visDescModelName, numSimLocReq):
        filePath = "../devset/descvis/img/acropolis_athens CM.csv"
        allLocationDetails = self.getAllLocationDetails()
        allLocationReqSemFeat = {}
        for locationId in allLocationDetails:
            #if locationId not in ["1","2"]:
            #    continue
            print(locationId)
            print(type(locationId))
            locationFilePath = self.RELATIVE_DEV_SET_PATH + "descvis/img/" + allLocationDetails[locationId]['title'] +" "+ visDescModelName + ".csv"
            print(locationFilePath)
            allLocationReqSemFeat[locationId] = glf.get_latent_features_vis_disc(locationFilePath, dimRedAlgo, numLatSemFeat)
            #allLocationReqSemFeat[locationId] = glf.get_latent_features_vis_disc("../devset/descvis/img/acropolis_athens CM.csv","LDA",2)
        #print(json.dumps(allLocationReqSemFeat))
        #print(allLocationReqSemFeat)
        allLocationsMatch = {}

        for locationId in allLocationDetails:
            if locationId != givlocId:
                allLocationsMatch[locationId] = self.getLocationSimilarity(allLocationReqSemFeat, givlocId, locationId, visDescModelName)


        #allLocationsMatch["2"] = self.getLocationSimilarity(allLocationReqSemFeat, givlocId, "2", visDescModelName)
        print(allLocationsMatch)
        algo = self.modelWiseDistSimAlgo(visDescModelName)
        sorted_list = [x for x in allLocationsMatch.items()]
        sorted_list.sort(key=lambda x: x[1])  # sort by value
        if algo == self.ALGO_COSINE_SIMILARITY:
            sorted_list.reverse()
        print("Found following most similar locations:-")
        print(sorted_list[:numSimLocReq])
        return sorted_list[:numSimLocReq]

    def getLocationSimilarity(self, allLocationsData, locationId1, locationId2, algo):
        location1Data = allLocationsData[locationId1]
        location2Data = allLocationsData[locationId2]

        location1features = location1Data[:, 1:]
        location2features = location2Data[:, 1:]
        nr1 =  location1features.shape[0]
        print(type(location1Data))
        count = 0
        dist = 0
        start = time.time()
        pairwiseDist = []
        for i in range(nr1):
            individual_vector = location1features[:][i]
            if algo == self.ALGO_COSINE_SIMILARITY:
                imgPairDistSim = generic_apis.consine_similarity(location2features, individual_vector)
            elif algo == self.ALGO_CHI_SQUARE_DISTANCE:
                imgPairDistSim = generic_apis.eucledian_distance(location2features, individual_vector)
            else:
                imgPairDistSim = generic_apis.chi_squared(location2features, individual_vector)
            #print("imgPairDistSim:")
            #print(imgPairDistSim)
            dist += np.sum(imgPairDistSim)
            count += imgPairDistSim.shape[0]
            #print(dist)
            #print(count)
            #exit()
        avgDist = dist / count
        end = time.time()
        t3 = end - start
        print(str(locationId2), ' time:', t3, ' avgDist:', avgDist)
        if algo == self.ALGO_COSINE_SIMILARITY:
            pairwiseDist.reverse()
        return avgDist

    def modelWiseDistSimAlgo(self,modelName):
        algo = {"CM" : self.ALGO_CHI_SQUARE_DISTANCE,
                "CM3x3" : self.ALGO_EUCLIDEAN_DISTANCE,
                "CN" : self.ALGO_EUCLIDEAN_DISTANCE,
                "CN3x3" : self.ALGO_EUCLIDEAN_DISTANCE,
                "CSD": self.ALGO_COSINE_SIMILARITY,
                "GLRLM" : self.ALGO_EUCLIDEAN_DISTANCE,
                "GLRLM3x3" : self.ALGO_CHI_SQUARE_DISTANCE,
                "HOG" : self.ALGO_COSINE_SIMILARITY,
                "LBP" : self.ALGO_CHI_SQUARE_DISTANCE,
                "LBP3x3" : self.ALGO_CHI_SQUARE_DISTANCE
        }
        return algo[modelName]


obj = VisDescParser()
obj.getTask5Items("LDA",2,"1","CM",5)