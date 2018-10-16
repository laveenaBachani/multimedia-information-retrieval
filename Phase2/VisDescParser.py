import locationInfoParser
import numpy as np
import get_latent_features as glf
from APIs import generic_apis

class VisDescParser:

    ALGO_EUCLIDEAN_DISTANCE = "euclidean_distance"

    ALGO_COSINE_SIMILARITY = "cosine_similarity"

    ALGO_CHI_SQUARE_DISTANCE = "chi_square_distance"

    RELATIVE_DEV_SET_PATH = "../devset/"

    TOPICS_INFO_FILE = "devset_topics.xml"

    def getVisDiscFilePath(self, allLocationDetails, locationId, visDescModelName):
        locationFilePath = self.RELATIVE_DEV_SET_PATH + "descvis/img/" + allLocationDetails[locationId]['title'] +" "+ visDescModelName + ".csv"
        return locationFilePath

    def getAllLocationDetails(self):
        topicsFilePath = self.RELATIVE_DEV_SET_PATH + self.TOPICS_INFO_FILE
        locInfoParser = locationInfoParser.LocationInfoParser()
        allLocationDetails = locInfoParser.parse_xml_topic(topicsFilePath)
        return allLocationDetails



    def getTask4Items(self, dimRedAlgo, numLatSemFeat, givlocId, visDescModelName, numSimLocReq):
        allLocationDetails = self.getAllLocationDetails()
        allLocationReqSemFeat = {}
        for locationId in allLocationDetails:
            print("Generating semantic features for location id:"+locationId)
            locationFilePath = self.getVisDiscFilePath(allLocationDetails, locationId, visDescModelName)
            allLocationReqSemFeat[locationId] = glf.get_latent_features_vis_disc(locationFilePath, dimRedAlgo, numLatSemFeat)
        allLocationsMatch = {}

        for locationId in allLocationDetails:
            if locationId != givlocId:
                allLocationsMatch[locationId] = self.getLocationSimilarity(allLocationReqSemFeat, givlocId, locationId, visDescModelName)
        algo = self.modelWiseDistSimAlgo(visDescModelName)
        sorted_list = [x for x in allLocationsMatch.items()]
        sorted_list.sort(key=lambda x: x[1])  # sort by value
        if algo == self.ALGO_COSINE_SIMILARITY:
            sorted_list.reverse()
        return sorted_list[:numSimLocReq]

    def getLocationSimilarity(self, allLocationsData, locationId1, locationId2, algo):
        location1Data = allLocationsData[locationId1]
        location2Data = allLocationsData[locationId2]

        location1features = location1Data[:, 1:]
        location2features = location2Data[:, 1:]
        nr1 = location1features.shape[0]

        count = 0
        dist = 0
        for i in range(nr1):
            individual_vector = location1features[:][i]
            if algo == self.ALGO_COSINE_SIMILARITY:
                imgPairDistSim = generic_apis.consine_similarity(location2features, individual_vector)
            elif algo == self.ALGO_CHI_SQUARE_DISTANCE:
                imgPairDistSim = generic_apis.chi_squared(location2features, individual_vector)
            else:
                imgPairDistSim = generic_apis.eucledian_distance(location2features, individual_vector)
            dist += np.sum(imgPairDistSim)
            count += imgPairDistSim.shape[0]
        avgDist = dist / count
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
out = obj.getTask4Items("LDA",2,"1","CM",5)
print(out)