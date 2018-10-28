from Phase2.Modules import get_latent_features as glf, locationInfoParser
import numpy as np
from Phase2.APIs import generic_apis


class VisDescParser:

    ALGO_EUCLIDEAN_DISTANCE = "euclidean_distance"

    ALGO_COSINE_SIMILARITY = "cosine_similarity"

    ALGO_CHI_SQUARE_DISTANCE = "chi_square_distance"

    RELATIVE_DEV_SET_PATH = "../Data/"

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
        locInKSem = []
        KSemInFet = []
        for locationId in allLocationDetails:
            locationFilePath = self.getVisDiscFilePath(allLocationDetails, locationId, visDescModelName)
            allLocationReqSemFeat[locationId], comp = glf.get_latent_features_vis_disc(locationFilePath, dimRedAlgo, numLatSemFeat)
            if locationId == givlocId:
                locInKSem = allLocationReqSemFeat[locationId]
                KSemInFet = comp

        allLocationsMatch = {}
        algo = self.modelWiseDistSimAlgo(visDescModelName)
        for locationId in allLocationDetails:
            allLocationsMatch[locationId] = self.getLocationSimilarity(allLocationReqSemFeat, givlocId, locationId, algo)
        sorted_list = [x for x in allLocationsMatch.items()]
        sorted_list.sort(key=lambda x: x[1])  # sort by value
        if algo == self.ALGO_COSINE_SIMILARITY:
            sorted_list.reverse()

        return sorted_list[:numSimLocReq], locInKSem, KSemInFet

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
            maxSimilarity = np.min(imgPairDistSim)
            if algo == self.ALGO_COSINE_SIMILARITY:
                maxSimilarity = np.max(imgPairDistSim)
            dist += np.sum(maxSimilarity)
            count += 1
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

    def write_latent_semantics_task5(self, vd_model, locInKSem, KSemInFet):

        f = open("task5output.txt", "a+")
        f.write("\nVD Model " + vd_model + " -\n")

        f.write("latent semantics - \n")

        # f.write(np.array2string(KSemInFet)+"\n")
        np.savetxt('task5npsave.txt', KSemInFet)

        npsave = open('task5npsave.txt')

        for line in npsave:
            f.write("["+line+"]\n")

        f.close()

    def getTask5Items(self, loc_id , k, dimRedAlgo ):

        print("\nInput -\nlocation_id - " + loc_id + "\nDimension reduction model - " + dimRedAlgo + "\nk = " + str(k))

        f = open("task5output.txt", "w")
        f.write("\nInput \nlocation_id - " + loc_id + "\nDimension reduction model - " + dimRedAlgo + "\nk = " + str(
            k) + "\n")
        f.close()
        vd_model_list = ["CM", "CM3x3", "CN", "CN3x3", "CSD", "GLRLM", "GLRLM3x3", "HOG", "LBP", "LBP3x3"]
        # vd_model_list = ["CM", "CM3x3", "CN", "CN3x3"]
        score_list = {}
        # vd_model_weight = [9, 81, 11, 99, 64, 44, 396, 81, 16, 144]
        print("Calculating rank of locations for each visual descriptor model-")
        for vd_model in vd_model_list:
            out, locInKSem, KSemInFet = self.getTask4Items(dimRedAlgo, k, loc_id, vd_model, 30)
            self.write_latent_semantics_task5(vd_model, locInKSem, KSemInFet)
            # print("VD Model " + vd_model + " -")
            score_list[vd_model] = out

        print("Calculating cumulative rank -")
        all_location_details = self.getAllLocationDetails()
        location_ids = all_location_details.keys()
        location_with_model_scores = {}

        for location in location_ids:
            rank_sum = 0
            for mod in score_list:
                for var in score_list[mod]:
                    if location == var[0]:
                        rank_sum += score_list[mod].index(var) + 1
                        break
            location_with_model_scores[location] = rank_sum / len(vd_model_list)

        location_with_model_scores_sorted = sorted(location_with_model_scores.items(), key=lambda kv: kv[1])


        f = open("task5output.txt", "a+")

        print("\nOutput -\n5 Similar locations -")
        f.write("\nOutput -\n5 Similar locations -\n")
        print("location_id\t\tScore")
        f.write("location_id\t\tScore\n")
        for i in range(5):
            loc = location_with_model_scores_sorted[i][0]
            print("\t" + str(loc) + "\t\t\t " + str(location_with_model_scores_sorted[i][1]))
            f.write("\t" + str(loc) + "\t\t\t " + str(location_with_model_scores_sorted[i][1])+"\n")
        f.close()
# out = obj.getTask5Items("LDA",2,"1","CM",5)
# print(out)
