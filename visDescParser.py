import math
from pprint import pprint
import time
import numpy

class VisDescParser:

    ALGO_EUCLIDEAN_DISTANCE = "euclidean_distance"

    ALGO_COSINE_SIMILARITY = "cosine_similarity"

    ALGO_CHI_SQUARE_DISTANCE = "chi_square_distance"

    @staticmethod
    def euclideanDistance(vector1, vector2):
        npv1 = numpy.asarray(vector1)
        mpv2 = numpy.asarray(vector2)
        distance = numpy.linalg.norm(npv1 - mpv2)
        return distance

    @staticmethod
    def cosineSimilarity(vector1, vector2):
        npv1 = numpy.asarray(vector1)
        npv2 = numpy.asarray(vector2)
        dotproduct = numpy.sum(npv1*npv2)
        npv1abs = math.sqrt(numpy.sum(npv1 * npv1))
        npv2abs = math.sqrt(numpy.sum(npv2 * npv2))
        similarity = dotproduct/(npv1abs*npv2abs)
        return similarity

    @staticmethod
    def chiSquareDistance(vector1, vector2):
        npv1 = numpy.asarray(vector1)
        npv2 = numpy.asarray(vector2)
        numerator = (npv1 - npv2)**2
        denominator = (npv1 + npv2)
        npdistance = numpy.divide(numerator, denominator, where=denominator!=0)
        distance = numpy.sum(npdistance)
        return distance

    def getLocationModelData(self, visDesBasePath, modelName, locationTitle):
        filePath = self.getFilePath(visDesBasePath, modelName, locationTitle)
        locationModelData = self.parseVisDesc(filePath)
        return locationModelData

    @staticmethod
    def getFilePath(visDesBasePath, modelName, locationTitle):
        filePath = visDesBasePath+locationTitle+' '+modelName+'.csv'
        return filePath

    @staticmethod
    def parseVisDesc(filePath):
        #print(filePath)
        newformat = {}
        with open(filePath) as openfileobject:
            for line in openfileobject:
                line = line.strip()
                lineArr = line.split(',')
                item_id = lineArr[0]
                lineArr.pop(0)
                for i in range(0, len(lineArr)):
                    lineArr[i] = float(lineArr[i])
                newformat[item_id] = lineArr
        return newformat

    def getAllLocationData(self, visDesBasePath, topicsData, modelName, allLocationsData):
        for locationId in topicsData:
            if not(locationId in allLocationsData):
                allLocationsData[locationId] = {}
            locationTitle = topicsData[locationId]['title']
            if not (modelName in allLocationsData[locationId]):
                allLocationsData[locationId][modelName] = self.getLocationModelData(visDesBasePath, modelName, locationTitle)
        return allLocationsData

    def getAllLocationAllModelsData(self, visDesBasePath, topicsData, allLocationsData):
        allModels = self.getAllModels()
        for locationId in topicsData:
            locationTitle = topicsData[locationId]['title']
            if not(locationId in allLocationsData):
                allLocationsData[locationId] = {}
            for modelName in allModels:
                if not (modelName in allLocationsData[locationId]):
                    allLocationsData[locationId][modelName] = self.getLocationModelData(visDesBasePath, modelName, locationTitle)
        return allLocationsData

    def getKSimilarItems(self, allLocationsData, poiLocationId, modelName, k):
        allLocationsMatch = {}
        algo = self.modelWiseDistSimAlgo(modelName)
        print("Finding ", k, " similar locations for locationId:", str(poiLocationId), ' using model:', modelName, 'and algo:', algo)
        for locationId in allLocationsData:
            if locationId != poiLocationId:
                allLocationsMatch[locationId] = self.getLocationSimilarity(allLocationsData, poiLocationId, locationId, modelName)

        sorted_list = [x for x in allLocationsMatch.items()]
        sorted_list.sort(key=lambda x: x[1]['sim'])  # sort by value
        print("Found following most similar locations:-")
        print(sorted_list[:k])
        print()
        return sorted_list[:k]

    def getAllModelsKSimilarItems(self, allLocationsData, poiLocationId, k):
        allModelsLocationsMatch = {}
        allModels = self.getAllModels()
        for model in allModels:
            totalLocations = len(allLocationsData)
            allModelsLocationsMatch[model] = self.getKSimilarItems(allLocationsData, poiLocationId, model, totalLocations)
        modelWiseLocationRank = self.getLocationRanksInAllModels(allModelsLocationsMatch)
        sorted_list = [x for x in modelWiseLocationRank.items()]
        sorted_list.sort(key=lambda x: x[1]['average'])  # sort by value
        return sorted_list[:k]

    def getLocationRanksInAllModels(self, allModelsLocationsMatch):
        modelWiseLocationRank = {}
        for model, allLocationsMatch in allModelsLocationsMatch.items():
            for item in allLocationsMatch:
                locationId = item[0]
                index = allLocationsMatch.index(item)
                if locationId not in modelWiseLocationRank:
                    modelWiseLocationRank[locationId] = {}
                    modelWiseLocationRank[locationId]['modelWise'] = {}
                    modelWiseLocationRank[locationId]['average'] = -1
                modelWiseLocationRank[locationId]['modelWise'][model] = index

        for locationId in modelWiseLocationRank:
            modelRanks = modelWiseLocationRank[locationId]['modelWise']
            rankSum = 0
            for model in modelRanks:
                rankSum += modelRanks[model]
            rankAverage = rankSum/len(modelRanks)
            modelWiseLocationRank[locationId]['average'] =  rankAverage
        return modelWiseLocationRank


    @staticmethod
    def printKSimilarItems(kSimilarItems, timeTaken):
        output = []
        for item in kSimilarItems:
            major_contri = []
            for contrib in item[1]['major_contri']:
                major_contri.append([contrib['id1'], contrib['id2']])
            outputstr = str(item[0]) + " " + str(item[1]['sim']) + " " + str(major_contri)
            output.append(outputstr)
        output.append("Time Taken :" + str(timeTaken))
        return output

    @staticmethod
    def printAllModelsKSimilarItems(allModelsKSimilarItems, timeTaken):
        output = []
        for item in allModelsKSimilarItems:
            outputstr = str(item[0]) + " " + str(item[1]['average']) + " " + str(item[1]['modelWise'])
            output.append(outputstr)
        output.append("Time Taken :" + str(timeTaken))
        return output

    def getLocationSimilarity(self, allLocationsData, locationId1, locationId2, modelName):
        location1Images = allLocationsData[locationId1][modelName]
        location2Images = allLocationsData[locationId2][modelName]
        count = 0
        dist = 0
        start = time.time()
        pairwiseDist = []
        algo = self.modelWiseDistSimAlgo(modelName)
        for location1Image, vector1 in location1Images.items():
            for location2Image, vector2 in location2Images.items():
                if algo == self.ALGO_COSINE_SIMILARITY:
                    imgPairDistSim = self.cosineSimilarity(vector1, vector2)
                elif algo == self.ALGO_CHI_SQUARE_DISTANCE:
                    imgPairDistSim = self.chiSquareDistance(vector1, vector2)
                else:
                    imgPairDistSim = self.euclideanDistance(vector1, vector2)
                dist += imgPairDistSim
                pairInfo = {'id1':location1Image, 'id2':location2Image, 'distSim': imgPairDistSim}
                pairwiseDist.append(pairInfo)
                count += 1
        avgDist = dist / count
        end = time.time()
        t3 = end - start
        print(str(locationId2), ' time:', t3, ' avgDist:', avgDist)
        pairwiseDist.sort(key=lambda x: x['distSim'])
        if algo == self.ALGO_COSINE_SIMILARITY:
            pairwiseDist.reverse()
        matchingDict = {'sim': avgDist, 'major_contri': pairwiseDist[:3]}
        return matchingDict


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

    @staticmethod
    def getAllModels():
        allModels = ["CM", "CM3x3", "CN", "CSD", "GLRLM", "GLRLM3x3", "HOG", "LBP","LBP3x3"]
        #allModels = ["GLRLM", "CN3x3"]
        return allModels