#!/usr/bin/python2.7
"""Driver."""

from pymongo import MongoClient
import database_utilities as db_util
import operator
import math
from numpy import argsort
from bs4 import BeautifulSoup
from multiprocessing import Pool


def printMenu():
    """Print Initial Client Options"""
    print """\n 1) Task 1 \n 2) Task 2 \n 3) Task 3 \n 4) Task 4 \n 5) Task 5 \n 6) EXIT"""

# TODO: consolidate
def processTopicsXML():
    mapping = {}
    fileHandler = open("devset_topics.xml", 'r')
    contents = fileHandler.read()
    soup = BeautifulSoup(contents, "xml")
    for topic in soup.find_all('topic'):
        title = topic.find('title').text
        location_id = topic.find('number').text
        mapping[title] = location_id

    return mapping


def dotProduct(v1, v2):
    return sum(map(operator.mul, v1, v2))


def euclidean_dist_sim(v1, v2):
    """Calculates euclidian distance and converts/returns value as a similarity measure between 0-1"""
    diffs = map(operator.sub, v1, v2)
    return 1 / (1 + math.sqrt(dotProduct(diffs, diffs)))


def cosine_sim(v1, v2):
    if dotProduct(v1, v2) != 0:
        return dotProduct(v1, v2) / (math.sqrt(dotProduct(v1, v1)) * math.sqrt(dotProduct(v2, v2)))

    return 0


def chi_squared_sim(v1, v2):

    def perElement(x, y):
        if (x+y) == 0:
            return 0
        return math.pow(x - y, 2) / (x + y)

    return 1 / (1 + sum(map(perElement, v1, v2)))

def termsWithHighSim(v1, v2, m):
    """Utility func. for tasks 1-3: returns indices of m terms contributing the most to cosine similarity"""
    elementWiseMult = map(operator.mul, v1, v2)
    sortedBySim = argsort(elementWiseMult)[::-1]
    return sortedBySim[:m]


def task4(visualCollection):
    locationIDMatch, model, k = raw_input("Enter parameters locationID, model, and k, as single line (e.g, 10 CN3x3 7)\t: ")\
                                    .strip()\
                                    .split()
    k = int(k)

    queryImageIDs = []    # Ordered list of imageId's for queried location
    queryFeatureList = [] # Ordered list of features for each image above
    for doc in visualCollection.find({"locationId": locationIDMatch}):
        queryImageIDs.append(doc['imageId'])
        queryFeatureList.append(doc[model])

    # Grab locationId's of all other locations
    otherLocations = processTopicsXML().values()
    otherLocations.remove(locationIDMatch)

    locSimMap = {}  # locationId->similarity_score , used to hold/return final similarity scores
    locTopImagePairMap = {}  # Holds top (3) image pairs per location

    for loc in otherLocations:
        """Logic Overview - "Brute-force"/cross-product approach using Euclidean distance/similarity 
        - For each location:
            1) compare all m images to all n images in queried location
                - keep the max similarity pair for each 1 (of m) to n distance calculation
            2) Take the average of top-z percent similarity scores for final sim. score for location
        """

        filteredImageToDistSim = {}  # imageID_pair->similarity_score for max() of each 1 (of m) to n comparisons
        for img in visualCollection.find({"locationId": loc}):

            # Logic for point 1 above
            imageToDistSim = {}  # imageID_pairs->similarity_score for each 1 (of m) to n comparisons
            v2 = img[model]
            for i, v1 in enumerate(queryFeatureList):
                imageToDistSim[img['imageId'] + '-' + queryImageIDs[i]] = euclidean_dist_sim(v1, v2)

            key_max = max(imageToDistSim.iterkeys(), key=(lambda key: imageToDistSim[key]))
            filteredImageToDistSim[key_max] = imageToDistSim[key_max]

        # Logic for point 2 above
        z = 0.15  # percentage of top similarity scores to use towards average
        topZ_percent_index = int(len(filteredImageToDistSim) * z)
        sortedDict = sorted(filteredImageToDistSim.iteritems(), key=operator.itemgetter(1), reverse=True)

        avg_loc_dist_sim = sum([x[1] for x in sortedDict][:topZ_percent_index]) / topZ_percent_index # locImageCount
        locSimMap[loc] = avg_loc_dist_sim

        # Logic to get top 3 images contributing to sim. score
        locTopImagePairMap[loc] = []
        for i in range(3):
            locTopImagePairMap[loc].append(sortedDict[i][0])

    sortedBySimScore = sorted(locSimMap.items(), key=lambda x: x[1], reverse=True)

    print "\n**************************************************"
    print "Results given - locationId " + locationIDMatch + " - " + model

    for locId, simScore in sortedBySimScore[:k]:
        print "\nSimilarity Score: " + str(simScore) + "\t For location: " + locId
        print "3 image pairs w/ highest sim. contribution:"
        for imagePair in locTopImagePairMap[locId]:
            z = imagePair.split('-')
            print "\t" + z[0] + ' - ' + z[1]

    print "************************************************** \n"


def locationSimilarityByModel(input):
    """
    Logic of task4() (similarity between two locations using inputted model type) but tweaked to be used
    by multiprocessing module.
    :param input: list of arguments [model, similarity function, locationID]
    :return: sorted list of similarities (score and locationID) by score
    """
    model, similarityFunction, locationIDMatch = input
    # Getting Client Connection from MongoDB
    client = MongoClient('mongodb://localhost:27017/')

    db = client[db_util.DB_NAME]
    visualCollection = db[db_util.COLLECTION_VISUAL_NAME]

    print "Starting " + model + " on " + locationIDMatch
    queryImageIDs = []  # Ordered list of imageId's for queried location
    queryFeatureList = []  # Ordered list of features for each image above
    for doc in visualCollection.find({"locationId": locationIDMatch}):
        queryImageIDs.append(doc['imageId'])
        queryFeatureList.append(doc[model])

    # Grab locationId's of all other locations
    otherLocations = processTopicsXML().values()
    otherLocations.remove(locationIDMatch)

    locSimMap = {}  # locationId->similarity_score , used to hold/return final similarity scores
    locTopImagePairMap = {}  # Holds top (3) image pairs per location

    for loc in otherLocations:
        """Logic Overview - "Brute-force"/cross-product approach using Euclidean distance/similarity 
        - For each location:
            1) compare all m images to all n images in queried location
                - keep the max similarity pair for each 1 (of m) to n distance calculation
            2) Take the average of top-z percent similarity scores for final sim. score for location
        """

        filteredImageToDistSim = {}  # imageID_pair->similarity_score for max() of each 1 (of m) to n comparisons
        for img in visualCollection.find({"locationId": loc}):

            # Logic for point 1 above
            imageToDistSim = {}  # imageID_pairs->similarity_score for each 1 (of m) to n comparisons
            v2 = img[model]
            for i, v1 in enumerate(queryFeatureList):
                imageToDistSim[img['imageId'] + '-' + queryImageIDs[i]] = similarityFunction(v1, v2)

            key_max = max(imageToDistSim.iterkeys(), key=(lambda key: imageToDistSim[key]))
            filteredImageToDistSim[key_max] = imageToDistSim[key_max]

        # Logic for point 2 above
        z = 0.15  # percentage of top similarity scores to use towards average
        topZ_percent_index = int(len(filteredImageToDistSim) * z)
        sortedDict = sorted(filteredImageToDistSim.iteritems(), key=operator.itemgetter(1), reverse=True)

        avg_loc_dist_sim = sum([x[1] for x in sortedDict][:topZ_percent_index]) / topZ_percent_index  # locImageCount
        locSimMap[loc] = avg_loc_dist_sim

        # Logic to get top 3 images contributing to sim. score
        locTopImagePairMap[loc] = []
        for i in range(3):
            locTopImagePairMap[loc].append(sortedDict[i][0])

    sortedBySimScore = sorted(locSimMap.items(), key=lambda x: x[1], reverse=True)
    return sortedBySimScore


def task5():
    locationIDMatch, k = raw_input(
        "Enter parameters locationID and k, as single line (e.g, 10 7)\t: ") \
        .strip() \
        .split()
    k = int(k)

    models = ["CM", "CM3x3", "CN", "CN3x3", "CSD", "GLRLM", "GLRLM3x3", "HOG", "LBP", "LBP3x3"]
    #models = ["CN", "LBP"]
    # Maps the similarity measure function to use for each model
    models_to_functions = {"CM": euclidean_dist_sim,
                           "CM3x3": euclidean_dist_sim,
                           "CN": euclidean_dist_sim,
                           "CN3x3": euclidean_dist_sim,
                           "CSD": cosine_sim,
                           "GLRLM": chi_squared_sim,
                           "GLRLM3x3": chi_squared_sim,
                           "HOG": cosine_sim,
                           "LBP": chi_squared_sim,
                           "LBP3x3": chi_squared_sim}

    # Create 10 processes, one for each model
    p = Pool(processes=len(models))
    args = [[x, models_to_functions[x], locationIDMatch] for x in models]
    results = p.map(locationSimilarityByModel, args)
    p.close()

    # locationId->(similarity_score, [CM_contribution, CM3x3_contribution, ...])
    # Used to hold/return final similarity scores
    locSimMap = {}
    for i, result in enumerate(results):
        """Populate locSimMap with similarity scores and individual model contributions"""

        for locID, simScore in result:
            model_contribution = simScore * 0.10
            if i == 0:
                locSimMap[locID] = [model_contribution, [model_contribution]]
            else:
                locSimMap[locID][0] += model_contribution
                locSimMap[locID][1].append(model_contribution)

    sortedBySimScore = sorted(locSimMap.items(), key=lambda x: x[1][0], reverse=True)

    print "\n**************************************************"
    print "Results given - locationId " + locationIDMatch

    for locId, data in sortedBySimScore[:k]:
        simScore = data[0]
        print "\nSimilarity Score: " + str(simScore) + "\t For location: " + locId
        print "Individual contributions of visual models:"
        for i, model_contribution in enumerate(data[1]):
            print "\t" + models[i] + ' - ' + str(model_contribution)

    print "************************************************** \n"


def textualTask(textualCollection, IDType):
    matcherID, model_input, k = raw_input("Enter parameters uniqueID, model, and k, as single line (e.g, 39052554@N00 TF 5)\t: ")\
                                    .strip()\
                                    .split()
    k = int(k)

    # Logic to map model input to correct usage
    options = {'TF': 'tf', 'DF': 'df', 'TF-IDF': 'tf-idf'}
    model_type = options.get(model_input, None)
    if model_type is None:
        print "Invalid Model"
        exit(-1)

    query = textualCollection.find_one({IDType: matcherID})

    terms = []
    v1 = []
    tf_sum = 0
    for entry in query['desc']:
        terms.append(entry['term'])
        v1.append(entry[model_type])
        tf_sum += entry['tf']

    # Normalize TF if it is used
    if model_type == 'tf' or model_type == 'tf-idf':
        v1 = map(lambda x: x / tf_sum, v1)

    simMap = {}  # Holds final similarity scores for query entity to every other entity

    for doc in textualCollection.find({IDType: {"$exists": True, "$nin": [matcherID]}}):
        termToModelMap = dict(map(lambda x: (x['term'], x[model_type]), doc['desc']))
        termToTFMap = termToModelMap
        if model_type != 'tf':
            termToTFMap = dict(map(lambda x: (x['term'], x['tf']), doc['desc']))

        v2 = []
        tf_sum = 0
        for term in terms:
            v2.append(termToModelMap.get(term, 0))
            tf_sum += termToTFMap.get(term, 0)

        # Normalize TF if it is used
        if model_type == 'tf' or model_type == 'tf-idf':
            v2 = map(lambda x: x / tf_sum, v2)

        simMap[doc[IDType]] = cosine_sim(v1, v2)

    sortedBySimScore = sorted(simMap.items(), key=lambda x: x[1], reverse=True)

    print "\n**************************************************"
    print "Results given - " + IDType + " - " + matcherID + " - " + model_type
    for matcherID, simScore in sortedBySimScore[:k]:
        """
        - Logic to print results.
        - Post-processing to find 3 terms w/ highest similarity contribution for each match.
        """

        doc = textualCollection.find_one({IDType: matcherID})
        termToModelMap = dict(map(lambda x: (x['term'], x[model_type]), doc['desc']))
        termToTFMap = termToModelMap
        if model_type != 'tf':
            termToTFMap = dict(map(lambda x: (x['term'], x['tf']), doc['desc']))

        v2 = []
        tf_sum = 0
        for term in terms:
            v2.append(termToModelMap.get(term, 0))
            tf_sum += termToTFMap.get(term, 0)

        # Normalize TF if it is used
        if model_type == 'tf' or model_type == 'tf-idf':
            v2 = map(lambda x: x / tf_sum, v2)

        termIndices = termsWithHighSim(v1, v2, 3)
        print "\nSimilarity Score: " + str(simScore) + "\t" + IDType + ": " + matcherID
        print "3 Terms w/ highest sim. contribution:"
        for index in termIndices:
            print "\t" + terms[index]

    print "************************************************** \n"


def main():
    # Getting Client Connection from MongoDB
    client = MongoClient('mongodb://localhost:27017/')

    db = client[db_util.DB_NAME]
    textualCollection = db[db_util.COLLECTION_TEXTUAL_NAME]
    visualCollection = db[db_util.COLLECTION_VISUAL_NAME]

    while(1):
        printMenu()
        option = int(raw_input("-- Please select an option from above --"))
        if option == 1:
            textualTask(textualCollection, "userId")
        elif option == 2:
            textualTask(textualCollection, "imageId")
        elif option == 3:
            textualTask(textualCollection, "locationId")
        elif option == 4:
            task4(visualCollection)
        elif option == 5:
            task5()
        elif option == 6:
            exit()
        else:
            print "Invalid input, try again"


if __name__ == "__main__":
    main()