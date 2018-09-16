#!/usr/bin/python2.7

from pymongo import MongoClient
import os
import traceback
from bs4 import BeautifulSoup

DB_NAME = "phase1"
COLLECTION_TEXTUAL_NAME = "textual"
COLLECTION_VISUAL_NAME = "visual"
LOCATION_NAME_TO_ID_MAP = {}

'''
Useful commands for using MongoDB from terminal

client.drop_database('phase1')
client = MongoClient()
db = client.phase1
col = db.visual
pprint.pprint(col.find_one())
'''


def processTopicsXML():
    mapping = {}
    fileHandler = open("devset_topics.xml", 'r')
    contents = fileHandler.read()
    soup = BeautifulSoup(contents, "xml")
    for topic in soup.find_all('topic'):
        title = topic.find('title').text
        location_id = topic.find('number').text
        mapping[title] = location_id

    global LOCATION_NAME_TO_ID_MAP
    LOCATION_NAME_TO_ID_MAP = mapping
    return mapping


def loadTextualData(collection):
    textualFilenames = {"devset_textTermsPerUser.txt": "userId",
                        "devset_textTermsPerImage.txt": "imageId",
                        "devset_textTermsPerPOI.txt": "locationId"}

    try:
        for fileName, identifier in textualFilenames.iteritems():
            print "Persisting - " + fileName + "into mongoDB\n"
            with open(fileName, "r") as textualUser:
                all_json = []
                for line in textualUser:
                    text_list = []
                    data = {}
                    entry = None

                    if fileName == "devset_textTermsPerPOI.txt":
                        location = line[:(line.index('"')) - 1]
                        location_id = LOCATION_NAME_TO_ID_MAP.get(location.replace(' ', '_'), None)
                        if location_id is None:
                            print "ouch"
                            exit(-1)

                        data[identifier] = location_id
                        entry = line[line.index('"'):].split()
                    else:
                        entry = line.split()
                        data[identifier] = entry[0]
                        entry = entry[1:]

                    for index in range(len(entry)):
                        if index % 4 == 0 and index < len(entry):
                            inner_data = {}
                            inner_data['term'] = entry[index].replace('"', '')
                            inner_data['tf'] = float(entry[index + 1])
                            inner_data['df'] = float(entry[index + 2])
                            inner_data['tf-idf'] = float(entry[index + 3])

                            text_list.append(inner_data)

                        index += 1

                    data['desc'] = text_list
                    all_json.append(data)

                collection.insert_many(all_json)

    except Exception as e:
        traceback.print_exc()


def loadVisualData(collection):
    cwd = os.getcwd()
    csv_location = cwd + '/img'
    all_json = {}

    try:
        for fileName in os.listdir(csv_location):
            print fileName
            locationId = LOCATION_NAME_TO_ID_MAP[fileName.split()[0]]
            model = fileName.split()[1].split('.')[0]
            with open(csv_location + '/' + fileName, 'r') as f:
                for line in f:
                    data = {}
                    entry = line.strip().split(',')
                    imageId = entry[0]
                    features = [float(x) for x in entry[1:]]

                    if imageId in all_json:
                        all_json[imageId][model] = features
                    else:
                        data['imageId'] = imageId
                        data['locationId'] = locationId
                        data[model] = features
                        all_json[imageId] = data

        collection.insert_many(all_json.values())


    except Exception as e:
        raise e
        print "Error in loadVisualData()!"


def main():
    # Getting Client Connection from MongoDB
    client = MongoClient('mongodb://localhost:27017/')

    # Creating a New DB in MongoDB
    print "Creating database in MongoDB named as " + DB_NAME
    db = client[DB_NAME]

    # Creating a collection named textual in MongoDB
    print "Creating a collection in " + DB_NAME + " named as " + COLLECTION_TEXTUAL_NAME
    textualCollection = db[COLLECTION_TEXTUAL_NAME]

    processTopicsXML()

    print "Loading textual descriptors into mongoDB"
    loadTextualData(textualCollection)

    # Creating a collection named visual in MongoDB
    print "Creating a collection in " + DB_NAME + " named as " + COLLECTION_VISUAL_NAME
    visualCollection = db[COLLECTION_VISUAL_NAME]

    print "Loading visual descriptors into mongoDB"
    loadVisualData(visualCollection)


if __name__ == "__main__":
    main()