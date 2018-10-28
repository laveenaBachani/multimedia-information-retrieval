import xml.etree.ElementTree as ET
import numpy as np
import argparse
from Phase2.Modules import get_latent_features
from Phase2.APIs import generic_apis
from collections import OrderedDict

tree = ET.parse('../Data/devset_topics.xml')  # to parse location XML
root = tree.getroot()  # root node
mapping={}  # dictionary for location id to location name mapping
c = 0
for child1 in root:
            id = root[c][0].text
            mapping[root[c][1].text] = id
            c = c + 1

# input from user
def parse_args_process():
    argument_parse = argparse.ArgumentParser()
    argument_parse.add_argument('--model_name', help='The visual descriptor model you want to use (CM/CM3x3/CN/CN3x3/CSD/GLRLM/GLRLM3x3/HOG/LBP/LBP3x3 )', type=str, required=True,
                                choices=['CM','CM3x3','CN', 'CN3x3','CSD','GLRLM','GLRLM3x3','HOG','LBP', 'LBP3x3'])
    argument_parse.add_argument('--no_of_features', help='The number semantics you want to find', type=int, required=True)
    argument_parse.add_argument('--dm_model_name', help='The model you want to use (PCA/SVD/LDA)', type=str, required=True,
                                choices=['PCA', 'SVD', 'LDA'])
    argument_parse.add_argument('--input_id', help='The image id', type=str, required=True)
    args = argument_parse.parse_args()
    return args
args = parse_args_process()
modelname = args.model_name
nooffeatures = args.no_of_features
dmmodelname = args.dm_model_name
inputid=args.input_id

def algotype(modelname):  # function to select similarity algorithm
    if modelname in {"CN", "CN3x3", "CM3x3", "GLRLM","CM"}:
        algtype = "eucledean"
    elif modelname in {"GLRLM3x3", "LBP3x3", "LBP" }:
        algtype = "chisquare"
    elif modelname in {"HOG", "CSD"}:
        algtype = "cosine"
    return algtype

algtype = algotype(modelname)

# to get latent symantics for all locations and store in dictionary with location as key
allLocationdata = {}
mainfolder = "../Data/descvis/img"
for id in mapping:
    filename2 = id + ' ' + modelname + ".csv"
    allLocationdata[id], comp = get_latent_features.get_latent_features_vis_disc(mainfolder + "/" + filename2, dmmodelname, int(nooffeatures))
    print("Latent symantics for ", id, "\n", comp)
# display latent symantics

#  search inputted image id

found = 0
for id in allLocationdata:
    imageids = allLocationdata[id][:, 0:1]
    for i in range(len(imageids)):
        if int(imageids[i]) == int(inputid):
            found=1
            input_features = allLocationdata[id][i, 1:]
            input_location_id = id
            break
    if found == 1:
        break
if found == 0:
    print("Inputted image id not found in dataset")
    exit()

alldistance=np.array([])
loc_to_similarity_avg = {}
avg = 0

for location in allLocationdata:
    curr_loc_data = allLocationdata[location]
    curr_loc_feature = curr_loc_data[:, 1:]
    curr_loc_imageids = curr_loc_data[:, 0]

    if algtype == "cosine":
        distance = generic_apis.consine_similarity(curr_loc_feature, input_features)
    elif algtype == "chisquare":
        distance = generic_apis.chi_squared(curr_loc_feature, input_features)
    elif algtype == "eucledean":
        distance = generic_apis.eucledian_distance(curr_loc_feature, input_features)

    avg = distance.sum(0)
    avg = avg/distance.size
    loc_to_similarity_avg[location] = avg
    img_to_distance = np.vstack((curr_loc_imageids, distance)).T


    if alldistance.size == 0:
        alldistance = img_to_distance
    else:
         alldistance = np.concatenate((alldistance, img_to_distance), axis=0)

if algtype == "cosine":
    alldistance = alldistance[alldistance[:, 1].argsort()[::-1]]
else:
    alldistance = alldistance[alldistance[:, 1].argsort()]

most_similar_images = alldistance[:5, :]
print("5 most similar images are <imageid, similarity score>")
for row in most_similar_images:
    print(str(int(row[0])), "->", row[1])

# sort location dictionary
if algtype == "cosine":
    loc_to_similarity_avg = OrderedDict(sorted(loc_to_similarity_avg.items(), key=lambda x: x[1], reverse=True))
else:
    loc_to_similarity_avg = OrderedDict(sorted(loc_to_similarity_avg.items(), key=lambda x: x[1]))
print("5 most similar locations are <location id, similarity score>")
counter = 1
for loc in loc_to_similarity_avg:
    if counter <= 5:
        print(int(mapping[loc]), "->", loc_to_similarity_avg[loc])
        counter = counter+1
    elif counter > 5:
        break
