# TODO:
# - creates an user-image-location tensor (based on number of terms shared by all three)
# - performs rank-k CP decomposition of this tensor
# - creates k non-overlapping groups of users, images, and locations based on the discovered latent semantics.

from APIs.generic_apis import *
import xmltodict
import numpy as np
import operator
import os

def dotProduct(v1, v2):
    return sum(map(operator.mul, v1, v2))

locations = []
with open("../Data/devset_topics.xml") as fd:
    doc = xmltodict.parse(fd.read())
    for topic in doc['topics']['topic']:
        locations.append(topic['title'])

def task7():
    # k = int(input("Enter a value for k: "))
    tensorFileName = "userImageLocation-tensor.npy"
    cwd = os.getcwd()

    print ("Loading User Space...")
    userFile = '../Data/devset_textTermsPerUser.txt'
    userVector, userTexts, users = tDictionary_to_vector(read_text_descriptor_files(userFile))
    users = list(users)

    # print str(users.index("53074617@N00"))

    print ("Loading Image Space...")
    imageFile = '../Data/devset_textTermsPerImage.txt'
    imageVector, imageTexts, images = tDictionary_to_vector(read_text_descriptor_files(imageFile))
    images = list(images)

    print ("Loading Location Space...")
    locFile = '../Data/devset_textTermsPerPOI.txt'
    locVector, locTexts, locations = tDictionary_to_vector(read_text_descriptor_files(locFile))
    locations = list(locations)

    if os.path.exists(cwd + '/' + tensorFileName):
        print ("Loading Tensor...")
        tensor = np.load(tensorFileName)
    else:
        print ("Creating Tensor...")
        tensor_shape = (len(users), len(images), len(locations)) # user, image, location
        tensor = np.zeros(tensor_shape, dtype=int)

        # Only interested in if term exists per vector, convert elements to only 0's and 1's
        userVector[userVector > 0] = 1
        imageVector[imageVector > 0] = 1
        locVector[locVector > 0] = 1

        # User and Image number of shared terms
        # userImageMatrix = userVector @ np.transpose(imageVector)

        # Brute force approach: get # of terms shared per user, image, location triplet
        for locationIndex in range(len(locations)):
            print ("Location... ", locations[locationIndex], " - ", str(locationIndex))
            for imageIndex in range(len(images)):
                if imageIndex % int(len(images) * 0.10) == 0:
                    print (str(imageIndex / len(images) * 100))
                for userIndex in range(len(users)):
                    numSharedTerms = sum(userVector[userIndex] * imageVector[imageIndex] * locVector[locationIndex])
                    tensor[userIndex][imageIndex][locationIndex] = numSharedTerms

        np.save(tensorFileName, tensor)

    print (tensor.shape)
    print ("OI")





if __name__ == '__main__':
    task7()