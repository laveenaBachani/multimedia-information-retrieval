from APIs.generic_apis import *
import xmltodict
import numpy as np
import os
from tensorly.decomposition import parafac
from joblib import Parallel, delayed
import multiprocessing


# Load name for each locationId
locations = []
with open("../Data/devset_topics.xml") as fd:
    doc = xmltodict.parse(fd.read())
    for topic in doc['topics']['topic']:
        locations.append(topic['title'])


def printGroups(groups, i, f):
    """
    Utility function for writing group information to file

    :param groups: list of length k, each bucket holds member id's for kth-group
    :param i: index of group to be printed
    :param f: file handler
    :return:
    """
    f.write("Group #" + str(i + 1) + " with " + str(len(groups[i])) + " members:\n")
    f.write(' '.join(groups[i]) + '\n')


def task7():
    """
    Core logic for task7
    """
    num_cores = multiprocessing.cpu_count()
    k = int(input("Enter a value for k: "))
    tensorFileName = "userImageLocation-tensor.npy"
    factorMatricesFileName = "factor-matrices" + str(k) + ".npy"
    cwd = os.getcwd()

    # Load each object space into dictionary where d_obj = {'id': {'term': df, ...}, ...}
    print("Loading User Space...")
    userFile = '../Data/devset_textTermsPerUser.txt'
    d_user = read_text_descriptor_files(userFile)

    print("Loading Image Space...")
    imageFile = '../Data/devset_textTermsPerImage.txt'
    d_images = read_text_descriptor_files(imageFile)

    print("Loading Location Space...")
    locFile = '../Data/devset_textTermsPerPOI.txt'
    d_locations = read_text_descriptor_files(locFile)

    user_list = list(d_user.keys())
    image_list = list(d_images.keys())
    loc_list = list(d_locations.keys())
    print(len(user_list), len(image_list), len(loc_list))

    if os.path.exists(cwd + '/' + tensorFileName):
        print("Loading Tensor...")
        tensor = np.load(tensorFileName)
    else:
        print("Creating Tensor...")

        def processInput(i):
            # Create a slice of 3-D tensor (combinations of loc & image per user)
            print('Started for user' + str(i))
            user = user_list[i]
            array = [[0 for _ in range(len(loc_list))] for _ in range(len(image_list))]
            for j in range(len(image_list)):
                for l in range(len(loc_list)):
                    image = image_list[j]
                    loc = loc_list[l]
                    # Number of terms shared by all three entities
                    union_words = d_user[user].keys() & d_images[image].keys() & d_locations[loc].keys()
                    array[j][l] += len(union_words)

            print('Ended for user' + str(i))
            return array

        tensor = Parallel(n_jobs=num_cores - 1)(delayed(processInput)(i) for i in range(len(user_list)))
        tensor = np.array(tensor)
        print(tensor.shape)
        np.save(tensorFileName, tensor)

    print('Tensor created')

    if not os.path.exists(cwd + '/' + factorMatricesFileName):
        # Perform CP decomposition via ALS
        tensor = tensor.astype(float)
        print("Performing CP Decomposition...")
        factors = parafac(tensor=tensor, rank=k, n_iter_max=150, init='random')
        np.save(factorMatricesFileName, factors)
    else:
        factors = np.load(factorMatricesFileName)

    print("Factor Matrices created")

    indexToSpaceIds = {0: user_list, 1: image_list, 2: loc_list}

    def createGroups(factor_index):
        # Create k non-overlapping groups
        f_matrix = factors[factor_index]  # factor matrix to be used
        groups = []
        for i in range(k):
            groups.append([])

        for j in range(f_matrix.shape[0]):
            # Assign object to one of k groups/latent-features that it has highest membership towards
            object_id = indexToSpaceIds[factor_index][j] # Map indices back to user/image/location id's
            group_index = np.argmax(f_matrix[j])
            groups[group_index].append(object_id)

        return groups

    groupsList = Parallel(n_jobs=num_cores - 1)(delayed(createGroups)(i) for i in [0, 1, 2])

    # Output Results
    with open("task7_output.txt", "w") as f:
        userGroups = groupsList[0]
        f.write("\n********** K-USER GROUPS **********\n")
        for i in range(k):
            printGroups(userGroups, i, f)

        imageGroups = groupsList[1]
        f.write("\n********** K-IMAGE GROUPS **********\n")
        for i in range(k):
            printGroups(imageGroups, i, f)

        locGroups = groupsList[2]
        f.write("\n********** K-LOCATION GROUPS **********\n")
        for i in range(k):
            printGroups(locGroups, i, f)


if __name__ == '__main__':
    task7()