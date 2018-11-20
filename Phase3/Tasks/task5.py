import argparse
from os import listdir
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

np.random.seed(50)


# Functions to validate if the data is correct and if all the images has all the modals.
# return types
# ans is the 2d image-feature vector, where feature is all the concatenated feature modals
# image_names-> image- id map where id is the index of that image in the ans.
# original_image -> Sorted order of the image names
def validate_data():
    location = '../Data/img'
    dictionary = defaultdict(lambda: defaultdict(lambda: set()))
    temp = defaultdict(lambda: set())
    names = set()
    files = set()
    image_names = set()
    for x in listdir(location):
        name, file = x.split()
        names.add(name)
        files.add(file)
        with open('{0}/{1}'.format(location, x), 'r') as f:
            line = f.readline()
            while line:
                data = line.split(',')
                image_name = data[0]
                temp[name].add(image_name)
                if image_name not in dictionary[name][file]:
                    dictionary[name][file].add(image_name)
                    image_names.add(image_name)
                else:
                    raise LookupError('Repitative image_names')
                line = f.readline()

    for x in dictionary:
        for y in dictionary[x]:
            if temp[x] != dictionary[x][y]:
                raise LookupError('The images are mismatched')
    original_names = sorted(list(image_names))
    image_names = {x: i for i, x in enumerate(original_names)}
    names = sorted(list(names))
    files = sorted(list(files))
    ans = [[] for _ in range(len(image_names))]

    for x in names:
        for y in files:
            file_to_read = '{0}/{1} {2}'.format(location, x, y)
            with open(file_to_read, 'r') as f:
                line = f.readline()
                while line:
                    data = line.split(',')
                    image_name = data[0]
                    if len(ans[image_names[image_name]]) < 945:
                        ans[image_names[image_name]] = ans[image_names[image_name]] + list(
                            map(lambda x: float(x), data[1:]))
                    line = f.readline()
    ans = np.array(ans)
    return ans, image_names, original_names


# One Hash Layer
class HashTable:
    def __init__(self, number_hashes, hash_size, input_dimensions):
        self.hash_size = hash_size
        self.input_dimensions = input_dimensions
        self.number_hashes = number_hashes
        self.hashes = []
        self.hashes_dict = []
        for i in range(number_hashes):
            self.hashes.append(np.random.randn(hash_size, input_dimensions))
            self.hashes_dict.append(defaultdict(lambda: set()))

    def generate_hash(self, input_vector):
        ans = []
        for i in range(self.number_hashes):
            temp = (np.matmul(self.hashes[i], input_vector.T).T > 0).astype('int')
            temp = temp.tolist()
            for j in range(len(temp)):
                temp[j] = ''.join(str(x) for x in temp[j])
            ans.append(temp)
        return ans

    def set_item(self, input_vectors, label):
        hashes = self.generate_hash(input_vectors)

        for i in range(self.number_hashes):
            for j in range(len(input_vectors)):
                self.hashes_dict[i][hashes[i][j]].add(j)

    def get_item(self, input_vectors):
        hashes = self.generate_hash(input_vectors)
        ans_set = set()
        for i in range(self.number_hashes):
            for j in range(len(input_vectors)):
                ans_set = ans_set | self.hashes_dict[i][hashes[i][j]]
        return ans_set


def euclidean_dst(vector, original_vector):
    return np.sum((vector - original_vector) ** 2, axis=1) ** 0.5


def parse_args_process():
    argument_parse = argparse.ArgumentParser()
    argument_parse.add_argument('--k', help='The number of hash layers per layer', type=int, required=True)
    argument_parse.add_argument('--l', help='The number of layers', type=int, required=True)
    args = argument_parse.parse_args()
    return args


if __name__ == '__main__':
    # number of hashes per layer
    args = parse_args_process()
    k = args.k
    # similarity_with = '2482756687'
    # number of similar images
    # t = 5
    # number of layers
    l = args.l
    ans, image_names, names_sorted = validate_data()
    layers = []
    for _ in range(l):
        hash_table = HashTable(2, 20, ans.shape[1])
        hash_table.set_item(ans, names_sorted)
        layers.append(hash_table)
    processing_required = input('Do you want to find something similar(y/n)')
    while processing_required == 'y':
        similarity_with = input('Enter the image id')
        if similarity_with not in image_names:
            print('Following image id does not exist')
            continue
        t = int(input('Enter the number of similar images you want'))

        vector = ans[image_names[similarity_with]].reshape(1, ans.shape[1])
        similarity_indexes = set()
        for i in range(l):
            temp = layers[i].get_item(vector)
            similarity_indexes = similarity_indexes | temp
        vectors = np.take(ans, list(similarity_indexes), axis=0)
        labels = np.take(names_sorted, list(similarity_indexes))
        dst = euclidean_dst(vectors, vector)
        print(len(dst))
        temp_ans = np.argpartition(dst, min(t, len(dst) - 1))[:min(t, len(dst))]
        ans_labels = np.take(labels, temp_ans)
        fig = plt.figure()
        n = math.ceil(len(ans_labels) / 2)
        for i in range(len(ans_labels)):
            img = mpimg.imread('../Data/images/{0}.jpg'.format(ans_labels[i]))
            plt.subplot(n, 2, i + 1)
            plt.imshow(img)
        plt.show()
        ans_distance = np.take(dst, temp_ans)
        processing_required = input('Do you want to find something similar(y/n)')

    # similarity_indexes = set()
    # vector = ans[image_names[similarity_with]].reshape(1, ans.shape[1])
    # for i in range(t):
    #     temp = layers[i].get_item(vector)
    #     similarity_indexes = similarity_indexes | temp
    #
    # vectors = np.take(ans, list(similarity_indexes), axis=0)
    # labels = np.take(names_sorted, list(similarity_indexes))
    # dst = euclidean_dst(vectors, vector)
    # ans = np.argpartition(dst, t)[:t]
    # ans_labels = np.take(labels, ans)
    # ans_distance = np.take(dst, ans)
    # print(ans_labels)
    # print(ans_distance)
