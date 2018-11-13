import argparse
from os import listdir
from collections import defaultdict
import numpy as np


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
                    image_names.add(name + "_" + image_name)
                else:
                    raise LookupError('Repitative image_names')
                line = f.readline()

    for x in dictionary:
        for y in dictionary[x]:
            if temp[x] != dictionary[x][y]:
                raise LookupError('The images are mismatched')
    image_names = {x: i for i, x in enumerate(sorted(list(image_names)))}
    names = sorted(list(names))
    files = sorted(list(files))
    ans = [[] for _ in range(len(image_names))]

    m = 0
    for x in names:
        for y in files:
            file_to_read = '{0}/{1} {2}'.format(location, x, y)
            with open(file_to_read, 'r') as f:
                line = f.readline()
                while line:
                    data = line.split(',')
                    image_name = data[0]
                    ans[image_names[x + "_" + image_name]] = ans[image_names[x + "_" + image_name]] + list(
                        map(lambda x: float(x), data[1:]))
                    m = max(m, len(ans[image_names[x + "_" + image_name]]))

                    line = f.readline()
    ans = np.array(ans)
    return ans, image_names


class HashTable:
    def __init__(self, number_hashes, hash_size, input_dimensions):
        self.hash_size = hash_size
        self.input_dimensions = input_dimensions
        self.number_hashes = number_hashes
        self.hashes = []
        self.hashes_dict = []
        for i in range(number_hashes):
            np.random.seed((i + 1))
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
            max_check = 0
            for j in range(len(input_vectors)):
                self.hashes_dict[i][hashes[i][j]].add(label[j])
                max_check = max(max_check, len(self.hashes_dict[i][hashes[i][j]]))
            print(max_check)

    def get_item(self, input_vectors):
        hashes = self.generate_hash(input_vectors)


if __name__ == '__main__':
    ans, image_names = validate_data()
    hash_table = HashTable(2, 50, ans.shape[1])
    hash_table.set_item(ans, list(image_names.keys()))
