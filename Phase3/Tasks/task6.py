import numpy as np
from os import listdir
from collections import defaultdict, Counter
from sklearn.decomposition import PCA
import json


def pageRank(G, mover, s=0.85, maxerr=0.001):
    n = G.shape[0]
    G = G / np.sum(G, axis=0).reshape(1, n)
    ro, r = np.zeros((n, 1)), mover.reshape(n, 1).copy()  # np.ones((n, 1)) / n
    mover = mover.reshape(n, 1)
    k = 0
    error = float('inf')
    while error > maxerr:
        ro = r.copy()
        r = s * np.matmul(G, r) + (1 - s) * mover
        error = np.sum(np.abs(r - ro))
        k += 1
    return r


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


matrix, names, original_names = validate_data()

names_for_page_rank = names.copy()


def normalize_vector(vector):
    minimum, maximum = np.min(vector, axis=0), np.max(vector, axis=0)
    ans = (vector - minimum)
    diff = maximum - minimum
    return ans / diff


def get_PCA(vector, k):
    pca = PCA(n_components=k)
    new_vector = pca.fit_transform(vector)
    return new_vector, pca.components_


#


matrix = normalize_vector(matrix)
matrix, _ = get_PCA(matrix, 250)


def euclidean_dst(vector, original_vector):
    return np.sum((vector - original_vector) ** 2, axis=1) ** 0.5


def knn():
    with open('../Data/Classification_Input.txt') as f:
        line = f.readline()
        vector = []
        labels = []
        classification_dict = defaultdict(lambda: [])
        while line:
            id, label = line.split('\n')[0].split('\t')
            vector.append(matrix[names[id]])
            labels.append(label)
            line = f.readline()
            del names[id]
        vector = np.array(vector)
        lbls = np.array(labels)
        f = open('../Data/task6_knn.json', 'w')
        classification_file = open('../Data/KNN_Output.txt', 'w')
        for id in names:
            distances = euclidean_dst(vector, matrix[names[id]])
            distindex = np.argpartition(distances, 7)[:7]
            label = Counter(lbls[distindex].tolist())
            labels = sorted(label.items(), key=lambda x: -x[1])
            ans = []
            for x in labels:
                if x[1] == labels[0][1]:
                    ans.append((x[0], x[1]))
                    classification_dict[x[0]].append('images/{0}.jpg'.format(id))
                else:
                    break
            classification_file.write('{0} {1}\n'.format(id, ans))
        classification_file.flush()
        classification_file.close()
        f.write(json.dumps(classification_dict))
        f.flush()
        f.close()
        #     f.write('{0} {1}\n'.format(id, ans))
        # f.flush()
        # f.close()


def personalized_page_rank():
    data = np.load(input('Enter where is adjacency matrix for 5b:'))
    images = original_names
    d = names_for_page_rank
    classes = defaultdict(lambda: set())
    labels = []
    dont_classify = set()
    with open('../Data/Classification_Input.txt') as f:
        line = f.readline()
        while line:
            name, label = line.split('\n')[0].split('\t')
            dont_classify.add(d[name])
            classes[label].add(d[name])
            line = f.readline()
    ans = []
    for x in classes:
        movers = np.zeros(len(data))

        for y in classes[x]:
            movers[y] = 1 / len(classes[x])
        page_ranks = pageRank(data.T, movers, s=0.85, maxerr=0.001).reshape(-1)
        ans.append(page_ranks)
        labels.append(x)
    ans = np.array(ans)
    indexes = np.argmax(ans, axis=0)
    json_output = open('../Data/task6_ppr.json', 'w')
    classification_dict = defaultdict(lambda: [])
    f = open('../Data/PPR_Output.txt', 'w')
    for i, x in enumerate(indexes):
        if i not in dont_classify:
            f.write('{0} {1}\n'.format(images[i], labels[x]))
            classification_dict[labels[x]].append('images/' + images[i] + '.jpg')
    json_output.write(json.dumps(classification_dict))
    json_output.flush()
    json_output.close()
    f.flush()
    f.close()


if __name__ == '__main__':
    knn()
    personalized_page_rank()
