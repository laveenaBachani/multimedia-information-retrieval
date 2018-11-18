import numpy as np
from os import listdir
from collections import defaultdict, Counter
from scipy import stats


def pageRank(G, mover, s=0.85, maxerr=0.001):
    n = G.shape[0]
    G = G / np.sum(G, axis=0).reshape(1, n)
    ro, r = np.zeros((n, 1)), np.ones((n, 1)) / n
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
    image_names = {x: i for i, x in enumerate(sorted(list(image_names)))}
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

    return ans, image_names


matrix, names = validate_data()


def euclidean_dst(vector, original_vector):
    return np.sum((vector - original_vector) ** 2, axis=1) ** 0.5


def knn():
    with open('../Data/Classification_Input.txt') as f:
        line = f.readline()
        vector = []
        labels = []

        while line:
            id, label = line.split('\n')[0].split('\t')
            vector.append(matrix[names[id]])
            labels.append(label)
            line = f.readline()
            del names[id]
        vector = np.array(vector)
        lbls = np.array(labels)
        for id in names:
            distances = euclidean_dst(vector, matrix[names[id]])
            distindex = np.argpartition(distances, 6)[:6]
            label = Counter(lbls[distindex].tolist())
            labels = sorted(label.items(), key=lambda x: -x[1])
            ans = []
            for x in labels:
                if x[1] == labels[0][1]:
                    ans.append((x[0], x[1]))
                else:
                    break
            print(id, ans)


def personalized_page_rank():
    data = np.load('../Data/adj_matrix_new.npy')
    images = []
    d = {}
    with open('../Data/devset_textTermsPerImage.txt') as f:
        line = f.readline()
        i = 0
        while line:
            line = line.split()[0]
            d[line] = i
            images.append(line)
            line = f.readline()
            i += 1
    classes = defaultdict(lambda: set())
    labels = []
    with open('../Data/Classification_Input.txt') as f:
        line = f.readline()
        while line:
            name, label = line.split('\n')[0].split('\t')
            classes[label].add(d[name])
            line = f.readline()
    ans = []
    for x in classes:
        movers = np.zeros(len(data))
        for y in classes[x]:
            movers[y] = 1
        page_ranks = pageRank(data.T, movers).reshape(-1)
        ans.append(page_ranks)
        labels.append(x)
    ans = np.array(ans)
    indexes = np.argmax(ans, axis=0)
    classification = []
    for i, x in enumerate(indexes):
        print(images[i], labels[x])


if __name__ == '__main__':
    personalized_page_rank()
