import argparse, numpy as np
from heapq import *
from collections import defaultdict

np.seterr(divide='ignore', invalid='ignore')
locations = ['acropolis_athens', 'agra_fort', 'albert_memorial', 'altes_museum', 'amiens_cathedral',
             'angel_of_the_north', 'angkor_wat', 'ara_pacis', 'arc_de_triomphe', 'aztec_ruins', 'berlin_cathedral',
             'big_ben', 'bok_tower_gardens', 'brandenburg_gate', 'cabrillo', 'casa_batllo', 'casa_rosada',
             'castillo_de_san_marcos', 'chartres_cathedral', 'chichen_itza', 'christ_the_redeemer_rio',
             'civic_center_san_francisco', 'cn_tower', 'cologne_cathedral', 'colosseum', 'hearst_castle',
             'la_madeleine', 'montezuma_castle', 'neues_museum', 'pont_alexandre_iii']


def parse_args():
    argument_parse = argparse.ArgumentParser()
    argument_parse.add_argument('--location_id', help='The location you want to choose', type=str, required=True)
    argument_parse.add_argument('--model', help='The model you want to use(TF/DF/TF-IDF)', type=str, required=True)
    argument_parse.add_argument('--k', help='The number of neighbours you want to find', type=int, required=True)
    args = argument_parse.parse_args()
    return args


def read_data(model, location_to_read_from):
    m = defaultdict(lambda: defaultdict(lambda: []))
    image_ids = defaultdict(lambda: defaultdict(lambda: []))
    for l in locations:
        array = []
        image_id = []
        f = open('{0}/{1} {2}.csv'.format(location_to_read_from, l, model))
        x = f.readline()
        while x:
            content = x.split(',')
            array.append(list(map(lambda x: float(x), content[1:])))
            image_id.append(content[0])
            x = f.readline()
        m[l] = np.array(array, np.float64)
        image_ids[l] = image_id
    return m, image_ids


def eucledian(dictionary, location, image_ids, k, num_matches=50):
    d = {}
    heap_setup = [[] for _ in range(len(dictionary[location]))]
    heap = []
    counter = defaultdict(lambda: [])
    scoring = defaultdict(lambda: [])
    ans = []

    def heap_key(x):
        return x[0][0]

    for x in dictionary:
        if x != location:
            for i, y in enumerate(dictionary[location]):
                matrix = ((dictionary[x] - y.reshape(1, y.shape[0])) ** 2).sum(axis=1).reshape(dictionary[x].shape[0],
                                                                                               1)
                distances = np.sqrt(matrix).tolist()
                image_id = image_ids[x]
                new_loc = [x] * len(image_id)
                new_loc_im = [str(image_ids[location][i])] * len(image_id)
                new_array = list(zip(distances, image_id, new_loc, new_loc_im))
                heap_setup[i].extend(new_array)
    for i in range(len(heap_setup)):
        heap_setup[i] = sorted(heap_setup[i], key=heap_key)
        heappush(heap, (heap_setup[i][0][0][0], i, 0))

    while len(ans) < k:
        score, index, index_heap = heappop(heap)
        # print(score)
        loc, id = heap_setup[index][index_heap][2], heap_setup[index][index_heap][1]
        if len(counter[loc]) < num_matches:

            counter[loc].append(heap_setup[index][index_heap][3])

            scoring[loc].append((id, score))
            if len(counter[loc]) == num_matches:
                ans.append((loc, list(zip(scoring[loc], counter[loc]))))
        if index_heap + 1 < len(heap_setup[index]):
            heappush(heap, (heap_setup[index][index_heap + 1][0][0], index, index_heap + 1))

    return ans


def chi_squared(dictionary, location, image_ids, k, num_matches=50):
    d = {}
    heap_setup = [[] for _ in range(len(dictionary[location]))]
    heap = []
    counter = defaultdict(lambda: [])
    scoring = defaultdict(lambda: [])
    ans = []

    def heap_key(x):
        return x[0][0]

    for x in dictionary:
        if x != location:
            for i, y in enumerate(dictionary[location]):
                y = y.reshape(1, y.shape[0])
                matrix = ((dictionary[x] - y) ** 2) / np.abs(y)

                matrix = matrix.sum(
                    axis=1).reshape(dictionary[x].shape[0],
                                    1)
                distances = matrix.tolist()
                image_id = image_ids[x]
                new_loc = [x] * len(image_id)
                new_loc_im = [str(image_ids[location][i])] * len(image_id)
                new_array = list(zip(distances, image_id, new_loc, new_loc_im))
                heap_setup[i].extend(new_array)
    for i in range(len(heap_setup)):
        heap_setup[i] = sorted(heap_setup[i], key=heap_key)
        heappush(heap, (heap_setup[i][0][0][0], i, 0))

    while len(ans) < k:
        score, index, index_heap = heappop(heap)
        # print(score)
        loc, id = heap_setup[index][index_heap][2], heap_setup[index][index_heap][1]
        if len(counter[loc]) < num_matches:

            counter[loc].append(heap_setup[index][index_heap][3])

            scoring[loc].append((id, score))
            if len(counter[loc]) == num_matches:
                ans.append((loc, list(zip(scoring[loc], counter[loc]))))
        if index_heap + 1 < len(heap_setup[index]):
            heappush(heap, (heap_setup[index][index_heap + 1][0][0], index, index_heap + 1))

    return ans


def cosine(dictionary, location, image_ids, k, num_matches=50):
    d = {}
    heap_setup = [[] for _ in range(len(dictionary[location]))]
    heap = []
    counter = defaultdict(lambda: [])
    scoring = defaultdict(lambda: [])
    ans = []

    def heap_key(x):
        return x[0][0]

    for x in dictionary:
        if x != location:
            for i, y in enumerate(dictionary[location]):
                y = y.reshape(1, y.shape[0])
                temp = np.sqrt((y ** 2).sum(axis=1).reshape(1, 1))
                matrix = (dictionary[x] * y)

                matrix = matrix.sum(
                    axis=1).reshape(dictionary[x].shape[0],
                                    1)

                denom = (np.sqrt((dictionary[x] ** 2).sum(axis=1)).reshape(matrix.shape[0], 1)) * temp
                matrix = matrix / denom
                distances = matrix.tolist()
                image_id = image_ids[x]
                new_loc = [x] * len(image_id)
                new_loc_im = [str(image_ids[location][i])] * len(image_id)
                new_array = list(zip(distances, image_id, new_loc, new_loc_im))
                heap_setup[i].extend(new_array)
    for i in range(len(heap_setup)):
        heap_setup[i] = sorted(heap_setup[i], key=heap_key)
        heappush(heap, (-heap_setup[i][0][0][0], i, 0))

    while len(ans) < k:
        score, index, index_heap = heappop(heap)
        # print(score)
        loc, id = heap_setup[index][index_heap][2], heap_setup[index][index_heap][1]
        if len(counter[loc]) < num_matches:

            counter[loc].append(heap_setup[index][index_heap][3])

            scoring[loc].append((id, -score))
            if len(counter[loc]) == num_matches:
                ans.append((loc, list(zip(scoring[loc], counter[loc]))))
        if index_heap + 1 < len(heap_setup[index]):
            heappush(heap, (-heap_setup[index][index_heap + 1][0][0], index, index_heap + 1))

    return ans


def question_4_entry(args):
    model, images = read_data(args.model, '../Data/img')
    if args.model in {'CM', 'GLRLM3x3', 'LBP3x3', 'LBP'}:
        return (chi_squared(model, locations[int(args.location_id)], images, int(args.k))), 'chi'
    elif args.model in {'CM3x3', 'GLRLM', 'CN', 'CN3x3'}:
        return (eucledian(model, locations[int(args.location_id)], images, int(args.k))), 'eucledian'
    else:
        return (cosine(model, locations[int(args.location_id)], images, int(args.k))), 'cosine'


if __name__ == '__main__':
    args = parse_args()
    ans, distance = question_4_entry(args)
    print('Location: ', locations[int(args.location_id)])
    print('Model: ', args.model)
    print('Distance Measure: ', distance)
    for name, images in ans:
        images = images[:3]
        print('#######################  Location {0} ########################'.format(name))
        for x in images:
            print('Image Id:', x[0][0])
            print('Score: ', x[0][1])
