import argparse

from task4 import *
from collections import defaultdict

import xmltodict

locations = []
with open("../Data/devset_topics.xml") as fd:
    doc = xmltodict.parse(fd.read())
    for topic in doc['topics']['topic']:
        locations.append(topic['title'])


def parse_args():
    argument_parse = argparse.ArgumentParser()
    argument_parse.add_argument('--location_id', help='The location you want to choose', type=str, required=True)
    argument_parse.add_argument('--k', help='The number of neighbours you want to find', type=int, required=True)
    args = argument_parse.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    args.location_id = int(args.location_id) - 1
    models = ['CM', 'CM3x3', 'CN', 'CN3x3', 'CSD', 'GLRLM', 'GLRLM3x3', 'HOG', 'LBP', 'LBP3x3']
    K = args.k
    d = defaultdict(lambda: [])
    print('This is for location: {0}'.format(locations[args.location_id]))
    for x in models:
        args.model = x
        args.k = 29
        result, distance = question_4_entry(args)
        for i, y in enumerate(result):
            d[y[0]].append(i + 1)
        print(args.model, distance)
    items = sorted(map(lambda x: (x[0], sum(x[1]) / 10., x[1]), d.items()), key=lambda x: x[1])[:K]

    for x in items:
        print('######################## Matched Location ########################')
        print('Name :{0}'.format(x[0]))
        print('Score :{0}'.format(x[1]))
        print('Ranking from each model :{0}'.format(list(zip(models, x[2]))))
        print('-------------------------------------------------------------------')
