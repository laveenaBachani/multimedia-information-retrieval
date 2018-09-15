import argparse

from task4 import *
from collections import defaultdict

locations = ['acropolis_athens', 'agra_fort', 'albert_memorial', 'altes_museum', 'amiens_cathedral',
             'angel_of_the_north', 'angkor_wat', 'ara_pacis', 'arc_de_triomphe', 'aztec_ruins', 'berlin_cathedral',
             'big_ben', 'bok_tower_gardens', 'brandenburg_gate', 'cabrillo', 'casa_batllo', 'casa_rosada',
             'castillo_de_san_marcos', 'chartres_cathedral', 'chichen_itza', 'christ_the_redeemer_rio',
             'civic_center_san_francisco', 'cn_tower', 'cologne_cathedral', 'colosseum', 'hearst_castle',
             'la_madeleine', 'montezuma_castle', 'neues_museum', 'pont_alexandre_iii']


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
        print('######################## Matched Locations ########################')
        print('Name :{0}'.format(x[0]))
        print('Score :{0}'.format(x[1]))
        print('Ranking from each model :{0}'.format(list(zip(models, x[2]))))
        print('-------------------------------------------------------------------')
