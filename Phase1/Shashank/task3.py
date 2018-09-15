import argparse
from heapq import *
from common_function import *


def parse_args_process():
    argument_parse = argparse.ArgumentParser()
    argument_parse.add_argument('--location_id', help='The user you want to choose', type=str, required=True)
    argument_parse.add_argument('--model', help='The model you want to use(TF/DF/TF-IDF)', type=str, required=True)
    argument_parse.add_argument('--k', help='The number of neighbours you want to find', type=int, required=True)
    args = argument_parse.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args_process()
    user_data = fetch_data('devset_textTermsPerPOI.txt')
    heap = []
    for x in user_data:
        if x != args.location_id:
            similarity, top_3 = cosine_similarity(user_data[x], user_data[args.location_id], args.model)
            heappush(heap, (similarity, x, top_3))
            if len(heap) > args.k:
                heappop(heap)

    ans = []
    while heap:
        ans.append(heappop(heap))

    for i, x in enumerate(ans[::-1]):
        print('################### RESULT FOR LOCATION :{0} ###################'.format(i + 1))
        print('Location ID:{0}'.format(x[1]))
        print('Score:{0}'.format(x[0]))
        print('Top terms: {0}'.format(x[2]))
        print('----------------------------------------------------------')
