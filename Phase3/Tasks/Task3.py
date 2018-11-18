import numpy as np
import argparse


image_file = "../../Phase2/Data/devset_textTermsPerImage.txt"


# Argument Parser
def parse_args_process():
    argument_parse = argparse.ArgumentParser()
    argument_parse.add_argument('--k', help='The number of most dominant images you want to find', type=int, required=True)
    args = argument_parse.parse_args()
    return args


def page_rank(k):

    max_iter = 100
    alph = 0.85

    adj_mtrx = np.load('adjMatrix_new.npy')
    row_sum = sum(adj_mtrx[0])
    adj_mtrx /= row_sum
    adj_mtrx = adj_mtrx.transpose()

    n = np.size(adj_mtrx[0])
    pr = (np.ones((n, 1)))/n

    for i in range(max_iter):
        pr = alph*np.matmul(adj_mtrx, pr) + (1 - alph)*pr

    find_k_most_relevant_images(pr, k)


def find_k_most_relevant_images(score, k):

    in_file = open(image_file, encoding="utf8")
    ids = []
    id_score_dic = {}
    for line in in_file:
        ids.append(line.split()[0])

    i = 0
    for val in np.nditer(score):
        id_score_dic[ids[i]] = val
        i += 1

    sorted_by_score = sorted(id_score_dic.items(), key=lambda kv: kv[1])
    sorted_by_score.reverse()
    for i in range(k):
        print("id - "+sorted_by_score[i][0]+"\t score - "+str(sorted_by_score[i][1]))


if __name__ == '__main__':
    args = parse_args_process()
    page_rank(args.k)
