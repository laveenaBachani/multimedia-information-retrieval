import argparse
from Phase2.APIs.generic_apis import *
import numpy as np
import os


def parse_args_process():
    argument_parse = argparse.ArgumentParser()
    argument_parse.add_argument('--file', help='What you want to choose user/image/location?', type=str, required=True,
                                )
    argument_parse.add_argument('--model', help='The model you want to use(PCA/SVD/LDA)', type=str, required=True,
                                choices=['PCA', 'SVD', 'LDA'])
    argument_parse.add_argument('--k', help='The number semantics you want to find', type=int, required=True)
    args = argument_parse.parse_args()
    return args


file_to_read = {
    'user': '../Data/devset_textTermsPerUser.txt',
    'image': '../Data/devset_textTermsPerImage.txt',
    'location': '../Data/devset_textTermsPerPOI.txt'
}
feature_reduce = {
    'PCA': get_PCA,
    'SVD': get_SVD,
    'LDA': get_LDA
}
import warnings

warnings.simplefilter('ignore', RuntimeWarning)
args = parse_args_process()
vector, columns, rows = tDictionary_to_vector(read_text_descriptor_files(file_to_read[args.file]))
vector = normalize_vector(vector)
reduced_vector, latent_semantics = feature_reduce[args.model](vector, args.k)
columns = np.array(columns)
with open('task1_output.txt', 'w') as f:
    ans, l = [], 0
    string = ''
    for i, x in enumerate(latent_semantics):
        indexes = np.argsort(-x)
        this = list(zip(np.take(columns, indexes), np.take(x, indexes)))
        ans.append(this)
        l = len(this)
        string += 'LS {0}\t'.format(i + 1)
    f.write(string + '\n')
    for i in range(l):
        string = ''
        for j in range(args.k):
            string += str(ans[j][i]) + '\t'
        f.write(string + '\n')

    f.flush()
    f.close()
