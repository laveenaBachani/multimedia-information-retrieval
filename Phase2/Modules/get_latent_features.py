from Phase2.APIs import generic_apis
import numpy as np


def get_latent_features_vis_disc(filename_with_path, model, k):

    orig_dataset = np.loadtxt(filename_with_path, delimiter=',')

    features = orig_dataset[:, 1:]
    features = generic_apis.normalize_vector(features)
    objects = orig_dataset[:, 0:1]

    if model == "PCA":
        latent_l_features, comp = generic_apis.get_PCA(features, k)
    elif model == "SVD":
        latent_l_features, comp = generic_apis.get_SVD(features, k)
    elif model == "LDA":
        latent_l_features, comp = generic_apis.get_LDA(features, k)
    else:
        print("Wrong Dimension Reduction Model - " + model)
        return np.empty(0)
    return np.concatenate((objects, latent_l_features), axis=1),comp
