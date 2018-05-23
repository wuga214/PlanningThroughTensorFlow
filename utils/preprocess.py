import numpy as np


def getnorm(features, mean=[], std=[]):
    if mean == []:
        mean = np.mean(features, axis = 0)
        std = np.std(features, axis = 0)
    new_feature = (features.T - mean[:,None]).T
    new_feature = (new_feature.T / std[:,None]).T
    new_feature[np.isnan(new_feature)]=0
    return new_feature, mean, std