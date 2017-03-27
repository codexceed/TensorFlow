import numpy as np

def normalize(featureMat):
    mean = np.mean(featureMat, axis=0)
    std = np.std(featureMat, axis=0, ddof=1)
    return (featureMat-mean)/std