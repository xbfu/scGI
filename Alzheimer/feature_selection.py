import numpy as np


def variance_threshold(features, k, labels=None):
    var = np.nanvar(features, axis=0)
    top_k_genes = np.argsort(var)[-k:]
    return top_k_genes


def t_score(features, k, labels):
    var = np.nanvar(features, axis=0)
    features = features[:, np.where(var > 0.0)[0]]
    list0 = np.where(labels == 0)[0].tolist()
    list1 = np.where(labels == 1)[0].tolist()
    n0 = len(list0)
    n1 = len(list1)
    mu0 = np.nanmean(features[list0], axis=0)
    mu1 = np.nanmean(features[list1], axis=0)
    var0 = np.nanvar(features[list0], axis=0)
    var1 = np.nanvar(features[list1], axis=0)
    t_score = np.abs(mu0 - mu1) / np.sqrt(var0 / n0 + var1 / n1)
    top_k_genes = np.argsort(t_score)[-k:]

    return np.where(var > 0.0)[0][top_k_genes]


def fisher_score(features, k, labels):
    var = np.nanvar(features, axis=0)
    features = features[:, np.where(var > 0.0)[0]]
    mu = np.nanmean(features, axis=0)
    list0 = np.where(labels == 0)[0].tolist()
    list1 = np.where(labels == 1)[0].tolist()
    n0 = len(list0)
    n1 = len(list1)
    mu0 = np.nanmean(features[list0], axis=0)
    mu1 = np.nanmean(features[list1], axis=0)
    var0 = np.nanvar(features[list0], axis=0)
    var1 = np.nanvar(features[list1], axis=0)
    numerator = n0 * (mu0 - mu) ** 2 + n1 * (mu1 - mu) ** 2
    denominator = n0 * var0 + n1 * var1
    fisher_score = numerator / denominator
    top_k_genes = np.argsort(fisher_score)[-k:]

    return np.where(var > 0.0)[0][top_k_genes]
