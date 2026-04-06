import random
import numpy as np
import csv
import scanpy as sc
import hdf5plugin


def get_data(file_name, indices, num_samples=20000, is_training=True):
    """
    read data from h5ad file
    :param file_name: name of h5ad file
    :param name_list: a list containing protein-coding genes
    :param num_classes: number of classes
    :return: inputs, targets
    :param file_name:
    :param indices:
    :param num_samples:
    :param is_training:
    :return:
    """
    data = sc.read_h5ad(file_name)
    all_genes = data.var.gene.values

    expression_matrix = data.X
    metadata = data.obs
    states = metadata.values[:, 2]  # state: Normal, Tumor
    labels = np.zeros(states.size) - 1
    labels[states == 'Normal'] = 0
    labels[states == 'Tumor'] = 1
    if is_training:
        normal = np.where(labels == 0)[0].tolist()
        tumor = np.where(labels == 1)[0].tolist()
        cell_list = random.sample(normal, int(num_samples / 2)) + random.sample(tumor, int(num_samples / 2))
        inputs = expression_matrix[cell_list]
        targets = labels[cell_list]
    else:
        inputs = expression_matrix
        targets = labels
    return inputs[:, indices], targets


def get_gene_peg(dataset_list):
    """
    get the index of protein-encoding genes (peg)
    :param dataset_list:
    :return: indices
    """
    dataset = 'Pericytes'
    file_name = f'./data/{dataset_list[dataset]}.h5ad'

    print(f'loading {dataset} dataset')
    data = sc.read_h5ad(file_name)
    all_genes = data.var.gene.values

    indices = []
    gene_list = []
    with open('gene_list.csv') as f:
        reader = csv.reader(f)
        next(reader)

        for i, row in enumerate(reader):
            gene_name = row[0]
            gene_list.append(gene_name)
            idx = all_genes.tolist().index(gene_name)
            indices.append(idx)

    return indices
