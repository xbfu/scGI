import scanpy as sc
import hdf5plugin
import numpy as np
import csv
import random


def get_data(file_name, indices, num_samples, gender_dict):
    """
    read data from h5ad file
    :param file_name: name of h5ad file
    :param indices: an index list containing protein-coding genes
    :param num_samples: number of samples
    :return: inputs, targets
    """
    data = sc.read_h5ad(file_name)
    all_genes = data.var_names.values

    expression_matrix = data.X
    metadata = data.obs
    states = metadata.values[:, 0]  # state: Not AD, High, Intermediate, Low
    print(list(set(states.tolist())))
    cell_names = data.obs_names.values.tolist()
    labels = np.zeros(states.shape, dtype=np.float32) + 10
    labels[states == 'Not AD'] = 0
    labels[states == 'High'] = 1
    labels[states == 'Intermediate'] = 2
    labels[states == 'Low'] = 3

    female_n = []
    female_h = []
    male_n = []
    male_h = []
    for i in range(data.n_obs):
        cell_name = cell_names[i]
        if cell_name in gender_dict.keys():
            if gender_dict[cell_name] == 0:
                if labels[i] == 0:
                    female_n.append(i)
                if labels[i] == 1:
                    female_h.append(i)
            else:
                if labels[i] == 0:
                    male_n.append(i)
                if labels[i] == 1:
                    male_h.append(i)

    print('Not AD + High: ', np.where(labels < 2)[0].size)
    print('Four groups: ', len(female_n)+len(female_h)+len(male_n)+len(male_h))
    female_n = random.sample(female_n, int(num_samples / 4))
    female_h = random.sample(female_h, int(num_samples / 4))
    male_n = random.sample(male_n, int(num_samples / 4))
    male_h = random.sample(male_h, int(num_samples / 4))
    cell_list = female_n + female_h + male_n + male_h
    inputs = expression_matrix[cell_list]
    targets = labels[cell_list]
    return inputs[:, indices], targets


if __name__ == '__main__':
    indices = []
    with open('name_list.txt', 'r') as f:
        for gene in f:
            indices.append(int(gene.split(' ')[0]))

    gender2binary = {'Female': 0, 'Male': 1}
    gender_information_name = 'cell_information.csv'
    gender_dict = dict()

    with open(gender_information_name) as f:
        reader = csv.reader(f)
        next(reader)
        for i, row in enumerate(reader):
            gender_dict[row[0]] = gender2binary[row[1]]

    for seed in range(5):
        random.seed(seed)
        dataset_list = [9, 11, 20]
        num_samples = 20000
        for dataset in dataset_list:
            file_name = f'./data/cell_type_{dataset}.h5ad'
            print(f'loading {dataset} dataset')
            inputs, targets = get_data(file_name, indices, num_samples, gender_dict)
            np.save(f'./data/inputs_{dataset}_{seed}.npy', inputs)
            np.save(f'./data/labels_{dataset}_{seed}.npy', targets)
            print(f'Num. of cells in {dataset}: ', len(targets))

        dataset_list = [18, 19, 23]
        num_samples = 4000
        for dataset in dataset_list:
            file_name = f'./data/cell_type_{dataset}.h5ad'
            print(f'loading {dataset} dataset')
            inputs, targets = get_data(file_name, indices, num_samples, gender_dict)
            np.save(f'./data/inputs_{dataset}_{seed}.npy', inputs)
            np.save(f'./data/labels_{dataset}_{seed}.npy', targets)
            print(f'Num. of cells in {dataset}: ', len(targets))
