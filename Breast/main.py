import os
import logging
import random
import numpy as np

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score

from model import FM
from load_data import get_data, get_gene_peg
from feature_selection import variance_threshold, t_score, fisher_score
from logger import Logger


def train(train_datasets, test_datasets, model, optimizer, k, device, logger):
    for i in range(1, 3000 + 1):
        for dataset in train_datasets:
            train_inputs = torch.from_numpy(dataset['train_inputs']).to(device)
            train_targets = torch.from_numpy(dataset['train_targets']).to(device)
            outputs = model(train_inputs)
            predictions = (outputs > 0.).float()
            dataset['train_ce_loss'] = F.binary_cross_entropy_with_logits(outputs, train_targets)
            dataset['train_acc'] = accuracy_score(y_true=train_targets.tolist(), y_pred=predictions.tolist())

        ce_loss = torch.stack([dataset['train_ce_loss'] for dataset in train_datasets]).mean()
        penalty_loss = torch.stack([dataset['train_ce_loss'] for dataset in train_datasets]).var()
        train_acc = np.nanmean([dataset['train_acc'] for dataset in train_datasets])

        N, d = model.emb.shape
        normalized_emb = (model.emb - model.emb.mean(0)) / model.emb.std(0)
        correlation = torch.matmul(normalized_emb.t(), normalized_emb) / N
        decorrelation = (correlation).pow(2).mean()

        loss = ce_loss + decorrelation + 10 * penalty_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            for dataset in train_datasets:
                test_inputs = torch.from_numpy(dataset['test_inputs']).to(device)
                outputs = model(test_inputs)
                predictions = (outputs > 0.).float().tolist()
                prob_list = torch.sigmoid(outputs).tolist()
                dataset['test_acc'] = accuracy_score(y_true=dataset['test_targets'].tolist(), y_pred=predictions)
                dataset['test_auc'] = roc_auc_score(y_true=dataset['test_targets'].tolist(), y_score=prob_list)

            test_acc = np.nanmean([dataset['test_acc'] for dataset in train_datasets])
            logger.critical('Iter: {:5d} | ce loss : {:7.5f} | train loss : {:7.5f} | penalty : {:8.6f} | train acc: {:7.5f} | test acc: {:7.5f}'
                            .format(i, ce_loss, loss, penalty_loss, train_acc, test_acc))

        if i % 50 == 0:
            for dataset in train_datasets:
                logger.critical('{:11s} | test acc: {:7.5f} | test auc: {:7.5f}'
                                .format(dataset['dataset'], dataset['test_acc'], dataset['test_auc']))

            model.eval()
            for dataset in test_datasets:
                inputs = torch.from_numpy(dataset['inputs']).to(device)
                outputs = model(inputs)
                predictions = (outputs > 0.).float().tolist()
                prob_list = torch.sigmoid(outputs).tolist()
                dataset['test_acc'] = accuracy_score(y_true=dataset['targets'].tolist(), y_pred=predictions)
                dataset['test_auc'] = roc_auc_score(y_true=dataset['targets'].tolist(), y_score=prob_list)
                logger.critical('{:11s} | test acc: {:7.5f} | test auc: {:7.5f}'
                                .format(dataset['dataset'], dataset['test_acc'], dataset['test_auc']))


def run(dataset_list, indices, num_samples, k, lr, seed, device, logger):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    dataset = 'Epithelial'
    file_name = f'./data/{dataset_list[dataset]}.h5ad'
    print(f'loading {dataset} dataset')
    inputs_Ep, targets_Ep = get_data(file_name, indices, num_samples=20000, is_training=True)

    dataset = 'Fibroblasts'
    file_name = f'./data/{dataset_list[dataset]}.h5ad'
    print(f'loading {dataset} dataset')
    inputs_Fi, targets_Fi = get_data(file_name, indices, num_samples=20000, is_training=True)

    top_k_genes = t_score(np.concatenate([inputs_Ep, inputs_Fi]), k, np.concatenate([targets_Ep, targets_Fi]))

    inputs_Ep = inputs_Ep[:, top_k_genes]
    inputs_Fi = inputs_Fi[:, top_k_genes]

    sparsity = np.float_([inputs_Ep > 0][0]).sum() / inputs_Ep.shape[0] / inputs_Ep.shape[1]
    print(f'Num. of cells in Epithelial: ', len(targets_Ep), '| Sparsity: {:6.2f}'.format(sparsity))
    sparsity = np.float_([inputs_Fi > 0][0]).sum() / inputs_Fi.shape[0] / inputs_Fi.shape[1]
    print(f'Num. of cells in Fibroblasts: ', len(targets_Fi), '| Sparsity: {:6.2f}'.format(sparsity))

    cur_state = np.random.get_state()
    np.random.shuffle(inputs_Ep)
    np.random.set_state(cur_state)
    np.random.shuffle(targets_Ep)

    cur_state = np.random.get_state()
    np.random.shuffle(inputs_Fi)
    np.random.set_state(cur_state)
    np.random.shuffle(targets_Fi)

    with open(f'{k}_genes_{seed}.txt', mode='w') as f:
        for gene in top_k_genes:
            f.write(f'{gene}\n')

    train_datasets = []
    for dataset, inputs, targets in [('Epithelial', inputs_Ep, targets_Ep), ('Fibroblasts', inputs_Fi, targets_Fi)]:
        train_inputs = inputs[: int(0.8 * num_samples)]
        train_targets = targets[: int(0.8 * num_samples)]
        test_inputs = inputs[int(0.8 * num_samples):]
        test_targets = targets[int(0.8 * num_samples):]
        train_dataset = {
            'dataset': dataset,
            'train_inputs': train_inputs,
            'train_targets': train_targets,
            'test_inputs': test_inputs,
            'test_targets': test_targets,
        }
        train_datasets.append(train_dataset)

    test_datasets = []
    for dataset in ['Endothelial', 'Pericytes']:
        file_name = f'./data/{dataset_list[dataset]}.h5ad'
        inputs, targets = get_data(file_name, indices, num_samples=8000, is_training=True)
        inputs = inputs[:, top_k_genes]
        sparsity = np.float_([inputs > 0][0]).sum() / inputs.shape[0] / inputs.shape[1]
        print(f'Num. of cells in {dataset}: ', len(targets), '| Sparsity: {:6.2f}'.format(sparsity) )
        test_dataset = {
            'dataset': dataset,
            'inputs': inputs,
            'targets': targets
        }
        test_datasets.append(test_dataset)

    model = FM(inputs_Ep.shape[1]).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    train(train_datasets, test_datasets, model, optimizer, k, device, logger)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lr = 0.001
    num_samples = 20000
    k = 1000
    seed = 0

    dataset_list = {'Endothelial': 'cell_type_3',
                    'Epithelial': 'cell_type_4',
                    'Fibroblasts': 'cell_type_5',
                    'Malignant': 'cell_type_6',
                    'Pericytes': 'cell_type_9'}

    indices = get_gene_peg(dataset_list)

    arch_name = os.path.basename(__file__).split('.')[0][4:]
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    filename = f'log/{arch_name}_{k}_{lr}_{seed}.log'
    logger = Logger(filename, formatter)

    run(dataset_list, indices, num_samples, k, lr, seed, device, logger)