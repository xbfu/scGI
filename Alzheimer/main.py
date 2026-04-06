import os
import logging
import scanpy as sc
import hdf5plugin
import numpy as np
import random

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from model import FM
from logger import Logger
from feature_selection import variance_threshold, t_score, fisher_score


def train(train_datasets, test_datasets, model, optimizer, k, lr, seed, device, logger):
    for i in range(1, 3000 + 1):
        for dataset in train_datasets:
            train_inputs = torch.from_numpy(dataset['train_inputs']).to(device)
            train_targets = torch.from_numpy(dataset['train_targets']).to(device)
            outputs = model(train_inputs)
            predictions = (outputs > 0.).float().tolist()
            dataset['train_ce_loss'] = F.binary_cross_entropy_with_logits(outputs, train_targets)
            dataset['train_acc'] = accuracy_score(y_true=train_targets.tolist(), y_pred=predictions)

        ce_loss = torch.stack([dataset['train_ce_loss'] for dataset in train_datasets]).mean()
        penalty_loss = torch.stack([dataset['train_ce_loss'] for dataset in train_datasets]).var()
        train_acc = np.nanmean([dataset['train_acc'] for dataset in train_datasets])

        N, d = model.emb.shape
        normalized_emb = (model.emb - model.emb.mean(0)) / model.emb.std(0)
        correlation = torch.matmul(normalized_emb.t(), normalized_emb) / N
        decorrelation = (correlation).pow(2).mean()

        loss = ce_loss + decorrelation
        _lambda = 2
        loss += _lambda * penalty_loss

        if i > 1500:
            _lambda = 1000000
            loss += _lambda * penalty_loss
            loss /= _lambda

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
                logger.critical('{:02d}      | test acc: {:7.5f} | test auc: {:7.5f}'
                                .format(dataset['dataset'], dataset['test_acc'], dataset['test_auc']))

            model.eval()
            for dataset in test_datasets:
                inputs = torch.from_numpy(dataset['inputs']).to(device)
                outputs = model(inputs)
                predictions = (outputs > 0.).float().tolist()
                prob_list = torch.sigmoid(outputs).tolist()
                dataset['test_acc'] = accuracy_score(y_true=dataset['targets'].tolist(), y_pred=predictions)
                dataset['test_auc'] = roc_auc_score(y_true=dataset['targets'].tolist(), y_score=prob_list)
                logger.critical('{:02d}      | test acc: {:7.5f} | test auc: {:7.5f}'
                                .format(dataset['dataset'], dataset['test_acc'], dataset['test_auc']))


def run(train_celltypes, test_celltypes, num_samples, k, lr, seed, device, logger):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    dataset = train_celltypes[0]
    inputs_O = np.load(f'data/inputs_{dataset}.npy')
    targets_O = np.load(f'data/labels_{dataset}.npy')

    dataset = train_celltypes[1]
    inputs_A = np.load(f'data/inputs_{dataset}.npy')
    targets_A = np.load(f'data/labels_{dataset}.npy')

    top_k_genes = t_score(np.concatenate([inputs_O, inputs_A]), k, np.concatenate([targets_O, targets_A]))
    inputs_O = inputs_O[:, top_k_genes]
    inputs_A = inputs_A[:, top_k_genes]

    sparsity = np.float_([inputs_O > 0][0]).sum() / inputs_O.shape[0] / inputs_O.shape[1]
    print(f'Num. of cells in {train_celltypes[0]}: ', len(targets_O), '| Sparsity: {:6.2f}'.format(sparsity))
    sparsity = np.float_([inputs_A > 0][0]).sum() / inputs_A.shape[0] / inputs_A.shape[1]
    print(f'Num. of cells in {train_celltypes[1]}: ', len(targets_A), '| Sparsity: {:6.2f}'.format(sparsity))

    cur_state = np.random.get_state()
    np.random.shuffle(inputs_O)
    np.random.set_state(cur_state)
    np.random.shuffle(targets_O)

    cur_state = np.random.get_state()
    np.random.shuffle(inputs_A)
    np.random.set_state(cur_state)
    np.random.shuffle(targets_A)

    train_datasets = []
    for dataset, inputs, targets in [(train_celltypes[0], inputs_O, targets_O),
                                     (train_celltypes[1], inputs_A, targets_A)]:
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
    for dataset in test_celltypes:
        inputs = np.load(f'data/inputs_{dataset}.npy')
        inputs = inputs[:, top_k_genes]
        targets = np.load(f'data/labels_{dataset}.npy')
        sparsity = np.float_([inputs > 0][0]).sum() / inputs.shape[0] / inputs.shape[1]
        print(f'Num. of cells in {dataset}: ', len(targets), '| Sparsity: {:6.2f}'.format(sparsity) )
        test_dataset = {
            'dataset': dataset,
            'inputs': inputs,
            'targets': targets
        }
        test_datasets.append(test_dataset)

    model = FM(inputs_O.shape[1]).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    train(train_datasets, test_datasets, model, optimizer, k, lr, seed, device, logger)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lr = 0.001
    num_samples = 20000
    k = 1000
    seed = 0
    num_classes = 2

    arch_name = os.path.basename(__file__).split('.')[0][4:]
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    filename = f'log/{arch_name}_{k}_{lr}_{seed}.log'
    logger = Logger(filename, formatter)

    # Neuron: 9, 11
    # Glial: 20 | 18, 19, 23
    # GABAergic: 4, 7 | 1, 6

    train_celltypes = [11, 20]
    test_celltypes = [18, 19, 23, 9]

    run(train_celltypes, test_celltypes, num_samples, k, lr, seed, device, logger)
