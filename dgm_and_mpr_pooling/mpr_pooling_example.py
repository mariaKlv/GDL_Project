"""
BASED ON https://github.com/crisbodnar/dgm/blob/master/eval.py
"""

from typing import List
import os.path as osp
from time import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold

from torch_geometric.nn.conv import GCNConv
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import OneHotDegree
from torch_geometric.nn.dense import MPRPooling

class GEmbedNet(torch.nn.Module):
    def __init__(self, input_dim: int = 89, hidden_dim: int = 128,
                 out_nodes: int = 20, overlap: float = 0.10,
                 pooling: bool = True):
        super(GEmbedNet, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.pooling = pooling
        self.pool = MPRPooling(out_nodes=out_nodes, overlap=overlap)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        if self.pooling:
            x, edge_index = self.pool(x, edge_index)
        return x, edge_index


class GClassifier(torch.nn.Module):
    def __init__(self, dataset, input_dim=128, hidden_dim=128):
        super(GClassifier, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.mlp = nn.Linear(hidden_dim, dataset.num_classes)

    def forward(self, x, edge_index, edge_attr):
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))

        x = F.dropout(x, training=self.training)
        x = torch.mean(x, dim=0)
        x = self.mlp(x)
        return x


class MPRModel(nn.Module):
    def __init__(self, dataset: TUDataset, hidden_dim: List[int], out_nodes: List[int], pooling_overlap: List[float],
                 pooling: bool = True):
        assert len(hidden_dim) == len(out_nodes) == len(pooling_overlap), \
            f"the hidden dimension, number of output nodes and pooling overlap should be specified for each embedding layer"

        super(MPRModel, self).__init__()

        self.input_nodes_features = dataset.num_node_features
        self.embed_nets = nn.ModuleList()
        self.n_embed_nets = len(hidden_dim)

        input_node_features = self.input_nodes_features
        for layer in range(len(hidden_dim)):
            self.embed_nets.append(GEmbedNet(input_dim=input_node_features,
                                            hidden_dim=hidden_dim[layer],
                                            out_nodes=out_nodes[layer],
                                            overlap=pooling_overlap[layer],
                                            pooling=pooling
                                            ))
            input_node_features = hidden_dim[layer]

        self.classifier_net = GClassifier(dataset=dataset, input_dim=input_node_features, hidden_dim=hidden_dim[-1])

    def forward(self, x, edge_index):
        for i in range(self.n_embed_nets):
            x, edge_index = self.embed_nets[i](x, edge_index)
        return self.classifier_net(x, edge_index, None)


def get_graph_classification_dataset(dataset: str) -> TUDataset:
    node_transform = None
    if dataset in ['COLLAB', 'REDDIT-BINARY', 'IMDB-BINARY', 'IMDB-MULTI',
                   'REDDIT-MULTI-5K']:
        node_transform = OneHotDegree(max_degree=64)

    path = osp.join(osp.dirname('./graph_datasets/'), dataset)
    dataset = TUDataset(path, name=dataset, pre_transform=node_transform)

    return dataset


def select_subset(this_dataset, indices):
    subset = this_dataset[torch.LongTensor(indices)]
    y = np.concatenate([d.y for d in subset])
    return subset, y


def evaluate_loss(y_pred, target):
    loss = F.cross_entropy(y_pred, target)
    return loss

def update_lrate(lr_anneal_coef, epochs, optimizer, epoch):
    if lr_anneal_coef and epoch >= epochs // 2:
        optimizer.param_groups[0]['lr'] = (optimizer.param_groups[0]['lr'] *
                                           lr_anneal_coef)


def run():
    epochs = 30
    lr = 0.0005
    lr_anneal_coef = 0.0
    batch_size = 32
    dataset_name = 'DD'
    out_nodes = [20, 5]
    hidden_dims = [128, 128]
    overlap = [.10, .10]
    pooling = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = get_graph_classification_dataset(dataset_name)

    kfold = StratifiedKFold(n_splits=10, shuffle=False)

    current_fold = 0
    test_accs = []

    for train_val_idxs, test_idxs in kfold.split(np.zeros(len(dataset.data.y)), dataset.data.y):
        current_fold += 1
        fold_train_start_time = time()
        fold_test_time = []
        s = '>>> 10-fold cross-validation --- fold %d' % current_fold
        print(s)

        # Split into train-val and test
        train_val_dataset, train_val_y = select_subset(dataset, train_val_idxs)
        test_dataset, test_y = select_subset(dataset, test_idxs)

        # Split first set into train and val
        kfold2 = StratifiedKFold(n_splits=9, shuffle=False)
        for train_idxs, val_idxs in kfold2.split(np.zeros(len(train_val_y)), train_val_y):
            train_dataset, train_y = select_subset(train_val_dataset, train_idxs)
            val_dataset, val_y = select_subset(train_val_dataset, val_idxs)
            break

        # Shuffle the training data
        shuffled_idx = torch.randperm(len(train_dataset))
        train_dataset, train_y = select_subset(train_dataset, shuffled_idx)

        model = MPRModel(dataset, out_nodes=out_nodes, hidden_dim=hidden_dims,
                         pooling_overlap=overlap, pooling=pooling
                         ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        max_val_acc = 0.0
        for epoch in range(epochs):
            # Train model
            train_loss = 0
            model.train()
            optimizer.zero_grad()

            for i, data in enumerate(train_dataset):
                data = data.to(device)

                y_pred = model(data.x, data.edge_index)
                y_pred = y_pred.unsqueeze(0)

                loss = evaluate_loss(y_pred, data.y)
                train_loss += loss

                (loss / batch_size).backward()
                if i % batch_size == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            train_loss /= len(train_dataset)

            # Run validation set
            val_acc = 0
            val_loss = 0
            model.eval()

            with torch.no_grad():
                for _, data in enumerate(val_dataset):
                    data = data.to(device)
                    y_pred = model(data.x, data.edge_index)
                    y_pred = y_pred.unsqueeze(0)

                    loss = evaluate_loss(y_pred, data.y).detach().cpu().numpy()
                    val_loss += loss
                    val_acc += y_pred.max(1)[1].eq(data.y)

                val_acc = float(val_acc) / len(val_dataset)
                val_loss /= len(val_dataset)

                s = ('Epoch %d - train loss %.4f, val loss %.4f, val accuracy %.4f' %
                     (epoch, train_loss, val_loss, val_acc))
                print(s)

                if val_acc > max_val_acc:
                    start_test_time = time()
                    s = 'New best validation accuracy at epoch %d: %.4f' % (epoch, val_acc)
                    max_val_acc = val_acc
                    print(s)

                    # Run test set
                    test_acc = 0
                    test_loss = 0
                    model.eval()

                    for _, data in enumerate(test_dataset):
                        data = data.to(device)

                        y_pred = model(data.x, data.edge_index)
                        y_pred = y_pred.unsqueeze(0)
                        loss = evaluate_loss(y_pred, data.y).detach().cpu().numpy()
                        test_loss += loss

                        test_acc += y_pred.max(1)[1].eq(data.y)

                    test_acc = float(test_acc) / len(test_dataset)
                    test_loss /= len(test_dataset)

                    s = ('Epoch %d - test loss %.4f, test accuracy %.4f' %
                         (epoch, test_loss, test_acc))
                    print(s)

                    fold_test_time.append(time()-start_test_time)

            update_lrate(lr_anneal_coef=lr_anneal_coef, epochs=epochs,
                         optimizer=optimizer, epoch=epoch)

        test_accs.append(test_acc)
        fold_training_time = time() - fold_train_start_time - sum(fold_test_time)
        print("Training for fold {0} took {1:.2}s".format(current_fold, fold_training_time))

    s = 'Test accuracies: %s, %.4f +- %.4f' % (str(test_accs),
                                               np.mean(np.array(test_accs)),
                                               np.std(np.array(test_accs)))
    print(s)


if __name__ == '__main__':
    run()