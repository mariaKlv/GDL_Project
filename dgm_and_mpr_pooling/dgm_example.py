"""
BASED ON https://github.com/crisbodnar/dgm/blob/master/run_dgm.py
"""

import os, random
from copy import deepcopy

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, DeepGraphInfomax, DGM

class GCNNet(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(inp_dim, 32)
        self.conv2 = GCNConv(32, 64)
        self.conv3 = GCNConv(64, 128)
        self.conv4 = GCNConv(128, 128)

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x, edge_index, edge_attr):
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = F.relu(self.conv3(x, edge_index, edge_attr))
        x = F.relu(self.conv4(x, edge_index, edge_attr))

        x = self.dropout(x)
        x = self.fc(x)
        return x


class GraphClassifier:
    def __init__(self, inp_dim, out_dim, device):
        self.gcn = GCNNet(inp_dim, out_dim)
        self.gcn = self.gcn.to(device)
        self.optimizer = torch.optim.Adam(self.gcn.parameters())

    def evaluate_loss(self, data, mode):
        # use masking for loss evaluation

        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        if mode == 'train':
            loss = F.cross_entropy(self.gcn(x, edge_index, edge_attr)[data.train_mask], data.y[data.train_mask])
        else:
            loss = F.cross_entropy(self.gcn(x, edge_index, edge_attr)[data.test_mask], data.y[data.test_mask])
        return loss

    def embed(self, data):
        return self.gcn(data.x, data.edge_index, data.edge_attr)

    def train(self, data):
        # training
        self.gcn.train()
        self.optimizer.zero_grad()
        loss = self.evaluate_loss(data, mode='train')
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def test(self, data):
        # testing
        self.gcn.eval()
        logits, accs = self.gcn(data.x, data.edge_index, data.edge_attr), []
        loss = self.evaluate_loss(data, mode='test').item()

        for _, mask in data('train_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)
        return [loss] + accs


class DGIEncoderNet(torch.nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(DGIEncoderNet, self).__init__()
        self.conv1 = GCNConv(inp_dim, 32)
        self.conv2 = GCNConv(32, 64)
        self.conv3 = GCNConv(64, out_dim)

    def forward(self, x, edge_index, edge_attr, msk=None):
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        x = self.conv3(x, edge_index, edge_attr)
        return x


class DGILearner:
    def __init__(self, inp_dim, out_dim, device):
        self.encoder = DGIEncoderNet(inp_dim, out_dim)
        self.dgi = DeepGraphInfomax(out_dim, encoder=self.encoder, summary=self.readout, corruption=self.corrupt)
        self.dgi = self.dgi.to(device)

        self.optimizer = torch.optim.Adam(self.dgi.parameters())

    def embed(self, data):
        pos_z, _, _ = self.dgi(data.x, data.edge_index, data.edge_attr, msk=None)
        return pos_z

    def readout(self, z, x, edge_index, edge_attr, msk=None):
        if msk is None:
            return torch.sigmoid(torch.mean(z, 0))
        else:
            return torch.sigmoid(torch.sum(z[msk], 0) / torch.sum(msk))

    def corrupt(self, x, edge_index, edge_attr, msk=None):
        shuffled_rows = torch.randperm(len(x))
        shuffled_x = x[shuffled_rows, :]
        return shuffled_x, edge_index, edge_attr

    def evaluate_loss(self, data, mode):
        # use masking for loss evaluation
        pos_z_train, neg_z_train, summ_train = self.dgi(data.x, data.edge_index, data.edge_attr, msk=data.train_mask)
        pos_z_test, neg_z_test, summ_test = self.dgi(data.x, data.edge_index, data.edge_attr, msk=data.test_mask)

        if mode == 'train':
            return self.dgi.loss(pos_z_train, neg_z_train, summ_train)
        else:
            return self.dgi.loss(pos_z_test, neg_z_test, summ_test)

    def train(self, data):
        # training
        self.dgi.train()
        self.optimizer.zero_grad()
        loss = self.evaluate_loss(data, mode='train')
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def test(self, data):
        # testing
        self.dgi.eval()
        return self.evaluate_loss(data, mode='test').item()


def train_model(dataset, train_mode, num_classes, device, save_path):
    if train_mode == 'supervised':
        model = GraphClassifier(dataset.num_node_features, num_classes, device)
    elif train_mode == 'unsupervised':
        model = DGILearner(dataset.num_node_features, 512, device)
    else:
        raise ValueError('Unsupported train mode {}'.format(train_mode))

    train_epochs = 81 if train_mode == 'supervised' else 201
    for epoch in range(0, train_epochs):
        train_loss = model.train(dataset)

        if epoch % 5 == 0:
            if train_mode == 'unsupervised':
                test_loss = model.test(dataset)
                log = 'Epoch: {:03d}, train_loss: {:.3f}, test_loss:{:.3f}'
                print(log.format(epoch, train_loss, test_loss))
            else:
                log = 'Epoch: {:03d}, train_loss: {:.3f}, test_loss:{:.3f}, train_acc: {:.2f}, test_acc: {:.2f}'
                print(log.format(epoch, train_loss, *model.test(dataset)))

    torch.save(obj=model, f=save_path)

    return model.embed(dataset).detach().cpu().numpy()

CORA_LEGEND = {
        0: 'Theory',
        1: 'Reinforcement Learning',
        2: 'Genetic Algorithms',
        3: 'Neural Networks',
        4: 'Probabilistic Methods',
        5: 'Case Based',
        6: 'Rule Learning',
    }
if __name__ == '__main__':
    random.seed(444)
    np.random.seed(444)
    torch.manual_seed(444)

    dataset_name = 'cora'
    train_mode = 'supervised'
    true_labels_lens = False
    use_true_labels_for_coloring = True
    model_path = "./{}_{}_model.npy".format(dataset_name, train_mode)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = Planetoid(root='/tmp/Cora', name=dataset_name)
    num_classes = data.num_classes
    legend_dict = CORA_LEGEND

    data = data[0].to(device)

    if os.path.isfile(model_path):
        print('Using existing model')
    else:
        print('No model found. Training a new model...')
        train_model(dataset=data, num_classes=num_classes, train_mode=train_mode, device=device,
                    save_path=model_path)
    model = torch.load(model_path)

    model_embeddings = model.embed(data).detach().cpu().numpy()
    def lens_function(data):
        return deepcopy(model_embeddings)


    def lens_function_true_labels(data):
        embeddings = data.y.cpu().numpy().astype(np.float32)
        embeddings = np.expand_dims(embeddings, axis=1)

        return embeddings

    lens = lens_function_true_labels if true_labels_lens else lens_function

    dgm = DGM(lens_function=lens,
              num_intervals=10, overlap=0.15,
              embedding_dims=1, embedding_reduction_method='tsne',
              reduction_random_state=None,
              min_component_size=5,
              use_sdgm=False)

    dgm.make_visualization(data=deepcopy(data),
                           use_true_labels=use_true_labels_for_coloring,
                           legend_dict=legend_dict,
                           )


