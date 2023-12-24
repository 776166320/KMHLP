import sys

sys.path.append('.')

import argparse
import yaml
import torch
import numpy as np

from modelling import build_model, build_loss
from solver import make_optimizer
from data_load import make_data_loader
from dotmap import DotMap
from sklearn.metrics import roc_auc_score, average_precision_score
from utils.pytorchtools import EarlyStopping
from utils.data import load_data
from utils.tools import parse_data_minibatch

import warnings
warnings.filterwarnings('ignore')


def train_val_test(cfgs):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # origin features
    features_list = []
    in_dims = cfgs.data.num_nodes
    for i in range(cfgs.data.num_ntype):
        indices = np.vstack((np.arange(torch.tensor(in_dims[i])), np.arange(torch.tensor(in_dims[i]))))
        indices = torch.LongTensor(indices)
        values = torch.FloatTensor(np.ones(in_dims[i]))
        features_list.append(torch.sparse.FloatTensor(indices, values, torch.Size([torch.tensor(in_dims[i]),
                                                                                   torch.tensor(in_dims[i])])).to(device))

    loss_computer = build_loss(cfgs)

    train_loader, val_loader, test_loader, loaded_data = make_data_loader(cfgs)
    adjlists, edge_metapath_indices_list, matrix_adj, type_mask, \
    train_val_test_pos, train_val_test_neg = loaded_data

    model = build_model(cfgs, matrix_adj)
    optimizer = make_optimizer(cfgs, model)

    test_pos = train_val_test_pos['test_pos' + cfgs.data.target_path]
    test_neg = train_val_test_neg['test_neg' + cfgs.data.target_path]
    y_true_test = np.array([1] * len(test_pos) + [0] * len(test_neg))

    # training loop
    early_stopping = EarlyStopping(patience=cfgs.train.patience, verbose=True,
                                   save_path='checkpoint/checkpoint_{}_{}.pt'.format(cfgs.model.name, cfgs.data.dataset))
    neighbor_samples = cfgs.train.neighbor_samples

    for epoch in range(cfgs.train.num_epochs):
        model.train()
        for iteration, (train_pos_batch, train_neg_batch) in enumerate(train_loader):

            train_pos_g_lists, train_pos_indices_lists, train_pos_idx_batch_mapped_lists = parse_data_minibatch(
                adjlists, edge_metapath_indices_list, train_pos_batch, device,
                neighbor_samples, cfgs.data.use_masks, cfgs.data.num_nodes[0])
            train_neg_g_lists, train_neg_indices_lists, train_neg_idx_batch_mapped_lists = parse_data_minibatch(
                adjlists, edge_metapath_indices_list, train_neg_batch, device,
                neighbor_samples, cfgs.data.no_masks, cfgs.data.num_nodes[0])

            pos_embedding_cls1, pos_embedding_cls2 = model((train_pos_g_lists, features_list, type_mask,
                                                            train_pos_indices_lists, train_pos_idx_batch_mapped_lists,
                                                            train_pos_batch))
            neg_embedding_cls1, neg_embedding_cls2 = model((train_neg_g_lists, features_list, type_mask,
                                                            train_neg_indices_lists, train_neg_idx_batch_mapped_lists,
                                                            train_neg_batch))

            train_loss = loss_computer(pos_embedding_cls1, pos_embedding_cls2, neg_embedding_cls1, neg_embedding_cls2)

            # autograd
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            # print training info
            if iteration % 100 == 0:
                print('Epoch {:05d} | Iteration {:05d} | Train_Loss {:.4f}'.format(epoch, iteration, train_loss.item()))

        # validation
        model.eval()
        val_loss = []
        with torch.no_grad():
            for val_pos_batch, val_neg_batch in val_loader:
                val_pos_g_lists, val_pos_indices_lists, val_pos_idx_batch_mapped_lists = parse_data_minibatch(
                    adjlists, edge_metapath_indices_list, val_pos_batch, device, neighbor_samples, cfgs.data.no_masks, cfgs.data.num_nodes[0])
                val_neg_g_lists, val_neg_indices_lists, val_neg_idx_batch_mapped_lists = parse_data_minibatch(
                    adjlists, edge_metapath_indices_list, val_neg_batch, device, neighbor_samples, cfgs.data.no_masks, cfgs.data.num_nodes[0])

                pos_embedding_cls1, pos_embedding_cls2 = model(
                    (val_pos_g_lists, features_list, type_mask, val_pos_indices_lists, val_pos_idx_batch_mapped_lists, val_pos_batch))
                neg_embedding_cls1, neg_embedding_cls2 = model(
                    (val_neg_g_lists, features_list, type_mask, val_neg_indices_lists, val_neg_idx_batch_mapped_lists, val_neg_batch))

                val_loss.append(loss_computer(pos_embedding_cls1, pos_embedding_cls2, neg_embedding_cls1, neg_embedding_cls2))
            val_loss = torch.mean(torch.tensor(val_loss))

        print('Epoch {:05d} | Val_Loss {:.4f}'.format(epoch, val_loss.item()))
        # early stopping

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print('Early stopping!')
            break

    model.load_state_dict(torch.load('checkpoint/checkpoint_{}_{}.pt'.format(cfgs.model.name, cfgs.data.dataset)))
    model.eval()

    pos_proba_list = []
    neg_proba_list = []
    with torch.no_grad():
        for test_pos_batch, test_neg_batch in test_loader:

            test_pos_g_lists, test_pos_indices_lists, test_pos_idx_batch_mapped_lists = parse_data_minibatch(
                adjlists, edge_metapath_indices_list, test_pos_batch, device, neighbor_samples, cfgs.data.no_masks, cfgs.data.num_nodes[0])
            test_neg_g_lists, test_neg_indices_lists, test_neg_idx_batch_mapped_lists = parse_data_minibatch(
                adjlists, edge_metapath_indices_list, test_neg_batch, device, neighbor_samples, cfgs.data.no_masks, cfgs.data.num_nodes[0])

            pos_embedding_cls1, neg_embedding_cls2 = model(
                (test_pos_g_lists, features_list, type_mask, test_pos_indices_lists, test_pos_idx_batch_mapped_lists, test_pos_batch))
            neg_embedding_cls1, neg_embedding_cls2 = model(
                (test_neg_g_lists, features_list, type_mask, test_neg_indices_lists, test_neg_idx_batch_mapped_lists, test_neg_batch))
            pos_embedding_cls1 = pos_embedding_cls1.view(-1, 1, pos_embedding_cls1.shape[1])
            neg_embedding_cls2 = neg_embedding_cls2.view(-1, neg_embedding_cls2.shape[1], 1)
            neg_embedding_cls1 = neg_embedding_cls1.view(-1, 1, neg_embedding_cls1.shape[1])
            neg_embedding_cls2 = neg_embedding_cls2.view(-1, neg_embedding_cls2.shape[1], 1)

            pos_out = torch.bmm(pos_embedding_cls1, neg_embedding_cls2).flatten()
            neg_out = torch.bmm(neg_embedding_cls1, neg_embedding_cls2).flatten()
            pos_proba_list.append(torch.sigmoid(pos_out))
            neg_proba_list.append(torch.sigmoid(neg_out))
        y_proba_test = torch.cat(pos_proba_list + neg_proba_list)
        y_proba_test = y_proba_test.cpu().numpy()
    auc = roc_auc_score(y_true_test, y_proba_test)
    ap = average_precision_score(y_true_test, y_proba_test)
    print('Link Prediction Test')
    print('AUC = {}'.format(auc))
    print('AP = {}'.format(ap))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="../configs/IMDB/KMHLP_IMDB.yaml")

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    cfgs = DotMap(config)

    train_val_test(cfgs)
