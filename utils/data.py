import networkx as nx
import numpy as np
import scipy
import pickle

import torch


def load_IMDB_data(dataset_root='/home/featurize/work/KMHLP/Data_process/data/preprocessed/IMDB_processed_LP',
                   fusion_matrix=True):
    in_file = open(dataset_root + '/0/0-1-0.adjlist', 'r')
    adjlist00 = [line.strip() for line in in_file]
    adjlist00 = adjlist00
    in_file.close()

    in_file = open(dataset_root + '/0/0-1-2-1-0.adjlist', 'r')
    adjlist01 = [line.strip() for line in in_file]
    adjlist01 = adjlist01
    in_file.close()

    in_file = open(dataset_root + '/1/1-0-1.adjlist', 'r')
    adjlist10 = [line.strip() for line in in_file]
    adjlist10 = adjlist10
    in_file.close()

    in_file = open(dataset_root + '/1/1-2-1.adjlist', 'r')
    adjlist11 = [line.strip() for line in in_file]
    adjlist11 = adjlist11
    in_file.close()

    in_file = open(dataset_root + '/0/0-1-0_idx.pickle', 'rb')
    idx00 = pickle.load(in_file)
    in_file.close()

    in_file = open(dataset_root + '/0/0-1-2-1-0_idx.pickle', 'rb')
    idx01 = pickle.load(in_file)
    in_file.close()

    in_file = open(dataset_root + '/1/1-0-1_idx.pickle', 'rb')
    idx10 = pickle.load(in_file)
    in_file.close()

    in_file = open(dataset_root + '/1/1-2-1_idx.pickle', 'rb')
    idx11 = pickle.load(in_file)
    in_file.close()

    adj_matrix = scipy.sparse.load_npz(dataset_root + 'adjM.npz').toarray()
    adj_matrix = torch.from_numpy(adj_matrix) + torch.diag(torch.ones(adj_matrix.shape[0]))

    if fusion_matrix:
        in_file = open(dataset_root + 'S_IMDB.pkl', 'rb')
        s_matrix = pickle.load(in_file)
        in_file.close()

        adj_matrix = torch.stack([adj_matrix, s_matrix], dim=0)

    type_mask = np.load(dataset_root + '/node_types.npy')
    train_val_test_pos_user_artist = np.load(dataset_root + '/train_val_test_pos_actor_movie.npz')
    train_val_test_neg_user_artist = np.load(dataset_root + '/train_val_test_neg_actor_movie.npz')

    return [[adjlist00, adjlist01], [adjlist10, adjlist11]],\
           [[idx00, idx01], [idx10, idx11]],\
           adj_matrix.float(), type_mask, train_val_test_pos_user_artist, train_val_test_neg_user_artist


def load_DBLP_data(dataset_root='data/preprocessed/DBLP_processed'):

    with open(dataset_root + '/0/0-1-0.adjlist', 'r') as in_file:
        adjlist00 = [line.strip() for line in in_file]
        print(adjlist00[3])
        print(adjlist00[4])
        print(adjlist00[5])
        raise RuntimeError
        adjlist00 = adjlist00[3:]

    with open(dataset_root + '/0/0-1-2-1-0.adjlist', 'r') as in_file:
        adjlist01 = [line.strip() for line in in_file]
        adjlist01 = adjlist01[3:]

    with open(dataset_root + '/0/0-1-3-1-0.adjlist', 'r') as in_file:
        adjlist02 = [line.strip() for line in in_file]
        adjlist02 = adjlist02[3:]

    with open(dataset_root + '/0/0-1-0_idx.pickle', 'rb') as in_file:
        idx00 = pickle.load(in_file)

    with open(dataset_root + '/0/0-1-2-1-0_idx.pickle', 'rb') as in_file:
        idx01 = pickle.load(in_file)

    with open(dataset_root + '/0/0-1-3-1-0_idx.pickle', 'rb') as in_file:
        idx02 = pickle.load(in_file)

    features_0 = scipy.sparse.load_npz(dataset_root + '/features_0.npz').toarray()
    features_1 = scipy.sparse.load_npz(dataset_root + '/features_1.npz').toarray()
    features_2 = np.load(dataset_root + '/features_2.npy')
    features_3 = np.eye(20, dtype=np.float32)

    adjM = scipy.sparse.load_npz(dataset_root + '/adjM.npz')
    type_mask = np.load(dataset_root + '/node_types.npy')
    labels = np.load(dataset_root + '/labels.npy')
    train_val_test_idx = np.load(dataset_root + '/train_val_test_idx.npz')

    return [adjlist00, adjlist01, adjlist02], \
           [idx00, idx01, idx02], \
           [features_0, features_1, features_2, features_3],\
           adjM, \
           type_mask,\
           labels,\
           train_val_test_idx


def load_LastFM_data(dataset_root='data/preprocessed/LastFM_processed',
                     fusion_matrix=True):
    in_file = open(dataset_root + '/0/0-1-0.adjlist', 'r')
    adjlist00 = [line.strip() for line in in_file]
    adjlist00 = adjlist00
    in_file.close()

    in_file = open(dataset_root + '/0/0-0.adjlist', 'r')
    adjlist02 = [line.strip() for line in in_file]
    adjlist02 = adjlist02
    in_file.close()

    in_file = open(dataset_root + '/1/1-0-1.adjlist', 'r')
    adjlist10 = [line.strip() for line in in_file]
    adjlist10 = adjlist10
    in_file.close()

    in_file = open(dataset_root + '/1/1-2-1.adjlist', 'r')
    adjlist11 = [line.strip() for line in in_file]
    adjlist11 = adjlist11
    in_file.close()

    in_file = open(dataset_root + '/0/0-1-0_idx.pickle', 'rb')
    idx00 = pickle.load(in_file)
    in_file.close()

    in_file = open(dataset_root + '/0/0-0_idx.pickle', 'rb')
    idx02 = pickle.load(in_file)
    in_file.close()

    in_file = open(dataset_root + '/1/1-0-1_idx.pickle', 'rb')
    idx10 = pickle.load(in_file)
    in_file.close()

    in_file = open(dataset_root + '/1/1-2-1_idx.pickle', 'rb')
    idx11 = pickle.load(in_file)
    in_file.close()

    adj_matrix = scipy.sparse.load_npz(dataset_root + 'adjM.npz').toarray()
    adj_matrix = torch.from_numpy(adj_matrix) + torch.diag(torch.ones(adj_matrix.shape[0]))

    if fusion_matrix:
        in_file = open(dataset_root + 'S_LastFM.pkl', 'rb')
        s_matrix = pickle.load(in_file)
        in_file.close()

        adj_matrix = torch.stack([adj_matrix, s_matrix], dim=0)

    type_mask = np.load(dataset_root + '/node_types.npy')
    train_val_test_pos_user_artist = np.load(dataset_root + '/train_val_test_pos_user_artist.npz')
    train_val_test_neg_user_artist = np.load(dataset_root + '/train_val_test_neg_user_artist.npz')

    return [[adjlist00, adjlist02], [adjlist10, adjlist11]],\
           [[idx00, idx02], [idx10, idx11]],\
           adj_matrix.float(), type_mask, train_val_test_pos_user_artist, train_val_test_neg_user_artist


def load_data(dataset_name, dataset_root, model_name, sub_gcn_name):

    allowed_datasets = ['IMDB', 'DBLP', 'LastFM']
    assert dataset_name in allowed_datasets, \
        f"Unknown dataset name: {dataset_name}, try: {allowed_datasets}"

    fusion_matrix = True
    if (model_name == 'GCN') or (model_name == 'KMHLP' and sub_gcn_name == 'GCN'):
        fusion_matrix = False

    if dataset_name == 'IMDB':
        return load_IMDB_data(dataset_root, fusion_matrix)
    elif dataset_name == 'LastFM':
        return load_LastFM_data(dataset_root, fusion_matrix)
    else:
        return load_DBLP_data(dataset_root, fusion_matrix)




