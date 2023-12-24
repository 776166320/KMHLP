import pickle

import networkx as nx
import numpy as np
import scipy
import torch
from tqdm import tqdm
import torch.nn.functional as F

# adj_matrix = scipy.sparse.load_npz('../processed_data/IMDB/adjM.npz').toarray()
# G = nx.from_numpy_array(adj_matrix)
#
# betweenness_centrality = nx.betweenness_centrality(G)
# f = open('./b_IMDB.pkl', 'wb')
# pickle.dump(betweenness_centrality, f)
# f.close()

adj_matrix = torch.from_numpy(scipy.sparse.load_npz('../processed_data/IMDB/adjM.npz').toarray())

# n = 20612
n = 13449

degree = 1 / torch.sum(adj_matrix, dim=1)
degree = F.normalize(degree, dim=0)

with open('./b_IMDB.pkl', 'rb') as f:
    b = pickle.load(f)
    b = torch.tensor(list(b.values()))
    b = F.normalize(b, dim=0)

s = torch.zeros_like(adj_matrix)

all = 0.5 * degree + 0.5 * b
for i in tqdm(range(n)):
    for j in range(n):
        # print(torch.where((adj_matrix[i] > 0)))
        # print(torch.where((adj_matrix[100] > 0)))
        common_neighbor = (adj_matrix[i] > 0) * (adj_matrix[j] > 0)
        s[i][j] = torch.sum(common_neighbor * all, dim=0)

f = open('./S_IMDB.pkl', 'wb')
pickle.dump(s, f)
f.close()
