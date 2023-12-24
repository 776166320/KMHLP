import torch
import torch.nn as nn
import torch.nn.functional as F

class K_GCN_layer(nn.Module):

    def __init__(self, f_dim, hidden_dim):

        super(K_GCN_layer, self).__init__()
        self.fc1 = nn.Linear(f_dim, hidden_dim)

        self.weights = nn.Parameter(torch.rand(2))

        self.activate_fn = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, node_feature, adj_s):
        # node_feature: [N, f_dim]
        num_node = node_feature.shape[0]

        if adj_s.shape[0] == 2:
            w = F.softmax(self.weights, dim=0)
            weighted_adj = adj_s * w.unsqueeze(-1).unsqueeze(-1)
            weighted_adj = torch.sum(weighted_adj, dim=0)
        else:
            weighted_adj = adj_s

        # out = torch.matmul(weighted_adj, self.fc1(node_feature))
        # return out

        norm = torch.diag(torch.sum(weighted_adj, dim=1) ** (-0.5))
        norm = torch.tensor(norm, dtype=torch.float32)

        out = self.fc1(torch.matmul(norm, node_feature))
        out = self.activate_fn(out)
        out = torch.matmul(norm, out)
        return self.fc2(out)


class K_GCN(nn.Module):

    def __init__(self, num_layers, f_dim, hidden_dim):
        super(K_GCN, self).__init__()

        self.layers = nn.ModuleList([
            K_GCN_layer(f_dim, hidden_dim) for i in range(num_layers)
        ])

    def forward(self, node_feature, adj_s):

        for layer in self.layers:
            last_feature = node_feature
            node_feature = layer(node_feature, adj_s) + last_feature

        return node_feature


# class K_GCN_layer(nn.Module):
#     def __init__(self, in_features, out_features, bias=True):
#         super(K_GCN_layer, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
#         self.use_bias = bias
#         if self.use_bias:
#             self.bias = nn.Parameter(torch.FloatTensor(out_features))
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         nn.init.kaiming_uniform_(self.weight)
#         if self.use_bias:
#             nn.init.zeros_(self.bias)
#
#     def forward(self, input_features, adj):
#         support = torch.mm(input_features, self.weight)
#         output = torch.mm(adj, support)
#         if self.use_bias:
#             return output + self.bias
#         else:
#             return output


# class GCN(nn.Module):
#     def __init__(self, input_dim=1433):
#         super(GCN, self).__init__()
#         self.gcn1 = GraphConvolution(input_dim, 16)
#         self.gcn2 = GraphConvolution(16, 7)
#         pass
#
#     def forward(self, X, adj):
#         X = F.relu(self.gcn1(X, adj))
#         X = self.gcn2(X, adj)
#
#         return F.log_softmax(X, dim=1)

