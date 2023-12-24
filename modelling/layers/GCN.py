import torch
import torch.nn as nn

class K_GCN(nn.Module):
    def __init__(self, f_dim, hidden_dim):
        super(K_GCN, self).__init__()
        self.fc1 = nn.Linear(f_dim, hidden_dim)

        self.activate_fn = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, node_feature, S):
        # node_feature: [N, f_dim]
        num_node = node_feature.shape[0]

        S = S + torch.diag(torch.ones((num_node))).to(S.device)
        norm = torch.diag(torch.sum(S, dim=1) ** (-0.5))
        norm = torch.tensor(norm, dtype=torch.float32)

        out = self.fc1(torch.matmul(norm, node_feature))
        out = self.activate_fn(out)
        out = torch.matmul(norm, out)
        return self.fc2(out)
