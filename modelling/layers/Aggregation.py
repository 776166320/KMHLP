import torch
import torch.nn as nn
import torch.nn.functional as F
from .MHSA import Block


class MeanP_agg(nn.Module):

    def __init__(self):
        super(MeanP_agg, self).__init__()

    def forward(self, metapath_features):
        # metapath_features: [num_paths, num_nodes, f_dim]
        # ouput: [num_paths, f_dim]
        return torch.mean(metapath_features, dim=1)


class LSTM_agg(nn.Module):

    def __init__(self, bidirectional, in_dim, num_layers):
        super(LSTM_agg, self).__init__()

        self.lstm = nn.LSTM(input_size=in_dim,
                            hidden_size=in_dim,
                            num_layers=num_layers,
                            bidirectional=bidirectional)

        self.bidirectional = bidirectional
        if self.bidirectional:
            self.fc = nn.Linear(in_dim * 2, in_dim)

    def forward(self, metapath_features):
        # metapath_features: [num_paths, num_nodes, f_dim]
        # ouput: [num_paths, f_dim]
        _, (hidden, _) = self.lstm(metapath_features.permute(1, 0, 2))
        if self.bidirectional:
            return self.fc(torch.cat([hidden[-2, :, :], hidden[-1, :, :]], dim=1))
        else:
            return hidden[-1, :, :]


class Rotation_agg(nn.Module):

    def __init__(self, r_type, num_edge_type, etypes, in_dim, xavier_init=False):
        super(Rotation_agg, self).__init__()

        allowed_types = ['RotatE0', 'RotatE1']
        assert r_type in allowed_types, f"Unknown rotation type: {r_type}, try: {allowed_types}"
        self.r_type = r_type
        self.out_dim = in_dim
        self.etypes = etypes

        if r_type == 'RotatE0':
            self.r_vec = nn.Parameter(torch.empty(size=(num_edge_type // 2, in_dim // 2, 2)))
        else:
            self.r_vec = nn.Parameter(torch.empty(size=(num_edge_type, in_dim // 2, 2)))
        if xavier_init:
            nn.init.xavier_normal_(self.r_vec.data, gain=1.414)

    def forward(self, metapath_features):
        # metapath_features: [num_paths, num_nodes, f_dim]
        # ouput: [num_paths, f_dim]
        r_vec = F.normalize(self.r_vec, p=2, dim=2)
        if self.r_type == 'RotatE0':
            r_vec = torch.stack((r_vec, r_vec), dim=1)
            r_vec[:, 1, :, 1] = -r_vec[:, 1, :, 1]
            r_vec = r_vec.reshape(self.r_vec.shape[0] * 2, self.r_vec.shape[1], 2)

        edata = metapath_features.reshape(metapath_features.shape[0], metapath_features.shape[1], metapath_features.shape[2] // 2, 2)
        final_r_vec = torch.zeros([edata.shape[1], self.out_dim // 2, 2], device=edata.device)
        final_r_vec[-1, :, 0] = 1
        for i in range(final_r_vec.shape[0] - 2, -1, -1):
            # consider None edge (symmetric relation)
            if self.etypes[i] is not None:
                final_r_vec[i, :, 0] = final_r_vec[i + 1, :, 0].clone() * r_vec[self.etypes[i], :, 0] - \
                                       final_r_vec[i + 1, :, 1].clone() * r_vec[self.etypes[i], :, 1]
                final_r_vec[i, :, 1] = final_r_vec[i + 1, :, 0].clone() * r_vec[self.etypes[i], :, 1] + \
                                       final_r_vec[i + 1, :, 1].clone() * r_vec[self.etypes[i], :, 0]
            else:
                final_r_vec[i, :, 0] = final_r_vec[i + 1, :, 0].clone()
                final_r_vec[i, :, 1] = final_r_vec[i + 1, :, 1].clone()
        for i in range(edata.shape[1] - 1):
            temp1 = edata[:, i, :, 0].clone() * final_r_vec[i, :, 0] - \
                    edata[:, i, :, 1].clone() * final_r_vec[i, :, 1]
            temp2 = edata[:, i, :, 0].clone() * final_r_vec[i, :, 1] + \
                    edata[:, i, :, 1].clone() * final_r_vec[i, :, 0]
            edata[:, i, :, 0] = temp1
            edata[:, i, :, 1] = temp2
        edata = edata.reshape(edata.shape[0], edata.shape[1], -1)
        hidden = torch.mean(edata, dim=1)
        return hidden


class Conv1d_agg(nn.Module):

    def __init__(self, in_dim, len_metapath):
        super(Conv1d_agg, self).__init__()

        self.in_dim = in_dim
        self.conv = nn.Conv1d(in_dim, in_dim, kernel_size=len_metapath)

    def forward(self, metapath_features):
        # metapath_features: [num_paths, num_nodes, f_dim]
        # output: [num_paths, f_dim]

        return self.conv(metapath_features.permute(0, 2, 1)).squeeze(-1)


class SelfAtt_agg(nn.Module):

    def __init__(self, len_metapath, depth, in_dim, nhead, mlp_ratio,
                 drop_ratio, attn_drop_ratio, norm_layer=nn.LayerNorm):
        super(SelfAtt_agg, self).__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, in_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, len_metapath + 1, in_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)
        self.blocks = nn.Sequential(*[
            Block(dim=in_dim, num_heads=nhead, mlp_ratio=mlp_ratio, drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio)
            for i in range(depth)
        ])
        self.norm = norm_layer(in_dim)

    def forward(self, metapath_features):
        # metapath_features: [num_paths, num_nodes, f_dim]
        # ouput: [num_paths, f_dim]
        cls_token = self.cls_token.repeat(metapath_features.shape[0], 1, 1)

        x = torch.cat((cls_token, metapath_features), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        x = self.blocks(x)
        x = self.norm(x)
        return x[:, 0, :]


class Expand(nn.Module):

    def __init__(self, in_dim, num_heads, expand_type='Linear'):
        super(Expand, self).__init__()

        allowed_expand_types = ['Linear', 'Copy']
        assert expand_type in allowed_expand_types, \
            f"Unknown expand type: {expand_type}, try: {allowed_expand_types}"
        self.expand_type = expand_type
        self.num_heads = num_heads

        if expand_type == 'Linear':
            self.linear = nn.Linear(in_dim, in_dim * num_heads)

    def forward(self, encoded_features):
        # encoder_features: [num_metapaths, in_dim]
        if self.expand_type == 'Linear':
            return self.linear(encoded_features)
        else:
            return torch.repeat(1, self.num_heads)


def build_agg_expand(agg_type, agg_construct_param, expand_construct_param):
    # 'MeanP', 'LSTM', 'Rotation', 'Conv1d', 'SelfAtt'
    if agg_type == 'MeanP':
        agg = MeanP_agg()
    elif agg_type == 'LSTM':
        agg = LSTM_agg(*agg_construct_param)
    elif agg_type == 'Rotation':
        agg = Rotation_agg(*agg_construct_param)
    elif agg_type == 'Conv1d':
        agg = Conv1d_agg(*agg_construct_param)
    else:
        agg = SelfAtt_agg(*agg_construct_param)

    return agg, Expand(*expand_construct_param)


if __name__ == '__main__':

    m = SelfAtt_agg(3, 1, 64, 1, 2, 0.5, 0.5)
    with torch.no_grad():
        print(m(torch.rand(6, 3, 64)).shape)
