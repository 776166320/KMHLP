import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from copy import deepcopy
from dgl.nn.pytorch import edge_softmax
from .Aggregation import build_agg_expand


class MHLP_layer(nn.Module):

    def __init__(self,
                 sub_att,
                 num_metapaths_list,
                 in_dim,
                 out_dim,
                 attn_dim,
                 num_heads,
                 agg_type,
                 attn_type,
                 agg_construct_param,
                 expand_construct_param,
                 xavier_init=True,
                 attn_drop=0.5,
                 offset_list=None):
        super(MHLP_layer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.offset_list = offset_list

        # ctr_ntype-specific layers
        agg_construct_param1 = deepcopy(agg_construct_param)
        agg_construct_param2 = deepcopy(agg_construct_param)
        if agg_type == 'Conv1d':
            agg_construct_param1[-1] = agg_construct_param1[-1][0]
            agg_construct_param2[-1] = agg_construct_param2[-1][1]
        if agg_type == 'SelfAtt':
            agg_construct_param1[0] = agg_construct_param1[0][0]
            agg_construct_param2[0] = agg_construct_param2[0][1]
        if agg_type == 'Rotation':
            agg_construct_param1[2] = agg_construct_param[2][0]
            agg_construct_param2[2] = agg_construct_param[2][1]

        self.cls1_layer = NodeTypeSpecific(sub_att,
                                           num_metapaths_list[0],
                                           in_dim,
                                           attn_dim,
                                           num_heads,
                                           agg_type,
                                           attn_type,
                                           agg_construct_param1,
                                           expand_construct_param,
                                           xavier_init,
                                           attn_drop,
                                           use_minibatch=True)
        self.cls2_layer = NodeTypeSpecific(sub_att,
                                           num_metapaths_list[0],
                                           in_dim,
                                           attn_dim,
                                           num_heads,
                                           agg_type,
                                           attn_type,
                                           agg_construct_param2,
                                           expand_construct_param,
                                           xavier_init,
                                           attn_drop,
                                           use_minibatch=True)

        # note that the acutal input dimension should consider the number of heads
        # as multiple head outputs are concatenated together
        self.fc_cls1 = nn.Linear(in_dim * num_heads, out_dim, bias=True)
        self.fc_cls2 = nn.Linear(in_dim * num_heads, out_dim, bias=True)
        nn.init.xavier_normal_(self.fc_cls1.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc_cls2.weight, gain=1.414)

    def forward(self, inputs):
        g_lists, features, type_mask, edge_metapath_indices_lists, target_idx_lists, cls_id_list = inputs

        # ctr_ntype-specific layers
        h_cls1 = self.cls1_layer(
            (g_lists[0], features, type_mask, edge_metapath_indices_lists[0], target_idx_lists[0], cls_id_list[0]))
        h_cls2 = self.cls2_layer(
            (g_lists[1], features, type_mask, edge_metapath_indices_lists[1], target_idx_lists[1], cls_id_list[1]))

        logits_cls1 = self.fc_cls1(h_cls1)
        logits_cls2 = self.fc_cls2(h_cls2)
        return logits_cls1, logits_cls2


class NodeTypeSpecific(nn.Module):
    def __init__(self,
                 sub_att,
                 num_metapaths,
                 in_dim,
                 attn_dim,
                 num_heads,
                 agg_type,
                 attn_type,
                 agg_construct_param,
                 expand_construct_param,
                 xavier_init=True,
                 attn_drop=0.5,
                 use_minibatch=True):
        super(NodeTypeSpecific, self).__init__()
        self.in_dim = in_dim
        self.num_heads = num_heads
        self.use_minibatch = use_minibatch

        # metapath-specific layers
        self.IntraPath_layers = nn.ModuleList()
        for i in range(num_metapaths):
            agg_param = deepcopy(agg_construct_param)
            if agg_type == 'Conv1d':
                agg_param[-1] = agg_param[-1][i]
            if agg_type == 'Rotation':
                agg_param[2] = agg_param[2][i]
            if agg_type == 'SelfAtt':
                agg_param[0] = agg_param[0][i]

            self.IntraPath_layers.append(IntraPathAgg(sub_att, in_dim, num_heads, agg_param, expand_construct_param,
                                                      agg_type=agg_type, attn_drop=attn_drop, use_minibatch=use_minibatch,
                                                      attn_type=attn_type, xavier_init=xavier_init))

        self.InterPath_layer = InterPathAtt(sub_att, in_dim * num_heads, attn_dim, xavier_init=xavier_init)

    def forward(self, inputs):
        if self.use_minibatch:
            g_list, features, type_mask, edge_metapath_indices_list, target_idx_list, cls_id_list = inputs

            metapath_outs = [F.elu(metapath_layer((g.to(0), features, type_mask, edge_metapath_indices, target_idx, cls_id_list)).view(-1, self.num_heads * self.in_dim))
                             for g, edge_metapath_indices, target_idx, metapath_layer in zip(g_list, edge_metapath_indices_list, target_idx_list, self.IntraPath_layers)]
        else:
            g_list, features, type_mask, edge_metapath_indices_list, cls_id_list = inputs

            metapath_outs = [F.elu(metapath_layer((g, features, type_mask, edge_metapath_indices, cls_id_list)).view(-1, self.num_heads * self.out_dim))
                             for g, edge_metapath_indices, metapath_layer in zip(g_list, edge_metapath_indices_list, self.IntraPath_layers)]

        metapath_outs = torch.stack(metapath_outs, dim=0)  # [num_metapath_types, num_nodes, num_heads * in_dim]
        return self.InterPath_layer(metapath_outs)


class InterPathAtt(nn.Module):

    def __init__(self, sub_att, hidden_dim, attn_dim, xavier_init=False):
        super(InterPathAtt, self).__init__()

        self.sub_att = sub_att
        self.fc1 = nn.Linear(hidden_dim, attn_dim, bias=True)
        self.fc2 = nn.Linear(attn_dim, 1, bias=False)

        # weight initialization
        if xavier_init:
            nn.init.xavier_normal_(self.fc1.weight, gain=1.414)
            nn.init.xavier_normal_(self.fc2.weight, gain=1.414)

    def forward(self, metapath_outs):
        # metapath_outs: [num_metapath_types, num_nodes, num_heads*head_dim]

        if self.sub_att == 'intra_path':
            return torch.sum(metapath_outs, dim=0)

        attn_weights = torch.tanh(self.fc1(metapath_outs))
        attn_weights = torch.mean(attn_weights, dim=1)
        attn_weights = self.fc2(attn_weights)
        attn_weights = F.softmax(attn_weights, dim=0).squeeze(-1)

        attend_outs = torch.einsum('ijk, i -> ijk', metapath_outs, attn_weights)
        return torch.sum(attend_outs, dim=0)


class IntraPathAgg(nn.Module):

    def __init__(self, sub_att, in_dim, num_heads, agg_construct_param, expand_construct_param,
                 agg_type='Rotation', attn_drop=0.5, alpha=0.01, use_minibatch=False, attn_type='switch_att', xavier_init=True):
        super(IntraPathAgg, self).__init__()

        allowed_agg_types = ['MeanP', 'LSTM', 'Rotation', 'Conv1d', 'SelfAtt']
        assert agg_type in allowed_agg_types, \
            f"Unknown aggregation type: {agg_type}, try: {allowed_agg_types}"
        self.agg_type = agg_type

        allowed_attn_types = ['switch_att', 'sim_att', 'no_att']
        assert attn_type in allowed_attn_types, \
            f"Unknown attention type: {attn_type}, try: {allowed_attn_types}"
        self.attn_type = attn_type

        allowed_sub_att_types = ['both', 'intra_path', 'inter_path']
        assert sub_att in allowed_sub_att_types, \
            f"Unknown sub_att type: {sub_att}, try: {allowed_sub_att_types}"
        self.sub_att = sub_att

        self.agg, self.expand = build_agg_expand(agg_type, agg_construct_param, expand_construct_param)

        # node-level attention
        if self.attn_type == 'switch_att':
            self.attn1 = nn.Linear(in_dim, num_heads, bias=False)
            self.attn2 = nn.Parameter(torch.empty(size=(1, num_heads, in_dim)))
            # weight initialization
            if xavier_init:
                nn.init.xavier_normal_(self.attn1.weight, gain=1.414)
                nn.init.xavier_normal_(self.attn2.data, gain=1.414)
        else:
            # sim_att
            self.attn = nn.Parameter(torch.empty(size=(1, num_heads, in_dim)))
            # weight initialization
            if xavier_init:
                nn.init.xavier_normal_(self.attn.data, gain=1.414)

        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

        self.in_dim = in_dim
        self.num_heads = num_heads
        self.use_minibatch = use_minibatch
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.softmax = edge_softmax

        # no intra_path fc
        self.fc = nn.Linear(self.in_dim, self.num_heads * self.in_dim)

    def edge_softmax(self, g):
        # E x num_heads x 1
        attention = self.softmax(g, g.edata.pop('att'))
        # Dropout attention scores and save them
        g.edata['att_drop'] = self.attn_drop(attention)

    def message_passing(self, edges):
        ft = edges.data['eft'] * edges.data['att_drop']
        return {'ft': ft}

    def forward(self, inputs):
        # features: num_all_nodes x node_dim
        # ...
        if self.use_minibatch:
            g, features, type_mask, edge_metapath_indices, target_idx, cls_id_list = inputs
        else:
            g, features, type_mask, edge_metapath_indices, cls_id_list = inputs

        if self.sub_att == 'inter_path':
            r = features[cls_id_list, :]
            return self.fc(r).reshape(-1, self.num_heads, self.in_dim)

        # Embedding layer
        edata = F.embedding(edge_metapath_indices, features)  # [num_edges, num_nodes, node_dim]
        hidden = self.expand(self.agg(edata))
        hidden = hidden.unsqueeze(dim=0).permute(1, 0, 2)  # [num_edges, 1, num_heads*node_dim]
        eft = hidden.view(-1, self.num_heads, self.in_dim)  # [num_edges, num_heads, node_dim]

        if self.attn_type == 'switch_att':
            center_node_feat = F.embedding(edge_metapath_indices[:, -1], features)  # [num_edges, node_dim]
            att1 = self.attn1(center_node_feat)  # [num_edges, num_heads]
            att2 = (eft * self.attn2).sum(dim=-1)  # [num_edges, num_heads]
            att = (att1 + att2).unsqueeze(dim=-1)  # [num_edges, num_heads, 1]
        elif self.attn_type == 'sim_att':
            center_node_feat = F.embedding(edge_metapath_indices[:, -1], features)  # [num_edges, node_dim]
            att = torch.einsum('ehd, ed -> eh', eft, center_node_feat).unsqueeze(dim=-1)
        else:
            att = (eft * self.attn).sum(dim=-1).unsqueeze(dim=-1)  # [num_edges, num_heads, 1]

        att = self.leaky_relu(att)
        g.edata.update({'eft': eft, 'att': att})
        # compute softmax normalized attention values
        self.edge_softmax(g)
        # compute the aggregated node features
        g.update_all(self.message_passing, fn.sum('ft', 'ft'))
        ret = g.ndata['ft']

        if self.use_minibatch:
            return ret[target_idx]  # [num_nodes, num_heads, in_dim]
        else:
            return ret


