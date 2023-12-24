# encoding: utf-8

import torch
import numpy as np
import torch.nn as nn
from .layers import MHLP_layer, K_GCN


class KMHLP(nn.Module):

    def __init__(self, inter_type, sub_att, num_metapaths_list, matrix_s, feats_dim_list, hidden_dim, num_gcn_layers,
                 attn_dim, num_heads, attn_type, agg_construct_param, expand_construct_param, offset_list,
                 agg_type='Rotation', xavier_init=True, dropout_rate=0.5):

        super(KMHLP, self).__init__()

        allowed_inter_types = ['KMHLP', 'MHLP', 'K_GCN', 'GCN']
        assert inter_type in allowed_inter_types, \
            f"Unknown inter type: {inter_type}, try: {allowed_inter_types}"
        self.inter_type = inter_type

        if self.inter_type != 'MHLP':
            self.K_GCN = K_GCN(num_layers=num_gcn_layers, f_dim=hidden_dim, hidden_dim=hidden_dim)

        if self.inter_type != 'K_GCN':
            self.MHLP = MHLP_layer(sub_att=sub_att,
                                   num_metapaths_list=num_metapaths_list,
                                   in_dim=hidden_dim,
                                   out_dim=hidden_dim,
                                   attn_dim=attn_dim,
                                   num_heads=num_heads,
                                   agg_type=agg_type,
                                   attn_type=attn_type,
                                   agg_construct_param=agg_construct_param,
                                   expand_construct_param=expand_construct_param,
                                   xavier_init=xavier_init,
                                   attn_drop=dropout_rate,
                                   offset_list=offset_list)

        self.matrix_s = matrix_s
        self.offset = offset_list

        if self.inter_type == 'KMHLP':
            self.fc_cls1 = nn.Linear(hidden_dim * 2, hidden_dim)
            self.fc_cls2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.hidden_dim = hidden_dim

        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, hidden_dim, bias=True) for feats_dim in feats_dim_list])

        if dropout_rate > 0:
            self.feat_drop = nn.Dropout(dropout_rate)
        else:
            self.feat_drop = lambda x: x
        # initialization of fc layers
        if xavier_init:
            for fc in self.fc_list:
                nn.init.xavier_normal_(fc.weight, gain=1.414)

    def forward(self, inputs):
        g_lists, features_list, type_mask, edge_metapath_indices_lists, target_idx_lists, cls1_cls2_batch = inputs

        # ntype-specific transformation
        transformed_features = torch.zeros(type_mask.shape[0], self.hidden_dim, device=features_list[0].device)
        for i, fc in enumerate(self.fc_list):
            node_indices = np.where(type_mask == i)[0]
            transformed_features[node_indices] = fc(features_list[i])
        transformed_features = self.feat_drop(transformed_features)

        cls1_id_list = [row[0] for row in cls1_cls2_batch]
        cls2_id_list = [row[1] + self.offset[0] for row in cls1_cls2_batch]

        if self.inter_type == 'K_GCN':
            K_GCN_out = self.K_GCN(transformed_features, self.matrix_s)
            return K_GCN_out[cls1_id_list, :], K_GCN_out[cls2_id_list, :]

        elif self.inter_type == 'MHLP':

            logits_cls1_MHLP, logits_cls2_MHLP = self.MHLP((g_lists, transformed_features, type_mask,
                                                            edge_metapath_indices_lists, target_idx_lists, [cls1_id_list, cls2_id_list]))
            return logits_cls1_MHLP, logits_cls2_MHLP

        else:
            logits_cls1_MHLP, logits_cls2_MHLP = self.MHLP((g_lists, transformed_features, type_mask,
                                                            edge_metapath_indices_lists, target_idx_lists, [cls1_id_list, cls2_id_list]))

            cls1_id_list = [row[0] for row in cls1_cls2_batch]
            cls2_id_list = [row[1] + self.offset[0] for row in cls1_cls2_batch]
            K_GCN_out = self.K_GCN(transformed_features, self.matrix_s)

            cat_feature_cls1 = torch.cat([K_GCN_out[cls1_id_list, :], logits_cls1_MHLP], dim=1)
            cat_feature_cls2 = torch.cat([K_GCN_out[cls2_id_list, :], logits_cls2_MHLP], dim=1)

            return self.fc_cls1(cat_feature_cls1), self.fc_cls2(cat_feature_cls2)

if __name__ == '__main__':

    pass





