# encoding: utf-8

from .loss import build_loss
from .KMHLP import KMHLP
from .layers import MHLP_layer
from .layers import K_GCN


def build_param(cfgs):
    if cfgs.model.agg.type == 'MeanP':
        agg_param = []
    elif cfgs.model.agg.type == 'LSTM':
        agg_param = [cfgs.model.agg.bidirectional,
                     cfgs.model.hidden_dim,
                     cfgs.model.agg.num_layers]
    elif cfgs.model.agg.type == 'Rotation':
        agg_param = [cfgs.model.agg.r_type,
                     cfgs.data.num_edge_type,
                     cfgs.data.etypes_lists,
                     cfgs.model.hidden_dim,
                     cfgs.model.xavier_init]
    elif cfgs.model.agg.type == 'Conv1d':
        agg_param = [cfgs.model.hidden_dim, cfgs.model.agg.length_list]
    else:
        agg_param = [cfgs.model.agg.length_list,
                     cfgs.model.agg.depth,
                     cfgs.model.hidden_dim,
                     cfgs.model.agg.nhead,
                     cfgs.model.agg.mlp_ratio,
                     cfgs.model.agg.drop_ratio,
                     cfgs.model.agg.attn_drop_ratio]

    expand_param = [cfgs.model.hidden_dim,
                    cfgs.model.num_heads,
                    cfgs.model.expand.expand_type]

    return agg_param, expand_param


def build_model(cfgs, matrix_s):

    agg_param, expand_param = build_param(cfgs)
    model = KMHLP(inter_type=cfgs.model.name,
                  sub_att=cfgs.model.sub_att,
                  num_metapaths_list=cfgs.model.num_metapaths_list,
                  matrix_s=matrix_s.to(cfgs.train.device),
                  feats_dim_list=cfgs.data.num_nodes,
                  hidden_dim=cfgs.model.hidden_dim,
                  num_gcn_layers=cfgs.model.num_gcn_layers,
                  attn_dim=cfgs.model.attn_dim,
                  num_heads=cfgs.model.num_heads,
                  attn_type=cfgs.model.attn_type,
                  agg_construct_param=agg_param,
                  expand_construct_param=expand_param,
                  offset_list=[cfgs.data.num_nodes[0]],
                  agg_type=cfgs.model.agg.type,
                  xavier_init=cfgs.model.xavier_init,
                  dropout_rate=cfgs.model.dropout_rate)
    model.to(cfgs.train.device)
    return model
