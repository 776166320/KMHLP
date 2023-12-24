# encoding: utf-8
import numpy as np
import torch


def link_prediction_collate(batch):

    pos, neg = zip(*batch)
    return torch.stack([torch.from_numpy(pos_sample) for pos_sample in pos], dim=0).tolist(), \
           torch.stack([torch.from_numpy(neg_sample) for neg_sample in neg], dim=0).tolist()



def build_collate_fn(dataset):

    if dataset in ['IMDB', 'LastFM']:
        return link_prediction_collate
    else:
        raise NotImplementedError
