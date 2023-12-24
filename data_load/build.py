# encoding: utf-8

from torch.utils.data import DataLoader
from .datasets import build_dataset
from .collate_batch import build_collate_fn


def make_data_loader(cfgs):

    train_set, val_set, test_set, loaded_data = build_dataset(cfgs)

    train_loader = DataLoader(train_set,
                              batch_size=cfgs.train.batch_size,
                              shuffle=True,
                              num_workers=cfgs.train.num_workers,
                              collate_fn=build_collate_fn(cfgs.data.dataset))
    val_loader = DataLoader(val_set,
                            batch_size=cfgs.train.batch_size,
                            shuffle=False,
                            num_workers=cfgs.train.num_workers,
                            collate_fn=build_collate_fn(cfgs.data.dataset))
    test_loader = DataLoader(test_set,
                             batch_size=cfgs.train.batch_size,
                             shuffle=False,
                             num_workers=cfgs.train.num_workers,
                             collate_fn=build_collate_fn(cfgs.data.dataset))

    return train_loader, val_loader, test_loader, loaded_data

