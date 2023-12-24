# encoding: utf-8

import torch.optim as optim
from .lr_scheduler import WarmupMultiStepLR, WarmupCosineAnnealingLR

def make_optimizer(cfgs, model):

    train_param = filter(lambda param: param.requires_grad, model.parameters())
    if cfgs.train.optim == 'adam':
        optimizer = optim.Adam(train_param,
                               lr=cfgs.train.lr,
                               weight_decay=cfgs.train.weight_decay)

        print('Adam')
    elif cfgs.train.optim == 'sgd':
        optimizer = optim.SGD(train_param,
                              lr=cfgs.train.lr,
                              momentum=cfgs.train.momentum,
                              weight_decay=cfgs.train.weight_decay)
        print('SGD')
    elif cfgs.train.optim == 'adamw':
        optimizer = optim.AdamW(train_param,
                                lr=cfgs.train.lr,
                                weight_decay=cfgs.train.weight_decay)
        print('AdamW')
    else:
        raise ValueError('Unknown optimizer: {}'.format(cfgs.train.optim))
    return optimizer

def make_lr_schedule(config, optimizer):
    if config.train.schedule_type == 'cosine':
        lr_scheduler = WarmupCosineAnnealingLR(
            optimizer,
            config.train.num_epochs,
            warmup_epochs=config.train.lr_warmup_step
        )
    elif config.train.schedule_type == 'multistep':
        if isinstance(config.train.lr_decay_step, list):
            milestones = config.train.lr_decay_step
        elif isinstance(config.train.lr_decay_step, int):
            milestones = [
                config.train.lr_decay_step * (i + 1)
                for i in range(config.train.num_epochs //
                               config.train.lr_decay_step)]
        else:
            raise ValueError("error learning rate decay step: {}".format(type(config.train.lr_decay_step)))
        lr_scheduler = WarmupMultiStepLR(
            optimizer,
            milestones,
            warmup_epochs=config.train.lr_warmup_step
        )
    else:
        raise ValueError('Unknown lr scheduler: {}'.format(config.train.schedule_type))

    return lr_scheduler
