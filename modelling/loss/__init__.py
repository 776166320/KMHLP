from .loss import CrossEntropyLoss, LogSigmoidLoss

def build_loss(cfgs):

    allowed_loss_types = ['CE', 'LS']

    assert cfgs.train.loss_type in allowed_loss_types, \
        f"Unknown loss type: {cfgs.train.loss_type}, try: {allowed_loss_types}"

    if cfgs.train.loss_type == "CE":
        loss = CrossEntropyLoss()

    if cfgs.train.loss_type == 'LS':
        loss = LogSigmoidLoss()

    return loss
