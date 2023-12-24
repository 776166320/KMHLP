import random

from torch.utils.data import Dataset


class LinkPrediction(Dataset):
    def __init__(self, pos, neg, mode: str = 'train'):
        super(LinkPrediction, self).__init__()

        allowed_modes = ['train', 'val', 'test']
        assert mode in allowed_modes, \
            f"Unknown mode: {mode}, try: {allowed_modes}"
        self.mode = mode
        self.pos = pos
        self.neg = neg
        self.len_neg = len(self.neg)

    def __len__(self):
        return len(self.pos)

    def __getitem__(self, item):
        if self.mode == 'train':
            return self.pos[item], self.neg[random.randint(0, self.len_neg - 1)]
        else:
            return self.pos[item], self.neg[item]
