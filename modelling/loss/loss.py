import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, output: torch.tensor, target: torch.Tensor):
        """
        Args:
            pred: shape = [batch_size, num_classes]
            gt: shape = [batch_size]

        Returns:
            loss
        """

        loss = self.loss_fn(output, target)
        return loss


class LogSigmoidLoss(nn.Module):
    def __init__(self):
        super(LogSigmoidLoss, self).__init__()

    def forward(self,
                pos_embedding_cls1,
                pos_embedding_cls2,
                neg_embedding_cls1,
                neg_embedding_cls2):
        # inputs shape: [batch_size, in_dim]

        pos_embedding_cls1 = pos_embedding_cls1.view(-1, 1, pos_embedding_cls1.shape[1])
        pos_embedding_cls2 = pos_embedding_cls2.view(-1, pos_embedding_cls2.shape[1], 1)
        neg_embedding_cls1 = neg_embedding_cls1.view(-1, 1, neg_embedding_cls1.shape[1])
        neg_embedding_cls2 = neg_embedding_cls2.view(-1, neg_embedding_cls2.shape[1], 1)

        pos_out = torch.bmm(pos_embedding_cls1, pos_embedding_cls2)
        neg_out = -torch.bmm(neg_embedding_cls1, neg_embedding_cls2)
        return -torch.mean(F.logsigmoid(pos_out) + F.logsigmoid(neg_out))
