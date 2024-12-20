import torch
from torch import nn
from torch.nn import functional as F


class SimCLR(nn.Module):
    def __init__(self, temp):
        super().__init__()

        self.temp = temp
        self.distributed = False

    def forward(self, hidden, idx):
        LARGE_NUM = 1e9
        batch_size = hidden.shape[0] // 2

        hidden1 = hidden[0:batch_size]
        hidden2 = hidden[batch_size:]
        
        hidden1 = F.normalize(hidden1, dim=1, p=2)
        hidden2 = F.normalize(hidden2, dim=1, p=2)
        
        # Gather hidden1/hidden2 across replicas and create local labels.
        if self.distributed:
            # TODO
            raise NotImplementedError("Distributed loss is not yet implemented.")
        else:
            hidden1_large = hidden1
            hidden2_large = hidden2
            labels = torch.arange(batch_size).cuda()
            masks = F.one_hot(torch.arange(batch_size), batch_size).cuda()

        logits_aa = torch.matmul(hidden1, hidden1_large.T) / self.temp
        logits_aa = logits_aa - masks * LARGE_NUM
        logits_bb = torch.matmul(hidden2, hidden2_large.T) / self.temp
        logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = torch.matmul(hidden1, hidden2_large.T) / self.temp
        logits_ba = torch.matmul(hidden2, hidden1_large.T) / self.temp

        loss_a = F.cross_entropy(torch.cat([logits_ab, logits_aa], dim=1), labels)
        loss_b = F.cross_entropy(torch.cat([logits_ba, logits_bb], dim=1), labels)
        loss = (loss_a + loss_b) / 2.0

        return loss