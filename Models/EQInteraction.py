import torch
from torch import nn


class EmptyBatch(Exception):
    pass


class RankingLoss(nn.Module):
    def __init__(self):
        super(RankingLoss, self).__init__()
        self.relu = nn.ReLU()
        self.delta = 10.0

    def forward(self, pred, target):
        pred = pred.squeeze()
        target = target.squeeze()
        batch_size = pred.size(0)
        loss = None
        loss_trivial = None
        N = 0
        F = 0
        T = 0
        for i in range(batch_size):
            for j in range(batch_size):
                delta = (pred[i] - pred[j])
                if target[i] == 0 and target[j] == 1:
                    if loss is None:
                        loss = self.relu(self.delta - delta) * self.relu(self.delta - delta)
                    else:
                        loss += self.relu(self.delta - delta) * self.relu(self.delta - delta)
                    N = N + 1
                    F += (delta > 0.0).to(dtype=torch.float)
                    T += (delta < 0.0).to(dtype=torch.float)

                elif target[i] == 1 and target[j] == 0:
                    if loss is None:
                        loss = self.relu(self.delta + delta) * self.relu(self.delta + delta)
                    else:
                        loss += self.relu(self.delta + delta) * self.relu(self.delta + delta)
                    N = N + 1
                    F += (delta < 0.0).to(dtype=torch.float)
                    T += (delta > 0.0).to(dtype=torch.float)

        if N > 0:
            return (loss) / float(N), T / N
        else:
            raise (EmptyBatch)


class EQInteraction(nn.Module):
    def __init__(self, model):
        super(EQInteraction, self).__init__()
        self.repr = model.repr
        self.scorer = model.scorer

    def forward(self, scores, angles):
        assert scores.ndimension() == 4
        batch_size = scores.size(0)
        num_angles = scores.size(1)
        L = scores.size(2)

        scores_flat = scores.view(batch_size, num_angles * L * L) / 200.0
        P = -torch.logsumexp(-scores_flat, dim=-1)
        return
