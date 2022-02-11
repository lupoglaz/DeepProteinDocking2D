import torch
from torch import nn
import numpy as np

from .Multiplication import ImageCrossMultiply
from .Convolution import ProteinConv2D
from math import *


def init_weights(module):
    if isinstance(module, torch.nn.Linear):
        # module.weight.data.fill_(0.0)
        pass


class EQScoringModel(nn.Module):
    def __init__(self, repr, num_features=1, prot_field_size=50):
        super(EQScoringModel, self).__init__()
        self.prot_field_size = prot_field_size

        self.mult = ImageCrossMultiply()
        self.repr = repr

        self.scorer = nn.Sequential(
            nn.Linear(4, 1, bias=False)
        )
        with torch.no_grad():
            self.scorer.apply(init_weights)

    def forward(self, receptor, ligand, alpha, dr):
        rec_feat = self.repr(receptor).tensor
        lig_feat = self.repr(ligand).tensor

        pos_repr, _, A = self.mult(rec_feat, lig_feat, alpha, dr)

        score = self.scorer(pos_repr)

        return score


class EQDockerGPU(nn.Module):
    def __init__(self, scoring_model, num_angles=360):
        super(EQDockerGPU, self).__init__()
        self.scoring_model = scoring_model
        self.conv = ProteinConv2D()
        self.num_angles = num_angles
        self.angles = torch.from_numpy(np.linspace(-np.pi, np.pi, num=num_angles)).to(device='cuda',
                                                                                      dtype=torch.float32)

    def rotate(self, repr, angle):
        alpha = angle.detach()
        T0 = torch.stack([torch.cos(alpha), -torch.sin(alpha), torch.zeros_like(alpha)], dim=1)
        T1 = torch.stack([torch.sin(alpha), torch.cos(alpha), torch.zeros_like(alpha)], dim=1)
        R = torch.stack([T0, T1], dim=1)
        curr_grid = nn.functional.affine_grid(R, size=repr.size(), align_corners=True)
        return nn.functional.grid_sample(repr, curr_grid, align_corners=True)

    def dock_global(self, rec_repr, lig_repr):
        lig_repr = lig_repr.repeat(self.num_angles, 1, 1, 1)
        rec_repr = rec_repr.repeat(self.num_angles, 1, 1, 1)
        rot_lig = self.rotate(lig_repr, self.angles)
        translations = self.conv(rec_repr, rot_lig)
        return translations

    def score(self, translations):
        batch_size = translations.size(0)
        num_features = translations.size(1)
        L = translations.size(2)

        translations = translations.view(batch_size, num_features, L * L)
        translations = translations.transpose(1, 2).contiguous().view(batch_size * L * L, num_features)
        scores = self.scoring_model.scorer(translations).squeeze()
        return scores.view(batch_size, L, L)

    def get_conformation(self, scores):
        minval_y, ind_y = torch.min(scores, dim=2, keepdim=False)
        minval_x, ind_x = torch.min(minval_y, dim=1)
        minval_angle, ind_angle = torch.min(minval_x, dim=0)
        x = ind_x[ind_angle].item()
        y = ind_y[ind_angle, x].item()

        best_score = scores[ind_angle, x, y].item()
        best_translation = torch.tensor([x - scores.size(1) / 2.0, y - scores.size(1) / 2.0], dtype=torch.float32)
        best_rotation = self.angles[ind_angle]

        # Best translations
        self.top_translations = scores[ind_angle, :, :].clone()
        # Best rotations
        self.top_rotations = [torch.min(scores[i, :, :]).item() for i in range(scores.size(0))]

        return best_score, best_rotation, best_translation

    def forward(self, receptor, ligand):
        assert ligand.size(0) == receptor.size(0)
        assert ligand.ndimension() == 4
        assert ligand.ndimension() == receptor.ndimension()
        batch_size = receptor.size(0)

        with torch.no_grad():
            rec_repr = self.scoring_model.repr(receptor).tensor
            lig_repr = self.scoring_model.repr(ligand).tensor
            translations = self.dock_global(rec_repr, lig_repr)
            scores = self.score(translations)

            score, rotation, translation = self.get_conformation(scores)

        return rotation, translation
