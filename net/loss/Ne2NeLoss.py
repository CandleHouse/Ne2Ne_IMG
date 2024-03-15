import torch
import torch.nn as nn


class RegularizedLoss(nn.Module):
    def __init__(self, beta=1, gamma=2):
        super().__init__()
        self.beta = beta
        self.gamma = gamma

    def mseloss(self, image, target):
        return torch.mean((image - target)**2)

    def regloss(self, g1, g2, G1, G2):
        return torch.mean((g1-g2-G1+G2)**2)

    def forward(self, fg1, fg2, G1f, G2f, epoch_ratio):
        return   self.beta * self.mseloss(fg1, fg2) + \
                 self.gamma * epoch_ratio * self.regloss(fg1, fg2, G1f, G2f)
               
               
class FIRELoss(nn.Module):
    def __init__(self, beta=1, gamma=2):
        super().__init__()
        self.beta = beta
        self.gamma = gamma

    def mseloss(self, image, target):
        return torch.mean((image - target)**2)

    def regloss(self, g1, g2, G1, G2):
        return torch.mean((g1-g2-G1+G2)**2)

    def forward(self, fg1, g2, g3, g4,
                      G1f, G2f, G3f, G4f, epoch_ratio):
        return   self.beta * self.mseloss(3*fg1, g2+g3+g4) + \
                 self.gamma * epoch_ratio * self.regloss(3*fg1, g2+g3+g4, 3*G1f, G2f+G3f+G4f)
