import torch.nn as nn

class FiLMFusion(nn.Module):
    def __init__(self, dim=768, hidden=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, dim * 2)
        )
    def forward(self, v_feat, c_feat):
        gamma_beta = self.mlp(c_feat)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return (1.0 + gamma) * v_feat + beta
