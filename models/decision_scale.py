# models/decision_scale.py
import torch
import torch.nn as nn

class DecisionScaler(nn.Module):
    """
    Simple 2-branch gate:
      - compute alpha over {image,text} from both embeddings
      - fuse: h = alpha_img * z_img + alpha_txt * e_txt
      - classify: logits = W h + b

    Add temperature tau (>1 softens the softmax) to avoid 0/1 collapse.
    """
    def __init__(self, dim: int, num_classes: int, hidden: int = 256, tau: float = 2.0):
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes
        self.tau = float(tau)  # ← softmax temperature

        # gate over two modalities
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2)  # logits for [img, txt]
        )

        # classifier after gated fusion
        self.classifier = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, z_img: torch.Tensor, e_txt: torch.Tensor):
        """
        z_img: [B, D]  image embedding after fusion block
        e_txt: [B, D]  text/question embedding
        returns:
           logits: [B, C]
           alpha:  [B, 2] (img, txt)
           h:      [B, D] fused representation
        """
        x = torch.cat([z_img, e_txt], dim=-1)        # [B, 2D]
        alpha_logits = self.gate(x)                  # [B, 2]
        alpha = torch.softmax(alpha_logits / self.tau, dim=-1)  # temperature-softened
        a_img = alpha[:, 0:1]
        a_txt = alpha[:, 1:2]

        h = a_img * z_img + a_txt * e_txt            # [B, D]
        logits = self.classifier(h)                  # [B, C]
        return logits, alpha, h
