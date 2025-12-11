# models/decision_scale.py
import torch
import torch.nn as nn

class DecisionScaler(nn.Module):
    """
    Improved 2-branch gate with balanced initialization:
      - Normalize features before gate to balance vision/text scales
      - Zero-init gate final layer for balanced start (alpha ≈ [0.5, 0.5])
      - Optionally use sigmoid(alpha_img) to align with single-α description while
        still leveraging both gate logits.
      - fuse: h = alpha_img * z_img + alpha_txt * e_txt
      - classify: logits = W h + b

    Add temperature tau (>1 softens the softmax) to avoid 0/1 collapse.
    """
    def __init__(self, dim: int, num_classes: int, hidden: int = 256, tau: float = 2.0, use_sigmoid: bool = True):
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes
        self.tau = float(tau)  # ← softmax temperature
        self.use_sigmoid = use_sigmoid

        # Normalize inputs to gate to balance vision/text feature scales
        self.gate_norm = nn.LayerNorm(dim * 2, eps=1e-6)
        
        # gate over two modalities
        gate_layers = [
            nn.Linear(dim * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2)  # logits for [img, txt]
        ]
        self.gate = nn.Sequential(*gate_layers)
        
        # Zero-initialize final gate layer for balanced start
        # This ensures softmax([0, 0]) ≈ [0.5, 0.5] at initialization
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

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
        x_norm = self.gate_norm(x)                   # Normalize to balance scales
        alpha_logits = self.gate(x_norm)             # [B, 2]
        if self.use_sigmoid:
            # Sigmoid on logit difference approximates a single-α gate
            alpha_diff = (alpha_logits[:, 0:1] - alpha_logits[:, 1:2]) / self.tau
            a_img = torch.sigmoid(alpha_diff)
            a_txt = 1.0 - a_img
            alpha = torch.cat([a_img, a_txt], dim=-1)
        else:
            alpha = torch.softmax(alpha_logits / self.tau, dim=-1)  # temperature-softened
            a_img = alpha[:, 0:1]
            a_txt = alpha[:, 1:2]

        h = a_img * z_img + a_txt * e_txt            # [B, D]
        logits = self.classifier(h)                  # [B, C]
        return logits, alpha, h
